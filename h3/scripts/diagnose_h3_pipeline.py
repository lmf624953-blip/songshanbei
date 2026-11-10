#!/usr/bin/env python3
"""Trace the real public request and replay its model boundaries for diagnosis.

Capture stops after one real denoiser call unless --complete is requested.
Replays use released weights and identical saved inputs, never scored outputs.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'artifacts/h3-pipeline-diagnostic'


def model_path():
    return Path(json.loads((ROOT / 'artifacts/model-download.json').read_text())['model_path'])


def wait_for_capture(filename):
    deadline = time.monotonic() + 1800
    while not (OUT / filename).is_file():
        if time.monotonic() > deadline:
            raise TimeoutError('Capture did not produce '+filename)
        print('Waiting for real request capture: '+filename, flush=True)
        time.sleep(30)


def stats(t):
    import torch
    x = t.detach().float()
    return dict(shape=list(t.shape), dtype=str(t.dtype), finite=bool(torch.isfinite(x).all()),
                rms=float(x.square().mean().sqrt()), mean=float(x.mean()),
                minimum=float(x.min()), maximum=float(x.max()))


def compare(a, b):
    import torch
    # FP64 reductions keep cosine within roundoff even for millions of rows.
    x, y = a.detach().cpu().double(), b.detach().cpu().double()
    actual_stats, reference_stats = stats(x), stats(y)
    actual_stats['dtype'], reference_stats['dtype'] = str(a.dtype), str(b.dtype)
    return dict(relative_rmse=float((x-y).square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-8)),
                cosine_similarity=float(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0)),
                max_abs_error=float((x-y).abs().max()), actual=actual_stats, reference=reference_stats,
                reduction_dtype='float64')


def save(name, values):
    from safetensors.torch import save_file
    save_file({k: v.detach().cpu().contiguous() for k, v in values.items()}, str(OUT / (name+'.safetensors')))


def load_async_reference(kind):
    """Batch checkpoint transfers per shard; keep CPU sources alive until done.

    This changes loading only. The released model classes, weights and forward
    computations remain the numerical reference.
    """
    import torch
    from accelerate import init_empty_weights, dispatch_model
    from safetensors import safe_open
    from diffusers import MiniMaxH3Transformer3DModel
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    directory=model_path()/('transformer' if kind=='transformer' else 'text_encoder')
    with init_empty_weights(include_buffers=False):
        if kind=='transformer':
            model=MiniMaxH3Transformer3DModel.from_config(MiniMaxH3Transformer3DModel.load_config(str(directory)))
        else:
            config=AutoConfig.from_pretrained(str(directory),local_files_only=True)
            model=Qwen3VLForConditionalGeneration._from_config(config,torch_dtype=torch.bfloat16,
                                                              attn_implementation='eager')
    # Do not map a parent container as well as its sharded children: accelerate
    # would first move the entire container onto that parent's device.
    if kind=='transformer':
        device_map={name:0 for name,_ in model.named_children() if name!='transformer_blocks'}
    else:
        device_map={'model.visual':0, 'model.language_model.embed_tokens':0,
                    'model.language_model.rotary_emb':0,
                    'model.language_model.norm':7, 'lm_head':7}
    layers=model.transformer_blocks if kind=='transformer' else model.model.language_model.layers
    prefix='transformer_blocks' if kind=='transformer' else 'model.language_model.layers'
    for i in range(len(layers)):
        device_map[f'{prefix}.{i}']=min(7,i*8//len(layers))
    expected=model.state_dict()
    remaining=set(expected)
    indices=list(directory.glob('*.safetensors.index.json'))
    weight_map=json.loads(indices[0].read_text())['weight_map']
    fp32=getattr(model,'_keep_in_fp32_modules',[]) if kind=='transformer' else []
    for shard in sorted(set(weight_map.values())):
        print('Async reference shard:',shard,flush=True)
        sources=[]
        with safe_open(str(directory/shard),framework='pt',device='cpu') as handle:
            for key in handle.keys():
                if key not in remaining:
                    raise ValueError('Unexpected/duplicate checkpoint key: '+key)
                # Pageable HIP transfers were ~240x slower in the local probe.
                # Pinned sources are retained until the shard's transfers finish.
                value=handle.get_tensor(key).pin_memory()
                if tuple(value.shape)!=tuple(expected[key].shape):
                    raise ValueError('Reference shape mismatch: '+key)
                sources.append(value)
                target=0
                parts=key.split('.')
                for n in range(len(parts)-1,0,-1):
                    parent='.'.join(parts[:n])
                    if parent in device_map:
                        target=device_map[parent]
                        break
                dtype=torch.float32 if any(part in key for part in fp32) else torch.bfloat16
                tensor=value.to(device=torch.device('cuda',target),dtype=dtype,non_blocking=True)
                owner,name=key.rsplit('.',1) if '.' in key else ('',key)
                module=model.get_submodule(owner) if owner else model
                if name in module._parameters:
                    module._parameters[name]=torch.nn.Parameter(tensor,requires_grad=False)
                else:
                    module._buffers[name]=tensor
                remaining.remove(key)
            for device in range(8):
                torch.cuda.synchronize(device)
        del sources
    if remaining:
        raise ValueError('Missing reference parameters: '+str(sorted(remaining)))
    model.eval().requires_grad_(False)
    return dispatch_model(model,device_map=device_map,force_hooks=True)


def capture(complete):
    import torch
    from infinilm.multimodal.minimax_h3 import MiniMaxH3ConditionEncoder, MiniMaxH3TransformerRunner, MiniMaxH3DiffusersAdapter
    from songshan_h3.attention_bridge import install
    from songshan_h3.diffusers_attention import install_video_attention
    from diffusers import AutoencoderKLMiniMaxH3
    install()
    install_video_attention()
    report = dict(diagnostic_only=True, complete_requested=complete, steps=[])
    engines = []

    class CaptureFinished(Exception):
        pass

    original_encode = MiniMaxH3ConditionEncoder.encode
    def encode(self, inputs):
        engines.append(self)
        save('condition-input', {k:v for k,v in inputs.items() if isinstance(v, torch.Tensor)})
        result = original_encode(self, inputs)
        save('condition-native', {'hidden_states': result})
        report['condition_native'] = stats(result)
        print('[trace] condition:', report['condition_native'], flush=True)
        return result
    MiniMaxH3ConditionEncoder.encode = encode

    original_forward = MiniMaxH3DiffusersAdapter.forward
    # Preserve the explicit signature: the denoise block introspects it.
    import functools
    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        index = len(report['steps'])
        if index == 0:
            engines.append(self)
            save('transformer-input', {k:v for k,v in kwargs.items() if isinstance(v, torch.Tensor)})
        result = original_forward(self, *args, **kwargs)
        video, audio = result if isinstance(result, tuple) else (result.sample, result.audio_sample)
        report['steps'].append(dict(index=index, video=stats(video), audio=stats(audio),
                                    input_video=stats(kwargs['hidden_states']),
                                    timestep=kwargs['timestep'].detach().cpu().tolist()))
        print('[trace] step:', report['steps'][-1], flush=True)
        if index == 0:
            save('transformer-native', {'video':video, 'audio':audio})
            # Test the actual large-input stream handoff on exactly these inputs.
            engine = self.runner.engine
            original_raw = engine.forward_raw
            def synchronized(*a, **kw):
                torch.cuda.synchronize(0)
                return original_raw(*a, **kw)
            engine.forward_raw = synchronized
            try:
                again = original_forward(self, *args, **kwargs)
            finally:
                del engine.forward_raw
            v2, a2 = again if isinstance(again, tuple) else (again.sample, again.audio_sample)
            report['stream_handoff'] = dict(video=compare(v2,video), audio=compare(a2,audio))
            save('transformer-native-synchronized', {'video':v2,'audio':a2})
            if not complete:
                raise CaptureFinished()
        (OUT / 'capture.json').write_text(json.dumps(report, indent=2)+'\n')
        return result
    MiniMaxH3DiffusersAdapter.forward = forward

    original_decode = AutoencoderKLMiniMaxH3.decode
    @functools.wraps(original_decode)
    def decode(self, latents, *args, **kwargs):
        save('vae-input', {'latents':latents})
        report['vae_input'] = stats(latents)
        return original_decode(self, latents, *args, **kwargs)
    AutoencoderKLMiniMaxH3.decode = decode

    previous = json.loads((ROOT / 'outputs/public_fl2va/20261103_001320_702524_infinilm/run.json').read_text())
    command = previous['command']
    entry = ROOT / 'sources/InfiniLM-build/examples/minimax_h3.py'
    arguments = command[command.index(str(entry))+1:]
    arguments = [('--output='+str(OUT/'video.mp4')) if x.startswith('--output=') else x for x in arguments]
    sys.argv = [str(entry), *arguments]
    try:
        runpy.run_path(str(entry), run_name='__main__')
        report['generation_completed'] = True
    except CaptureFinished:
        report['stopped_after_first_prediction'] = True
        print('[trace] Captured real inputs and first denoiser output; diagnostic stop.', flush=True)
    finally:
        for engine in reversed(engines):
            engine.close()
        (OUT/'capture.json').write_text(json.dumps(report, indent=2)+'\n')


def reference_transformer():
    import torch
    from safetensors.torch import load_file
    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.models.transformers import transformer_minimax_h3 as module
    from songshan_h3.diffusers_attention import video_attention
    module.dispatch_attention_fn = video_attention
    print('Loading released full Diffusers transformer across eight devices', flush=True)
    model = load_async_reference('transformer')
    model.eval().requires_grad_(False)
    # Align head outputs with the packed indices even when accelerate moves them.
    from accelerate.hooks import remove_hook_from_module
    input_device = next(model.context_embedder.parameters()).device
    for name in ('proj_out','audio_proj_out'):
        getattr(model,name).register_forward_hook(lambda _m,_a,o:o.to(input_device))
    wait_for_capture('transformer-native.safetensors')
    data={k:v.to(input_device) for k,v in load_file(str(OUT/'transformer-input.safetensors')).items()}
    print('Replaying actual packed input', {k:list(v.shape) for k,v in data.items()}, flush=True)
    with torch.inference_mode():
        result=model(**data,return_dict=False)
    save('transformer-reference',dict(zip(('video','audio'),result)))
    actual=load_file(str(OUT/'transformer-native.safetensors'))
    report={k:compare(a, b) for k,a,b in zip(('video','audio'),(actual['video'],actual['audio']),result)}
    (OUT/'transformer-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


def reference_condition():
    import torch
    from safetensors.torch import load_file
    from transformers import Qwen3VLForConditionalGeneration
    print('Loading released Qwen3-VL across eight devices; eager attention reference',flush=True)
    model=load_async_reference('condition')
    model.eval().requires_grad_(False)
    wait_for_capture('condition-native.safetensors')
    data={k:v.to(model.device) for k,v in load_file(str(OUT/'condition-input.safetensors')).items()}
    # Keep only the requested layer, avoiding all full-sequence hidden states.
    captured=[]
    def hook(_m,_a,output):
        hidden=output[0] if isinstance(output,tuple) else output
        captured.append(hidden.detach().cpu())
    model.model.language_model.layers[49].register_forward_hook(hook)
    with torch.inference_mode():
        model.model(**data)
    save('condition-reference',{'hidden_states':captured[0]})
    actual=load_file(str(OUT/'condition-native.safetensors'))['hidden_states']
    report=compare(actual,captured[0])
    (OUT/'condition-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


def native_replay():
    import torch
    from safetensors.torch import load_file
    from infinilm.multimodal.minimax_h3 import (MiniMaxH3TransformerRunner, MiniMaxH3TransformerInputs,
                                              minimax_h3_rotary_cache)
    import infinicore
    from songshan_h3.attention_bridge import install
    install()
    runner=MiniMaxH3TransformerRunner(model_path()/'transformer',tp_device_ids=list(range(8)))
    data={k:v.to('cuda:0') for k,v in load_file(str(OUT/'transformer-input.safetensors')).items()}
    original_data={k:v.detach().cpu().clone() for k,v in data.items()}
    data['video_hidden_states']=data.pop('hidden_states')
    original_raw=runner.engine.forward_raw
    mode='none'
    def forward(*a,**kw):
        if mode in ('input','both'):
            torch.cuda.current_stream(0).synchronize()
        return original_raw(*a,**kw)
    def copy(tensor,device):
        if mode in ('output','both'):
            torch.cuda.synchronize(device)
        # Explicitly retain the old copy-after-enqueue behavior for the
        # ablation, even when the installed production facade has been fixed.
        result=torch.empty(tuple(tensor.shape),dtype=infinicore.utils.to_torch_dtype(tensor.dtype),device=device)
        infinicore.from_torch(result).copy_(tensor)
        infinicore.sync_device()
        return result
    results=[]
    report=[]
    try:
        for i,mode in enumerate(('none','input','output','both','both','both','none','both')):
            print('Native replay',i,mode,flush=True)
            with torch.inference_mode():
                torch_inputs=dict(data)
                torch_inputs['rotary_cos_sin_cache']=minimax_h3_rotary_cache(
                    data['position_ids'],rope_freq_dim=runner.rope_freq_dim,
                    rope_theta=runner.rope_theta,dtype=infinicore.utils.to_torch_dtype(runner.engine.dtype))
                torch_inputs['position_ids']=torch.arange(data['position_ids'].shape[0],dtype=torch.int64,device='cuda:0')
                native_inputs={k:runner._to_infinicore(v) for k,v in torch_inputs.items()}
                output=forward(None,**native_inputs)
                video=copy(output['logits'],data['video_hidden_states'].device)
                audio=copy(output['hidden_states'],data['video_hidden_states'].device)
            values={'video':video.detach().cpu(),'audio':audio.detach().cpu()}
            save(f'replay-{i}-{mode}',values)
            results.append(values)
            item=dict(index=i,synchronization=mode,video=stats(values['video']),audio=stats(values['audio']))
            if i:
                item['vs_previous']={k:compare(values[k],results[i-1][k]) for k in values}
            report.append(item)
            (OUT/'native-replay.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(item),flush=True)
        mutations={}
        for k,v in original_data.items():
            key='video_hidden_states' if k=='hidden_states' else k
            mutations[k]=bool(torch.equal(data[key].detach().cpu(),v))
        (OUT/'native-input-unchanged.json').write_text(json.dumps(mutations,indent=2)+'\n')
    finally:
        runner.close()


def native_condition_replay():
    import torch
    from safetensors.torch import load_file
    from infinilm.multimodal.minimax_h3 import MiniMaxH3ConditionEncoder
    from songshan_h3.attention_bridge import install
    install()
    model=MiniMaxH3ConditionEncoder(model_path()/'text_encoder',tp_device_ids=list(range(8)))
    data={k:v.to('cuda:0') for k,v in load_file(str(OUT/'condition-input.safetensors')).items()}
    try:
        with torch.inference_mode():
            first=model.encode(data).detach().cpu()
            second=model.encode(data).detach().cpu()
        save('condition-native-fixed',{'hidden_states':first})
        reference=load_file(str(OUT/'condition-reference.safetensors'))['hidden_states']
        old=load_file(str(OUT/'condition-native.safetensors'))['hidden_states']
        report=dict(fixed_vs_reference=compare(first,reference),
                    fixed_vs_original=compare(first,old),repeat=compare(second,first))
        (OUT/'condition-fixed-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    finally:
        model.close()


def handoff_race():
    """Make the producer/consumer race deterministic with a delayed producer."""
    import torch
    import infinicore as ic
    from infinilm.infer_engine import InferEngine
    from infinilm.distributed import DistConfig
    from infinilm.modeling_utils import _remap_minimax_h3
    from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner, minimax_h3_rotary_cache
    from diagnose_h3_transformer import load_reference, inputs
    from songshan_h3.attention_bridge import install
    install()
    model,state=load_reference('transformer_block')
    data=inputs(model.config,'cuda:0')
    del model
    engine=InferEngine(str(model_path()/'transformer'),device=ic.device('cuda',0),
                       distributed_config=DistConfig(tp_device_ids=list(range(8))),
                       hf_config_overrides=dict(num_layers=1,num_refiner_layers=0))
    native_state=_remap_minimax_h3(dict(state),engine.hf_config)
    engine.load_state_dict({k:ic.from_torch(v) for k,v in native_state.items()},strict=True)
    engine.process_weights_after_loading()
    del state,native_state
    data['video_hidden_states']=data.pop('hidden_states')
    results={}
    try:
        for label,synchronize in [('missing_dependency',False),('ordered_handoff',True)]:
            torch.cuda.synchronize(0)
            # Only the diagnostic producer is delayed; the model and weights
            # are identical. Native workers use independent streams.
            torch.cuda._sleep(500_000_000)
            tensors=dict(data)
            tensors['rotary_cos_sin_cache']=minimax_h3_rotary_cache(data['position_ids'])
            tensors['position_ids']=torch.arange(data['position_ids'].shape[0],device='cuda:0',dtype=torch.int64)
            native={k:ic.from_torch(v.contiguous()) for k,v in tensors.items()}
            if synchronize:
                torch.cuda.current_stream(0).synchronize()
            output=engine.forward_raw(None,**native)
            results[label]={k:MiniMaxH3TransformerRunner._to_torch(output[name],torch.device('cuda:0')).cpu()
                            for k,name in [('video','logits'),('audio','hidden_states')]}
            save('producer-delay-'+label,results[label])
        report={k:compare(results['missing_dependency'][k],results['ordered_handoff'][k]) for k in ('video','audio')}
        report['diagnostic_only']=True
        report['producer_delay_cycles']=500_000_000
        (OUT/'producer-delay-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    finally:
        engine.close()


def main():
    import torch
    torch.set_num_threads(8)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('capture','reference-transformer','reference-condition','native-replay',
                                       'native-condition-replay','handoff-race'))
    parser.add_argument('--complete',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=='capture': capture(args.complete)
    elif args.mode=='reference-transformer': reference_transformer()
    elif args.mode=='native-replay': native_replay()
    elif args.mode=='native-condition-replay': native_condition_replay()
    elif args.mode=='handoff-race': handoff_race()
    else: reference_condition()


if __name__=='__main__':
    main()
