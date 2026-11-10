#!/usr/bin/env python3
"""Compare isolated H3 components using real checkpoint weights, without video generation.

Reduced layer counts and synthetic inputs are used ONLY for numerical diagnosis.
This is never a competition inference or performance result. Production config,
checkpoint files and the comparison/evaluation script are left unchanged.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / 'artifacts/h3-transformer-diagnostic'
STAGES = {'projections': (0, 0), 'text_refiner': (0, 2), 'transformer_block': (1, 0),
          # A downstream attention block makes text-refiner errors observable
          # in the video/audio outputs; zero backbone layers cannot do so.
          'refiner_with_block': (1, 2), 'four_blocks': (4, 2)}


def checkpoint_path():
    manifest = json.loads((ROOT / 'artifacts/model-download.json').read_text())
    return Path(manifest['model_path']) / 'transformer'


def load_reference(stage):
    import torch
    from diffusers import MiniMaxH3Transformer3DModel
    from safetensors import safe_open
    path = checkpoint_path()
    config = dict(MiniMaxH3Transformer3DModel.load_config(str(path)))
    config['num_layers'], config['num_refiner_layers'] = STAGES[stage]
    with torch.device('meta'):
        model = MiniMaxH3Transformer3DModel.from_config(config)
    expected = model.state_dict()
    index_files = list(path.glob('*.safetensors.index.json'))
    if len(index_files) != 1:
        raise ValueError(f'Expected one checkpoint shard index, found {index_files}')
    weight_map = json.loads(index_files[0].read_text())['weight_map']
    missing = set(expected) - set(weight_map)
    if missing:
        raise ValueError(f'Checkpoint does not cover diagnostic model keys: {sorted(missing)}')
    state = {}
    for filename in sorted({weight_map[key] for key in expected}):
        with safe_open(str(path / filename), framework='pt', device='cpu') as handle:
            for key in expected:
                if weight_map[key] != filename:
                    continue
                tensor = handle.get_tensor(key)
                if tuple(tensor.shape) != tuple(expected[key].shape):
                    raise ValueError(f'Shape mismatch for {key}: {tensor.shape} vs {expected[key].shape}')
                keep_fp32 = any(part in key for part in model._keep_in_fp32_modules)
                state[key] = tensor.to(torch.float32 if keep_fp32 else torch.bfloat16)
    model.load_state_dict(state, strict=True, assign=True)
    # The nonpersistent frequency buffer is not in a checkpoint. Recreate it
    # on CPU after the meta-only shape construction, using the released class.
    model.rope = type(model.rope)(rope_freq_dim=config['rope_freq_dim'], rope_theta=config['rope_theta'])
    model.eval().requires_grad_(False)
    return model, state


def inputs(config, device='cpu'):
    import torch
    generator = torch.Generator().manual_seed(42)
    # Interleave modalities to test scatter/gather and row-index handling.
    count = 29
    tags = torch.arange(count, dtype=torch.int64) % 3
    video, text, audio = (torch.where(tags == tag)[0] for tag in (0, 1, 2))
    def random(length, width):
        return torch.randn(1, length, width, generator=generator)
    data = dict(hidden_states=random(len(video), config.in_channels * 4),
                audio_hidden_states=random(len(audio), config.audio_in_channels),
                encoder_hidden_states=random(len(text), config.text_dim),
                timestep=torch.tensor([0.0, 0.23, 0.81]),
                timestep_indices=torch.arange(count, dtype=torch.int64) % 3,
                token_tags=tags,
                position_ids=torch.randn(count, 3, generator=generator) * 4,
                video_indices=video, audio_indices=audio, text_indices=text)
    return {key: value.to(device) for key, value in data.items()}


def metrics(actual, reference):
    import torch
    a, b = actual.float().cpu(), reference.float().cpu()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    difference = a - b
    relative = float(difference.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-8))
    cosine = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    return dict(finite=finite, max_abs_error=float(difference.abs().max()),
                relative_rmse=relative, cosine_similarity=cosine,
                actual_rms=float(a.square().mean().sqrt()),
                reference_rms=float(b.square().mean().sqrt()),
                passed=finite and relative < 0.08 and cosine > 0.995)


def cpu_prepare():
    import torch
    from safetensors.torch import save_file
    torch.set_num_threads(8)
    report = dict(diagnostic_only=True, gpu_execution_checked=False, stages={})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        started = time.monotonic()
        print(f'CPU reference: {stage}; loading selected real weights only', flush=True)
        model, state = load_reference(stage)
        parameters = sum(t.numel() for t in state.values())
        del state
        model.float()
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            result = model(**inputs(model.config), return_dict=False)
        save_file({'video': result[0].cpu().contiguous(), 'audio': result[1].cpu().contiguous()},
                  str(OUTPUT / (stage + '-cpu-fp32.safetensors')))
        report['stages'][stage] = dict(loaded_parameters=parameters,
                                      elapsed_seconds=time.monotonic() - started,
                                      outputs_finite=all(bool(torch.isfinite(t).all()) for t in result))
        print(json.dumps(report['stages'][stage]), flush=True)
        del model, result
        gc.collect()
    (OUTPUT / 'cpu-reference.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if all(item['outputs_finite'] for item in report['stages'].values()) else 1


def worker(stage, tp):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from safetensors.torch import save_file, load_file
    import infinicore as ic
    from infinilm.infer_engine import InferEngine
    from infinilm.distributed import DistConfig
    from infinilm.modeling_utils import _remap_minimax_h3
    from infinilm.multimodal.minimax_h3 import (MiniMaxH3TransformerRunner,
                                              MiniMaxH3TransformerInputs)
    from songshan_h3.attention_bridge import install
    torch.set_num_threads(8)
    install()
    report = dict(stage=stage, tp=tp, diagnostic_only=True, passed=False)
    engine = None
    try:
        print(f'Real checkpoint stage={stage}, TP={tp}: constructing reference', flush=True)
        model, state = load_reference(stage)
        data = inputs(model.config, 'cuda:0')
        model.to('cuda:0')
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            reference = tuple(value.float().cpu() for value in model(**data, return_dict=False))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        blocks, refiners = STAGES[stage]
        print('Loading native engine with exactly the same component weights', flush=True)
        engine = InferEngine(str(checkpoint_path()), device=ic.device('cuda', 0),
                             distributed_config=DistConfig(tp_device_ids=list(range(tp))),
                             hf_config_overrides=dict(num_layers=blocks, num_refiner_layers=refiners))
        native_state = _remap_minimax_h3(dict(state), engine.hf_config)
        missing = set(engine.state_dict_keyname()) - set(native_state)
        unexpected = set(native_state) - set(engine.state_dict_keyname())
        if missing or unexpected:
            raise ValueError(f'Native key mismatch: missing={sorted(missing)}, extra={sorted(unexpected)}')
        engine.load_state_dict({name: ic.from_torch(value) for name, value in native_state.items()}, strict=True)
        engine.process_weights_after_loading()
        del native_state, state
        # Reuse the production Python facade and RoPE construction. Avoid
        # loading a second checkpoint through its normal constructor.
        runner = object.__new__(MiniMaxH3TransformerRunner)
        runner.engine = engine
        runner.rope_freq_dim = int(engine.hf_config['rope_freq_dim'])
        runner.rope_theta = float(engine.hf_config['rope_theta'])
        native_data = dict(data)
        native_data['video_hidden_states'] = native_data.pop('hidden_states')
        torch.cuda.synchronize()
        with torch.inference_mode():
            result = runner(MiniMaxH3TransformerInputs(**native_data))
        actual = result.sample.float().cpu(), result.audio_sample.float().cpu()
        # Also isolate missing stream handoff at the Python/native boundary.
        # The production facade creates its RoPE cache inside runner(), after
        # the caller's synchronization, so wait immediately before forward_raw.
        original_forward = engine.forward_raw
        def synchronized_forward(*args, **kwargs):
            torch.cuda.current_stream(0).synchronize()
            return original_forward(*args, **kwargs)
        engine.forward_raw = synchronized_forward
        try:
            with torch.inference_mode():
                synchronized_result = runner(MiniMaxH3TransformerInputs(**native_data))
        finally:
            del engine.forward_raw
        synchronized = (synchronized_result.sample.float().cpu(),
                        synchronized_result.audio_sample.float().cpu())
        save_file({'native_video': actual[0].contiguous(), 'native_audio': actual[1].contiguous(),
                   'synchronized_video': synchronized[0].contiguous(), 'synchronized_audio': synchronized[1].contiguous(),
                   'reference_video': reference[0].contiguous(), 'reference_audio': reference[1].contiguous()},
                  str(OUTPUT / f'{stage}-tp{tp}.safetensors'))
        report['video'] = metrics(actual[0], reference[0])
        report['audio'] = metrics(actual[1], reference[1])
        report['with_input_stream_synchronization'] = {
            'video': metrics(synchronized[0], reference[0]),
            'audio': metrics(synchronized[1], reference[1])}
        baseline_path = OUTPUT / (stage + '-cpu-fp32.safetensors')
        if baseline_path.is_file():
            baseline = load_file(str(baseline_path))
            report['gpu_reference_vs_cpu_fp32'] = {
                'video': metrics(reference[0], baseline['video']),
                'audio': metrics(reference[1], baseline['audio'])}
        report['passed'] = report['video']['passed'] and report['audio']['passed']
        if 'gpu_reference_vs_cpu_fp32' in report:
            report['passed'] = report['passed'] and all(
                value['passed'] for value in report['gpu_reference_vs_cpu_fp32'].values())
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report['passed'] else 1
    except Exception:
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
        return 1
    finally:
        if engine is not None:
            engine.close()
        (OUTPUT / f'{stage}-tp{tp}.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-cpu', action='store_true', help='Create CPU FP32 real-weight component references')
    parser.add_argument('--stage', choices=tuple(STAGES), help=argparse.SUPPRESS)
    parser.add_argument('--tp', type=int, choices=(1, 8), help=argparse.SUPPRESS)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.prepare_cpu:
        return cpu_prepare()
    if args.stage:
        return worker(args.stage, args.tp)
    import torch
    if torch.cuda.device_count() < 8:
        raise SystemExit('Run from the ordinary terminal exposing all 8 devices')
    report = dict(diagnostic_only=True, inference_passed=False, cases=[])
    for stage in STAGES:
        for tp in (1, 8):
            path = ROOT / 'logs' / f'h3-parity-{stage}-tp{tp}.log'
            print(f'[{stage}, TP={tp}] Isolated real-weight component comparison', flush=True)
            with path.open('w') as log:
                try:
                    process = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                                              '--stage', stage, '--tp', str(tp)],
                                             cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                             env={**os.environ, 'HF_HUB_OFFLINE': '1', 'PYTHONFAULTHANDLER': '1'},
                                             timeout=360)
                    status = process.returncode
                except subprocess.TimeoutExpired:
                    status = 'timeout'
            report['cases'].append(dict(stage=stage, tp=tp, returncode=status, log=str(path)))
            print(f'  exit={status}; {path}', flush=True)
            print('\n'.join(path.read_text(errors='replace').splitlines()[-22:]), flush=True)
            report['passed'] = all(case['returncode'] == 0 for case in report['cases'])
            (OUTPUT / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
