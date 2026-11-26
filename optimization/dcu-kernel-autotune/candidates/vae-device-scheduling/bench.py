#!/usr/bin/env python3
"""复用原 VAE 窗口、batch=1 与混合权重，对照两卡/四卡调度；不缓存输出。"""
import argparse,json,os,sys,time
from pathlib import Path
from statistics import median
import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import video_latent_num_frames
from songshan_h3.diffusers_attention import install_video_attention


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--model',type=Path,required=True);cli.add_argument('--output',type=Path,required=True);a=cli.parse_args()
    sys.path.insert(0,str(Path(os.environ['H3_AUTOTUNE_CODE_ROOT'])/'optimization'))
    from video_decode_parallel import decode_parallel
    install_video_attention();torch.set_num_threads(4)
    vae=AutoencoderKLMiniMaxH3.from_pretrained(str(a.model/'vae'),dtype=torch.bfloat16,local_files_only=True).eval().requires_grad_(False).to('cuda:0')
    devices=[torch.device('cuda',i) for i in range(4)];gen=torch.Generator(device='cuda:0').manual_seed(43);records=[]
    with torch.inference_mode():
        for height,width in [(928,768),(768,768)]:
            spatial=vae.spatial_compression_ratio
            latents=torch.randn((1,vae.config.latent_channels,video_latent_num_frames(124,vae.config.clip_length,vae.tokens_chunk_size),height//spatial,width//spatial),device=devices[0],dtype=next(vae.post_quant_conv.parameters()).dtype,generator=gen)
            expected=None
            for trial,count in enumerate([2,4,2,4,4,2]):
                for device in devices:torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
                started=time.perf_counter();video=decode_parallel(vae,latents,devices[:count],clip_batch=1)
                for device in devices:torch.cuda.synchronize(device)
                elapsed=time.perf_counter()-started
                if expected is None:expected=video.detach().clone()
                exact=torch.equal(expected.view(torch.int16),video.view(torch.int16))
                record=dict(height=height,width=width,trial=trial,devices=count,warmup=trial<2,seconds=elapsed,exact=exact,finite=bool(torch.isfinite(video).all()),shape=list(video.shape),torch_peak_allocated_bytes={str(dev):torch.cuda.max_memory_allocated(dev) for dev in devices})
                print(json.dumps(record),flush=True);records.append(record);del video
                if not exact:raise RuntimeError('设备调度改变输出；不继续性能比较')
            del expected,latents
    summary=[]
    for h,w in [(928,768),(768,768)]:
        times={n:[r['seconds'] for r in records if (r['height'],r['width'],r['devices'],r['warmup'])==(h,w,n,False)] for n in [2,4]}
        b,c=map(median,(times[2],times[4]));summary.append(dict(height=h,width=w,baseline_seconds=b,candidate_seconds=c,reduction_percent=100*(1-c/b)))
    a.output.write_text(json.dumps(dict(scope='独立 VAE 原窗口重算，排除每设备首次预热；不作为端到端收益',records=records,summary=summary),ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
