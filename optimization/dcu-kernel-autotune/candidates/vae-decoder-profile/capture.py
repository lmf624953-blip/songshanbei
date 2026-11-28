#!/usr/bin/env python3
"""预热后采集一次现用两卡视频 VAE 解码，分段计时用于诊断。"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
from time import perf_counter, sleep, time_ns
from unittest.mock import patch

import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import video_latent_num_frames
from songshan_h3.diffusers_attention import install_video_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() or torch.cuda.device_count() != 2:
        parser.error("需要两张可见卡与新输出路径")
    sys.path.insert(0, str(Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization"))
    import video_decode_parallel as decode

    install_video_attention()
    torch.set_num_threads(4)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(str(args.model / "vae"), dtype=torch.bfloat16,
                                                local_files_only=True).eval().requires_grad_(False).to("cuda:0")
    primary, secondary = devices = tuple(torch.device("cuda", index) for index in range(2))
    ratio = vae.spatial_compression_ratio
    frames = video_latent_num_frames(124, vae.config.clip_length, vae.tokens_chunk_size)
    latents = torch.randn((1, vae.config.latent_channels, frames, 928 // ratio, 768 // ratio),
                          device=primary, dtype=vae.post_quant_conv.weight.dtype,
                          generator=torch.Generator(device=primary).manual_seed(43))
    synchronize = lambda: [torch.cuda.synchronize(device) for device in devices]
    phases = {}
    replicas = decode.decoder_replicas
    schedule = decode.schedule_clips

    def timed_replicas(*call_args):
        start = perf_counter()
        result = replicas(*call_args)
        synchronize()
        phases["replica_seconds"] = perf_counter() - start
        return result

    def timed_schedule(*call_args):
        start = perf_counter()
        result = schedule(*call_args)
        synchronize()
        phases["schedule_seconds"] = perf_counter() - start
        phases["schedule_end_epoch_ns"] = time_ns()
        phases["clips"] = [list(clip.shape) for clip in result]
        return result

    hip = ctypes.CDLL("libamdhip64.so")
    with torch.inference_mode():
        output = decode.decode_parallel(vae, latents, devices)
        synchronize()
        reference = output.view(torch.int16).clone()
        del output
        synchronize()
        print("两卡预热完成，开始独立采样", flush=True)
        assert hip.hipProfilerStart() == 0
        with patch.object(decode, "decoder_replicas", timed_replicas), patch.object(decode, "schedule_clips", timed_schedule):
            begin = time_ns()
            print("[hipprof.begin]", json.dumps(dict(name="vae.decode", epoch_ns=begin)), flush=True)
            output = decode.decode_parallel(vae, latents, devices)
            synchronize()
            end = time_ns()
            print("[hipprof.end]", json.dumps(dict(name="vae.decode", epoch_ns=end)), flush=True)
        sleep(2)
        assert hip.hipProfilerStop() == 0
        exact = torch.equal(output.view(torch.int16), reference)
        finite = bool(torch.isfinite(output).all())
        assert exact and finite and list(output.shape) == [1, 3, 124, 928, 768]
    report = dict(complete=True, exact=exact, finite=finite,
                  scope="独立两卡 VAE 诊断，固定种子 latent，含额外分段同步",
                  shape=list(output.shape), latent_shape=list(latents.shape), begin_ns=begin, end_ns=end,
                  total_instrumented_seconds=(end - begin) / 1e9, phases=phases,
                  dtypes={"q": str(vae.decoder.transformer_blocks[0].attn.to_q.weight.dtype),
                          "conv": str(vae.post_quant_conv.weight.dtype)})
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
