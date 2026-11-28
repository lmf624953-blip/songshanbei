#!/usr/bin/env python3
"""比较原时序分派和空间 tile 分派，核对完整解码像素位模式。"""

import argparse
import json
import os
from pathlib import Path
from statistics import median
import sys
import time
from unittest.mock import patch

import torch

from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import video_latent_num_frames
from songshan_h3.diffusers_attention import install_video_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--height", type=int, default=928)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or not 2 <= args.devices <= torch.cuda.device_count() or args.pairs < 1:
        parser.error("输出路径需为空，且需确认可用设备数和配对数")
    sys.path.insert(0, str(Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization"))
    import video_decode_parallel
    from schedule import schedule_spatial

    install_video_attention()
    torch.set_num_threads(4)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(str(args.model / "vae"), dtype=torch.bfloat16,
                                                local_files_only=True).eval().requires_grad_(False).to("cuda:0")
    devices = [torch.device("cuda", rank) for rank in range(args.devices)]
    ratio = vae.spatial_compression_ratio
    generator = torch.Generator(device="cuda:0").manual_seed(43)
    frames = video_latent_num_frames(124, vae.config.clip_length, vae.tokens_chunk_size)
    latents = torch.randn((1, vae.config.latent_channels, frames, args.height // ratio, args.width // ratio),
                          device=devices[0], dtype=next(vae.post_quant_conv.parameters()).dtype,
                          generator=generator)
    records = []
    reference = None

    def measure(mode, warmup):
        nonlocal reference
        for device in devices:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        if mode == "spatial":
            with patch.object(video_decode_parallel, "schedule_clips", schedule_spatial):
                result = video_decode_parallel.decode_parallel(vae, latents, devices)
        else:
            result = video_decode_parallel.decode_parallel(vae, latents, devices[:2])
        torch.cuda.synchronize(devices[0])
        seconds = time.perf_counter() - started
        if reference is None:
            reference = result.detach().clone()
        exact = torch.equal(result.view(torch.int16), reference.view(torch.int16))
        row = dict(mode=mode, warmup=warmup, seconds=seconds, exact=exact,
                   finite=bool(torch.isfinite(result).all()), shape=list(result.shape),
                   peak_bytes=[torch.cuda.max_memory_allocated(device) for device in devices])
        if not exact:
            row["mismatches"] = int(torch.count_nonzero(result.view(torch.int16) != reference.view(torch.int16)))
        records.append(row)
        args.output.write_text(json.dumps(dict(records=records), ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if not exact or not row["finite"]:
            raise ValueError("完整解码位模式或有限值检查失败")
        return seconds

    with torch.inference_mode():
        measure("temporal", True)
        measure("spatial", True)
        paired = []
        for index in range(args.pairs):
            order = ("temporal", "spatial") if index % 2 == 0 else ("spatial", "temporal")
            paired.append({mode: measure(mode, False) for mode in order})
    summary = dict(complete=True, scope="独立原 VAE 窗口和空间 tile 解码，含复制与原序拼接",
                   height=args.height, width=args.width, input_latent_frames=frames,
                   spatial_devices=args.devices, records=records, pairs=paired,
                   baseline_median_seconds=median(pair["temporal"] for pair in paired),
                   candidate_median_seconds=median(pair["spatial"] for pair in paired),
                   paired_gain_percent=100 * median(1 - pair["spatial"] / pair["temporal"] for pair in paired))
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print({key: summary[key] for key in ("baseline_median_seconds", "candidate_median_seconds", "paired_gain_percent")},
          flush=True)


if __name__ == "__main__":
    main()
