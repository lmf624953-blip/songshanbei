#!/usr/bin/env python3
"""用当前解码器比较时间窗 batch，并在计时前逐位检查。"""

import argparse
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import video_latent_num_frames
from songshan_h3.diffusers_attention import install_video_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or torch.cuda.device_count() != 2:
        parser.error("需要新输出路径和两张可见卡")
    if args.pairs < 1 or not args.batches or any(batch < 2 for batch in args.batches):
        parser.error("候选 batch 至少为 2")
    sys.path.insert(0, str(Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization"))
    from video_decode_parallel import decode_parallel

    install_video_attention()
    torch.set_num_threads(4)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(str(args.model / "vae"), dtype=torch.bfloat16,
                                                local_files_only=True).eval().requires_grad_(False).to("cuda:0")
    devices = tuple(torch.device("cuda", rank) for rank in range(2))
    ratio = vae.spatial_compression_ratio
    latent_frames = video_latent_num_frames(124, vae.config.clip_length, vae.tokens_chunk_size)
    seed = torch.Generator(device=devices[0]).manual_seed(43)
    latents = torch.randn((1, vae.config.latent_channels, latent_frames, 928 // ratio, 768 // ratio),
                          device=devices[0], dtype=vae.post_quant_conv.weight.dtype, generator=seed)
    report = dict(scope="两卡同实例固定 latent 独立解码", complete=False, records=[], comparisons={})
    reference = None

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def measure(batch, warmup):
        nonlocal reference
        for device in devices:
            torch.cuda.synchronize(device)
        started = perf_counter()
        output = decode_parallel(vae, latents, devices, clip_batch=batch)
        torch.cuda.synchronize(devices[0])
        elapsed = perf_counter() - started
        bits = output.view(torch.int16)
        if reference is None:
            reference = bits.clone()
        finite = bool(torch.isfinite(output).all())
        exact = torch.equal(bits, reference)
        shape = list(output.shape)
        row = dict(batch=batch, warmup=warmup, seconds=elapsed, finite=finite, exact=exact, shape=shape)
        if not exact:
            row["mismatches"] = int(torch.count_nonzero(bits != reference))
        report["records"].append(row)
        save()
        print(json.dumps(row), flush=True)
        assert shape == [1, 3, 124, 928, 768] and finite
        return elapsed, exact

    with torch.inference_mode():
        measure(1, True)
        for batch in dict.fromkeys(args.batches):
            _, exact = measure(batch, True)
            if not exact:
                report["comparisons"][batch] = dict(qualified=False, reason="完整解码输出未逐位一致")
                continue
            pairs = []
            for trial in range(args.pairs):
                order = (1, batch) if trial % 2 == 0 else (batch, 1)
                sample = {}
                for setting in order:
                    elapsed, exact = measure(setting, False)
                    assert exact, f"batch {setting} 重复数值不一致"
                    sample[setting] = elapsed
                pairs.append(dict(base=sample[1], candidate=sample[batch]))
            report["comparisons"][batch] = dict(qualified=True, pairs=pairs,
                baseline_median=median(row["base"] for row in pairs),
                candidate_median=median(row["candidate"] for row in pairs),
                paired_gain_percent=100 * median(1 - row["candidate"] / row["base"] for row in pairs))
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
