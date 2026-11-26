#!/usr/bin/env python3
"""衡量 FP16 PV 的代价和相对于独立 FP32 计算的误差。"""

import argparse
import json
from pathlib import Path
import runpy

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    helper = runpy.run_path(str(Path(__file__).resolve().parents[2] / "adapters/bench-h3-attention.py"))
    load = helper["native_loader"]()
    baseline, candidate = load(args.reference), load(args.candidate)
    performance = [helper["compare"](baseline, candidate, n, 11) for n in (27570, 22016)]
    generator = torch.Generator(device="cuda").manual_seed(915)
    quality = []
    for n in (513, 22016, 27570):
        q, k, v = [
            torch.randn((1, n, 7, 128), device="cuda", dtype=torch.bfloat16, generator=generator).transpose(1, 2)
            for _ in range(3)
        ]
        subset = torch.linspace(0, n - 1, 64, device="cuda").long()
        attention_scale = 128**-0.5
        k_t = k.transpose(-1, -2).float()
        for stress in (False, True):
            if stress:
                q.mul_(2)
                k_t.mul_(2)
                k.mul_(2)
            scores = (q[:, :, subset, :].float() @ k_t) * attention_scale
            expected = scores.softmax(-1) @ v.float()
            for name, launch in (("bf16_pv", baseline), ("fp16_pv", candidate)):
                result = torch.full_like(q, float("nan"))
                launch(q, k, v, attention_scale, result)
                difference = result[:, :, subset, :].float() - expected
                quality.append(dict(
                    sequence=n, stress=stress, variant=name, finite=bool(torch.isfinite(result).all()),
                    rmse=float(difference.square().mean().sqrt()), max_abs=float(difference.abs().max())))
    report = dict(performance=performance, quality_samples=quality, query_rows_per_head=64,
                  quality_reference="PyTorch FP32 QK、softmax、PV，抽样 64 个查询，各自使用完整序列 K/V")
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
