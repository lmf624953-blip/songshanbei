#!/usr/bin/env python3
"""在同一进程内成对比较 v3 和候选九齿 AOT。"""

import argparse
import json
import os
from pathlib import Path
import runpy
from statistics import median
import sys

import torch


def compare(baseline, candidate, sequence, iterations):
    device = "cuda:0"
    generator = torch.Generator(device=device).manual_seed(42)

    def packed(heads):
        return torch.randn((1, sequence, heads, 128), device=device,
                           dtype=torch.bfloat16, generator=generator)

    query, key = (packed(7).transpose(1, 2) for _ in range(2))
    value = packed(21)[:, :, -7:].transpose(1, 2)
    outputs = [torch.empty_like(query), torch.empty_like(query)]
    scale = 128 ** -0.5
    launches = (baseline, candidate)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = [[], []]

    def launch(index):
        launches[index](query, key, value, scale, outputs[index])

    for _ in range(2):
        launch(0)
        launch(1)
    torch.cuda.synchronize()
    for index in range(iterations):
        for variant in (index % 2, 1 - index % 2):
            start.record()
            launch(variant)
            end.record()
            end.synchronize()
            samples[variant].append(start.elapsed_time(end))

    def errors():
        difference = outputs[1].float() - outputs[0].float()
        return {
            "finite": all(bool(torch.isfinite(output).all()) for output in outputs),
            "exact": torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16)),
            "max_abs": float(difference.abs().max()),
            "rmse": float(difference.square().mean().sqrt()),
        }

    normal = errors()
    query.mul_(2)
    key.mul_(2)
    launch(0)
    launch(1)
    stress = errors()
    return {
        "sequence": sequence,
        "reference_ms": median(samples[0]),
        "candidate_ms": median(samples[1]),
        "paired_speedup_pct": 100 * median(1 - b / a for a, b in zip(*samples)),
        "samples_ms": samples,
        "normal": normal,
        "stress": stress,
    }



def native_loader():
    paths = [Path(directory) for directory in sys.path]
    if os.environ.get("H3_AUTOTUNE_CODE_ROOT"):
        paths.insert(0, Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization")
    for directory in paths:
        source = directory / "bench-attention.py"
        if source.is_file():
            return runpy.run_path(str(source))["native_launcher"]
    raise FileNotFoundError("找不到 bench-attention.py；先 source adapters/env/h3.sh 或将 optimization 加入 PYTHONPATH")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--iterations", required=True, type=int)
    args = parser.parse_args()
    if args.iterations < 3:
        parser.error("至少需要三次配对计时")
    load = native_loader()
    baseline, candidate = load(args.baseline), load(args.candidate)
    total = 0
    for sequence in (27570, 22016):
        row = compare(baseline, candidate, sequence, args.iterations)
        print(json.dumps(row), flush=True)
        total += row["candidate_ms"]
    print(f"H3_KERNEL_MEDIAN_SUM_MS={total:.6f}")


if __name__ == "__main__":
    main()
