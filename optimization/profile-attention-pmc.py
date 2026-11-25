#!/usr/bin/env python3
"""Launch only the production NineToothed attention binary for hardware counters."""

import argparse
import json
from pathlib import Path
import runpy
from statistics import median

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("--sequence", nargs="+", type=int, default=(27570, 22016))
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1 or any(length < 1 for length in args.sequence):
        parser.error("Sequence lengths and the number of iterations must be positive")

    build = args.build.resolve(strict=True)
    functions = runpy.run_path(str(Path(__file__).with_name("bench-attention.py")))
    launch = functions["native_launcher"](build)
    config = json.loads((build / "dense-attention-aot-build.json").read_text())
    generator = torch.Generator(device="cuda:0").manual_seed(42)
    for sequence in args.sequence:
        def packed(heads):
            return torch.randn((1, sequence, heads, 128), generator=generator,
                               device="cuda:0", dtype=torch.bfloat16)

        query = packed(7).transpose(1, 2)
        key = packed(7).transpose(1, 2)
        value = packed(21)[:, :, -7:].transpose(1, 2)
        output = torch.empty_like(query)
        tensors = (query, key, value, output)
        print(json.dumps({"sequence": sequence, "shape": list(query.shape),
                          "row_strides_in_elements": [tensor.stride(2) for tensor in tensors],
                          "iterations": args.iterations, "hsaco_sha256": config["hsaco_sha256"]}), flush=True)
        samples = functions["timings"](lambda: launch(query, key, value, 128 ** -0.5, output), args.iterations)
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError(f"Nonfinite attention output for {sequence} tokens")
        print(json.dumps({"sequence": sequence, "samples_ms": samples,
                          "median_ms": median(samples), "finite": True}), flush=True)


if __name__ == "__main__":
    main()
