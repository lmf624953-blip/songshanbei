#!/usr/bin/env python3
"""Compare packed attention on BF16 exceptional inputs and block boundaries."""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

import torch


CASES = {
    "min_bf16_subnormal": 0x0001,
    "negative_zero": 0x8000,
    "max_finite": 0x7F7F,
    "inf": 0x7F80,
    "quiet_nan": 0x7FC1,
    "signaling_nan": 0x7F81,
}


def compare(launchers, length, name, word, generator):
    def packed(heads):
        return torch.randn((1, length, heads, 128), device="cuda", dtype=torch.bfloat16,
                           generator=generator)

    q, k, v = packed(7).transpose(1, 2), packed(7).transpose(1, 2), packed(21)[:, :, -7:].transpose(1, 2)
    i16 = word if word < 32768 else word - 65536
    for value in (q, k):
        value.view(torch.int16)[0, 0, ::13, ::7] = i16
    outputs = [torch.empty((1, length, 7, 128), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
               for _ in launchers]
    for launch, output in zip(launchers, outputs, strict=True):
        launch(q, k, v, 128**-0.5, output)
    mismatches = int(torch.count_nonzero(outputs[0].view(torch.int16) != outputs[1].view(torch.int16)))
    print(json.dumps(dict(sequence=length, case=name, mismatches=mismatches)), flush=True)
    if mismatches:
        raise ValueError(f"Different BF16 output for length {length}, {name}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("candidate", type=Path)
    args = cli.parse_args()
    code = Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization"
    sys.path.insert(0, str(code))
    load = runpy.run_path(str(code / "bench-attention.py"))["native_launcher"]
    launchers = load(args.reference), load(args.candidate)
    generator = torch.Generator(device="cuda").manual_seed(42)
    for length in (31, 64, 257, 1023):
        for name, word in CASES.items():
            compare(launchers, length, name, word, generator)


if __name__ == "__main__":
    main()
