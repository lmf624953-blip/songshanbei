#!/usr/bin/env python3
"""在独立 PMC 会话中按固定顺序各采一条参考与 D64 注意力。"""

import argparse
import ctypes
import json
from pathlib import Path
import runpy
from time import sleep

import torch
from songshan_h3.attention_kernel import kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query_block", type=int)
    parser.add_argument("warps", type=int)
    parser.add_argument("validation", type=Path)
    args = parser.parse_args()
    build = runpy.run_path(str(Path(__file__).with_name("bench-head64.py")))["build"]
    candidate = build(args.query_block, args.warps)
    reference = kernel()
    tokens = 1797
    generator = torch.Generator(device="cuda").manual_seed(20261128 + tokens)
    inputs = [torch.randn((1, tokens, 32, 64), device="cuda", dtype=torch.bfloat16,
                          generator=generator).transpose(1, 2) for _ in range(3)]
    outputs = [torch.empty_like(inputs[0]) for _ in range(2)]
    launchers = (reference, candidate)

    def launch_both():
        for launch, output in zip(launchers, outputs, strict=True):
            launch(*inputs, False, 64 ** -0.5, output)
            torch.cuda.synchronize()

    launch_both()
    assert torch.equal(*(output.view(torch.int16) for output in outputs))
    hip = ctypes.CDLL("libamdhip64.so")
    assert hip.hipProfilerStart() == 0
    launch_both()
    sleep(2)
    assert hip.hipProfilerStop() == 0
    assert torch.equal(*(output.view(torch.int16) for output in outputs))
    report = dict(exact=True, tokens=tokens, heads=32, head_dim=64,
                  order=["reference", f"q{args.query_block}-w{args.warps}-d64"])
    with args.validation.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
