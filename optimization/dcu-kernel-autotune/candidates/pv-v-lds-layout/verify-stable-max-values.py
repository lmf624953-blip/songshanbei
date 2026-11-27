#!/usr/bin/env python3
"""为恒等重缩放候选覆盖极小 V、抵消、符号零和非有限 V 的最终输出位模式。"""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

import torch


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("builds", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    code = Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization"
    sys.path.insert(0, str(code))
    load = runpy.run_path(str(code / "bench-attention.py"))["native_launcher"]
    reference = load(args.reference)
    variants = {name: load(args.builds / name) for name in ("deferred", "guard-scale", "guarded")}
    generator = torch.Generator(device="cuda").manual_seed(20261127)
    with args.output.open("x") as stream:
        for rows in (4096, 4097, 27570):
            for case in ("tiny-normal", "minimum-bf16-subnormal", "signed-zero",
                         "alternating-tiny", "alternating-large", "random-large",
                         "positive-inf", "negative-inf", "mixed-inf", "nan"):
                def packed(heads):
                    return torch.randn((1, rows, heads, 128), device="cuda",
                                       dtype=torch.bfloat16, generator=generator)
                # Q/K 为零使第 2 个完整块起 max 必然不变，实际经过快路径。
                q, k = (packed(7).transpose(1, 2).zero_() for _ in range(2))
                v = packed(21)[:, :, -7:].transpose(1, 2)
                if case == "tiny-normal":
                    v.fill_(torch.finfo(torch.bfloat16).tiny)
                elif case == "minimum-bf16-subnormal":
                    v.fill_(2 ** -133)
                elif case == "signed-zero":
                    v.fill_(-0.0)
                    v[:, :, ::2] = 0.0
                elif case in ("alternating-tiny", "alternating-large"):
                    value = 2 ** -133 if case == "alternating-tiny" else 2 ** 120
                    v.fill_(-value)
                    v[:, :, ::2] = value
                elif case == "random-large":
                    v.mul_(2 ** 100)
                elif case == "positive-inf":
                    v.fill_(float("inf"))
                elif case == "negative-inf":
                    v.fill_(float("-inf"))
                elif case == "mixed-inf":
                    v.fill_(float("inf"))
                    v[:, :, ::2] = float("-inf")
                else:
                    v.fill_(float("nan"))
                expected = torch.empty((1, rows, 7, 128), device="cuda",
                                       dtype=torch.bfloat16).transpose(1, 2)
                reference(q, k, v, 128 ** -0.5, expected)
                for name, launch in variants.items():
                    storage = torch.full((1, rows + 2, 7, 128), -19, device="cuda",
                                         dtype=torch.bfloat16)
                    actual = storage[:, :rows].transpose(1, 2)
                    launch(q, k, v, 128 ** -0.5, actual)
                    torch.cuda.synchronize()
                    exact = torch.equal(expected.view(torch.int16), actual.view(torch.int16))
                    tail_ok = bool((storage[:, -2:] == -19).all())
                    record = dict(sequence=rows, case=case, variant=name, bitwise_equal=exact,
                                  tail_ok=tail_ok, reference_nan=int(torch.isnan(expected).sum()))
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    if not exact or not tail_ok:
                        raise ValueError(record)
    print("90 组稳定 max 的 V 边界输入逐位一致，尾部保护通过")


if __name__ == "__main__":
    main()
