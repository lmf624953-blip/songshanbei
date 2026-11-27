#!/usr/bin/env python3
"""复核全部 BF16 NaN payload、带符号零和最大有限值的最终输出位模式。"""

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
    baseline = load(args.reference)
    candidates = {name: load(args.builds / name) for name in ("full-kv", "ror4", "combined")}
    generator = torch.Generator(device="cuda").manual_seed(20261127)
    bits = [sign | 0x7f80 | payload for sign in (0, 0x8000) for payload in range(1, 128)]
    payloads = torch.tensor([x if x < 32768 else x - 65536 for x in bits],
                            device="cuda", dtype=torch.int16).view(torch.bfloat16)
    with args.output.open("x") as stream:
        for rows in (4096, 4097, 27570):
            for case in ("all-bf16-q-nan-payloads", "all-bf16-k-nan-payloads",
                         "signed-zero", "largest-finite-qk"):
                def packed(heads):
                    return torch.randn((1, rows, heads, 128), device="cuda",
                                       dtype=torch.bfloat16, generator=generator)
                q, k = (packed(7).transpose(1, 2) for _ in range(2))
                v = packed(21)[:, :, -7:].transpose(1, 2)
                if case == "all-bf16-q-nan-payloads":
                    q[0, 0, :254] = payloads[:, None]
                elif case == "all-bf16-k-nan-payloads":
                    k[0, 0, :254] = payloads[:, None]
                elif case == "signed-zero":
                    q.fill_(0.0)
                    k.fill_(-0.0)
                else:
                    q.fill_(torch.finfo(torch.bfloat16).max)
                    k.fill_(torch.finfo(torch.bfloat16).max)
                reference = torch.empty((1, rows, 7, 128), device="cuda",
                                        dtype=torch.bfloat16).transpose(1, 2)
                baseline(q, k, v, 128 ** -0.5, reference)
                for name, launch in candidates.items():
                    storage = torch.full((1, rows + 2, 7, 128), -19, device="cuda",
                                         dtype=torch.bfloat16)
                    result = storage[:, :rows].transpose(1, 2)
                    launch(q, k, v, 128 ** -0.5, result)
                    torch.cuda.synchronize()
                    exact = torch.equal(reference.view(torch.int16), result.view(torch.int16))
                    tail_ok = bool((storage[:, -2:] == -19).all())
                    record = dict(sequence=rows, case=case, variant=name, bitwise_equal=exact,
                                  tail_ok=tail_ok, reference_nan=int(torch.isnan(reference).sum()))
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()
                    if not exact or not tail_ok:
                        raise ValueError(record)
    print("36 组 payload / 零 / 最大有限值输出逐位一致，尾部保护通过")


if __name__ == "__main__":
    main()
