#!/usr/bin/env python3
"""先比较非有限输入和边界的最终 BF16 位模式，再做正常输入配对计时。"""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

import torch


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "output"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    code = Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization"
    sys.path.insert(0, str(code))
    load = runpy.run_path(str(code / "bench-attention.py"))["native_launcher"]
    baseline, candidate = load(args.reference), load(args.candidate)
    generator = torch.Generator(device="cuda").manual_seed(20261127)
    with args.output.open("x") as stream:
        def record(row):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)

        def tensors(rows):
            def packed(heads):
                return torch.randn((1, rows, heads, 128), device="cuda", dtype=torch.bfloat16,
                                   generator=generator)
            q, k = (packed(7).transpose(1, 2) for _ in range(2))
            v = packed(21)[:, :, -7:].transpose(1, 2)
            return q, k, v

        def compare(q, k, v, scale, label):
            rows = q.shape[2]
            storage = [torch.full((1, rows + 2, 7, 128), -19, device="cuda",
                                  dtype=torch.bfloat16) for _ in range(2)]
            outputs = [buffer[:, :rows].transpose(1, 2) for buffer in storage]
            for launch, output in zip((baseline, candidate), outputs):
                launch(q, k, v, scale, output)
            torch.cuda.synchronize()
            exact = torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
            tails = all(bool((buffer[:, -2:] == -19).all()) for buffer in storage)
            mismatches = int(torch.count_nonzero(outputs[0].view(torch.int16) != outputs[1].view(torch.int16)))
            row = dict(kind="bitwise", case=label, sequence=rows, exact=exact, tail_ok=tails,
                       mismatches=mismatches, reference_nan=int(torch.isnan(outputs[0]).sum()),
                       candidate_nan=int(torch.isnan(outputs[1]).sum()))
            record(row)
            if not exact or not tails:
                raise ValueError(row)

        cases = ("random", "zero-qk", "all-q-nan", "one-q-nan", "one-k-nan", "all-k-nan",
                 "one-k-inf", "all-q-inf", "qk-overflow", "one-v-nan", "all-v-inf",
                 "tiny-qk", "negative-scale", "zero-scale", "nan-scale", "inf-scale")
        for rows in (32, 33, 129, 4097, 27570):
            for case in cases:
                q, k, v = tensors(rows)
                scale = 128 ** -0.5
                if case == "zero-qk":
                    q.zero_(); k.zero_()
                elif case == "all-q-nan":
                    q.fill_(float("nan"))
                elif case == "one-q-nan":
                    q[0, 0, 0, 0] = float("nan")
                elif case == "one-k-nan":
                    k[0, 0, 0, 0] = float("nan")
                elif case == "all-k-nan":
                    k.fill_(float("nan"))
                elif case == "one-k-inf":
                    k[0, 0, 0, 0] = float("inf")
                elif case == "all-q-inf":
                    q.fill_(float("inf"))
                elif case == "qk-overflow":
                    q.mul_(1e20); k.mul_(1e20)
                elif case == "one-v-nan":
                    v[0, 0, 0, 0] = float("nan")
                elif case == "all-v-inf":
                    v.fill_(float("inf"))
                elif case == "tiny-qk":
                    q.mul_(1e-30); k.mul_(1e-30)
                elif case == "negative-scale":
                    scale = -scale
                elif case == "zero-scale":
                    scale = 0.0
                elif case == "nan-scale":
                    scale = float("nan")
                elif case == "inf-scale":
                    scale = float("inf")
                compare(q, k, v, scale, case)
        for rows in (1, 2, 31, 63, 64, 65, 127, 128, 255, 256, 257, 4095, 4096, 8193, 22016):
            compare(*tensors(rows), 128 ** -0.5, "boundary")
        screen = runpy.run_path(str(Path(__file__).with_name("screen.py")))["screen"]
        for rows in (27570, 22016):
            result = screen(baseline, candidate, rows, 19)
            record(dict(kind="paired", **result))
            if not result["passed"]:
                raise ValueError(result)


if __name__ == "__main__":
    main()
