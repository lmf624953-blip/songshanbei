#!/usr/bin/env python3
"""验证 P 行布局与 V 预排布的组合，并保留无 profiler 逐对样本。"""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

import torch


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "plain", "previous", "combined", "prepack", "output"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    code = Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization"
    sys.path.insert(0, str(code))
    load = runpy.run_path(str(code / "bench-attention.py"))["native_launcher"]
    scripts = Path(__file__).parent
    screen = runpy.run_path(str(scripts / "screen.py"))["screen"]
    entry = runpy.run_path(str(scripts / "screen-prepacked-v.py"))
    baseline = load(args.reference)
    plain = load(args.plain)
    previous = entry["launcher"](args.previous, args.prepack, baseline)
    combined = entry["launcher"](args.combined, args.prepack, baseline)
    with args.output.open("x") as stream:
        def record(row):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)

        for label, left, right in (("row-vs-reference", baseline, plain),
                                   ("combined-vs-reference", baseline, combined),
                                   ("combined-vs-prepacked", previous, combined)):
            for length in (27570, 22016):
                row = screen(left, right, length, 19)
                record(dict(kind="paired", comparison=label, **row))
                if not row["passed"]:
                    raise ValueError(f"{label} at {length}")
        lengths = (1, 2, 31, 32, 33, 63, 64, 65, 95, 96, 127, 128, 129,
                   255, 256, 257, 4095, 4096, 4097, 8193)
        for length in lengths:
            row = screen(baseline, combined, length, 1)
            record(dict(kind="boundary", **row))
            if not row["passed"]:
                raise ValueError(f"Boundary at {length}")
        generator = torch.Generator(device="cuda").manual_seed(20261127)
        for length in (129, 4097, 27570):
            q = torch.randn((1, length, 7, 128), device="cuda", dtype=torch.bfloat16,
                            generator=generator).transpose(1, 2)
            k = torch.randn_like(q)
            v = torch.randn((1, length, 21, 128), device="cuda", dtype=torch.bfloat16,
                            generator=generator)[:, :, -7:].transpose(1, 2)
            for factor in (0, 1, 16):
                query, key = q * factor, k * factor
                for scale in (0.0, 128 ** -0.5):
                    left, right = torch.empty_like(q), torch.empty_like(q)
                    baseline(query, key, v, scale, left)
                    combined(query, key, v, scale, right)
                    exact = torch.equal(left.view(torch.int16), right.view(torch.int16))
                    finite = bool(torch.isfinite(right).all())
                    record(dict(kind="special", sequence=length, factor=factor, scale=scale,
                                exact=exact, finite=finite))
                    if not exact or not finite:
                        raise ValueError("Special input mismatch")
            for iteration in range(3):
                v.add_(0.25)
                left, right = torch.empty_like(q), torch.empty_like(q)
                baseline(q, k, v, 128 ** -0.5, left)
                combined(q, k, v, 128 ** -0.5, right)
                exact = torch.equal(left.view(torch.int16), right.view(torch.int16))
                record(dict(kind="fresh-value", sequence=length, iteration=iteration, exact=exact))
                if not exact:
                    raise ValueError("Workspace must be recomputed for each value")


if __name__ == "__main__":
    main()
