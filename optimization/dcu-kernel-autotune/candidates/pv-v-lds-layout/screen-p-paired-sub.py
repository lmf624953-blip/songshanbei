#!/usr/bin/env python3
"""单进程比较 P 减法候选，并在首尾加入自身配对以识别计时漂移。"""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("builds", type=Path)
    cli.add_argument("guard_reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--pairs", type=int, default=39)
    cli.add_argument("--variants", nargs="+", default=(
        "rename", "neg-scalar", "packed", "guard-rename", "guard-neg-scalar", "guard-packed"))
    args = cli.parse_args()
    here = Path(__file__).resolve().parent
    code = Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization"
    sys.path.insert(0, str(code))
    load = runpy.run_path(str(code / "bench-attention.py"))["native_launcher"]
    screen = runpy.run_path(str(here / "screen.py"))["screen"]
    reference = load(args.reference)
    candidates = [("self-before", reference), ("guard-reference", load(args.guard_reference))]
    candidates += [(name, load(args.builds / name)) for name in args.variants]
    candidates += [("self-after", reference)]
    with args.output.open("x") as stream:
        for name, launch in candidates:
            for rows in (27570, 22016):
                result = screen(reference, launch, rows, args.pairs)
                record = dict(variant=name, hip_visible_devices=os.environ.get("HIP_VISIBLE_DEVICES"), **result)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(name, rows, result.get("medians_ms"), result.get("paired_gain_percent"), flush=True)
                if not result["passed"]:
                    raise ValueError(record)


if __name__ == "__main__":
    main()
