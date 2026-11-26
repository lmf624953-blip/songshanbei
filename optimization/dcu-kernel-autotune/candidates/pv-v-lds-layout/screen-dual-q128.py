#!/usr/bin/env python3
"""Validate and time the dual-Q128 experiment, including its spill-heavy controls."""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("builds", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    code = Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization"
    sys.path.insert(0, str(code))
    load = runpy.run_path(str(code / "bench-attention.py"))["native_launcher"]
    screen = runpy.run_path(str(Path(__file__).with_name("screen.py")))["screen"]
    names = ("reference", "independent-kv", "shared-kv")
    launchers = dict(zip(names, map(load, (args.reference, args.builds / names[1], args.builds / names[2]))))
    pairs = (("reference", "independent-kv"), ("reference", "shared-kv"), ("independent-kv", "shared-kv"))
    rows = []
    for base, candidate in pairs:
        lengths = (31, 32, 33, 127, 128, 129, 255, 256, 257, 383, 384, 385, 27570, 22016)
        if base == "independent-kv":
            lengths = lengths[-2:]
        for length in lengths:
            samples = 11 if length in (27570, 22016) else 1
            result = screen(launchers[base], launchers[candidate], length, samples)
            item = dict(base=base, candidate=candidate, **result)
            rows.append(item)
            print(json.dumps(item), flush=True)
            args.output.write_text(json.dumps(rows, indent=2) + "\n")
            if not result["passed"]:
                raise RuntimeError(f"Numerical validation failed for {candidate} at {length} tokens")


if __name__ == "__main__":
    main()
