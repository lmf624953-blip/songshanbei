#!/usr/bin/env python3
"""检查短长度、分块边界与尾部掩码，不把诊断计时当作性能收益。"""

import argparse
import json
from pathlib import Path
import runpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    benchmark = runpy.run_path(str(Path(__file__).with_name("bench-h3-attention.py")))
    load = benchmark["native_loader"]()
    baseline, candidate = load(args.baseline.resolve()), load(args.candidate.resolve())
    results = []
    for sequence in (1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 895, 896, 897, 1023):
        row = benchmark["compare"](baseline, candidate, sequence, iterations=3)
        result = {key: row[key] for key in ("sequence", "normal", "stress")}
        results.append(result)
        print(json.dumps(result), flush=True)
    passed = all(row[mode]["finite"] and row[mode]["exact"]
                 for row in results for mode in ("normal", "stress"))
    result = {"verdict": "PASS" if passed else "FAIL", "cases": results}
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
