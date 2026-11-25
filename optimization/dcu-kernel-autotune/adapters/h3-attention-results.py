#!/usr/bin/env python3
"""解析长注意力配对结果，核验编译调度变体是否保持 BF16 逐位一致。"""

import json
from pathlib import Path
import sys


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.startswith('{"sequence":')]


def parse_run_log(path):
    data = rows(path)
    return {
        "A+B kernel median sum (ms)": sum(row["candidate_ms"] for row in data),
        **{f'{row["sequence"]} kernel (ms)': row["candidate_ms"] for row in data},
    }


def main():
    base, candidate, output = sys.argv[1:]
    reference, tested = rows(base), rows(candidate)
    passed = [r["sequence"] for r in reference] == [r["sequence"] for r in tested] == [27570, 22016]
    passed &= all(row[mode]["finite"] and row[mode]["exact"]
                  for row in reference + tested for mode in ("normal", "stress"))
    summary = ["verdict: " + ("PASS" if passed else "FAIL")]
    summary.extend(json.dumps({k: row[k] for k in ("sequence", "normal", "stress")}) for row in tested)
    Path(output).write_text("\n".join(summary) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
