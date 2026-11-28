#!/usr/bin/env python3
"""核对四个生产形状的库候选覆盖、无 profiler 样本和独立 PMC。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median


SHAPES = {(2688, 27570, 5376), (3584, 27570, 5376),
          (5376, 27570, 1792), (5376, 27570, 896)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--label", default="a02r3n04")
    args = parser.parse_args()
    root = args.experiment
    docs = Path(__file__).resolve().parents[2] / "docs"
    previous_path = docs / "gemm-coverage-f13r1n16.json"
    previous = json.loads(previous_path.read_text())
    previous_rows = [row for row in previous["screen"]["records"]
                     if (row["m"], row["n"], row["k"]) in SHAPES]
    assert len(previous_rows) == 168
    parts = {}
    combined = []
    for part, expected in [("screen", 300), ("remaining", 5056), ("historical-numerics", 260)]:
        assert (root / f"{part}-exit-code").read_text().strip() == "0"
        rows = read_lines(root / f"{part}.jsonl")
        assert len(rows) == expected and all(row["status"] == 0 for row in rows)
        for row in rows:
            assert (row["m"], row["n"], row["k"]) in SHAPES
            assert row["elements"] == row["m"] * row["n"]
            if row["mismatches"] == 0:
                assert len(row["baseline_ms"]) == len(row["candidate_ms"]) == 9
            else:
                assert not row["baseline_ms"] and not row["candidate_ms"]
        parts[part] = dict(rows=len(rows), exact=sum(row["mismatches"] == 0 for row in rows),
                           raw_sha256=digest(root / f"{part}.jsonl"))
        combined.extend(dict(phase=part, **row) for row in rows)
    combined.extend(dict(phase="previous-42", **row) for row in previous_rows)
    keys = [(row["m"], row["n"], row["k"], row["index"]) for row in combined]
    assert len(keys) == len(set(keys)) == 5784
    coverage = Counter((row["m"], row["n"], row["k"]) for row in combined)
    assert set(coverage) == SHAPES and set(coverage.values()) == {1446}
    sets = [{row["index"] for row in combined if (row["m"], row["n"], row["k"]) == shape}
            for shape in sorted(SHAPES)]
    assert all(indices == sets[0] for indices in sets)
    controls = read_lines(root / "controls.jsonl")
    assert (root / "controls-exit-code").read_text().strip() == "0"
    assert len(controls) == 4 and all(row["mismatches"] == 0 for row in controls)
    for name in ("build", "pmc-build", "solution-pmc"):
        assert (root / f"{name}-exit-code").read_text().strip() == "0"
    profile_rows = read_lines(root / "solution-pmc.jsonl")
    assert len(profile_rows) == 3 and [row["mismatches"] == 0 for row in profile_rows] == [True, False, True]
    assert all(not row["baseline_ms"] and not row["candidate_ms"] for row in profile_rows)
    assert "HIP_PROF:finish" in (root / "solution-pmc.log").read_text()
    counters = []
    for section in (root / "solution-pmc.txt").read_text().split("kernel-name:")[1:]:
        lines = section.splitlines()
        counters.append(dict(name=lines[0].strip('"'),
                             **dict(line.strip().rsplit(maxsplit=1) for line in lines[1:] if line.strip())))
    assert len(counters) == 6 and all(row["gpu id"] == "0" for row in counters)
    for index, counter in enumerate(counters):
        assert counter["name"].startswith("Cijk_")
        if index % 2 == 0:
            assert counter["name"] == counters[0]["name"]
    catalog_path = Path(previous["experiment"]) / "current-solutions.json"
    catalog = json.loads(catalog_path.read_text())
    by_index = {row["index"]: row for row in catalog}
    assert set(by_index) == sets[0]
    for index, row in enumerate(profile_rows):
        assert counters[2 * index + 1]["name"] == by_index[row["index"]]["name"]
    library = Path(previous["coverage"]["library"])
    assert digest(library) == previous["coverage"]["library_sha256"]
    assert digest(root / "screen-solutions-pmc.cc") == digest(Path(__file__).with_name("screen-solutions.cc"))
    assert digest(root / "prepare-remaining.py") == digest(Path(__file__).with_name("prepare-unmeasured.py"))
    shape_summary = {}
    for shape in sorted(SHAPES):
        rows = [row for row in combined if (row["m"], row["n"], row["k"]) == shape]
        new_exact = [row for row in rows if row["phase"] in ("screen", "remaining") and row["mismatches"] == 0]
        best = min(new_exact, key=lambda row: row["candidate_median_ms"])
        shape_summary[str(shape)] = dict(considered=len(rows),
            exact=sum(row["mismatches"] == 0 for row in rows), different=sum(row["mismatches"] != 0 for row in rows),
            best_new=dict(index=best["index"], reference_ms=best["baseline_median_ms"],
                          candidate_ms=best["candidate_median_ms"],
                          paired_gain_percent=100 * median(1 - b / a for a, b in
                                                          zip(best["baseline_ms"], best["candidate_ms"], strict=True))))
    prefix = f"gemm-unmeasured-{args.label}"
    targets = {suffix: args.output / f"{prefix}{suffix}" for suffix in (".json", "-screen.jsonl", "-pmc.txt")}
    assert not any(path.exists() for path in targets.values())
    targets["-screen.jsonl"].write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined))
    targets["-pmc.txt"].write_bytes((root / "solution-pmc.txt").read_bytes())
    report = dict(node=args.label, experiment=str(root), adopted=False,
        scope="固定DTK BF16 TN目录、四个生产投影形状和确定性合成输入的完整候选筛选",
        library_sha256=digest(library), catalog_json_sha256=digest(catalog_path),
        covered_unique_indices=1446, covered_shape_index_pairs=len(combined), parts=parts,
        exact_combinations=sum(row["mismatches"] == 0 for row in combined), shape_summary=shape_summary,
        controls=controls, cpu_full_array_cross_checks_in_screen=40,
        reused_previous_report=dict(path=previous_path.name, sha256=digest(previous_path), rows=168),
        pmc=dict(rows=profile_rows, dispatches=counters),
        raw_files={path.name: digest(path) for suffix, path in targets.items() if suffix != ".json"},
        source_sha256={name: digest(root / name) for name in
                       ("screen-solutions.cc", "screen-solutions-pmc.cc", "prepare-unmeasured.py", "prepare-remaining.py")},
        binary_sha256={name: digest(root / name) for name in ("screen", "screen-pmc")},
        decision="新增逐位通过的库解均较慢，保留已有默认；结论限于当前目录和形状，不代表GEMM全局最优")
    targets[".json"].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("covered_unique_indices", "covered_shape_index_pairs", "exact_combinations", "shape_summary")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
