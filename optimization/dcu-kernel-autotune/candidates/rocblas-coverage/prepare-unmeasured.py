#!/usr/bin/env python3
"""复用库目录与历史实测清单，提取未测的大 tile 单 split GEMM。"""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--catalog-json", type=Path)
    parser.add_argument("--scope", choices=("large", "remaining"), default="large")
    parser.add_argument("--exclude-selection", type=Path)
    args = parser.parse_args()
    docs = Path(__file__).resolve().parents[2] / "docs"
    previous = json.loads((docs / "gemm-coverage-f13r1n16.json").read_text())
    old = json.loads((docs / "rocblas-bf16-a14r1n11.json").read_text())
    library_sha = digest(args.library)
    if library_sha != previous["coverage"]["library_sha256"]:
        parser.error("库版本变化，需要重新导出目录和审核历史映射")
    if args.catalog_json:
        solutions = json.loads(args.catalog_json.read_text())
    else:
        import msgpack
        solutions = msgpack.unpackb(args.library.read_bytes(), strict_map_key=False)["solutions"]
    if len(solutions) != previous["coverage"]["actual_solutions"]:
        parser.error("库目录候选数与已核验库不一致")
    seen = {name for group in old["subsets"].values() for name in group["kernels"]}
    seen.update(row["name"] for row in previous["screen"]["records"])
    if args.exclude_selection:
        completed = json.loads(args.exclude_selection.read_text())
        if completed["library_sha256"] != library_sha:
            parser.error("待排除的候选来自其他库")
        seen.update(row["short"] for row in completed["selected"])
    selected = []
    for solution in solutions:
        mapping = solution["sizeMapping"]
        if solution["info"]["SolutionNameMin"] in seen:
            continue
        if args.scope == "large" and (
                mapping["globalSplitU"] != 1 or mapping["macroTile"][0] * mapping["macroTile"][1] < 32768):
            continue
        selected.append(dict(index=solution["index"], short=solution["info"]["SolutionNameMin"],
                             actual=solution["name"], size_mapping=mapping))
    selected.sort(key=lambda s: (s["size_mapping"]["depthU"],
                                -s["size_mapping"]["macroTile"][0] * s["size_mapping"]["macroTile"][1],
                                s["index"]))
    args.output.mkdir()
    (args.output / "candidates.tsv").write_text(
        "".join(f'{row["index"]} {row["short"]}\n' for row in selected))
    report = dict(library_sha256=library_sha, previous_unique_names=len(seen),
                  catalog_size=len(solutions), selected_size=len(selected),
                  predicate=("排除历史实测和已指定候选后，macroTile面积>=32768、globalSplitU=1"
                             if args.scope == "large" else "排除历史实测和已指定候选后，遍历其余全部库候选"),
                  scope=args.scope,
                  excluded_selection_sha256=digest(args.exclude_selection) if args.exclude_selection else None,
                  catalog_json_sha256=digest(args.catalog_json) if args.catalog_json else None,
                  prior_report_sha256={name: digest(docs / name) for name in
                                       ("gemm-coverage-f13r1n16.json", "rocblas-bf16-a14r1n11.json")},
                  selected=selected)
    (args.output / "selection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"选择 {len(selected)} 个未测候选，排除 {len(seen)} 个历史名称")


if __name__ == "__main__":
    main()
