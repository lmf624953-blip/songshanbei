#!/usr/bin/env python3
"""归档独立两阶段/AllGather实验，保留原始配对，不把微测投影当作端到端。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def read(path):
    return json.loads(path.read_text())


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("owner", type=Path)
    cli.add_argument("allgather", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    assert (args.owner / "post-tests-exit-code").read_text().strip() == "0"
    assert (args.owner / "grid2d-tests-exit-code").read_text().strip() == "0"
    assert (args.allgather / "initial-tests-exit-code").read_text().strip() == "0"
    steady = 120.69804317139399
    sources, builds, experiments, performance, pmc = {}, {}, {}, {}, {}
    here = Path(__file__).resolve().parent
    for directory in (here, here.parent / "peer-allgather-publish"):
        for path in directory.iterdir():
            if path.is_file():
                sources[str(path.relative_to(here.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for group, root in (("owner", args.owner), ("allgather", args.allgather)):
        for path in sorted(root.glob("*.json")):
            data = read(path)
            experiments[f"{group}/{path.name}"] = data
            if "records" not in data or not path.name.startswith("clean-"):
                continue
            results = []
            for row in data["records"]:
                assert row["exact"] and not row["consumer_copy"] and not row["diagnostic_profile_regions"]
                assert len(row["samples"]) == 39
                saved = [s["reference"]["gpu_ms"] - s["candidate"]["gpu_ms"] for s in row["samples"]]
                wall = [s["reference"]["wall_ms"] - s["candidate"]["wall_ms"] for s in row["samples"]]
                median = statistics.median(saved)
                result = dict(count=row["count"], pairs=len(saved), reference_median_ms=row["reference_median_ms"],
                              candidate_median_ms=row["candidate_median_ms"],
                              median_paired_gpu_gain_percent=row["paired_gain_percent"],
                              median_paired_gpu_saved_ms=median,
                              median_paired_wall_saved_ms=statistics.median(wall))
                if row["count"] == 148216320:
                    seconds = median * 4900 / 1000
                    result["unverified_A_projection"] = dict(seconds_saved=seconds,
                        request_reduction_percent=seconds / steady * 100,
                        assumption="微测差额逐次适用于4900个模型融合批次；仅筛选投影，未在模型中验证，非上界或成绩")
                results.append(result)
            performance[f"{group}/{path.name}"] = results
        for path in sorted(root.glob("*/peer-runtime-build.json")):
            builds[f"{group}/{path.parent.name}"] = read(path)
        for path in sorted(root.glob("grid2d-*-pmc.txt")):
            text = path.read_text()
            blocks = text.split('kernel-name:')[1:]
            assert len(blocks) == 24, (path, len(blocks))
            devices = [int(re.search(r"^ gpu id\s+(\d+)", b, re.M)[1]) for b in blocks]
            assert set(devices) == set(range(8)) and all(devices.count(r) == 3 for r in range(8))
            fields = {}
            for block in blocks:
                for line in block.splitlines()[1:]:
                    match = re.fullmatch(r" (.*?)\s{2,}([^\s].*)", line)
                    if match:
                        fields.setdefault(match[1], []).append(match[2])
            pmc[f"{group}/{path.name}"] = dict(records=len(blocks), devices=devices, raw_fields=fields,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                scope="独立hipprof --pmc；原始字段，非正式性能")
    assert len(performance) == 10, list(performance)
    assert len(pmc) == 2
    record = dict(node="f13r1n16", adopted=False, qualified=False,
        scope="八卡独立原序BF16求和/门控/发布实验；没有新增真实模型或完整生成成绩",
        selected_release="validated-build/peer-gate-nt-f13r1n16",
        selected_steady_A_seconds=steady,
        formal_spec=dict(width=768, height=928, steps=50, frames=124, seed=42, transformer_calls=49),
        compilation_cache_optimization=False,
        full_A_three_percent_required_saving_per_batch_ms=steady * 0.03 / 4.9,
        roots=dict(owner=str(args.owner), allgather=str(args.allgather)),
        source_sha256=sources, builds=builds, clean_performance=performance, pmc=pmc,
        experiments=experiments,
        decision="公开A对应长度所有候选的微测投影不足3%完整请求收益，部分其他长度退化；归档，不修改默认。二维后端变换可重建且资源减少，但不将资源数字代替实际性能。")
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
