#!/usr/bin/env python3
"""核对无 profiler 配对、八卡校验和独立 PMC，归档现有记录。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import median


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    experiment = args.experiment
    for marker in ("bench-a-qkv-exit-code", "vector-suite-exit-code", "copy-suite-exit-code", "pmc/exit-code"):
        assert (experiment / marker).read_text().strip() == "0", marker
    formal = {}
    for name in ("bench-a-qkv", "vector-a-qkv", "vector-a-ffn", "copy-a-qkv", "copy-a-ffn"):
        data = json.loads((experiment / (name + ".json")).read_text())
        modes = set(data["median_wall_ms"])
        assert data["complete"] and len(data["pairs"]) == 39 and "profiler" not in data
        assert {(row["mode"], row["rank"]) for row in data["validation"]} == {
            (mode, rank) for mode in modes for rank in range(8)
        }
        assert all(all(row[key] for key in ("input_exact", "output_exact", "guards", "finite"))
                   for row in data["validation"])
        for mode in modes:
            assert data["median_wall_ms"][mode] == median(pair[mode]["wall_ms"] for pair in data["pairs"])
        formal[name] = data
    p = {}
    for mode in ("full", "split_overlap"):
        json_path = experiment / "pmc" / (mode + ".json")
        txt_path = json_path.with_suffix(".txt")
        data = json.loads(json_path.read_text())
        assert data["complete"] and data["profiler"] == mode and not data["pairs"]
        kernel_names = re.findall(r'^kernel-name:"(.*)"$', txt_path.read_text(), re.M)
        counts = Counter("norm" if name == "wide_norm_indexed_kernel" else
                         "publish" if "publish_rows" in name else
                         "gemm" if name.startswith("Cijk_") else "other" for name in kernel_names)
        expect = {"norm": 8, "publish": 8, "gemm": 8} if mode == "full" else {
            "norm": 8, "publish": 9, "gemm": 16
        }
        assert counts == expect, (mode, counts)
        p[mode] = dict(record=data, raw_txt_sha256=digest(txt_path), counts=counts,
                       kernel_names=Counter(kernel_names))
    qkv = formal["bench-a-qkv"]["median_wall_ms"]
    ffn = formal["copy-a-ffn"]["median_wall_ms"]
    seconds = ((qkv["full"] - qkv["split_overlap"]) * 2401 +
               (ffn["full"] - ffn["split_overlap"]) * 2450) / 1000
    report = dict(experiment=str(experiment), node="f12r1n15",
                  scope="八卡独立发布到投影，复用正式九齿归一化和 rocBLAS，输入为固定种子的生成张量",
                  selected=False, formal=formal, pmc=p, hypothetical_full_a_seconds=seconds,
                  equivalent_fraction_of_accepted_full_a=seconds / 116.975680189,
                  projection_scope="2401 次 QKV 和 2450 次 FFN 对应发布，微测中位数差外推，跨节点且未经完整请求实测",
                  source_sha256={path.name: digest(path) for path in Path(__file__).parent.iterdir()
                                 if path.is_file()})
    with args.report.open("x") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
