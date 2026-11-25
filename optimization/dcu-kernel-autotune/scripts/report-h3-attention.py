#!/usr/bin/env python3
"""生成按生产长度拆分的 H3 长注意力调优报告。"""

import csv
import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
ROUNDS = ROOT / "rounds/h3-long-attention"
REPORT = ROOT / "reports/h3-long-attention/按长度对照.md"
SEQUENCES = 27570, 22016


def field(row, key):
    return int(row[key])


def resources(round_dir, sequence, config):
    path = round_dir / "pmc/full.csv"
    if not path.is_file():
        return ["—"] * 4
    query_block = config["query_block"]
    threads = config["warps"] * 64
    grid = ((sequence + query_block - 1) // query_block) * 7 * threads
    with path.open(newline="") as stream:
        records = [r for r in csv.DictReader(stream)
                   if r["KernelName"] == "dense_attention_aot_kernel" and field(r, "grd") == grid]
    samples = records[1:]
    if not samples:
        return ["—"] * 4
    conflicts = [100 * field(r, "SQ_LDS_BANK_CONFLICT") / (80 * field(r, "GRBM_GUI_ACTIVE"))
                 for r in samples]
    return [str(int(median(field(r, name) for r in samples))) for name in ("arch_vgpr", "lds", "scr")] + [
        f"{median(conflicts):.2f}%"
    ]


def main():
    lines = [
        "# MiniMax-H3 长注意力逐轮实测",
        "",
        "同一张 BW1000 上交替测量固定 v3 与候选，每个长度 11 组 GPU event 配对。"
        "正百分比表示候选节省耗时。PMC 与计时分开采集，资源数据排除各长度的首条预热发射。",
        "",
        "| 轮次 | 变更 | 长度 | v3 中位数 (ms) | 候选中位数 (ms) | 配对节省 | 数值 | VGPR | LDS (B) | Scratch (B) | LDS 冲突 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for directory in sorted(ROUNDS.iterdir()):
        status = directory / "eval/status.txt"
        if not status.exists():
            continue
        log = directory / "eval/run.log"
        config_path = directory / "install/attention/dense-attention-aot-build.json"
        label = "正式 v3"
        if directory.name != "baseline":
            label = next((line.removeprefix("description: ").strip()
                          for line in (directory / "report.md").read_text().splitlines()
                          if line.startswith("description: ")), directory.name)
        round_link = f"[{directory.name}](../../rounds/h3-long-attention/{directory.name}/)"
        if not log.is_file() or not config_path.is_file():
            lines.append(f"| {round_link} | {label} | — | — | — | — | {status.read_text().strip()} | — | — | — | — |")
            continue
        config = json.loads(config_path.read_text())
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{"sequence":')]
        assert [row["sequence"] for row in rows] == list(SEQUENCES), directory
        for row in rows:
            sequence = row["sequence"]
            exact = all(row[mode]["exact"] and row[mode]["finite"] for mode in ("normal", "stress"))
            columns = [round_link, label, str(sequence), f'{row["reference_ms"]:.3f}',
                       f'{row["candidate_ms"]:.3f}', f'{row["paired_speedup_pct"]:+.2f}%',
                       "逐位一致" if exact else "有差异", *resources(directory, sequence, config)]
            lines.append("| " + " | ".join(columns) + " |")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT)


if __name__ == "__main__":
    main()
