#!/usr/bin/env python3
"""核对同轮 v3/候选 Transformer 输出哈希，汇总真实输入回放计时。"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    if (directory / "exit-code").read_text().strip() != "0":
        raise RuntimeError("回放进程未成功退出")
    if not (directory / "replay/complete").is_file():
        raise RuntimeError("缺少回放完成记录")
    records = json.loads((directory / "replay/results.json").read_text())
    baseline = "candidate-tf-no-tf-reducecast"
    candidate = "candidate-bf16-wire-fast-attention"
    variants = (baseline, "candidate-tf", candidate)
    expected = {(step, variant) for step in (1, 25, 49) for variant in variants}
    actual = [(row["step"], row["variant"]) for row in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError(f"回放条目不完整或有重复：{actual}")
    indexed = dict(zip(actual, records))
    rows = []
    for step in (1, 25, 49):
        reference = indexed[step, baseline]
        trial = indexed[step, candidate]
        if reference["capture_sha256"] != trial["capture_sha256"]:
            raise RuntimeError("两侧真实输入哈希不一致")
        outputs = []
        for a, b in zip(reference["against_capture"], trial["against_capture"], strict=True):
            outputs.append({"name": a["name"], "shape": a["shape"],
                            "finite": a["finite"] and b["finite"],
                            "exact": a["name"] == b["name"] and a["shape"] == b["shape"]
                            and a["output_sha256"] == b["output_sha256"],
                            "v3_sha256": a["output_sha256"],
                            "candidate_sha256": b["output_sha256"]})
        for row in (reference, trial):
            if row["diagnostic_profiler"] or len(row["samples_seconds"]) != 3:
                raise RuntimeError("需要无 profiler 的三次计时")
        rows.append({"step": step, "capture_sha256": reference["capture_sha256"],
                     "v3_median_seconds": reference["median_seconds"],
                     "candidate_median_seconds": trial["median_seconds"],
                     "saving_pct": 100 * (1 - trial["median_seconds"] / reference["median_seconds"]),
                     "v3_samples_seconds": reference["samples_seconds"],
                     "candidate_samples_seconds": trial["samples_seconds"],
                     "outputs": outputs})
    passed = all(item["finite"] and item["exact"] for row in rows for item in row["outputs"])
    total_a = sum(row["v3_median_seconds"] for row in rows)
    total_b = sum(row["candidate_median_seconds"] for row in rows)
    result = {"verdict": "PASS" if passed else "FAIL", "scope": "真实八卡 Transformer 输入回放",
              "saving_pct_from_sum_of_medians": 100 * (1 - total_b / total_a), "steps": rows}
    (directory / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    lines = ["# 八卡 Transformer 回放对照", "",
             "在同一个模型实例中比较正式 v3 和候选，每个 step 各一次预热、三次同步墙钟计时。"
             "两侧按变体依次测量，未交替配对，存在测量顺序影响；它是去噪调用回放，不是完整视频生成测速。", "",
             "| 去噪调用 | v3 中位数 (s) | 候选中位数 (s) | 节省 | video/audio 输出 |",
             "| ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        exact = all(x["exact"] and x["finite"] for x in row["outputs"])
        lines.append(f"| {row['step']} | {row['v3_median_seconds']:.6f} | "
                     f"{row['candidate_median_seconds']:.6f} | {row['saving_pct']:+.2f}% | "
                     f"{'哈希一致且有限' if exact else '未通过'} |")
    lines += ["", f"三个中位数合计节省 {result['saving_pct_from_sum_of_medians']:.2f}%。"
              f"数值结论为 {result['verdict']}。", "",
              "[逐项输出哈希及原始计时](comparison.json) · "
              "[原生库与源码快照](replay/source/manifest.json) · "
              "[候选注意力产物](replay/attention-overrides.json)", ""]
    (directory / "Transformer回放.md").write_text("\n".join(lines))
    print(json.dumps({"verdict": result["verdict"],
                      "saving_pct": result["saving_pct_from_sum_of_medians"]}, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
