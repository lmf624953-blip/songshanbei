#!/usr/bin/env python3
"""汇总预先登记的完整请求配对；严格排除预热、诊断和未完成运行。"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median, stdev


def read(path):
    return json.loads(path.read_text())


def stats(values):
    return dict(samples=values, mean=mean(values), median=median(values),
                min=min(values), max=max(values), stdev=stdev(values) if len(values) > 1 else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    if Path(str(run) + ".exit-code").read_text().strip() != "0":
        raise RuntimeError("推理进程未正常结束")
    plan, complete = read(args.plan), read(run / "complete.json")
    if any(plan[k] != complete[k] for k in ("sequence", "attention_variants")):
        raise RuntimeError("实际请求与预登记方案不一致")
    overrides = read(run / "attention-overrides.json")
    if overrides["sequence"] != plan["attention_variants"]:
        raise RuntimeError("注意力覆盖清单与请求序列不一致")
    validations = read(run / "sequence-validation.json")
    if len(validations) != len(complete["sequence"]) or len({v["index"] for v in validations}) != len(validations):
        raise RuntimeError("成片验证记录不完整或索引重复")
    by_index = {v["index"]: v for v in validations}
    samples = [json.loads(line) for line in Path(str(run) + ".vram.jsonl").read_text().splitlines()]
    records = []
    for i, (case, variant) in enumerate(zip(plan["sequence"], plan["attention_variants"])):
        directory = run / f"{i:02d}-{case}"
        record = read(directory / "record.json")
        if (record["index"] != i or record["case"] != case or record["attention_variant"] != variant
                or record["variant"] != "candidate" or record["transformer_calls"] != 49
                or record["diagnostic_capture"] or record.get("diagnostic_finite_checks")
                or not record["media"]["video_specs_passed"]):
            raise RuntimeError(f"请求 {i} 不满足完整公平计时要求")
        options = record["command_options"]
        if any(options[k] != v for k, v in dict(steps=50, num_frames=124, seed=42).items()):
            raise RuntimeError(f"请求 {i} 修改了规定工作量")
        quality = by_index[i]
        if quality["case"] != case:
            raise RuntimeError(f"请求 {i} 的质量索引错误")
        video = quality["comparison"]
        if not (video["decoded_length_match"] and video["metadata_size_match"]
                and video["shape_mismatches"] == 0 and video["ssim_match_percent"] is not None):
            raise RuntimeError(f"请求 {i} 未通过视频检查")
        if case == "A" and (record["first_frame_sha256"] != "b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54"
                            or options["text"] != "A dog falls asleep" or options["last_frame"] is not None):
            raise RuntimeError(f"请求 {i} 的公开输入不正确")
        window = [s for s in samples if record["started_epoch_ns"] <= s["epoch_ns"] <= record["finished_epoch_ns"]]
        if not window:
            raise RuntimeError(f"请求 {i} 无显存采样")
        peak = {}
        for sample in window:
            for card, used in sample["used_mib"].items():
                peak[card] = max(peak.get(card, 0), used)
        record.update(video_sha256=hashlib.sha256((directory / "video.mp4").read_bytes()).hexdigest(),
                      quality=quality, sampled_device_peak_mib=peak)
        records.append(record)
    warmups = plan["warmup_indices"]
    if {records[i]["attention_variant"] for i in warmups} != {"base", "fast"}:
        raise RuntimeError("两侧都需要独立完整预热")
    pairs = plan["pairs"]
    measured = [i for pair in pairs for i in pair]
    if len(pairs) < 3 or len(set(measured)) != len(measured) or set(measured).intersection(warmups):
        raise RuntimeError("需要至少三组不重复、排除预热的完整配对")
    for base, fast in pairs:
        a, b = records[base], records[fast]
        if (a["case"] != "A" or b["case"] != "A"
                or a["attention_variant"] != "base" or b["attention_variant"] != "fast"):
            raise RuntimeError("配对必须是相同公开 A 的 base/fast 请求")
    fields = ["wall_seconds"] + sorted(set.intersection(*(set(records[i]["stage_seconds"]) for i in measured)))
    timings = {}
    for field in fields:
        def value(i):
            return records[i]["wall_seconds"] if field == "wall_seconds" else records[i]["stage_seconds"][field]
        base_values, fast_values = ([value(pair[k]) for pair in pairs] for k in (0, 1))
        timings[field] = dict(base=stats(base_values), fast=stats(fast_values),
                             mean_reduction_percent=100 * (1 - mean(fast_values) / mean(base_values)),
                             paired_reduction_percent=stats([100 * (1 - b / a) for a, b in zip(base_values, fast_values)]))
    all_a = [r for r in records if r["case"] == "A"]
    report = dict(run=str(run), plan=plan, timings=timings, requests=records,
                  all_A_mp4_equal=len({r["video_sha256"] for r in all_a}) == 1,
                  model_load_seconds=complete["model_load_seconds"],
                  resource_note="同一实例，仅长注意力回调交替；显存为五秒整卡采样，不能据此宣称独立实例显存节省。")
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({k: timings[k] for k in ("wall_seconds", "generation.pipeline_total", "compute.h3_transformer")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
