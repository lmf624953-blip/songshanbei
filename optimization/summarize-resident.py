#!/usr/bin/env python3
"""汇总连续请求的预热、正式计时、成片质量与整卡显存采样。"""
import argparse
import json
from pathlib import Path
from statistics import mean, median, stdev


def stats(values):
    return {"samples": values, "mean": mean(values), "median": median(values),
            "min": min(values), "max": max(values), "stdev": stdev(values) if len(values) > 1 else 0}


def peaks(samples):
    result = {}
    for sample in samples:
        for device, value in sample["used_mib"].items():
            result[device] = max(result.get(device, 0), value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--measured-indices", default="1,2,3")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    complete = json.loads((run / "complete.json").read_text())
    validations = json.loads((run / "sequence-validation.json").read_text())
    measured = set(map(int, args.measured_indices.split(",")))
    if len(measured) < 3 or 0 in measured:
        parser.error("正式计时至少三次，并排除第一个预热请求")
    samples = [json.loads(line) for line in Path(str(run) + ".vram.jsonl").read_text().splitlines()]
    records = []
    for index, case in enumerate(complete["sequence"]):
        record = json.loads((run / f"{index:02d}-{case}" / "record.json").read_text())
        if (record["diagnostic_capture"] or record.get("diagnostic_finite_checks") or
                record["transformer_calls"] != 49 or not record["media"]["video_specs_passed"]):
            raise RuntimeError(f"诊断或规格不合格请求不能进入正式汇总：{index}")
        matching = [s for s in samples if record["started_epoch_ns"] <= s["epoch_ns"] <= record["finished_epoch_ns"]]
        if not matching:
            raise RuntimeError(f"请求 {index} 缺少整卡显存采样")
        record["sampled_device_peak_mib"] = peaks(matching)
        record["vram_samples"] = len(matching)
        record["quality"] = next(v["comparison"] for v in validations if v["index"] == index)
        record["audio_quality"] = next(v.get("audio_comparison") for v in validations if v["index"] == index)
        records.append(record)
    timed = [r for r in records if r["index"] in measured]
    if len(timed) != len(measured) or len({r["case"] for r in timed}) != 1:
        raise RuntimeError("正式测量索引不存在或混合了不同用例")
    fields = ("generation.pipeline_total", "compute.h3_transformer", "compute.condition_encoder",
              "input.video_vae_encode", "output.video_vae_decode", "output.audio_vae_decode", "output.media_encode")
    result = dict(run=str(run), measured_indices=sorted(measured),
                  wall_seconds=stats([r["wall_seconds"] for r in timed]),
                  stage_seconds={f: stats([r["stage_seconds"][f] for r in timed]) for f in fields},
                  ssim_percent=stats([r["quality"]["ssim_match_percent"] for r in timed]),
                  sampled_process_device_peak_mib=peaks(samples),
                  sampled_measured_device_peak_mib={d: max(r["sampled_device_peak_mib"].get(d, 0) for r in timed)
                                                   for d in peaks(samples)},
                  requests=records, model_load_seconds=complete["model_load_seconds"],
                  resource_note="hy-smi 每五秒整卡已用显存采样；不是瞬时精确峰值，也不等同于 PyTorch allocator 统计。")
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("run", "wall_seconds", "ssim_percent", "sampled_measured_device_peak_mib")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
