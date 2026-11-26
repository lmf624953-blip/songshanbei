#!/usr/bin/env python3
"""Compare complete resident runs and report whether to select the P LDS candidate."""

import argparse
import hashlib
import json
from pathlib import Path


ORDER = "AAAABCA"
MEASURED = [1, 2, 3]
MIN_GAIN_PERCENT = 0.3


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def inspect(run):
    if run.with_name(run.name + ".exit-code").read_text().strip() != "0":
        raise RuntimeError(f"推理进程未正常退出: {run}")
    report = read_json(run / "measurement.json")
    quality = read_json(run / "sequence-validation.json")
    if report["measured_indices"] != MEASURED or len(quality) != len(ORDER):
        raise RuntimeError(f"计时位置或质量记录数量不匹配: {run}")
    requests = report["requests"]
    if [item["case"] for item in requests] != list(ORDER):
        raise RuntimeError(f"请求顺序不匹配: {run}")
    return report, quality


def compare_requests(old_path, new_path, baseline, candidate, old_quality, new_quality):
    records = []
    for index, case in enumerate(ORDER):
        a, b = baseline["requests"][index], candidate["requests"][index]
        previous, current = old_quality[index], new_quality[index]
        if any(r["index"] != index or r["case"] != case for r in (a, b, previous, current)):
            raise RuntimeError(f"请求序号不匹配: {index}")
        for record in (a, b):
            options = record["command_options"]
            media = record["media"]
            if (record["transformer_calls"] != 49 or options["steps"] != 50
                    or options["num_frames"] != 124 or options["seed"] != 42
                    or not media["video_specs_passed"]):
                raise RuntimeError(f"请求规格或工作量不匹配: {index}")
            if case == "A" and (media["video"]["width"], media["video"]["height"]) != (768, 928):
                raise RuntimeError(f"公开 A 的尺寸不匹配: {index}")
        if a["candidate_calls"] != b["candidate_calls"] or b["candidate_calls"]["attention"] <= 0:
            raise RuntimeError(f"实际算子调用数不匹配: {index}")
        if previous["reference"] != current["reference"]:
            raise RuntimeError(f"独立质量参考不匹配: {index}")
        video = f"{index:02d}-{case}/video.mp4"
        current_hash = sha(new_path / video)
        previous_ssim = previous["comparison"]["ssim_match_percent"]
        current_ssim = current["comparison"]["ssim_match_percent"]
        records.append(dict(index=index, case=case, video_sha256=current_hash,
                            video_exact=current_hash == sha(old_path / video),
                            previous_ssim_percent=previous_ssim, current_ssim_percent=current_ssim,
                            quality_pass=current_ssim >= previous_ssim,
                            previous_wall_seconds=a["wall_seconds"], current_wall_seconds=b["wall_seconds"]))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline, old_quality = inspect(args.baseline)
    candidate, new_quality = inspect(args.candidate)
    requests = compare_requests(args.baseline, args.candidate, baseline, candidate,
                                old_quality, new_quality)
    timings = {}
    for name, old, new in (
        ("wall", baseline["wall_seconds"], candidate["wall_seconds"]),
        ("transformer", baseline["stage_seconds"]["compute.h3_transformer"],
         candidate["stage_seconds"]["compute.h3_transformer"]),
    ):
        timings[name] = dict(baseline=old, candidate=new,
                             reduction_percent=100 * (1 - new["mean"] / old["mean"]))
    quality_pass = all(r["video_exact"] and r["quality_pass"] for r in requests)
    speed_pass = all(item["reduction_percent"] >= MIN_GAIN_PERCENT for item in timings.values())
    report = dict(verdict="PASS" if quality_pass and speed_pass else "REVIEW",
                  quality_pass=quality_pass, speed_pass=speed_pass,
                  minimum_gain_percent=MIN_GAIN_PERCENT, requests=requests, timings=timings,
                  selected_default_changed=False,
                  sampled_device_peak_mib=candidate["sampled_process_device_peak_mib"])
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
