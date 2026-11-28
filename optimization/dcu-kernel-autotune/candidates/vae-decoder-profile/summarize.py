#!/usr/bin/env python3
"""归类两卡 VAE 时间线并保留设备事件、分段同步和重叠口径。"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import runpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.experiment
    wrapper_exit = int((root / "exit-code").read_text())
    assert wrapper_exit in (0, 1)
    capture = json.loads((root / "capture.json").read_text())
    assert capture["complete"] and capture["exact"] and capture["finite"]
    assert "HIP_PROF:finish" in (root / "run.log").read_text()
    traces = sorted(root.glob("hipprof_stream*.json"))
    assert traces
    assert wrapper_exit == 0 or all(path.stem.split("_")[-1].isdigit() for path in traces)
    analysis = Path(__file__).resolve().parents[3] / "analyze-hipprof-timeline.py"
    timeline = runpy.run_path(str(analysis))
    roi_begin, roi_end = capture["begin_ns"], capture["end_ns"]
    events = []
    for trace in traces:
        package = json.loads(trace.read_text())
        for event in package["traceEvents"] if isinstance(package, dict) else package:
            metadata = event.get("args", {})
            if event.get("ph") == "X" and event.get("cat") in timeline["DEVICE_CATEGORIES"]:
                if "BeginNs" in metadata and "EndNs" in metadata:
                    if int(metadata["BeginNs"]) < roi_end and int(metadata["EndNs"]) > roi_begin:
                        events.append(event)
        del package
    summary = timeline["summarize_window"](events, roi_begin, roi_end, "vae.decode", {0, 1})
    totals, counts, names = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for event in events:
        category = event.get("cat")
        metadata = event.get("args", {})
        if event.get("ph") != "X" or category not in timeline["DEVICE_CATEGORIES"]:
            continue
        if "BeginNs" not in metadata or "EndNs" not in metadata:
            continue
        duration = min(roi_end, int(metadata["EndNs"])) - max(roi_begin, int(metadata["BeginNs"]))
        if duration <= 0:
            continue
        device = int(metadata["devId"])
        name = event["name"]
        lower = name.lower()
        if category in {"HIPCOPY", "RCCLOPS"}:
            group = "copy_or_collective"
        elif name.startswith("Cijk_") or "gemm" in lower:
            group = "gemm"
        elif name == "application_kernel" or "attention" in lower:
            group = "attention"
        elif "conv" in lower:
            group = "convolution"
        elif "norm" in lower or "reduce" in lower:
            group = "norm_or_reduction"
        else:
            group = "other"
        totals[device][group] += duration
        counts[device][group] += 1
        names[device][name] += duration
    assert set(totals) == {0, 1}
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    report = dict(capture=capture, trace_files={path.name: digest(path) for path in traces},
                  wrapper_exit_code=wrapper_exit, device_events_in_roi=len(events),
                  scope="同一诊断请求的设备事件累计，可重叠，分组按内核符号",
                  per_device={rank: dict(group_ms={group: time / 1e6 for group, time in totals[rank].items()},
                                         group_counts=counts[rank],
                                         largest_kernel_names_ms=[(name, time / 1e6)
                                                                  for name, time in names[rank].most_common(25)])
                              for rank in totals}, roi_summary=summary)
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
