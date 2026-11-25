#!/usr/bin/env python3
"""Export per-device kernel timings and idle intervals from a bounded hipprof trace."""

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import json
from math import ceil
from pathlib import Path
import runpy
from statistics import median


DEVICE_CATEGORIES = {"HIPOPS", "RCCLOPS", "HIPCOPY"}
NS_PER_MS = 1_000_000


@dataclass(frozen=True)
class Activity:
    start: int
    end: int
    device: int
    category: str
    name: str
    stream: str

    def description(self):
        return {"device": self.device, "category": self.category, "name": self.name,
                "stream": self.stream}


def occupancy(activities, start, end):
    """Merge across streams and retain the activities adjacent to each idle interval."""
    previous = None
    cursor = start
    gaps = []
    for activity in sorted(activities, key=lambda item: (item.start, -item.end)):
        if activity.start > cursor:
            gaps.append((cursor, activity.start, previous, activity))
        if activity.end > cursor:
            cursor, previous = activity.end, activity
    if cursor < end:
        gaps.append((cursor, end, previous, None))
    idle = sum(right - left for left, right, _, _ in gaps)
    return end - start - idle, gaps


def gap_summary(gaps, start, end):
    internal = [gap for gap in gaps if gap[0] != start and gap[1] != end]
    return {
        "idle_ms": sum(right - left for left, right, _, _ in gaps) / NS_PER_MS,
        "internal_idle_ms": sum(right - left for left, right, _, _ in internal) / NS_PER_MS,
        "leading_idle_ms": (gaps[0][1] - start) / NS_PER_MS if gaps and gaps[0][0] == start else 0,
        "trailing_idle_ms": (end - gaps[-1][0]) / NS_PER_MS if gaps and gaps[-1][1] == end else 0,
        "internal_gap_counts_at_least_ms": {str(ms): sum(right - left >= ms * NS_PER_MS for left, right, _, _ in internal)
                                           for ms in (0.01, 0.1, 0.5, 1, 5, 10)},
        "largest_internal_gaps": [
            {"begin_ms": (left - start) / NS_PER_MS, "end_ms": (right - start) / NS_PER_MS,
             "duration_ms": (right - left) / NS_PER_MS, "previous": previous.description(),
             "next": following.description()}
            for left, right, previous, following in sorted(internal, key=lambda gap: gap[1] - gap[0], reverse=True)[:10]
        ],
    }


def summarize_window(events, start, end, name, expected_devices):
    all_activity = []
    by_device = defaultdict(list)
    host = defaultdict(list)
    durations = defaultdict(list)
    for event in events:
        category = event.get("cat")
        args = event.get("args", {})
        if event.get("ph") != "X" or category not in DEVICE_CATEGORIES | {"HIP", "RCCL"}:
            continue
        if "BeginNs" not in args or "EndNs" not in args:
            continue
        left, right = max(start, int(args["BeginNs"])), min(end, int(args["EndNs"]))
        if right <= left:
            continue
        if category not in DEVICE_CATEGORIES:
            host[category, event["name"]].append(right - left)
            continue
        activity = Activity(left, right, int(args["devId"]), category, event["name"], str(args["queueId"]))
        all_activity.append(activity)
        by_device[activity.device].append(activity)
        durations[activity.device, category, activity.name].append(right - left)
    if set(by_device) != set(expected_devices):
        raise ValueError(f"{name}: expected devices {expected_devices}, got {sorted(by_device)}")

    wall = end - start
    any_busy, all_idle = occupancy(all_activity, start, end)
    per_device = {}
    for device, activities in sorted(by_device.items()):
        busy, gaps = occupancy(activities, start, end)
        categories = {category: [event for event in activities if event.category == category]
                      for category in sorted(DEVICE_CATEGORIES)}
        computation, _ = occupancy(categories["HIPOPS"], start, end)
        communication, _ = occupancy(categories["RCCLOPS"], start, end)
        kernel_busy, _ = occupancy(categories["HIPOPS"] + categories["RCCLOPS"], start, end)
        per_device[str(device)] = {
            "device_activity_pct": busy / wall * 100,
            "busy_ms": busy / NS_PER_MS,
            "compute_busy_ms": computation / NS_PER_MS,
            "communication_busy_ms": communication / NS_PER_MS,
            "compute_communication_overlap_ms": (computation + communication - kernel_busy) / NS_PER_MS,
            "event_counts": {cat: len(items) for cat, items in categories.items()},
            **gap_summary(gaps, start, end),
        }

    rows = []
    for (device, category, kernel), samples in durations.items():
        samples.sort()
        rows.append({"phase": name, "device": device, "category": category, "name": kernel,
                     "calls": len(samples), "total_ms": sum(samples) / NS_PER_MS,
                     "median_ms": median(samples) / NS_PER_MS,
                     "p95_ms": samples[ceil(len(samples) * 0.95) - 1] / NS_PER_MS,
                     "max_ms": samples[-1] / NS_PER_MS})
    rows.sort(key=lambda row: (row["device"], -row["total_ms"]))
    host_rows = [{"category": category, "name": api, "calls": len(samples), "total_ms": sum(samples) / NS_PER_MS}
                 for (category, api), samples in host.items()]
    host_rows.sort(key=lambda row: -row["total_ms"])
    summary = {"wall_ms": wall / NS_PER_MS,
               "any_device_active_pct": any_busy / wall * 100,
               "all_devices_idle": gap_summary(all_idle, start, end),
               "devices": per_device, "host_api": host_rows}
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", type=int, nargs="+", default=list(range(8)))
    args = parser.parse_args()
    phase_windows = runpy.run_path(str(Path(__file__).with_name("summarize-profile.py")))["phase_windows"]
    trace = json.loads(args.trace.read_text())
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    result, kernels = {}, []
    for start, end, name, _ in phase_windows(args.log):
        result[name], rows = summarize_window(events, start, end, name, args.devices)
        kernels.extend(rows)
    with (args.output_dir / "timeline-summary.json").open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    with (args.output_dir / "kernel-times.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(kernels[0]))
        writer.writeheader()
        writer.writerows(kernels)


if __name__ == "__main__":
    main()
