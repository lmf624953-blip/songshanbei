#!/usr/bin/env python3
"""Summarize HIP, copy, kernel and RCCL durations in recorded FL2VA phases."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path


def phase_windows(log):
    phases = {}
    for line in log.read_text().splitlines():
        tag, separator, payload = line.partition(" ")
        if separator and tag in {"[hipprof.begin]", "[hipprof.end]", "[hipprof.inputs]"}:
            record = json.loads(payload)
            phases.setdefault(record["name"], {})[tag] = record
    windows = []
    for name, phase in phases.items():
        start = phase["[hipprof.begin]"]["epoch_ns"]
        end = phase["[hipprof.end]"]["epoch_ns"]
        windows.append((start, end, name, phase.get("[hipprof.inputs]", {}).get("tensors", {})))
    if not windows:
        raise ValueError("No profiled phases in inference log")
    return sorted(windows)


def merge_duration(intervals):
    """Count the union of intervals on one device across its streams."""
    elapsed = 0
    end = 0
    for begin, finish in sorted(intervals):
        elapsed += max(0, finish - max(end, begin))
        end = max(end, finish)
    return elapsed


def summarize(log, traces):
    windows = phase_windows(log)
    starts = [phase[0] for phase in windows]
    stats = [defaultdict(lambda: [0, 0]) for _ in windows]
    device_intervals = [defaultdict(list) for _ in windows]
    for trace in traces:
        payload = json.loads(trace.read_text())
        events = payload["traceEvents"] if isinstance(payload, dict) else payload
        for event in events:
            args = event.get("args", {})
            if event.get("ph") != "X" or "BeginNs" not in args or "EndNs" not in args:
                continue
            category = event.get("cat")
            if category not in {"HIP", "HIPCOPY", "HIPOPS", "RCCL", "RCCLOPS"}:
                continue
            start, end = int(args["BeginNs"]), int(args["EndNs"])
            index = bisect_right(starts, start) - 1
            if index < 0 or start >= windows[index][1]:
                continue
            end = min(end, windows[index][1])
            item = stats[index][category, event["name"]]
            item[0] += 1
            item[1] += end - start
            if category in {"HIPCOPY", "HIPOPS", "RCCLOPS"} and "devId" in args:
                device_intervals[index][str(args["devId"])].append((start, end))

    result = {}
    for (start, end, phase, inputs), counters, devices in zip(windows, stats, device_intervals):
        if not devices:
            raise ValueError(f"No timed device activities found for {phase}")
        by_category = defaultdict(list)
        for (category, name), (count, duration) in sorted(counters.items(), key=lambda item: item[1][1], reverse=True):
            by_category[category].append({"name": name, "calls": count, "total_ms": duration / 1e6})
        result[phase] = {
            "wall_ms_with_profiler": (end - start) / 1e6,
            "inputs": inputs,
            "device_activity_union_ms": {device: merge_duration(items) / 1e6 for device, items in devices.items()},
            "events": dict(by_category),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.log, args.traces)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for key, phase in result.items():
        print(key, f'{phase["wall_ms_with_profiler"]:.2f} ms')
        for category, events in phase["events"].items():
            print(category, round(sum(event["total_ms"] for event in events), 2), "aggregate ms")
            for event in events[:4]:
                print(" ", event["calls"], f'{event["total_ms"]:.2f}', event["name"][:160])


if __name__ == "__main__":
    main()
