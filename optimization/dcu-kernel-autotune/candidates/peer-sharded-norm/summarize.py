#!/usr/bin/env python3
"""保存分片归一化独立原型的数值、配对、原始PMC及实际trace窗口。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def union_ns(intervals):
    end, total = None, 0
    for first, last in sorted(intervals):
        if end is None or first > end:
            total += last-first
        elif last > end:
            total += last-end
        end = max(end or last, last)
    return total


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("experiment", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    root = args.experiment
    for name in ["screen-exit-code", "split-screen-exit-code", "profile-exit-code"]:
        assert (root/name).read_text().strip() == "0", name
    experiments, performance, pmc = {}, {}, {}
    for path in sorted(root.glob("*.json")):
        if path.name.endswith(("_stream.json", "summary.json")) or path.name.startswith("split-trace") or path.resolve() == args.output.resolve():
            continue
        data = json.loads(path.read_text())
        experiments[path.name] = data
        if path.name not in {"clean.json", "split-clean.json", "split-clean-repeat.json"}:
            continue
        results = []
        for row in data["records"]:
            assert not row["consumer_copy"] and len(row["samples"]) == 39
            assert all(all(v for k, v in r.items() if k != "rank") for r in row["validated"])
            assert len(row["residual_chain"]) == 3
            saved = statistics.median(s["reference"]["gpu_ms"] - s["candidate"]["gpu_ms"] for s in row["samples"])
            result = dict(rows=row["rows"], count=row["count"], reference_median_ms=row["reference_median_ms"],
                          candidate_median_ms=row["candidate_median_ms"], paired_gain_percent=row["paired_gain_percent"],
                          paired_saved_ms=saved)
            if row["rows"] == 27570:
                result["unverified_A_projection"] = dict(calls=4851, seconds=saved*4.851,
                    reduction_percent=saved*4.851/120.69804317139399*100,
                    note="按每次Transformer仅99个可替换norm计算；未计模型集成及末层物化差额，非模型成绩或上界")
            results.append(result)
        performance[path.name] = results
    assert len(performance) == 3
    for name, data in experiments.items():
        if "consumer-" not in name or "-chain.json" not in name:
            continue
        assert len(data["records"]) == 3, name
        for row in data["records"]:
            assert row["consumer_copy"] and len(row["residual_chain"]) == 3, name
            assert all(all(v for k,v in detail.items() if k != "rank") for detail in row["validated"]), name
            for step in row["residual_chain"]:
                assert all(all(v for k,v in detail.items() if k != "rank") for detail in step["validated"]), name
    fallback = experiments["tree-fallback.json"]["records"]
    assert all(r["candidate_stats"]["fused"] == 0 and r["candidate_stats"]["fallback"] > 0 for r in fallback)
    for path in sorted(root.glob("split-*-pmc.txt")):
        blocks = path.read_text().split("kernel-name:")[1:]
        assert len(blocks) == 24, (path, len(blocks))
        devices, fields = [], {}
        for block in blocks:
            devices.append(int(re.search(r"^ gpu id\s+(\d+)", block, re.M)[1]))
            for line in block.splitlines()[1:]:
                match = re.fullmatch(r" (.*?)\s{2,}([^\s].*)", line)
                if match:
                    fields.setdefault(match[1], []).append(match[2])
        assert all(devices.count(r) == 3 for r in range(8))
        pmc[path.name] = dict(records=24, raw_fields=fields, sha256=digest(path))
    assert len(pmc) == 3
    # 使用绝对epoch ns，避免hipprof的相对ts起点和主机日志混用。
    tags = re.findall(r"\[profile\.(begin|end)\] (\d+)", (root/"trace.log").read_text())
    assert [x[0] for x in tags] == ["begin", "end"]*3
    begin, end = int(tags[-2][1]), int(tags[-1][1])
    events = json.loads((root/"split-trace_stream.json").read_text())["traceEvents"]
    selected, intervals = [], []
    stages = {}
    for event in events:
        if event.get("cat") not in {"HIPOPS", "RCCLOPS", "HIPCOPY"}:
            continue
        info = event.get("args", {})
        if "BeginNs" not in info or "EndNs" not in info:
            continue
        first, last = int(info["BeginNs"]), int(info["EndNs"])
        if first >= end or last <= begin:
            continue
        first, last = max(first, begin), min(last, end)
        intervals.append((first, last))
        name, device = event["name"], int(info["devId"])
        stage = "gate" if name == "h3_peer_gate_kernel" else "norm" if name == "wide_norm_indexed_kernel" else "publish" if "publish_rows" in name else "other"
        stages.setdefault(str(device), {}).setdefault(stage, []).append((last-first)/1e6)
        selected.append(dict(device=device, stage=stage, name=name,
                             start_ms=(first-begin)/1e6, duration_ms=(last-first)/1e6))
    assert len(selected) == 24 and all(set(stages[str(r)]) == {"gate", "norm", "publish"} for r in range(8))
    trace = dict(scope="独立profiler最后一次无校准窗口，不作性能成绩", begin_epoch_ns=begin, end_epoch_ns=end,
                 wall_ms=(end-begin)/1e6, all_devices_idle_ms=((end-begin)-union_ns(intervals))/1e6,
                 leading_idle_ms=(min(x[0] for x in intervals)-begin)/1e6,
                 trailing_idle_ms=(end-max(x[1] for x in intervals))/1e6,
                 internal_idle_ms=(max(x[1] for x in intervals)-min(x[0] for x in intervals)-union_ns(intervals))/1e6,
                 device_stage_ms=stages, kernels=selected, path=str(root/"split-trace_stream.json"))
    here = Path(__file__).resolve().parent
    sources = {p.name:digest(p) for p in here.iterdir() if p.is_file()}
    builds = {str(p.relative_to(root)):json.loads(p.read_text()) for p in root.glob("*/peer-runtime-build.json")}
    record = dict(node="f13r1n16", experiment=str(root), scope="独立八卡gate/norm/publish原型；尚未接入模型",
                  model_hooked=False, qualified=False, adopted=False,
                  selected_release="validated-build/peer-gate-nt-f13r1n16", selected_steady_A_seconds=120.69804317139399,
                  source_sha256=sources, builds=builds, performance=performance, pmc=pmc,
                  split_repeat_driver_snapshot=dict(path=str(root/"source-snapshot/bench-chain.py"),
                      sha256=digest(root/"source-snapshot/bench-chain.py"),
                      note="Tree回退诊断之前的独立repeat驱动快照；当前驱动另补充回退初始化及work恢复，计时内kernel链未变"),
                  trace=trace, experiments=experiments,
                  next="接入模型分片hidden生命周期与norm分发；首层保持原路径，末层及回退完整物化；实际八卡早中晚逐位与无profiler回放先于完整官方/稳态验收")
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
