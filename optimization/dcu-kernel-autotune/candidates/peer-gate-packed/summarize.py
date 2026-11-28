#!/usr/bin/env python3
"""汇总双 BF16 门控的完整位模式、无 profiler 配对及独立 PMC。"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
from statistics import mean


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root, output = args.experiment, args.output
    assert not output.exists()
    for name in ("gate-micro", "gate-paired", "packed-pmc", "packed-256-rebuild"):
        assert (root / f"{name}-exit-code").read_text().strip() == "0", name
    micro = json.loads((root / "gate-paired.json").read_text())
    assert micro["complete"] and all(micro["eligible"].values())
    assert len(micro["validation"]) == 27 and len(micro["timing"]) == 9
    assert all(row["exact"] and len(row["checks"]) == 8 for row in micro["validation"])
    assert all(len(row["pairs"]) == 39 for row in micro["timing"])
    pmc = json.loads((root / "packed-pmc.json").read_text())
    assert pmc["complete"] and pmc["profile_order"] == ["reference", "packed-v2-256"]
    assert "HIP_PROF:finish" in (root / "packed-pmc.log").read_text()
    rows = []
    for section in (root / "packed-pmc.txt").read_text().split("kernel-name:")[1:]:
        lines = section.splitlines()
        rows.append(dict(name=lines[0].strip('"'),
                         **dict(line.strip().rsplit(maxsplit=1) for line in lines[1:] if line.strip())))
    assert len(rows) == 16 and all(row["name"] == "h3_peer_gate_kernel" for row in rows)
    groups = {}
    for name, vgpr in (("reference", "60"), ("packed-v2-256", "24")):
        group = [row for row in rows if row["vgpr count"] == vgpr]
        assert len(group) == 8 and sorted(int(row["gpu id"]) for row in group) == list(range(8))
        groups[name] = dict(rows=group, counters={})
        for key in ("kernel time", "processed ALU instructions", "shared memory bank conflict",
                    "L1 cache unit is active", "L1 cache unit is stalled", "L2 cache hit rate"):
            values = [float(row[key].split("(")[0]) for row in group]
            groups[name]["counters"][key] = dict(min=min(values), mean=mean(values), max=max(values))
    elf = runpy.run_path(str(Path(__file__).resolve().parents[2] / "adapters/compare-h3-device.py"))["elf_loads"]
    original = elf(root / "packed-v2-256/peer-gate.hsaco")
    rebuilt = elf(root / "packed-256-rebuild/peer-gate.hsaco")
    assert all(original[key] == rebuilt[key] for key in ("sections", "segments", "elf_target"))
    source_names = ("build.py", "kernel.py", "bench.py")
    for name in source_names:
        assert digest(root / name) == digest(Path(__file__).with_name(name)), name
    metadata_source = Path(__file__).resolve().parent.parent / "peer-gate-metadata/bench.py"
    assert digest(metadata_source) == digest(root / "metadata-bench-used.py")
    result = dict(node="a02r3n04", experiment=str(root), adopted=False, micro=micro,
                  pmc=dict(report=pmc, groups=groups),
                  rebuild=dict(passed=True, original=original, rebuilt=rebuilt),
                  source_sha256={name: digest(root / name) for name in source_names},
                  shared_harness_sha256=digest(metadata_source),
                  first_timing_excluded=dict(path="gate-micro.json", sha256=digest(root / "gate-micro.json"),
                      reason="首次候选在事件内额外创建位视图；已对齐两侧准备与检查后重测，正式结论只用gate-paired.json"),
                  failed_builds_preserved=[f"packed-{tile}" for tile in (256, 512, 1024)],
                  scope=micro["scope"],
                  decision="降低rank选择指令和VGPR未形成可靠提速，归档；未接入真实模型或修改正式选版")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    output.with_name(output.stem + "-pmc.txt").write_bytes((root / "packed-pmc.txt").read_bytes())
    print(json.dumps(dict(adopted=False, rank_checks=216, timing_pairs=351, pmc_dispatches=16,
                          rebuild_loads_equal=True), ensure_ascii=False))


if __name__ == "__main__":
    main()
