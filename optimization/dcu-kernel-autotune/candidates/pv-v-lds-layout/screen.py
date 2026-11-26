#!/usr/bin/env python3
"""对 V 片上布局候选做全覆盖校验和无 profiler 交替计时。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
from statistics import median
import sys

import torch


def screen(reference, candidate, length, iterations):
    generator = torch.Generator(device="cuda:0").manual_seed(42)

    def packed(heads):
        return torch.randn((1, length, heads, 128), dtype=torch.bfloat16,
                           device="cuda:0", generator=generator)

    q, k = (packed(7).transpose(1, 2) for _ in range(2))
    v = packed(21)[:, :, -7:].transpose(1, 2)
    storage = [torch.empty((1, length + 2, 7, 128), dtype=q.dtype, device=q.device) for _ in range(2)]
    outputs = [x[:, :length].transpose(1, 2) for x in storage]
    launchers = (reference, candidate)

    def run(i):
        launchers[i](q, k, v, 128**-0.5, outputs[i])

    def check():
        for buffer in storage:
            buffer.fill_(float("nan"))
            buffer[:, length:].fill_(-19)
        run(0)
        run(1)
        torch.cuda.synchronize()
        x, y = outputs
        return dict(exact=torch.equal(x.view(torch.int16), y.view(torch.int16)),
                    finite=all(bool(torch.isfinite(out).all()) for out in outputs),
                    tail_ok=all(bool((out[:, length:] == -19).all()) for out in storage),
                    max_abs=float((x.float() - y.float()).abs().max()))

    normal = check()
    if not all(normal[k] for k in ("exact", "finite", "tail_ok")):
        return dict(sequence=length, normal=normal, passed=False)
    samples = [[], []]
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for trial in range(iterations + 2):
        for i in (trial % 2, 1 - trial % 2):
            start.record()
            run(i)
            end.record()
            end.synchronize()
            if trial >= 2:
                samples[i].append(start.elapsed_time(end))
    q.mul_(2)
    k.mul_(2)
    stress = check()
    return dict(sequence=length, normal=normal, stress=stress, samples_ms=samples,
                medians_ms=[median(series) for series in samples],
                paired_gain_percent=100 * median(1-b/a for a,b in zip(*samples, strict=True)),
                passed=all(stress[k] for k in ("exact", "finite", "tail_ok")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("new_builds", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    code = Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization"
    sys.path.insert(0, str(code))
    load = runpy.run_path(str(code / "bench-attention.py"))["native_launcher"]
    workspace = Path(__file__).resolve().parents[2]
    elf = runpy.run_path(str(workspace / "adapters/compare-h3-device.py"))["elf_loads"]
    args.out.mkdir()
    ref = (args.reference / "dense-attention-aot.hsaco").resolve()
    ctrl = (args.new_builds / "v0/dense-attention-aot.hsaco").resolve()
    a, b = elf(ref), elf(ctrl)
    assert all(a[k] == b[k] for k in ("sections", "segments", "elf_target")), "重编对照可加载区段发生变化"
    base = load(args.reference.resolve())
    rows = []
    for path in sorted(args.new_builds.glob("v*/dense-attention-aot-build.json")):
        candidate = load(path.parent.resolve())
        record = json.loads(path.read_text())
        cases = [screen(base, candidate, length, 9) for length in (27570, 22016)]
        result = dict(name=path.parent.name, cases=cases, layout=record["experiment"]["v_layout"],
                      shared=record["shared"], isa=record["experiment"]["isa"],
                      binary=record["hsaco_sha256"])
        rows.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        (args.out / "progress.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    winners = [r for r in rows if r["name"] != "v0"
               and all(c["passed"] and c["paired_gain_percent"] >= 0.5 for c in r["cases"])]
    winners.sort(key=lambda r: -min(c["paired_gain_percent"] for c in r["cases"]))
    result = dict(reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),
                  control_sections_exact=True, results=rows, winners=[r["name"] for r in winners])
    (args.out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
