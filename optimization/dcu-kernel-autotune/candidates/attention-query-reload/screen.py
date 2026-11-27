#!/usr/bin/env python3
"""先校验尾块与压力输入，再串行比较资源对照及 Q 重载的正式配对时间。"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import torch


def occupancy(path, kernel=b"dense_attention_aot_kernel", shared=16384):
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    hip.hipModuleUnload.argtypes = [ctypes.c_void_p]
    module, function, count = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_int()
    for status in (hip.hipModuleLoad(ctypes.byref(module), os.fsencode(path)),):
        if status:
            raise RuntimeError(f"hipModuleLoad: {status}")
    status = hip.hipModuleGetFunction(ctypes.byref(function), module, kernel)
    if status:
        raise RuntimeError(f"hipModuleGetFunction: {status}")
    status = hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(count), function, 256, shared)
    if status:
        raise RuntimeError(f"hipModuleOccupancyMaxActiveBlocksPerMultiprocessor: {status}")
    hip.hipModuleUnload(module)
    return count.value


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("experiment", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--pairs", type=int, default=39)
    args = cli.parse_args()
    code = Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization"
    sys.path.insert(0, str(code))
    sibling = Path(__file__).resolve().parents[1] / "pv-v-lds-layout"
    screen = runpy.run_path(str(sibling / "screen.py"))["screen"]
    load = runpy.run_path(str(code / "bench-attention.py"))["native_launcher"]
    torch.cuda.init()
    baseline = load(args.reference)
    builds = [args.experiment / name for name in ("resource-control", "reload-full", "reload-compact")]
    result = dict(reference=str(args.reference), reference_packed_sha256=hashlib.sha256(
        (args.reference / "dense-attention-packed.hsaco").read_bytes()).hexdigest(),
        scope="单卡正式ABI，含每次V预排布及完整注意力；无profiler；不代表端到端成绩",
        profiler=False, pairs=args.pairs, records=[])
    with args.output.open("x") as destination:
        def record(row):
            result["records"].append(row)
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            destination.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)
        record(dict(kind="reference-occupancy", active_blocks_per_CU=occupancy(
            args.reference / "dense-attention-packed.hsaco")))
        for build in builds:
            for original in args.reference.iterdir():
                if original.name not in ("dense-attention-packed.hsaco", "dense-attention-packed.amdgcn",
                                         "dense-attention-aot-build.json"):
                    assert (build / original.name).read_bytes() == original.read_bytes(), original.name
            record(dict(kind="candidate-occupancy", variant=build.name,
                active_blocks_per_CU=occupancy(build / "dense-attention-packed.hsaco"),
                packed_sha256=hashlib.sha256((build / "dense-attention-packed.hsaco").read_bytes()).hexdigest()))
            candidate = load(build)
            for rows in (4096, 4097, 4159, 4160, 4161, 4192, 4223, 4224, 4225, 22899, 22016, 27570):
                row = screen(baseline, candidate, rows, 1)
                record(dict(kind="normal-stress-tail", variant=build.name, **row))
                if not row["passed"]:
                    raise ValueError("逐位、有限值或尾部校验未通过，停止此轮")
        # 所有待测候选先通过数值校验；正式计时期间不编译。
        for build in builds:
            candidate = load(build)
            for rows in (27570, 22016):
                row = screen(baseline, candidate, rows, args.pairs)
                record(dict(kind="formal-paired", variant=build.name, **row))
                if not row["passed"]:
                    raise ValueError("正式计时后逐位校验失败")
    (args.output.parent / (args.output.stem + "-summary.json")).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
