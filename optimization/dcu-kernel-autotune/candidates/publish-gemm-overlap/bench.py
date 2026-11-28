#!/usr/bin/env python3
"""用八卡现用九齿归一化与 rocBLAS 测量分批发布和投影重叠。"""

import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time

import torch


RANKS = 8
WIDTH = 5376
GUARD = 0x3535
MODES = ("full", "split_serial", "split_overlap", "vector32", "vector64", "vector128", "peer_copy",
         "push_local", "push_links", "pull_links")


def addresses(tensors):
    return (ct.c_uint64 * RANKS)(*(tensor.data_ptr() for tensor in tensors))


def check(error):
    if error:
        raise RuntimeError(error.decode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("norm", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, default=27570)
    parser.add_argument("--m", type=int, default=2688)
    parser.add_argument("--split", type=int)
    parser.add_argument("--pairs", type=int, default=39)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=5120)
    parser.add_argument("--profile-mode", choices=MODES)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES[:3])
    args = parser.parse_args()
    if args.output.exists() or torch.cuda.device_count() != RANKS:
        raise ValueError("需要八卡和新的输出路径")
    split = args.split or args.rows // 2 // 256 * 256
    if not 0 < split < args.rows or args.pairs < 3 or args.warmup < 2:
        parser.error("分块边界或计时参数不合法")
    modes = tuple(args.modes)
    if modes[0] != "full" or len(set(modes)) != len(modes) or len(modes) < 2:
        parser.error("对照列表需以 full 开始，且不得重复")
    if args.profile_mode and args.profile_mode not in modes:
        parser.error("采样对象必须在对照列表中")

    library = ct.CDLL(str(args.library))
    norm = ct.CDLL(str(args.norm / "wide-norm-indexed-host.so"))
    metadata = json.loads((args.norm / "wide-norm-indexed-build.json").read_text())
    image = args.norm / "wide-norm-indexed.hsaco"
    assert hashlib.sha256(image.read_bytes()).hexdigest() == metadata["hsaco_sha256"]
    norm.songshan_prepare_widenorm_indexed.argtypes = [ct.c_char_p, ct.c_int, ct.c_int]
    norm.songshan_prepare_widenorm_indexed.restype = ct.c_char_p
    check(norm.songshan_prepare_widenorm_indexed(os.fsencode(image), metadata["warps"], metadata["shared"]))
    library.pipeline_setup.argtypes = [ct.c_void_p]
    library.pipeline_run.argtypes = [ct.c_int] * 5 + [ct.POINTER(ct.c_uint64)] * 9 + [ct.POINTER(ct.c_float)]
    library.pipeline_close.argtypes = []
    for name in ("pipeline_setup", "pipeline_run", "pipeline_close"):
        getattr(library, name).restype = ct.c_char_p

    generator = torch.Generator().manual_seed(295037 + args.rows)
    host = torch.randn((args.rows, WIDTH), generator=generator, dtype=torch.bfloat16)
    host_weight = torch.randn(WIDTH, generator=generator, dtype=torch.bfloat16)
    host_table = torch.randn((3, 6 * WIDTH), generator=generator, dtype=torch.bfloat16)
    hidden, norm_weights, tables, indices, partial, local, matrices = ([] for _ in range(7))
    outputs = [[] for _ in modes]
    reference_local = []
    for rank in range(RANKS):
        with torch.cuda.device(rank):
            hidden.append(host.to(rank))
            norm_weights.append(host_weight.to(rank))
            tables.append(host_table.to(rank))
            indices.append(torch.arange(args.rows, device=rank, dtype=torch.int64) % 3)
            owned = args.rows * (rank + 1) // RANKS - args.rows * rank // RANKS
            partial.append(torch.empty(owned * WIDTH + 16, device=rank, dtype=torch.bfloat16))
            local.append(torch.empty(args.rows * WIDTH + 16, device=rank, dtype=torch.bfloat16))
            reference_local.append(torch.empty_like(local[-1]))
            device_rng = torch.Generator(device=f"cuda:{rank}").manual_seed(492003 + rank + args.m)
            matrices.append(torch.randn((args.m, WIDTH), device=rank, dtype=torch.bfloat16, generator=device_rng))
            for output in outputs:
                output.append(torch.empty(args.rows * args.m + 16, device=rank, dtype=torch.bfloat16))
            torch.cuda.synchronize(rank)

    scale = (ct.c_uint64 * RANKS)(*(table.data_ptr() + 4 * WIDTH * 2 for table in tables))
    shift = (ct.c_uint64 * RANKS)(*(table.data_ptr() + 3 * WIDTH * 2 for table in tables))
    arguments = [addresses(hidden), addresses(norm_weights), scale, shift, addresses(indices),
                 addresses(partial), addresses(local), addresses(matrices)]
    output_arguments = [addresses(output) for output in outputs]

    def run(mode):
        device_ms = (ct.c_float * RANKS)()
        start = time.perf_counter()
        check(library.pipeline_run(MODES.index(modes[mode]), args.rows, args.m, split, args.blocks, *arguments,
                                   output_arguments[mode], device_ms))
        return dict(wall_ms=(time.perf_counter() - start) * 1000, device_ms=list(device_ms))

    report = dict(rows=args.rows, m=args.m, k=WIDTH, split=split, blocks=args.blocks,
                  scope="八卡九齿归一化、发布和原 rocBLAS 投影，包含主机及同步",
                  norm_sha256=metadata["hsaco_sha256"],
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  validation=[], pairs=[])
    try:
        check(library.pipeline_setup(ct.cast(norm.songshan_bench_widenorm_indexed, ct.c_void_p)))
        for mode in range(len(modes)):
            for rank in range(RANKS):
                for tensor in (local[rank], partial[rank], outputs[mode][rank]):
                    tensor.view(torch.int16).fill_(GUARD)
                torch.cuda.synchronize(rank)
            run(mode)
            for rank in range(RANKS):
                if mode == 0:
                    reference_local[rank].copy_(local[rank])
                exact_input = torch.equal(local[rank].view(torch.int16), reference_local[rank].view(torch.int16))
                exact_output = torch.equal(outputs[mode][rank].view(torch.int16),
                                           outputs[0][rank].view(torch.int16))
                guards = all(bool((tensor[-16:].view(torch.int16) == GUARD).all())
                             for tensor in (local[rank], partial[rank], outputs[mode][rank]))
                if modes[mode] in ("push_local", "push_links", "pull_links"):
                    guards = guards and bool((partial[rank].view(torch.int16) == GUARD).all())
                finite = bool(torch.isfinite(outputs[mode][rank][:-16]).all())
                row = dict(mode=modes[mode], rank=rank, input_exact=exact_input, output_exact=exact_output,
                           guards=guards, finite=finite)
                if not exact_output:
                    row["output_mismatches"] = int(torch.count_nonzero(outputs[mode][rank].view(torch.int16) !=
                                                                        outputs[0][rank].view(torch.int16)))
                report["validation"].append(row)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            if not all(all(row[key] for key in ("input_exact", "output_exact", "guards", "finite"))
                       for row in report["validation"]):
                raise ValueError(f"{modes[mode]} 逐位或边界校验失败")
        print("八卡归一化输入、投影输出和边界通过逐位验证", flush=True)
        for _ in range(args.warmup):
            for mode in range(len(modes)):
                run(mode)
        if args.profile_mode:
            hip = ct.CDLL("libamdhip64.so")
            if hip.hipProfilerStart():
                raise RuntimeError("hipProfilerStart failed")
            print("[hipprof.begin]", json.dumps(dict(name=args.profile_mode, epoch_ns=time.time_ns())), flush=True)
            report["profile_sample"] = run(modes.index(args.profile_mode))
            print("[hipprof.end]", json.dumps(dict(name=args.profile_mode, epoch_ns=time.time_ns())), flush=True)
            time.sleep(2)
            if hip.hipProfilerStop():
                raise RuntimeError("hipProfilerStop failed")
            report["profiler"] = args.profile_mode
        else:
            forward = list(range(len(modes)))
            orders = []
            for offset in range(len(modes)):
                rotated = forward[offset:] + forward[:offset]
                orders.extend((rotated, list(reversed(rotated))))
            for index in range(args.pairs):
                order = orders[index % len(orders)]
                report["pairs"].append(dict(order=list(order), **{modes[mode]: run(mode) for mode in order}))
            report["median_wall_ms"] = {mode: median(pair[mode]["wall_ms"] for pair in report["pairs"])
                                        for mode in modes}
            report["median_max_device_ms"] = {mode: median(max(pair[mode]["device_ms"]) for pair in report["pairs"])
                                              for mode in modes}
            comparisons = [("full", mode) for mode in modes[1:]]
            if "split_serial" in modes and "split_overlap" in modes:
                comparisons.append(("split_serial", "split_overlap"))
            report["paired_gain_percent"] = {
                f"{base}_to_{candidate}": 100 * median(1 - pair[candidate]["wall_ms"] / pair[base]["wall_ms"]
                                                        for pair in report["pairs"])
                for base, candidate in comparisons
            }
        report["complete"] = True
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print({key: value for key, value in report.items() if key.startswith(("median_", "paired_"))}, flush=True)
    finally:
        check(library.pipeline_close())


if __name__ == "__main__":
    main()
