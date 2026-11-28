#!/usr/bin/env python3
"""八卡独立行投影。完整结果逐位一致的候选才进入无 profiler 配对。"""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter, sleep

import torch


RANKS = 8
WIDTH = 5376
GUARD = 0x3535
MODES = ("full", "split_local", "split_peer")


def addresses(tensors):
    return (ct.c_uint64 * RANKS)(*(tensor.data_ptr() for tensor in tensors))


def check(error):
    if error:
        raise RuntimeError(error.decode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, default=27570)
    parser.add_argument("--k", type=int, choices=(896, 1792), required=True)
    parser.add_argument("--pairs", type=int, default=39)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.rows < RANKS or args.output.exists() or args.pairs < 3 or torch.cuda.device_count() != RANKS:
        parser.error("需要八卡、新输出路径、至少八行和三组配对")

    library = ct.CDLL(str(args.library))
    library.scatter_run.argtypes = ([ct.c_int] * 3 + [ct.c_size_t] + [ct.POINTER(ct.c_uint64)] * 3
                                    + [ct.POINTER(ct.c_float)])
    library.scatter_setup.argtypes = library.scatter_close.argtypes = []
    for function in (library.scatter_run, library.scatter_setup, library.scatter_close):
        function.restype = ct.c_char_p
    plane = ((args.rows + RANKS - 1) // RANKS) * WIDTH + 16
    inputs, weights = [], []
    outputs = {mode: [] for mode in MODES}
    report = dict(rows=args.rows, k=args.k, width=WIDTH, plane_stride=plane,
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  scope="独立八卡原 rocBLAS 行并行 GEMM，不含后继门控和完整推理",
                  complete=False, validation={}, timing={})

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    for rank in range(RANKS):
        device = torch.device("cuda", rank)
        generator = torch.Generator(device=device).manual_seed(923827 + 19 * rank)
        inputs.append(torch.randn((args.rows, args.k), device=device, dtype=torch.bfloat16, generator=generator))
        weights.append(torch.randn((WIDTH, args.k), device=device, dtype=torch.bfloat16, generator=generator))
        for mode, buffers in outputs.items():
            elements = (plane * RANKS if mode == "split_peer" else args.rows * WIDTH + 16)
            buffers.append(torch.full((elements,), GUARD, device=device, dtype=torch.int16))
        torch.cuda.synchronize(device)
    input_addresses, weight_addresses = addresses(inputs), addresses(weights)
    output_addresses = {mode: addresses(buffers) for mode, buffers in outputs.items()}

    def run(mode):
        device_ms = (ct.c_float * RANKS)()
        start = perf_counter()
        check(library.scatter_run(MODES.index(mode), args.rows, args.k, plane, input_addresses,
                                  weight_addresses, output_addresses[mode], device_ms))
        return dict(wall_ms=(perf_counter() - start) * 1000, device_ms=list(device_ms))

    try:
        check(library.scatter_setup())
        run("full")
        reference = outputs["full"]
        assert all(bool((tensor[-16:] == GUARD).all()) and bool(torch.isfinite(tensor[:-16].view(torch.bfloat16)).all())
                   for tensor in reference)
        for mode in MODES[1:]:
            row = report["validation"][mode] = []
            run(mode)
            if mode == "split_local":
                for rank, (actual, expected) in enumerate(zip(outputs[mode], reference, strict=True)):
                    mismatch = int(torch.count_nonzero(actual[:-16] != expected[:-16]))
                    row.append(dict(rank=rank, mismatches=mismatch, guard=bool((actual[-16:] == GUARD).all())))
            else:
                for owner, target in enumerate(outputs[mode]):
                    first = args.rows * owner // RANKS * WIDTH
                    end = args.rows * (owner + 1) // RANKS * WIDTH
                    length = end - first
                    for rank, expected in enumerate(reference):
                        actual = target[rank * plane:(rank + 1) * plane]
                        mismatch = int(torch.count_nonzero(actual[:length] != expected[first:end].to(target.device)))
                        row.append(dict(owner=owner, rank=rank, mismatches=mismatch,
                                        guard=bool((actual[length:] == GUARD).all())))
            exact = all(record["mismatches"] == 0 and record["guard"] for record in row)
            print(mode, "exact", exact, "mismatches", sum(record["mismatches"] for record in row), flush=True)
            save()
            if not exact or args.profile:
                continue
            for _ in range(4):
                run("full")
                run(mode)
            pairs = []
            for trial in range(args.pairs):
                order = ("full", mode) if trial % 2 == 0 else (mode, "full")
                pairs.append(dict(order=order, **{side: run(side) for side in order}))
            timing = dict(pairs=pairs, median_wall_ms={side: median(pair[side]["wall_ms"] for pair in pairs)
                                                      for side in ("full", mode)},
                          median_max_device_ms={side: median(max(pair[side]["device_ms"]) for pair in pairs)
                                                for side in ("full", mode)},
                          paired_gain_percent=100 * median(1 - pair[mode]["wall_ms"] / pair["full"]["wall_ms"]
                                                           for pair in pairs))
            report["timing"][mode] = timing
            print({key: value for key, value in timing.items() if key != "pairs"}, flush=True)
            save()
        if args.profile:
            if not all(record["mismatches"] == 0 and record["guard"]
                       for record in report["validation"]["split_peer"]):
                raise RuntimeError("直接发布候选未通过 PMC 前的完整逐位验证")
            hip = ct.CDLL("libamdhip64.so")
            assert hip.hipProfilerStart() == 0
            report["profile_order"] = ["full", "split_peer"]
            for side in report["profile_order"]:
                run(side)
            sleep(2)
            assert hip.hipProfilerStop() == 0
        report["complete"] = True
        save()
    finally:
        check(library.scatter_close())


if __name__ == "__main__":
    main()
