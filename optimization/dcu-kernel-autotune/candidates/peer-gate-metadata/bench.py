#!/usr/bin/env python3
"""八卡分片门控原镜像对照，检查 owner 混合块和 BF16 边界后交替计时。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import sys
from threading import Barrier
from time import perf_counter, sleep

import numpy as np
import torch


RANKS = 8
WIDTH = 5376
GUARD = 0x3535


class Image:
    def __init__(self, root, hip):
        self.meta = json.loads((root / "peer-gate-build.json").read_text())
        image = root / "peer-gate.hsaco"
        assert hashlib.sha256(image.read_bytes()).hexdigest() == self.meta["hsaco_sha256"]
        self.hip = hip
        self.module, self.function = ct.c_void_p(), ct.c_void_p()
        self.check(hip.hipModuleLoad(ct.byref(self.module), os.fsencode(image)))
        self.check(hip.hipModuleGetFunction(ct.byref(self.function), self.module, b"h3_peer_gate_kernel"))

    def check(self, status):
        if status:
            raise RuntimeError(self.hip.hipGetErrorString(status).decode())

    def launch(self, stream, values):
        cells = []
        for arg in self.meta["arguments"]:
            value = values[arg["source"]]
            if arg["kind"] == "tensor":
                cells.append(ct.c_void_p(value.data_ptr()))
            elif arg["kind"] == "shape":
                cells.append(ct.c_int(value.shape[arg["dim"]]))
            else:
                assert arg["kind"] == "scalar"
                cells.append(ct.c_int(value))
        cells.extend([ct.c_void_p(), ct.c_void_p()])
        argv = (ct.c_void_p * len(cells))(*(ct.addressof(value) for value in cells))
        blocks = (values["output"].numel() + self.meta["tile"] - 1) // self.meta["tile"]
        self.check(self.hip.hipModuleLaunchKernel(self.function, blocks, 1, 1,
            self.meta["warps"] * 64, 1, 1, self.meta["shared"], stream, argv, None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("builds", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variants", nargs="+", default=["rotate", "shared-v4", "shared-rotate-v4"])
    parser.add_argument("--pairs", type=int, default=39)
    parser.add_argument("--profile-variant")
    args = parser.parse_args()
    if args.output.exists() or torch.cuda.device_count() != RANKS or args.pairs < 3:
        parser.error("需要八卡、新输出文件与至少三组配对")
    if args.profile_variant and args.profile_variant not in args.variants:
        parser.error("采样对象须在候选列表中")
    root = Path(os.environ["H3_AUTOTUNE_CODE_ROOT"])
    sys.path.insert(0, str(root / "optimization/dcu-kernel-autotune/candidates/rccl-sequence-parallel"))
    from ring_order_map import table
    hip = ct.CDLL("libamdhip64.so")
    hip.hipGetErrorString.argtypes, hip.hipGetErrorString.restype = [ct.c_int], ct.c_char_p
    hip.hipModuleLoad.argtypes = [ct.POINTER(ct.c_void_p), ct.c_char_p]
    hip.hipModuleGetFunction.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p]
    hip.hipModuleLaunchKernel.argtypes = ([ct.c_void_p] + [ct.c_uint] * 7
                                        + [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_void_p])
    transport = ct.CDLL(str(args.reference / "peer-transport.so"))
    transport.transport_setup.argtypes = transport.transport_cleanup.argtypes = []
    assert transport.transport_setup() == 0
    names = ["reference", *args.variants]
    paths = [args.reference, *(args.builds / name for name in args.variants)]
    images, streams, starts, ends = [], [], [], []
    for rank in range(RANKS):
        with torch.cuda.device(rank):
            images.append({name: Image(path, hip) for name, path in zip(names, paths, strict=True)})
            streams.append(torch.cuda.Stream())
            starts.append(torch.cuda.Event(enable_timing=True))
            ends.append(torch.cuda.Event(enable_timing=True))
    report = dict(scope="八卡独立分片gate，合成输入和运行时rank排列，未校准当前communicator且未接入模型",
                  validation=[], timing=[], complete=False, eligible={name: True for name in args.variants},
                  identity={name: hashlib.sha256((path / "peer-gate.hsaco").read_bytes()).hexdigest()
                            for name, path in zip(names, paths, strict=True)})
    barrier = Barrier(RANKS)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def prepare(rows, case, mapping):
        count = rows * WIDTH
        if mapping == "ring":
            owners = table(count)
        elif mapping == "mixed":
            owners = (np.arange(count // 8) % 8).astype(np.uint8)
        else:
            # 每个tile的首尾owner相同、内部不同，检查端点判定的反例。
            owners = np.zeros(count // 8, dtype=np.uint8)
            owners[50::128] = 1
        ring = np.random.default_rng(104729 + rows).permutation(RANKS)
        ringbits = sum(int(value) << (3 * i) for i, value in enumerate(ring))
        arguments, storage = [], {name: [] for name in names}
        sources = []
        for rank in range(RANKS):
            device = torch.device("cuda", rank)
            generator = torch.Generator(device=device).manual_seed(904991 + rank)
            if case == "special":
                patterns = torch.tensor([0, -32768, 1, -32767, 0x7f7f, -129, 0x7f80, -128,
                                         0x7fc1, 0x7f81, -63, 0x3f80, -16512, 0x0080, -32640, 0x3f00],
                                        dtype=torch.int16, device=device)
                value = patterns.roll(rank).repeat((count + 15) // 16)[:count].view(torch.bfloat16)
            else:
                value = torch.randn(count, device=device, dtype=torch.bfloat16, generator=generator)
                value.mul_(2.0 ** (-130 if case == "tiny" else 120 if case == "wide" else rank % 4))
            sources.append(value)
        for rank in range(RANKS):
            device = torch.device("cuda", rank)
            generator = torch.Generator(device=device).manual_seed(30001)
            first, end = rows * rank // RANKS * WIDTH, rows * (rank + 1) // RANKS * WIDTH
            modulation = torch.randn((3, 6 * WIDTH), device=device, dtype=torch.bfloat16, generator=generator)
            values = {f"p{i}": value for i, value in enumerate(sources)}
            values.update(owners=torch.from_numpy(owners).to(device),
                          residual=torch.randn(count, device=device, dtype=torch.bfloat16, generator=generator),
                          gate=modulation[:, 2 * WIDTH:3 * WIDTH],
                          indices=torch.arange(rows, device=device, dtype=torch.int64) % 3,
                          offset=first, ringbits=ringbits)
            for name in names:
                storage[name].append(torch.full((end - first + 16,), GUARD, dtype=torch.int16, device=device))
            arguments.append(values)
            torch.cuda.synchronize(device)
        return arguments, storage

    try:
        with ThreadPoolExecutor(max_workers=RANKS) as workers:
            def execute(name, values, buffers):
                def worker(rank):
                    with torch.cuda.device(rank):
                        current = values[rank] | {"output": buffers[rank][:-16].view(torch.bfloat16)}
                        barrier.wait()
                        starts[rank].record(streams[rank])
                        images[rank][name].launch(streams[rank].cuda_stream, current)
                        ends[rank].record(streams[rank])
                begin = perf_counter()
                list(workers.map(worker, range(RANKS)))
                for event in ends:
                    event.synchronize()
                per_rank = [a.elapsed_time(b) for a, b in zip(starts, ends, strict=True)]
                return dict(wall_ms=(perf_counter() - begin) * 1000, gpu_ms=max(per_rank), per_rank_ms=per_rank)

            cases = [(27570, "random", "ring"), (22899, "random", "ring"), (24062, "random", "ring"),
                     (24577, "random", "ring"), (17, "random", "mixed"), (9, "random", "endpoint-trap"),
                     (4097, "tiny", "ring"), (4097, "wide", "ring"), (4097, "special", "ring")]
            if args.profile_variant:
                cases = cases[:1]
            for rows, case, mapping in cases:
                values, buffers = prepare(rows, case, mapping)
                execute("reference", values, buffers["reference"])
                for name in args.variants:
                    execute(name, values, buffers[name])
                    checks = []
                    for rank in range(RANKS):
                        output = buffers[name][rank]
                        checks.append(dict(rank=rank,
                            mismatches=int(torch.count_nonzero(output[:-16] != buffers["reference"][rank][:-16])),
                            guard=bool((output[-16:] == GUARD).all())))
                    exact = all(row["mismatches"] == 0 and row["guard"] for row in checks)
                    report["eligible"][name] &= exact
                    report["validation"].append(dict(rows=rows, case=case, mapping=mapping,
                                                      variant=name, exact=exact, checks=checks))
                print("validation", rows, case, mapping, report["eligible"], flush=True)
                save()
                del values, buffers
            if args.profile_variant:
                assert report["eligible"][args.profile_variant]
                values, buffers = prepare(27570, "random", "ring")
                for _ in range(2):
                    execute("reference", values, buffers["reference"])
                    execute(args.profile_variant, values, buffers[args.profile_variant])
                assert hip.hipProfilerStart() == 0
                order = ["reference", args.profile_variant]
                report["diagnostic_samples"] = {side: execute(side, values, buffers[side]) for side in order}
                sleep(2)
                assert hip.hipProfilerStop() == 0
                report["profile_order"] = order
                del values, buffers
            for rows in (() if args.profile_variant else (27570, 22899, 24062)):
                values, buffers = prepare(rows, "random", "ring")
                for name in args.variants:
                    if not report["eligible"][name]:
                        continue
                    for _ in range(3):
                        execute("reference", values, buffers["reference"])
                        execute(name, values, buffers[name])
                    pairs = []
                    for trial in range(args.pairs):
                        order = ("reference", name) if trial % 2 == 0 else (name, "reference")
                        pairs.append({side: execute(side, values, buffers[side]) for side in order})
                    result = dict(rows=rows, variant=name, pairs=pairs,
                        reference_ms=median(p["reference"]["gpu_ms"] for p in pairs),
                        candidate_ms=median(p[name]["gpu_ms"] for p in pairs),
                        gain_percent=100 * median(1 - p[name]["gpu_ms"] / p["reference"]["gpu_ms"] for p in pairs))
                    report["timing"].append(result)
                    print({key: value for key, value in result.items() if key != "pairs"}, flush=True)
                    save()
                del values, buffers
    finally:
        assert transport.transport_cleanup() == 0
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
