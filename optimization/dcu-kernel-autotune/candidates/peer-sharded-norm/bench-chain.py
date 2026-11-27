#!/usr/bin/env python3
"""八卡原序求和/门控→宽归一化→发布的独立链；不构成模型接入。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rccl-sequence-parallel"))
from bridge_micro import InfiniCCLProxy, install_adapter


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("candidate", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--reference", required=True, type=Path)
    cli.add_argument("--indexed", required=True, type=Path)
    cli.add_argument("--norm", required=True, type=Path)
    cli.add_argument("--rows", type=int, nargs="+", default=[27570, 22899, 24062])
    cli.add_argument("--pairs", type=int, default=39)
    cli.add_argument("--case", choices=["random", "tiny", "wide", "special"], default="random")
    cli.add_argument("--consumer-copy", action="store_true")
    cli.add_argument("--profile-candidate-only", action="store_true", help="独立profiler工作负载；跳过对照与数值资格，结果不得作正式性能")
    args = cli.parse_args()
    assert torch.cuda.device_count() == 8 and args.pairs > 0
    proxy = InfiniCCLProxy(Path(os.environ["INFINI_ROOT"]) / "lib/libinfiniccl.so")
    comms = (ct.c_void_p * 8)()
    proxy.ncclCommInitAll(comms, 8, (ct.c_int * 8)(*range(8)))
    indexed = ct.CDLL(str((args.indexed / "indexed-host.so").resolve()))
    indexed_meta = json.loads((args.indexed / "indexed-build.json").read_text())
    indexed.songshan_prepare_indexed.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_int]
    indexed.songshan_prepare_indexed.restype = ct.c_char_p
    error = indexed.songshan_prepare_indexed(str(args.indexed / "indexed_adaln.hsaco").encode(),
        str(args.indexed / "indexed_gate.hsaco").encode(), indexed_meta["indexed_gate"]["warps"])
    assert not error, error
    identities = {}
    for label, build in [("candidate", args.candidate), ("reference", args.reference)]:
        record = json.loads((build / "peer-runtime-build.json").read_text())
        identities[label] = {}
        for name in ["peer-gate.hsaco", "peer-gate-bridge.so", "peer-transport.so"]:
            digest = hashlib.sha256((build / name).read_bytes()).hexdigest()
            assert digest == record["sha256"][name], (build, name)
            identities[label][name] = digest
    candidate = ct.CDLL(str((args.candidate / "peer-gate-bridge.so").resolve()))
    split_arrivals = json.loads((args.candidate / "peer-runtime-build.json").read_text()).get("split_arrivals", False)
    if split_arrivals:
        candidate.h3_peer_chain_norm_raw.argtypes = [ct.c_int]
        candidate.h3_peer_chain_norm_raw.restype = ct.c_char_p
    reference = ct.CDLL(str((args.reference / "peer-gate-bridge.so").resolve()))
    for library, path in [(candidate, args.candidate), (reference, args.reference)]:
        install_adapter(library, path, args.indexed, proxy)
        assert library.peer_setup() == 0
    norm = ct.CDLL(str((args.norm / "wide-norm-indexed-host.so").resolve()))
    meta = json.loads((args.norm / "wide-norm-indexed-build.json").read_text())
    assert hashlib.sha256((args.norm / "wide-norm-indexed.hsaco").read_bytes()).hexdigest() == meta["hsaco_sha256"]
    norm.songshan_prepare_widenorm_indexed.argtypes = [ct.c_char_p, ct.c_int, ct.c_int]
    norm.songshan_prepare_widenorm_indexed.restype = ct.c_char_p
    norm.songshan_bench_widenorm_indexed.argtypes = [ct.c_int] * 3 + [ct.c_void_p] * 6 + [ct.c_uint64]
    norm.songshan_bench_widenorm_indexed.restype = ct.c_char_p
    error = norm.songshan_prepare_widenorm_indexed(str(args.norm / "wide-norm-indexed.hsaco").encode(), meta["warps"], meta["shared"])
    assert not error, error
    candidate.h3_peer_chain_prepare_norm.argtypes = [ct.c_void_p]
    candidate.h3_peer_chain_prepare_norm.restype = ct.c_char_p
    error = candidate.h3_peer_chain_prepare_norm(ct.cast(norm.songshan_bench_widenorm_indexed, ct.c_void_p))
    assert not error, error
    candidate.h3_peer_chain_set_norm.argtypes = [ct.c_uint64] * 4
    candidate.h3_peer_chain_set_norm.restype = None
    hip = ct.CDLL("libamdhip64.so")
    hip.hipSetDevice.argtypes = [ct.c_int]
    hip.hipSetDevice.restype = ct.c_int
    streams = [torch.cuda.Stream(device=r) for r in range(8)]
    starts = [torch.cuda.Event(enable_timing=True) for _ in streams]
    ends = [torch.cuda.Event(enable_timing=True) for _ in streams]
    records = []
    poison = 0x3535

    def sync():
        for stream in streams:
            stream.synchronize()

    def check_error(message):
        if message:
            raise RuntimeError(message.decode())

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for rows in args.rows:
                count = rows * 5376
                source, work, residual, tables, indices, weights = [], [], [], [], [], []
                by, bz, cy, cz, bc, cc = [], [], [], [], [], []
                for rank in range(8):
                    with torch.cuda.device(rank):
                        gen = torch.Generator(device=rank).manual_seed(104729 + rank)
                        if args.case == "special":
                            bits = [0, 0x8000, 1, 0x8001, 0x7f7f, 0xff7f, 0x7f80, 0xff80,
                                    0x7fc1, 0x7f81, 0xffc1, 0x3f80, 0xbf80, 0x0080, 0x8080, 0x3f00]
                            bits = bits[rank:] + bits[:rank]
                            bits = torch.tensor([v if v < 32768 else v - 65536 for v in bits], device=rank, dtype=torch.int16)
                            x = bits.view(torch.bfloat16).repeat(count // len(bits))
                        else:
                            x = torch.randn(count, device=rank, dtype=torch.bfloat16, generator=gen)
                            x.mul_(2.0 ** (rank % 4))
                            if args.case == "tiny":
                                x.mul_(2.0 ** -130)
                            if args.case == "wide":
                                x.mul_(2.0 ** 120)
                        source.append(x)
                        work.append(torch.empty_like(x))
                        gen.manual_seed(30001)
                        residual.append(torch.randn(count, device=rank, dtype=torch.bfloat16, generator=gen))
                        tables.append(torch.randn((3, 6 * 5376), device=rank, dtype=torch.bfloat16, generator=gen))
                        weights.append(torch.randn(5376, device=rank, dtype=torch.bfloat16, generator=gen))
                        indices.append(torch.arange(rows, device=rank, dtype=torch.int64) % 3)
                        for buffers in [by, bz, cy, cz, bc, cc]:
                            buffers.append(torch.empty(count + 8, device=rank, dtype=torch.bfloat16))
                        streams[rank].wait_stream(torch.cuda.current_stream(rank))
                sync()

                def reset(library=None):
                    selected = [by, bz, bc] if library is reference else [cy, cz, cc] if library is candidate else [by, bz, cy, cz, bc, cc]
                    for rank, stream in enumerate(streams):
                        with torch.cuda.stream(stream):
                            work[rank].copy_(source[rank])
                            for buffers in selected:
                                buffers[rank].view(torch.int16).fill_(poison)
                    sync()

                def reset_work():
                    # 原InfiniCCL回退原地修改work，诊断配对也必须恢复同一份局部输入。
                    sync()
                    for rank, stream in enumerate(streams):
                        with torch.cuda.stream(stream):
                            work[rank].copy_(source[rank])
                    sync()

                def pipeline(library, ys, zs, inputs):
                    def worker(rank):
                        assert hip.hipSetDevice(rank) == 0
                        table = tables[rank].data_ptr()
                        if library is candidate:
                            candidate.h3_peer_chain_set_norm(weights[rank].data_ptr(), table + 4 * 5376 * 2,
                                                             table + 3 * 5376 * 2, ys[rank].data_ptr())
                        destination = zs if library is candidate else ys
                        check_error(library.h3_peer_bridge_raw(rank, work[rank].data_ptr(), destination[rank].data_ptr(),
                            inputs[rank].data_ptr(), table + 2 * 5376 * 2, indices[rank].data_ptr(), count, 3,
                            comms[rank], streams[rank].cuda_stream))
                        if library is candidate and split_arrivals:
                            check_error(candidate.h3_peer_chain_norm_raw(rank))
                        if library is reference:
                            check_error(norm.songshan_bench_widenorm_indexed(rank, rows, 3, ys[rank].data_ptr(),
                                weights[rank].data_ptr(), table + 4 * 5376 * 2, table + 3 * 5376 * 2,
                                indices[rank].data_ptr(), zs[rank].data_ptr(), streams[rank].cuda_stream))
                    list(pool.map(worker, range(8)))
                    if args.consumer_copy:
                        consumers = cc if library is candidate else bc
                        for rank, stream in enumerate(streams):
                            with torch.cuda.stream(stream):
                                torch.bitwise_xor(zs[rank][:count].view(torch.int16), 0,
                                                  out=consumers[rank][:count].view(torch.int16))

                def measured(library, ys, zs):
                    reset(library)
                    for event, stream in zip(starts, streams):
                        event.record(stream)
                    begin = time.perf_counter()
                    if args.profile_candidate_only:
                        print('[profile.begin]', time.time_ns(), flush=True)
                    pipeline(library, ys, zs, residual)
                    for event, stream in zip(ends, streams):
                        event.record(stream)
                    sync()
                    if args.profile_candidate_only:
                        print('[profile.end]', time.time_ns(), flush=True)
                    return dict(wall_ms=(time.perf_counter() - begin) * 1000,
                                gpu_ms=max(a.elapsed_time(b) for a, b in zip(starts, ends)))

                def numerical(ys, zs):
                    calibration = next(c for c in json.loads(candidate.h3_peer_bridge_stats())["calibrations"] if c["count"] == count)
                    details = []
                    for rank in range(8):
                        first, end = rows * rank // 8 * 5376, rows * (rank + 1) // 8 * 5376
                        with torch.cuda.device(rank):
                            equal = torch.equal(bz[rank][:count].view(torch.int16), zs[rank][:count].view(torch.int16))
                            hidden = torch.equal(by[rank][first:end].view(torch.int16), ys[rank][first:end].view(torch.int16))
                            untouched = bool(torch.all(ys[rank][:first].view(torch.int16) == poison)) and bool(torch.all(ys[rank][end:].view(torch.int16) == poison))
                            fallback_full = calibration["ok"] or torch.equal(by[rank][:count].view(torch.int16), ys[rank][:count].view(torch.int16))
                            if not calibration["ok"]:
                                untouched = True  # 回退明确返回完整hidden，不具有分片所有权。
                            tails = all(bool(torch.all(buffers[rank][-8:].view(torch.int16) == poison)) for buffers in [by, bz, ys, zs, bc, cc])
                            consumer = not args.consumer_copy or torch.equal(bc[rank][:count], cc[rank][:count])
                            # BF16 NaN按原始位模式检查，不能使用浮点equal。
                            if args.consumer_copy:
                                consumer = torch.equal(bc[rank][:count].view(torch.int16), cc[rank][:count].view(torch.int16))
                            details.append(dict(rank=rank, normalized_exact=equal, hidden_local_exact=hidden,
                                                hidden_unowned_untouched=untouched, tails=tails, consumer_exact=consumer))
                            details[-1]["fallback_full_hidden_exact"] = fallback_full
                    return details

                if args.profile_candidate_only:
                    samples = [measured(candidate, cy, cz) for _ in range(3)]
                    records.append(dict(rows=rows, count=count, profile_only=True, numerical_qualified=False,
                                        diagnostic_samples=samples, identities=identities,
                                        candidate_stats=json.loads(candidate.h3_peer_bridge_stats())))
                    args.output.write_text(json.dumps(dict(scope='独立profiler诊断；不作数值或性能资格', records=records), ensure_ascii=False, indent=2)+'\n')
                    continue
                for _ in range(2):
                    measured(reference, by, bz)
                    measured(candidate, cy, cz)
                # reset()只清空输出，不改变输入residual。先同轮重新计算对照与候选再检查。
                reset();pipeline(reference, by, bz, residual);reset_work();pipeline(candidate, cy, cz, residual);sync()
                validation = numerical(cy, cz)
                assert all(all(v for k, v in row.items() if k != "rank") for row in validation), validation
                samples = []
                for pair in range(args.pairs):
                    if pair % 2:
                        c = measured(candidate, cy, cz);b = measured(reference, by, bz)
                    else:
                        b = measured(reference, by, bz);c = measured(candidate, cy, cz)
                    samples.append(dict(reference=b, candidate=c))
                    if args.consumer_copy:
                        current = numerical(cy, cz)
                        assert all(all(v for k, v in row.items() if k != "rank") for row in current), current
                # 计时结束后，连续三轮把分片hidden作为下轮残差；禁止读取未拥有的行。
                residual_chain = []
                for step in range(3):
                    old_by, old_cy = by, cy
                    by, bz, cy, cz = [], [], [], []
                    for rank, stream in enumerate(streams):
                        with torch.cuda.device(rank), torch.cuda.stream(stream):
                            work[rank].copy_(source[rank])
                            for buffers in [by, bz, cy, cz]:
                                buffers.append(torch.empty(count + 8, device=rank, dtype=torch.bfloat16))
                                buffers[-1].view(torch.int16).fill_(poison)
                            for consumers in [bc, cc]:
                                consumers[rank].view(torch.int16).fill_(poison)
                    sync()
                    pipeline(reference, by, bz, old_by)
                    reset_work()
                    pipeline(candidate, cy, cz, old_cy)
                    sync()
                    current = numerical(cy, cz)
                    assert all(all(v for k, v in row.items() if k != "rank") for row in current), (step, current)
                    residual_chain.append(dict(step=step + 1, validated=current))
                    del old_by, old_cy
                record = dict(identities=identities, split_arrivals=split_arrivals, rows=rows, count=count, case=args.case, consumer_copy=args.consumer_copy,
                              validated=validation, residual_chain=residual_chain, samples=samples,
                              reference_median_ms=statistics.median(s['reference']['gpu_ms'] for s in samples),
                              candidate_median_ms=statistics.median(s['candidate']['gpu_ms'] for s in samples),
                              paired_gain_percent=statistics.median(100 * (1-s['candidate']['gpu_ms']/s['reference']['gpu_ms']) for s in samples),
                              candidate_stats=json.loads(candidate.h3_peer_bridge_stats()))
                records.append(record)
                args.output.write_text(json.dumps(dict(scope='八卡独立gate→norm→publish链；不是模型成绩',
                    reference=str(args.reference), candidate=str(args.candidate), norm=str(args.norm), records=records), ensure_ascii=False, indent=2) + '\n')
                print(rows, record['reference_median_ms'], record['candidate_median_ms'], record['paired_gain_percent'], flush=True)
                del source, work, residual, tables, indices, weights, by, bz, cy, cz, bc, cc, x
                for rank in range(8):
                    with torch.cuda.device(rank):
                        torch.cuda.empty_cache()
    finally:
        candidate.peer_cleanup();reference.peer_cleanup()
        for rank, comm in enumerate(comms):
            with torch.cuda.device(rank):
                proxy.ncclCommDestroy(comm)


if __name__ == '__main__':
    main()
