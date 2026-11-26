#!/usr/bin/env python3
"""Measure in-place BF16 RCCL AllReduce on the same eight local devices."""

import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch


def load_rccl(path):
    library = ct.CDLL(str(path))
    library.ncclGetErrorString.argtypes = [ct.c_int]
    library.ncclGetErrorString.restype = ct.c_char_p

    def check(result, function, _):
        if result:
            raise RuntimeError(f"{function.__name__}: {library.ncclGetErrorString(result).decode()}")

    pointer = ct.c_void_p
    signatures = {
        "ncclCommInitAll": [ct.POINTER(pointer), ct.c_int, ct.POINTER(ct.c_int)],
        "ncclCommDestroy": [pointer],
        "ncclGroupStart": [],
        "ncclGroupEnd": [],
        "ncclAllReduce": [pointer, pointer, ct.c_size_t, ct.c_int, ct.c_int, pointer, pointer],
    }
    for name, arguments in signatures.items():
        function = getattr(library, name)
        function.argtypes = arguments
        function.restype = ct.c_int
        function.errcheck = check
    return library


def bench(library, comms, streams, source, buffers, warmups, iterations):
    starts = [torch.cuda.Event(enable_timing=True) for _ in streams]
    stops = [torch.cuda.Event(enable_timing=True) for _ in streams]
    gpu_ms, wall_ms = [], []
    for iteration in range(warmups + iterations):
        for rank, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                buffers[rank].copy_(source[rank])
        for stream in streams:
            stream.synchronize()

        t0 = time.perf_counter()
        for event, stream in zip(starts, streams):
            event.record(stream)
        library.ncclGroupStart()
        for rank, stream in enumerate(streams):
            with torch.cuda.device(rank):
                data = buffers[rank].data_ptr()
                # ncclBfloat16 = 9, ncclSum = 0 in the DTK RCCL header.
                library.ncclAllReduce(data, data, buffers[rank].numel(), 9, 0, comms[rank], stream.cuda_stream)
        library.ncclGroupEnd()
        for event, stream in zip(stops, streams):
            event.record(stream)
        for event in stops:
            event.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        if iteration >= warmups:
            gpu_ms.append(max(start.elapsed_time(stop) for start, stop in zip(starts, stops)))
            wall_ms.append(elapsed)

    hashes = [hashlib.sha256(x.view(torch.uint8).cpu().numpy()).hexdigest() for x in buffers]
    if len(set(hashes)) != 1:
        raise RuntimeError(f"AllReduce ranks disagree: {hashes}")
    return dict(gpu_ms=gpu_ms, wall_ms=wall_ms, median_gpu_ms=statistics.median(gpu_ms),
                median_wall_ms=statistics.median(wall_ms), sha256=hashes[0])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("library", type=Path)
    cli.add_argument("--counts", type=int, nargs="+", default=[148216320, 3806208])
    cli.add_argument("--warmups", type=int, default=5)
    cli.add_argument("--iterations", type=int, default=17)
    args = cli.parse_args()
    ranks = 8
    if torch.cuda.device_count() != ranks:
        raise ValueError(f"Expected {ranks} visible devices")
    library = load_rccl(args.library.resolve(strict=True))
    streams = [torch.cuda.Stream(device=rank) for rank in range(ranks)]
    comms = (ct.c_void_p * ranks)()
    devices = (ct.c_int * ranks)(*range(ranks))
    library.ncclCommInitAll(comms, ranks, devices)
    knobs = ("NCCL_ALGO", "NCCL_PROTO", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS")
    print(json.dumps(dict(settings={key: os.environ.get(key) for key in knobs}, library=str(args.library))), flush=True)
    try:
        for count in args.counts:
            source = []
            for rank in range(ranks):
                device = torch.device("cuda", rank)
                generator = torch.Generator(device=device).manual_seed(104729 + rank)
                values = torch.randn(count, device=device, generator=generator, dtype=torch.bfloat16)
                values.mul_(2 ** (rank % 4))
                streams[rank].wait_stream(torch.cuda.current_stream(device))
                source.append(values)
            buffers = [torch.empty_like(x) for x in source]
            result = bench(library, comms, streams, source, buffers, args.warmups, args.iterations)
            print(json.dumps(dict(count=count, bytes=count * 2, **result)), flush=True)
    finally:
        for rank, comm in enumerate(comms):
            with torch.cuda.device(rank):
                library.ncclCommDestroy(comm)


if __name__ == "__main__":
    main()
