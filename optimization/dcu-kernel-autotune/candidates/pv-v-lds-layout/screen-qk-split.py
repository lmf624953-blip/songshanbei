#!/usr/bin/env python3
"""验证两阶段QK拆分的最终位模式，并以完整双阶段计时，分段时间仅供诊断。"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import torch


def launcher(build, reference, batch, fallback):
    host = ctypes.CDLL(str(build / "split-host.so"))
    prepare = host.prepare_split
    prepare.argtypes = [ctypes.c_char_p] * 3
    prepare.restype = ctypes.c_char_p
    error = prepare(*(os.fsencode(p) for p in (build / "qk.hsaco", build / "softmax-pv.hsaco", reference / "prepack-value.hsaco")))
    if error:
        raise RuntimeError(error.decode())
    launch = host.bench_split
    launch.argtypes = [ctypes.c_int, ctypes.c_float] + [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3 + [ctypes.c_size_t, ctypes.c_int]
    launch.restype = ctypes.c_char_p
    meta = json.loads((reference / "dense-attention-aot-build.json").read_text())["prepacked_value_dispatch"]["prepack"]
    buffers = {}

    def call(q, k, v, scale, out, phase=0):
        rows = q.shape[2]
        if rows < 4096:
            return fallback(q, k, v, scale, out)
        key = (rows, q.device.index)
        if key not in buffers:
            blocks = rows // 32
            count = min(batch, 7 * ((rows + 127) // 128))
            buffers[key] = (torch.empty(7 * blocks * 4096, dtype=torch.bfloat16, device=q.device),
                            torch.empty(count * blocks * 4096, dtype=torch.float32, device=q.device))
        packed, scores = buffers[key]
        error = launch(rows, scale, *(x.data_ptr() for x in (q, k, v, out, packed, scores)),
                       batch, meta["tile"], meta["warps"] * 64,
                       torch.cuda.current_stream(q.device).cuda_stream, phase)
        if error:
            raise RuntimeError(error.decode())
    call.buffers = buffers
    return call


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("build", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--batches", nargs="+", type=int, default=[64, 256, 1024])
    cli.add_argument("--lengths", nargs="+", type=int, default=[4096, 4097, 8193, 22016, 27570])
    cli.add_argument("--pairs", type=int, default=19)
    args = cli.parse_args()
    code = Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization"
    sys.path.insert(0, str(code))
    fallback = runpy.run_path(str(code / "bench-attention.py"))["native_launcher"](args.reference)
    screen = runpy.run_path(str(Path(__file__).with_name("screen.py")))["screen"]
    with args.output.open("x") as output:
        for batch in args.batches:
            candidate = launcher(args.build, args.reference, batch, fallback)
            for length in args.lengths:
                row = dict(batch=batch, **screen(fallback, candidate, length, args.pairs))
                packed, scores = candidate.buffers[(length, 0)]
                row["workspace_bytes"] = sum(t.numel() * t.element_size() for t in (packed, scores))
                row["qk_roundtrip_bytes"] = 7 * ((length + 127) // 128) * (length // 32) * 4096 * 4 * 2
                output.write(json.dumps(row) + "\n");output.flush()
                print(json.dumps({k:v for k,v in row.items() if k != "samples_ms"}), flush=True)
                if not row["passed"]:
                    raise RuntimeError("两阶段最终输出或尾部保护不一致，停止性能结论")
            del candidate
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
