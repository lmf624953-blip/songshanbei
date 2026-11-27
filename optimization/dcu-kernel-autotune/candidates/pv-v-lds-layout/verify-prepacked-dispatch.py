#!/usr/bin/env python3
"""检验正式主机 ABI、八设备、动态长度、stride 回退与独立 stream 工作区。"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import runpy
import sys

import torch


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "output"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    code = Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization"
    sys.path.insert(0, str(code))
    native = runpy.run_path(str(code / "bench-attention.py"))["native_launcher"]
    baseline, candidate = native(args.reference), native(args.candidate)
    screen = runpy.run_path(str(Path(__file__).with_name("screen.py")))["screen"]
    with args.output.open("x") as destination:
        def record(row):
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            destination.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)

        for rows in (27570, 22016):
            result = screen(baseline, candidate, rows, 19)
            record(dict(kind="paired-native", **result))
            if not result["passed"]:
                raise ValueError("Native paired check failed")

        def prepare(rows, device, heads=(7, 7, 21, 7)):
            tensors = []
            for size in heads[:3]:
                tensor = torch.randn((1, rows, size, 128), device=device, dtype=torch.bfloat16)
                tensors.append(tensor[:, :, -7:].transpose(1, 2))
            storage = [torch.full((1, rows + 2, heads[3], 128), -19,
                                  device=device, dtype=torch.bfloat16) for _ in range(2)]
            outputs = [buffer[:, :rows, -7:].transpose(1, 2) for buffer in storage]
            return tensors, outputs, storage

        def check(tensors, outputs, storage):
            exact = torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
            finite = bool(torch.isfinite(outputs[1]).all())
            tail = all(bool((buffer[:, -2:] == -19).all()) for buffer in storage)
            if not (exact and finite and tail):
                raise ValueError(f"Native check failed: {exact}, {finite}, {tail}")
            return dict(exact=exact, finite=finite, tail_ok=tail)

        torch.manual_seed(20261127)
        for device in range(torch.cuda.device_count()):
            with torch.cuda.device(device):
                lengths = (1, 31, 32, 33, 127, 128, 129, 4095, 4096, 4097, 22016, 27570)
                if device:
                    lengths = (4097, 27570)
                for rows in lengths:
                    tensors, outputs, storage = prepare(rows, device)
                    baseline(*tensors, 128 ** -0.5, outputs[0])
                    candidate(*tensors, 128 ** -0.5, outputs[1])
                    torch.cuda.synchronize(device)
                    record(dict(kind="device-length", device=device, sequence=rows,
                                **check(tensors, outputs, storage)))
        with torch.cuda.device(0):
            for heads in ((7, 7, 28, 7), (8, 7, 21, 7), (7, 8, 21, 7), (7, 7, 21, 8)):
                tensors, outputs, storage = prepare(4097, 0, heads)
                baseline(*tensors, 128 ** -0.5, outputs[0])
                candidate(*tensors, 128 ** -0.5, outputs[1])
                torch.cuda.synchronize()
                record(dict(kind="stride-fallback", allocated_heads=heads,
                            **check(tensors, outputs, storage)))
            streams = [torch.cuda.Stream() for _ in range(2)]
            for rows in (4097, 27570, 4096, 22016):
                cases = [prepare(rows, 0) for _ in streams]
                for stream, (tensors, outputs, storage) in zip(streams, cases):
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        baseline(*tensors, 128 ** -0.5, outputs[0])
                        candidate(*tensors, 128 ** -0.5, outputs[1])
                for index, (stream, (tensors, outputs, storage)) in enumerate(zip(streams, cases)):
                    stream.synchronize()
                    record(dict(kind="independent-stream", sequence=rows, stream=index,
                                **check(tensors, outputs, storage)))


if __name__ == "__main__":
    main()
