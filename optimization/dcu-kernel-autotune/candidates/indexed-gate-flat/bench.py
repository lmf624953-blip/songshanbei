#!/usr/bin/env python3
"""用 HIP events 对照当前门控 AOT 和扁平候选，并核对整块输出。"""

import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import sys

import infinicore as ic
import torch

sys.path.insert(0, str(Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization/adaln"))
from indexed_runtime import load_indexed

WIDTH = 5376


class FlatGate:
    def __init__(self, directory, hip):
        self.metadata = json.loads((directory / "build.json").read_text())
        image = directory / "indexed_gate_flat.hsaco"
        if hashlib.sha256(image.read_bytes()).hexdigest() != self.metadata["hsaco_sha256"]:
            raise RuntimeError(f"镜像校验失败: {image}")
        self.image = image
        self.hip = hip
        self.module = ct.c_void_p()
        self.function = ct.c_void_p()
        self.check(hip.hipModuleLoad(ct.byref(self.module), os.fsencode(image)))
        self.check(hip.hipModuleGetFunction(ct.byref(self.function), self.module, b"indexed_gate_flat_kernel"))

    def check(self, error):
        if error:
            raise RuntimeError(f"{self.image}: {self.hip.hipGetErrorString(error).decode()}")

    def launch(self, stream, **inputs):
        if self.metadata.get("buffer_addressing") and (
            inputs["elements"] * 2 >= 2**31 or inputs["table_rows"] * 6 * WIDTH * 2 >= 2**31
        ):
            raise ValueError("buffer 版本需要有界的 32 位字节偏移")
        values = [
            ct.c_void_p(inputs[name].data_ptr()) if kind == "ptr" else ct.c_int(inputs[name])
            for kind, name in self.metadata["arguments"]
        ]
        values.extend([ct.c_void_p(), ct.c_void_p()])
        argv = (ct.c_void_p * len(values))(*(ct.addressof(value) for value in values))
        tile = self.metadata["tile"]
        blocks = (inputs["elements"] + tile - 1) // tile
        self.check(self.hip.hipModuleLaunchKernel(
            self.function, blocks, 1, 1, 64 * self.metadata["warps"], 1, 1, 0, stream, argv, None))


def measure(stream, launch, events):
    start, end = events
    start.record(stream)
    launch()
    end.record(stream)
    end.synchronize()
    return start.elapsed_time(end)


def compare(rows, table_rows, index_pattern, candidates, reference, generator, stream):
    def random_bf16(*shape):
        return torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)

    residual = random_bf16(rows, WIDTH)
    activation = random_bf16(rows, WIDTH)
    gate = random_bf16(table_rows, 6 * WIDTH)[:, 2 * WIDTH:3 * WIDTH]
    if index_pattern == "random":
        indices = torch.randint(table_rows, (rows,), device="cuda", generator=generator)
    else:
        indices = torch.arange(rows, device="cuda") * table_rows // rows
    expected = torch.empty_like(residual)
    destination = torch.empty(rows * WIDTH + 256, dtype=torch.bfloat16, device="cuda")
    output = destination[:rows * WIDTH].view(rows, WIDTH)
    torch.cuda.synchronize()
    pointers = [tensor.data_ptr() for tensor in (residual, activation, gate, indices, expected)]

    def run_reference():
        error = reference(True, 0, rows, table_rows, *pointers, stream.cuda_stream)
        if error:
            raise RuntimeError(error.decode())

    run_reference()
    stream.synchronize()
    events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(2))
    for directory, candidate in candidates:
        def run_candidate():
            candidate.launch(stream.cuda_stream, residual=residual, activation=activation, gate=gate,
                             indices=indices, output=output, rows=rows, table_rows=table_rows,
                             elements=rows * WIDTH)

        destination.fill_(float("nan"))
        destination[rows * WIDTH:].fill_(-17)
        torch.cuda.synchronize()
        run_candidate()
        stream.synchronize()
        exact = torch.equal(output.view(torch.int16), expected.view(torch.int16))
        finite = torch.isfinite(output).all().item()
        tail_ok = destination[rows * WIDTH:].eq(-17).all().item()
        record = dict(rows=rows, table_rows=table_rows, indices=index_pattern, variant=directory.name,
                      exact=exact, finite=finite, tail_ok=tail_ok)
        if exact and finite and tail_ok and rows >= 4096:
            for _ in range(3):
                run_reference()
                run_candidate()
            stream.synchronize()
            control_samples, candidate_samples = [], []
            for repeat in range(11):
                if repeat % 2:
                    candidate_samples.append(measure(stream, run_candidate, events))
                    control_samples.append(measure(stream, run_reference, events))
                else:
                    control_samples.append(measure(stream, run_reference, events))
                    candidate_samples.append(measure(stream, run_candidate, events))
            record.update(reference_ms=median(control_samples), candidate_ms=median(candidate_samples),
                          reduction_percent=100 * median(
                              1 - new / old for old, new in zip(control_samples, candidate_samples)),
                          reference_samples_ms=control_samples, candidate_samples_ms=candidate_samples)
        yield record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-buffer", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(0)
    ic.set_device(ic.device("cuda", 0))
    native_stream = ic.get_stream()
    if type(native_stream).__name__ == "PyCapsule":
        pointer = ct.pythonapi.PyCapsule_GetPointer
        pointer.argtypes = [ct.py_object, ct.c_char_p]
        pointer.restype = ct.c_void_p
        native_stream = pointer(native_stream, None)
    stream = torch.cuda.ExternalStream(native_stream, device=0)
    hip = ct.CDLL("libamdhip64.so")
    hip.hipGetErrorString.argtypes = [ct.c_int]
    hip.hipGetErrorString.restype = ct.c_char_p
    hip.hipModuleLoad.argtypes = [ct.POINTER(ct.c_void_p), ct.c_char_p]
    hip.hipModuleGetFunction.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p]
    hip.hipModuleLaunchKernel.argtypes = [ct.c_void_p] + [ct.c_uint] * 7 + [
        ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_void_p]
    reference_host = load_indexed(args.reference)
    reference = reference_host.songshan_bench_indexed
    reference.argtypes = [ct.c_bool] + [ct.c_int] * 3 + [ct.c_void_p] * 6
    reference.restype = ct.c_char_p
    candidates = []
    skipped = []
    for path in sorted(args.candidates.glob("*/build.json")):
        metadata = json.loads(path.read_text())
        if args.skip_buffer and metadata.get("buffer_addressing"):
            skipped.append(path.parent.name)
            continue
        candidates.append((path.parent, FlatGate(path.parent, hip)))
    if skipped:
        print(json.dumps(dict(skipped=skipped, reason="buffer 寻址镜像的 HIP 符号加载错误"),ensure_ascii=False),flush=True)
    if not candidates:
        raise RuntimeError("没有可测试的门控镜像")
    generator = torch.Generator(device="cuda").manual_seed(42)
    rows_and_patterns = [(1, 3, "random"), (33, 6, "random"), (4096, 3, "random"),
                         (22016, 6, "grouped"), (27570, 6, "grouped"), (27570, 6, "random")]
    records = []
    for rows, table_rows, index_pattern in rows_and_patterns:
        for record in compare(rows, table_rows, index_pattern, candidates, reference, generator, stream):
            records.append(record)
            print(json.dumps(record), flush=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    if any(not (r["exact"] and r["finite"] and r["tail_ok"]) for r in records):
        raise SystemExit("至少一项候选未通过数值核验")


if __name__ == "__main__":
    main()
