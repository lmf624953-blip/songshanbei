#!/usr/bin/env python3
"""同一设备 stream 比较宽 RMSNorm+索引调制及九齿融合，检查完整输出。"""

import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import sys

import infinicore as ic
from infinicore.nn.functional import rms_norm
import torch

sys.path.insert(0, str(Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization/adaln"))
from indexed_runtime import load_indexed


class Fusion:
    def __init__(self, root, symbol=b"wide_norm_indexed_kernel"):
        self.meta = json.loads((root / "build.json").read_text())
        image = root / "wide-norm-indexed.hsaco"
        assert hashlib.sha256(image.read_bytes()).hexdigest() == self.meta["hsaco_sha256"]
        self.hip = ct.CDLL("libamdhip64.so")
        self.hip.hipGetErrorString.argtypes = [ct.c_int]
        self.hip.hipGetErrorString.restype = ct.c_char_p
        self.hip.hipModuleLoad.argtypes = [ct.POINTER(ct.c_void_p), ct.c_char_p]
        self.hip.hipModuleGetFunction.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p]
        self.hip.hipModuleLaunchKernel.argtypes = (
            [ct.c_void_p] + [ct.c_uint] * 7
            + [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_void_p])
        self.module, self.function = ct.c_void_p(), ct.c_void_p()
        self.check(self.hip.hipModuleLoad(ct.byref(self.module), os.fsencode(image)))
        self.check(self.hip.hipModuleGetFunction(ct.byref(self.function), self.module, symbol))

    def check(self, code):
        if code:
            raise RuntimeError(self.hip.hipGetErrorString(code).decode())

    def launch(self, stream, **values):
        cells = []
        for kind, name in self.meta["arguments"]:
            cells.append(ct.c_void_p(values[name].data_ptr()) if kind == "ptr" else ct.c_int(values[name]))
        cells.extend([ct.c_void_p(), ct.c_void_p()])
        argv = (ct.c_void_p * len(cells))(*(ct.addressof(x) for x in cells))
        self.check(self.hip.hipModuleLaunchKernel(
            self.function, values["rows"], 1, 1, self.meta["warps"] * 64, 1, 1,
            self.meta["shared"], stream, argv, None))


def wrap(tensor):
    return ic.strided_from_blob(tensor.data_ptr(), list(tensor.shape), list(tensor.stride()),
                               dtype=ic.bfloat16, device=ic.device("cuda", 0))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("build", type=Path)
    cli.add_argument("--output", required=True, type=Path)
    args = cli.parse_args()
    torch.cuda.set_device(0)
    ic.set_device(ic.device("cuda", 0))
    stream = ic.get_stream()
    if type(stream).__name__ == "PyCapsule":
        get_pointer = ct.pythonapi.PyCapsule_GetPointer
        get_pointer.argtypes, get_pointer.restype = [ct.py_object, ct.c_char_p], ct.c_void_p
        stream = get_pointer(stream, None)
    events_stream = torch.cuda.ExternalStream(stream, device=0)
    host = load_indexed(Path(os.environ["H3_INDEXED_BUILD"]))
    indexed = host.songshan_bench_indexed
    indexed.argtypes = [ct.c_bool] + [ct.c_int] * 3 + [ct.c_void_p] * 6
    indexed.restype = ct.c_char_p
    images = {path.parent.name: Fusion(path.parent)
              for path in sorted(args.build.glob("*/build.json"))
              if json.loads(path.read_text())["mode"] != "cub"}
    assert images, "缺少行内并行归约的设备镜像"
    generator = torch.Generator(device="cuda").manual_seed(42)
    records = []
    width, table_rows = 5376, 6

    for rows, magnitude in ((1, 1), (33, 8), (22016, 1), (27570, 1)):
        def rand(*shape):
            return torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)

        hidden = rand(rows, width) * magnitude
        weight = rand(width)
        modulation = rand(table_rows, width * 6) / 8
        shift, scale = modulation[:, :width], modulation[:, width:2 * width]
        indices = torch.randint(table_rows, (rows,), device="cuda", generator=generator)
        normalized = torch.empty_like(hidden)
        reference = torch.full_like(hidden, float("nan"))
        destination = torch.empty((rows + 2, width), dtype=hidden.dtype, device="cuda")
        output = destination[:rows]
        ic_input, ic_weight, ic_norm = map(wrap, (hidden, weight, normalized))

        def baseline():
            rms_norm(ic_input, [width], ic_weight, eps=1e-5, out=ic_norm)
            error = indexed(False, 0, rows, table_rows, normalized.data_ptr(), scale.data_ptr(),
                            shift.data_ptr(), indices.data_ptr(), reference.data_ptr(), stream)
            if error:
                raise RuntimeError(error.decode())

        torch.cuda.synchronize()
        baseline()
        torch.cuda.synchronize()
        for name, module in images.items():
            def candidate():
                module.launch(stream, rows=rows, table_rows=table_rows, hidden=hidden, weight=weight,
                              scale=scale, shift=shift, indices=indices, output=output)

            destination.fill_(float("nan"))
            destination[rows:].fill_(-17)
            torch.cuda.synchronize()
            candidate()
            torch.cuda.synchronize()
            error = (reference.float() - output.float())
            case = dict(rows=rows, magnitude=magnitude, variant=name,
                        exact=torch.equal(reference.view(torch.int16), output.view(torch.int16)),
                        mismatches=int((reference.view(torch.int16) != output.view(torch.int16)).sum()),
                        finite=bool(torch.isfinite(output).all() and torch.isfinite(reference).all()),
                        tail_ok=bool((destination[rows:] == -17).all()),
                        max_abs=float(error.abs().max()), rmse=float(error.square().mean().sqrt()))
            if rows >= 22016 and case["finite"] and case["tail_ok"]:
                samples = [[], []]
                for trial in range(13):
                    for v in (trial % 2, 1 - trial % 2):
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record(events_stream)
                        (baseline, candidate)[v]()
                        end.record(events_stream)
                        end.synchronize()
                        if trial >= 2:
                            samples[v].append(begin.elapsed_time(end))
                case.update(device_ms=[median(x) for x in samples],
                            paired_reduction_percent=100 * median(1 - b / a for a, b in zip(*samples)),
                            samples_ms=samples)
            records.append(case)
            print(json.dumps(case, ensure_ascii=False), flush=True)

    with args.output.open("x") as stream_file:
        json.dump(dict(timing="同一个 InfiniCore stream 上的 GPU event", cases=records), stream_file, indent=2)
        stream_file.write("\n")


if __name__ == "__main__":
    main()
