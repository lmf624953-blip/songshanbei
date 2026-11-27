#!/usr/bin/env python3
"""对预排布 V 加完整注意力做逐位验证及无 profiler 配对计时。"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import torch

from v_prepack_kernel import kernel


def launcher(build, prepack=None, fallback=None):
    host = ctypes.CDLL(str(build / "prepacked-host.so"))
    prepare = host.prepare_prepacked_attention
    prepare.argtypes = [ctypes.c_char_p]
    prepare.restype = ctypes.c_char_p
    error = prepare(os.fsencode(build / "dense-attention-packed.hsaco"))
    if error:
        raise RuntimeError(error.decode())
    if prepack:
        prepare_value = host.prepare_value_prepack
        prepare_value.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        prepare_value.restype = ctypes.c_char_p
        metadata = json.loads((prepack / "prepack-value-build.json").read_text())
        image = prepack / "prepack-value.hsaco"
        assert hashlib.sha256(image.read_bytes()).hexdigest() == metadata["hsaco_sha256"]
        error = prepare_value(os.fsencode(image), metadata["tile"], metadata["warps"])
        if error:
            raise RuntimeError(error.decode())
        pack = host.bench_value_prepack
        pack.argtypes = [ctypes.c_int] * 2 + [ctypes.c_void_p] * 3
        pack.restype = ctypes.c_char_p

    def pack_value(value, packed):
        if prepack:
            error = pack(value.device.index, value.shape[0], value.data_ptr(), packed.data_ptr(),
                         torch.cuda.current_stream(value.device).cuda_stream)
            if error:
                raise RuntimeError(error.decode())
        else:
            kernel()(value, value.shape[0] // 32, packed)
    launch = host.bench_prepacked_attention
    launch.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float] + [ctypes.c_void_p] * 6
    launch.restype = ctypes.c_char_p
    workspaces = {}

    def run(q, k, v, scale, output):
        rows = q.shape[2]
        strides = tuple(tensor.stride(2) for tensor in (q, k, v, output))
        if rows < 32 or strides != (896, 896, 2688, 896):
            if fallback is None:
                raise ValueError("Experimental input requires the reference fallback")
            return fallback(q, k, v, scale, output)
        blocks = rows // 32
        stream = torch.cuda.current_stream(q.device)
        key = (q.device, rows, stream.cuda_stream)
        if key not in workspaces:
            workspaces[key] = torch.empty(7 * blocks * 4096, dtype=torch.bfloat16, device=q.device)
        packed = workspaces[key]
        if prepack is None:
            pack_value(v.transpose(1, 2).squeeze(0), packed)
        error = launch(q.device.index, rows, scale, q.data_ptr(), k.data_ptr(), v.data_ptr(),
                       output.data_ptr(), packed.data_ptr(), stream.cuda_stream)
        if error:
            raise RuntimeError(error.decode())
    run.pack_value = pack_value
    run.workspaces = workspaces
    return run


def verify_pack(rows, pack):
    blocks = rows // 32
    storage = torch.randint(-32768, 32767, (rows, 21, 128), device="cuda", dtype=torch.int16)
    value = storage[:, -7:].view(torch.bfloat16)
    packed = torch.empty(7 * blocks * 4096, device="cuda", dtype=torch.bfloat16)
    pack(value, packed)
    index = torch.arange(4096, device="cuda")
    column = index // 32
    row = 2 * (((index // 2) % 16) ^ (((column // 2) % 8) * 2)) + index % 2
    expected = storage[:, -7:].unfold(0, 32, 32).permute(1, 0, 3, 2)
    expected = expected[:, :, row, column].reshape(-1)
    actual = packed.view(torch.int16)
    different = int(torch.count_nonzero(actual != expected))
    if different:
        raise ValueError(f"V pack has {different} different words at length {rows}")
    return dict(sequence=rows, elements=packed.numel(), bitwise_equal=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("builds", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--prepack", type=Path)
    args = cli.parse_args()
    code = Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization"
    sys.path.insert(0, str(code))
    load = runpy.run_path(str(code / "bench-attention.py"))["native_launcher"]
    screen = runpy.run_path(str(Path(__file__).with_name("screen.py")))["screen"]
    baseline = load(args.reference)
    candidates = [(build, launcher(build, args.prepack, baseline)) for build in sorted(args.builds.iterdir())]
    result = dict(scope="每次调用均重新执行 V 预排布，事件计时包含预排布和完整注意力",
                  pack_checks=[], records=[], torch_version=torch.__version__, hip_version=torch.version.hip,
                  launch_mode="combined-aot" if args.prepack else "python-prepack")
    with args.output.open("x") as stream:
        for length in (32, 33, 65, 129, 257, 4097, 22016, 27570):
            result["pack_checks"].append(verify_pack(length, candidates[0][1].pack_value))
            print(json.dumps(result["pack_checks"][-1]), flush=True)
        for build, candidate in candidates:
            tests = []
            for length in (32, 33, 65, 127, 128, 129, 257, 4097, 27570, 22016):
                row = screen(baseline, candidate, length, 11 if length in (27570, 22016) else 1)
                tests.append(row)
                print(json.dumps(dict(variant=build.name, **row)), flush=True)
                if not row["passed"]:
                    break
            result["records"].append(dict(
                variant=build.name, cases=tests,
                packed_sha256=hashlib.sha256((build / "dense-attention-packed.hsaco").read_bytes()).hexdigest()))
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
