#!/usr/bin/env python3
"""在正式模型计时结束后运行，比较相同资源的 MMAC 累加依赖链。"""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
from statistics import median

import torch


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("build", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--blocks", type=int, nargs="+", default=[80, 320])
    args = cli.parse_args()
    if any(blocks <= 0 for blocks in args.blocks):
        cli.error("blocks 必须大于零")
    if args.output.exists():
        raise FileExistsError(args.output)
    meta = json.loads((args.build / "probe-build.json").read_text())
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    hip.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3

    def check(status):
        if status:
            raise RuntimeError(f"HIP status {status}")

    torch.cuda.init()
    modules, functions = [], []
    for variant in meta["variants"]:
        path = args.build / variant["image"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == variant["sha256"]
        module, function = ctypes.c_void_p(), ctypes.c_void_p()
        check(hip.hipModuleLoad(ctypes.byref(module), str(path).encode()))
        check(hip.hipModuleGetFunction(ctypes.byref(function), module, b"dense_attention_aot_kernel"))
        modules.append(module)
        functions.append(function)

    def measure(blocks):
        output = torch.empty(blocks * 256, dtype=torch.float32, device="cuda")
        storage = [ctypes.c_uint64(0) for _ in range(15)]
        storage[4].value = output.data_ptr()
        parameters = (ctypes.c_void_p * len(storage))(
            *(ctypes.addressof(value) for value in storage))

        def run(index):
            check(hip.hipModuleLaunchKernel(functions[index], blocks, 1, 1, 256, 1, 1,
                  16384, torch.cuda.current_stream().cuda_stream, parameters, None))

        for index, variant in enumerate(meta["variants"]):
            output.fill_(float("nan"))
            run(index)
            torch.cuda.synchronize()
            assert bool((output == variant["expected_output"]).all()), variant
        samples = [[] for _ in functions]
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for trial in range(21):
            order = list(range(len(functions)))
            offset = trial % len(functions)
            order = order[offset:] + order[:offset]
            if trial % 2:
                order.reverse()
            for index in order:
                start.record()
                run(index)
                end.record()
                end.synchronize()
                if trial >= 2:
                    samples[index].append(start.elapsed_time(end))
        return dict(blocks=blocks, records=[dict(variant, samples_ms=times, median_ms=median(times))
                   for variant, times in zip(meta["variants"], samples)])

    cases = [measure(blocks) for blocks in args.blocks]
    result = dict(scope="MMAC循环内无访存的诊断，不能直接推断真实注意力或端到端收益",
                  blocks=args.blocks, threads=256, build=meta, cases=cases,
                  device=str(torch.cuda.get_device_properties(0)), numerical_check_passed=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(cases, indent=2), flush=True)


if __name__ == "__main__":
    main()
