#!/usr/bin/env python3
"""验证长注意力的逐缓冲区范围守卫、两条发射路径与镜像摘要校验。"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    code = Path(os.environ["H3_AUTOTUNE_CODE_ROOT"])
    sys.path.insert(0, str(code / "optimization"))
    from adaln.aot_codegen import compile_host
    from adaln.aot_runtime import load_attention

    build = args.build.resolve(strict=True)
    reference = args.reference.resolve(strict=True)
    name = "dense-attention-aot"
    primary_image = build / f"{name}.hsaco"
    fallback_image = build / "dense-attention-fallback.hsaco"
    record_path = build / f"{name}-build.json"
    record = json.loads(record_path.read_text())
    elf = runpy.run_path(str(Path(__file__).with_name("compare-h3-device.py")))["elf_loads"]
    normal = elf(reference / f"{name}.hsaco")
    fallback = elf(fallback_image)
    assert all(normal[key] == fallback[key] for key in ("sections", "segments", "elf_target"))
    report = {"fallback_equal_to_reference": True, "cpu_cases": [], "gpu_cases": [], "digest_rejections": []}

    def eligible(host, rows, strides):
        function = host.songshan_attention_buffer_eligible
        function.argtypes = [ctypes.c_int] * 5
        function.restype = ctypes.c_int
        return bool(function(rows, *strides))

    production = load_attention(build)
    limit = record["buffer_ops_limit_bytes"]
    assert limit == 2**31
    stride_at_limit = (limit // 2 - 896)
    for axis in range(4):
        for delta, expected in ((-16, True), (0, True), (16, False)):
            strides = [896] * 4
            strides[axis] = stride_at_limit + delta
            assert eligible(production, 2, strides) == expected
            report["cpu_cases"].append(dict(axis=axis, rows=2, strides=strides, eligible=expected))
    for rows, strides in ((0, [896] * 4), (2, [896, 0, 896, 896]), (2**31 - 1, [2**31 - 1] * 4)):
        assert not eligible(production, rows, strides)
        report["cpu_cases"].append(dict(rows=rows, strides=strides, eligible=False))

    benchmark = runpy.run_path(str(code / "optimization/bench-attention.py"))
    reference_launch = benchmark["native_launcher"](reference)
    production_launch = benchmark["native_launcher"](build)
    generator = torch.Generator(device="cuda").manual_seed(42)

    with tempfile.TemporaryDirectory(prefix="h3-attention-buffer-check-") as directory:
        fixture = Path(directory)
        for path in (primary_image, fallback_image, record_path):
            shutil.copy2(path, fixture / path.name)
        for source in (primary_image, fallback_image):
            image = fixture / source.name
            image.write_bytes(source.read_bytes() + b"\0")
            try:
                load_attention(fixture)
            except RuntimeError as error:
                assert image.name in str(error)
                report["digest_rejections"].append(image.name)
            else:
                raise RuntimeError(f"已篡改的镜像通过校验：{image}")
            shutil.copy2(source, image)

        compile_host(fixture / f"{name}.so", ("attention_buffer_host.cc",),
                     defines=(("H3_ATTENTION_QUERY_BLOCK", 64), ("H3_ATTENTION_BUFFER_LIMIT", 8192)))
        forced_host = load_attention(fixture)
        forced_launch = benchmark["native_launcher"](fixture)
        for rows in (1, 4, 5, 31, 32, 33, 63, 64, 65, 129):
            qkv = [torch.randn((1, rows, 7, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
                   .transpose(1, 2) for _ in range(3)]
            output = [torch.full_like(qkv[0], float("nan")) for _ in range(3)]
            strides = [t.stride(2) for t in (*qkv, output[0])]
            production_eligible = eligible(production, rows, strides)
            forced_eligible = eligible(forced_host, rows, strides)
            assert production_eligible and forced_eligible == (rows <= 4)
            for launch, out in zip((reference_launch, production_launch, forced_launch), output, strict=True):
                launch(*qkv, 128**-0.5, out)
            torch.cuda.synchronize()
            exact = all(torch.equal(output[0].view(torch.int16), out.view(torch.int16)) for out in output[1:])
            finite = all(bool(torch.isfinite(out).all()) for out in output)
            report["gpu_cases"].append(dict(rows=rows, production_buffer=production_eligible,
                                            forced_buffer=forced_eligible, exact=exact, finite=finite))
            assert exact and finite

    report["verdict"] = "PASS"
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
