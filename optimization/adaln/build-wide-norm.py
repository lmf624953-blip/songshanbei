#!/usr/bin/env python3
"""将宽 5376 的九齿行归约预编译为可核验的 gfx936 内核。"""
import argparse
import json
from pathlib import Path

import ninetoothed as nt

from aot_codegen import compile_host, compile_image, lower_kernel
from wide_norm import arrange, naive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--warps", type=int, choices=(1, 2, 4, 8, 16), default=8)
    args = parser.parse_args()
    build = args.destination.resolve()
    build.mkdir()
    source, kernel, bindings = lower_kernel(build, arrange, naive,
        (nt.Tensor(2, dtype=nt.bfloat16), nt.Tensor(1, dtype=nt.bfloat16),
         nt.Tensor(0, dtype=nt.float32), nt.Tensor(2, dtype=nt.bfloat16)), "wide_norm", args.warps)
    signature, constants, dynamic = {}, {}, []
    for binding in bindings:
        name, kind, tensor, axis = binding.name, binding.kind, binding.source, binding.dim
        if kind == "tensor":
            signature[name] = "*bf16"
        elif kind == "shape" and tensor in {"input", "output"} and axis == 0:
            signature[name] = "i32"
            dynamic.append((kind, tensor))
        else:
            signature[name] = "constexpr"
            if kind == "scalar" and tensor == "epsilon":
                constants[name] = 1e-5
            elif kind == "shape":
                constants[name] = 5376
            elif kind == "stride":
                constants[name] = 1 if tensor == "weight" or axis == 1 else 5376
            else:
                raise RuntimeError(f"Unexpected wide norm argument: {binding}")
    if dynamic != [("shape", "input"), ("shape", "output")]:
        raise RuntimeError(f"Unexpected wide norm scalar order: {dynamic}")
    digest, shared = compile_image(build / "wide-norm.hsaco", kernel, signature, constants,
                                  ["*bf16"] * 3 + ["i32"] * 2, args.warps, block=8192, allow_shared=True)
    host = compile_host(build / "wide-norm-host.so", ("wide_norm_host.cc",))
    (build / "wide-norm-build.json").write_text(json.dumps(dict(
        source=str(source), hsaco_sha256=digest, warps=args.warps, shared=shared,
        block=8192, width=5376, epsilon=1e-5, host_command=host), indent=2) + "\n")


if __name__ == "__main__":
    main()
