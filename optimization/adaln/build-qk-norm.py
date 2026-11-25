#!/usr/bin/env python3
"""Ahead-of-time build of the NineToothed H3 Q/K RMSNorm candidate."""
import argparse
from functools import partial
import json
from pathlib import Path

import ninetoothed as nt

from aot_codegen import compile_host, compile_image, lower_kernel
from qk_norm import adjacent_rmsnorm, arrange, cub_order_rmsnorm, rmsnorm


HEADS, WIDTH, EPS = 7, 128, 1e-5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--warps", type=int, choices=(1, 2, 4), default=2)
    order = parser.add_mutually_exclusive_group()
    order.add_argument("--adjacent", action="store_true")
    order.add_argument("--cub-order", action="store_true")
    args = parser.parse_args()
    build = args.destination.resolve()
    build.mkdir()
    tensors = (nt.Tensor(3, dtype=nt.bfloat16), nt.Tensor(1, dtype=nt.bfloat16),
               nt.Tensor(0, dtype=nt.float32), nt.Tensor(3, dtype=nt.bfloat16))
    source, kernel, bindings = lower_kernel(
        build, partial(arrange, rows=1, heads=1, width=WIDTH),
        cub_order_rmsnorm if args.cub_order else adjacent_rmsnorm if args.adjacent else rmsnorm,
        tensors, "qk_norm", args.warps,
    )
    signature, constants, dynamic = {}, {}, []
    for binding in bindings:
        name = binding.name
        kind, tensor, axis = binding.kind, binding.source, binding.dim
        if kind == "tensor":
            signature[name] = "*bf16"
            continue
        if axis == 0 and (kind, tensor) in {("shape", "input"), ("stride", "input"), ("shape", "output")}:
            signature[name] = "i32"
            dynamic.append((kind, tensor))
            continue
        signature[name] = "constexpr"
        if kind == "scalar" and tensor == "epsilon":
            constants[name] = EPS
        elif kind == "shape":
            constants[name] = HEADS if axis == 1 else WIDTH
        elif kind == "stride":
            constants[name] = 1 if tensor == "weight" else (HEADS * WIDTH, WIDTH, 1)[axis]
        else:
            raise RuntimeError(f"Unexpected Q/K RMSNorm argument: {binding}")
    if dynamic != [("shape", "input"), ("stride", "input"), ("shape", "output")]:
        raise RuntimeError(f"Unexpected Q/K RMSNorm scalar order: {dynamic}")
    digest, shared = compile_image(build / "qk-norm.hsaco", kernel, signature, constants,
                                   ["*bf16"] * 3 + ["i32"] * 3, args.warps, block=WIDTH, allow_shared=True)
    host = compile_host(build / "qk-norm-host.so", ("qk_norm_host.cc",))
    record = {"source": str(source), "hsaco_sha256": digest, "warps": args.warps,
              "shared": shared, "block": WIDTH, "adjacent": args.adjacent,
              "cub_order": args.cub_order, "host_command": host}
    (build / "qk-norm-build.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
