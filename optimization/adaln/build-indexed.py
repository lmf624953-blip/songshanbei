#!/usr/bin/env python3
"""Compile and package the indexed AdaLN and gate kernels for BW1000."""
import argparse
from functools import partial
import json
from pathlib import Path

import ninetoothed as nt

from aot_codegen import WIDTH, compile_host, compile_image, lower_kernel
from indexed_kernel import arrange_adaln, arrange_gate, indexed_adaln, indexed_gate


def compile_device(build, arrange, application, tables, expected_scalar_sources, warps, tile):
    name = application.__name__
    tensors = (
        nt.Tensor(2, dtype=nt.bfloat16), nt.Tensor(2, dtype=nt.bfloat16),
        nt.Tensor(2, dtype=nt.bfloat16), nt.Tensor(1, dtype=nt.int64),
        nt.Tensor(2, dtype=nt.bfloat16),
    )
    source, kernel, bindings = lower_kernel(build, partial(arrange, tile=tile), application, tensors, name, warps)
    signature, constants = {}, {}
    scalars = []
    for binding in bindings:
        variable, arg = binding.name, binding.source
        if binding.kind == "tensor":
            signature[variable] = "*i64" if arg == "indices" else "*bf16"
        elif binding.kind == "shape" and binding.dim == 0:
            signature[variable] = "i32"
            scalars.append(arg)
        else:
            signature[variable] = "constexpr"
            if binding.kind == "shape" and binding.dim == 1:
                constants[variable] = WIDTH
            elif binding.kind == "stride":
                stride = WIDTH * (6 if arg in tables else 1)
                constants[variable] = stride if binding.dim == 0 and arg != "indices" else 1
            else:
                raise RuntimeError(f"Unexpected NineToothed launch binding: {binding}")
    if scalars != expected_scalar_sources:
        raise RuntimeError(f"Unexpected {name} scalar parameter order: {scalars}")
    runtime_signature = ["*bf16", "*bf16", "*bf16", "*i64", "*bf16"] + ["i32"] * 5
    digest, _ = compile_image(build / f"{name}.hsaco", kernel, signature, constants, runtime_signature, warps, block=tile)
    return {"source": str(source), "hsaco_sha256": digest, "scalar_sources": scalars, "warps": warps, "tile": tile}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--warps", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--tile", type=int, choices=(256, 512, 1024, 2048, 4096, 8192), default=256)
    args = parser.parse_args()
    build = args.destination.resolve()
    build.mkdir()
    cases = (
        (arrange_adaln, indexed_adaln, {"scale", "shift"}, ["normalized", "scale", "shift", "indices", "output"]),
        (arrange_gate, indexed_gate, {"gate"}, ["residual", "activation", "gate", "indices", "output"]),
    )
    record = {app.__name__: compile_device(build, arrange, app, tables, scalars, args.warps, args.tile)
              for arrange, app, tables, scalars in cases}
    record["host_command"] = compile_host(build / "indexed-host.so", ("indexed_host.cc",), defines=(("H3_INDEXED_TILE", args.tile),))
    (build / "indexed-build.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
