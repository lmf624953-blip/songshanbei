#!/usr/bin/env python3
"""Compile the NineToothed AdaLN for the H3 BF16 hidden width on BW1000."""
import argparse
import json
from pathlib import Path

import ninetoothed

from adaln_kernel import adaln
from aot_codegen import TILE, WARPS, WIDTH, compile_host, compile_image, lower_kernel


def arrangement(normalized, scale, shift, output):
    return tuple(tensor.tile((1, TILE)) for tensor in (normalized, scale, shift, output))


def compile_device(build):
    tensors = tuple(ninetoothed.Tensor(2, dtype=ninetoothed.bfloat16) for _ in range(4))
    source, kernel, bindings = lower_kernel(build, arrangement, adaln, tensors, "adaln_aot")
    row_strides = {"normalized": WIDTH, "scale": 6 * WIDTH, "shift": 6 * WIDTH, "output": WIDTH}
    signature = {}
    constants = {}
    for binding in bindings:
        name = binding.name
        if binding.kind == "tensor":
            signature[name] = "*bf16"
        elif binding.kind == "shape" and binding.dim == 0:
            signature[name] = "i32"
        else:
            signature[name] = "constexpr"
            if binding.kind == "shape" and binding.dim == 1:
                constants[name] = WIDTH
            elif binding.kind == "stride":
                constants[name] = row_strides[binding.source] if binding.dim == 0 else 1
            else:
                raise RuntimeError(f"Unexpected NineToothed launch binding: {binding}")
    digest, _ = compile_image(build / "adaln-aot.hsaco", kernel, signature, constants,
                              ["*bf16"] * 4 + ["i32"] * 4)
    return {
        "source": str(source),
        "hsaco_sha256": digest,
        "arch": "gfx936",
        "row_strides": row_strides,
        "threads": WARPS * 64,
        "tile": TILE,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    build = parser.parse_args().destination.resolve()
    build.mkdir()
    record = compile_device(build)
    record["host_command"] = compile_host(build / "adaln-aot-host.so", ("adaln_aot_host.cc", "adaln_host.cc"))
    (build / "adaln-aot-build.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
