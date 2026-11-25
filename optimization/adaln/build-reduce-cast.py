#!/usr/bin/env python3
"""预编译 BF16 与通信浮点类型之间的九齿转换。"""
import argparse
import json
from pathlib import Path

import ninetoothed as nt
from aot_codegen import compile_host, compile_image, lower_kernel
from reduce_cast import application, arrange, restore_from_wire, scale_down_for_wire


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--scale-power", type=int, choices=(0, 8), default=0,
                        help="For FP16, scale by 1/256 before reduction and restore after")
    args = parser.parse_args()
    if args.scale_power and args.dtype != "float16":
        parser.error("Scaling is only used for FP16 communication")
    build = args.destination.resolve()
    build.mkdir()
    wide = getattr(nt, args.dtype)
    wire_signature = "fp32" if args.dtype == "float32" else "fp16"
    record = {"dtype": args.dtype, "block": 1024, "warps": 4, "scale_power": args.scale_power}
    for name, input_type, output_type, pointers in (
        ("up", nt.bfloat16, wide, ("*bf16", "*" + wire_signature)),
        ("down", wide, nt.bfloat16, ("*" + wire_signature, "*bf16")),
    ):
        tensors = (nt.Tensor(1, dtype=input_type), nt.Tensor(1, dtype=output_type))
        operation = application
        if args.scale_power:
            operation = restore_from_wire if name == "down" else scale_down_for_wire
        source, kernel, bindings = lower_kernel(build, arrange, operation, tensors, "reducecast_" + name)
        signature, constants = {}, {}
        for binding in bindings:
            if binding.kind == "tensor":
                signature[binding.name] = pointers[0 if binding.source == "input" else 1]
            elif binding.kind == "shape":
                signature[binding.name] = "i32"
            elif binding.kind == "stride":
                signature[binding.name] = "constexpr"
                constants[binding.name] = 1
            else:
                raise RuntimeError(f"Unexpected cast binding: {binding}")
        digest, _ = compile_image(build / f"reducecast-{name}.hsaco", kernel, signature, constants,
                                   [*pointers, "i32", "i32"], block=1024)
        record[name] = {"source": str(source), "hsaco_sha256": digest}
    record["host_command"] = compile_host(build / "reducecast-host.so", ("reduce_cast_host.cc",))
    (build / "reducecast-build.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
