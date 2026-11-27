#!/usr/bin/env python3
"""构建九齿 V 预排布 AOT，固定真实 packed 输入步长。"""

import argparse
from functools import partial
import json
from pathlib import Path
import sys

import ninetoothed as nt

from v_prepack_kernel import arrangement, application


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("--tile", type=int, choices=(256, 512, 1024, 2048, 4096), default=1024)
    cli.add_argument("--warps", type=int, choices=(2, 4, 8), default=4)
    args = cli.parse_args()
    code = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(code / "adaln"))
    from aot_codegen import lower_kernel, compile_image
    args.output.mkdir()
    source, kernel, bindings = lower_kernel(args.output, partial(arrangement, tile=args.tile), application, (
        nt.Tensor(shape=(None, 7, 128), dtype=nt.bfloat16),
        nt.Tensor(0, dtype=nt.int32), nt.Tensor(1, dtype=nt.bfloat16),
    ), "h3_prepack_value", args.warps)
    print([str(binding) for binding in bindings], flush=True)
    signature, constants, runtime, arguments = {}, {}, [], []
    for binding in bindings:
        if binding.kind == "tensor":
            dtype = "*bf16"
            signature[binding.name] = dtype
            runtime.append(dtype)
            arguments.append(dict(kind=binding.kind, source=binding.source))
        elif binding.kind == "scalar" and binding.source == "blocks":
            signature[binding.name] = "i32"
            runtime.append("i32")
            arguments.append(dict(kind="scalar", source="blocks"))
        elif binding.kind == "shape":
            if binding.source == "value" and binding.dim in (1, 2):
                signature[binding.name] = "constexpr"
                constants[binding.name] = (7, 128)[binding.dim - 1]
            else:
                signature[binding.name] = "i32"
                runtime.append("i32")
                arguments.append(dict(kind="shape", source=binding.source, dim=binding.dim))
        elif binding.kind == "stride":
            signature[binding.name] = "constexpr"
            constants[binding.name] = (2688, 128, 1)[binding.dim] if binding.source == "value" else 1
        else:
            raise ValueError(f"Unexpected binding {binding}")
    sha, shared = compile_image(args.output / "prepack-value.hsaco", kernel, signature, constants,
                                 runtime, args.warps, block=args.tile)
    assert runtime == ["*bf16", "i32", "*bf16", "i32", "i32"]
    record = dict(source=str(source), hsaco_sha256=sha, shared=shared, warps=args.warps,
                  tile=args.tile, arguments=arguments)
    (args.output / "prepack-value-build.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
