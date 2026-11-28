#!/usr/bin/env python3
"""生成连续布局的 VAE RoPE HSACO，用于去掉通用 JIT 发射路径的对照。"""

import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys

import ninetoothed as nt

from rope_flat import apply_rope_flat, arrange

sys.path.insert(0, str(Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization/adaln"))
from aot_codegen import compile_image, lower_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tiles", type=int, nargs="+", default=[512, 2048])
    args = parser.parse_args()
    args.output.mkdir()
    flat, frequency = (None,), (None, 48)
    shapes = flat, flat, frequency, frequency, flat, flat
    tensors = tuple(nt.Tensor(shape=shape, dtype=nt.bfloat16) for shape in shapes)
    for tile in args.tiles:
        directory = args.output / str(tile)
        directory.mkdir()
        _, kernel, bindings = lower_kernel(directory, partial(arrange, tile=tile), apply_rope_flat,
                                           tensors, "apply_rope_flat", 4)
        signature, constants, runtime, arguments = {}, {}, [], []
        for arg in bindings:
            if arg.kind == "tensor":
                signature[arg.name] = "*bf16"
                arguments.append(("ptr", arg.source))
            elif arg.kind == "shape" and arg.dim == 0:
                signature[arg.name] = "i32"
                arguments.append(("i32", "tokens" if arg.source in ("cosine", "sine") else "elements"))
            elif arg.kind == "stride":
                signature[arg.name] = "constexpr"
                constants[arg.name] = 48 if arg.source in ("cosine", "sine") and arg.dim == 0 else 1
            else:
                raise RuntimeError(f"不支持的绑定 {arg}")
            if signature[arg.name] != "constexpr":
                runtime.append(signature[arg.name])
        digest, shared = compile_image(directory / "wide-norm-indexed.hsaco", kernel, signature,
                                       constants, runtime, 4, block=tile)
        report = dict(arguments=arguments, tile=tile, warps=4, shared=shared, hsaco_sha256=digest)
        (directory / "build.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
