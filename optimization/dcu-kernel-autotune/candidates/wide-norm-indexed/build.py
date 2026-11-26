#!/usr/bin/env python3
"""构建同一动态长度 ABI 下的宽归约与索引调制融合候选。"""

import argparse
import json
import os
from pathlib import Path
import sys

import ninetoothed as nt

sys.path.insert(0, str(Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization/adaln"))
from aot_codegen import compile_image, lower_kernel
from kernel import arrange_cub, arrange_naive, cub, cub_vector, naive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("naive", "cub", "cub-vector"), required=True)
    parser.add_argument("--warps", type=int, choices=(4, 8, 16), required=True)
    args = parser.parse_args()
    args.output.mkdir()
    hidden = nt.Tensor(shape=(None, 5376), dtype=nt.bfloat16)
    scale = nt.Tensor(shape=(None, 5376), dtype=nt.bfloat16)
    tensors = (hidden, nt.Tensor(shape=(5376,), dtype=nt.bfloat16),
               scale, nt.Tensor(shape=scale.shape, dtype=nt.bfloat16),
               nt.Tensor(shape=(hidden.shape[0],), dtype=nt.int64),
               nt.Tensor(shape=hidden.shape, dtype=nt.bfloat16))
    arrange, app = {
        "naive": (arrange_naive, naive),
        "cub": (arrange_cub, cub),
        "cub-vector": (arrange_naive, cub_vector),
    }[args.mode]
    source, kernel, bindings = lower_kernel(
        args.output, arrange, app, tensors, "wide_norm_indexed", args.warps)
    signature, constants, runtime, arguments = {}, {}, [], []
    for binding in bindings:
        name, source_name = binding.name, binding.source
        if binding.kind == "tensor":
            signature[name] = "*i64" if source_name == "indices" else "*bf16"
            arguments.append(("ptr", source_name))
        elif binding.kind == "shape" and binding.dim == 0 and source_name in ("hidden", "scale"):
            signature[name] = "i32"
            arguments.append(("i32", "rows" if source_name == "hidden" else "table_rows"))
        elif binding.kind == "stride":
            signature[name] = "constexpr"
            if binding.dim == 0 and source_name not in ("weight", "indices"):
                constants[name] = 5376 * (6 if source_name in ("scale", "shift") else 1)
            else:
                constants[name] = 1
        else:
            raise RuntimeError(f"不支持的绑定 {binding}")
        if signature[name] != "constexpr":
            runtime.append(signature[name])
    assert len(arguments) == 8
    digest, shared = compile_image(
        args.output / "wide-norm-indexed.hsaco", kernel, signature, constants, runtime,
        args.warps, block=8192, allow_shared=True)
    (args.output / "build.json").write_text(json.dumps(dict(
        arguments=arguments, warps=args.warps, shared=shared, source=str(source),
        mode=args.mode, hsaco_sha256=digest), indent=2) + "\n")


if __name__ == "__main__":
    main()
