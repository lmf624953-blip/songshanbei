#!/usr/bin/env python3
"""构建同一动态长度 ABI 下的宽归约与索引调制融合候选。"""

import argparse
import json
import os
from pathlib import Path
import sys

import ninetoothed as nt

from aot_codegen import compile_host, compile_image, lower_kernel
from wide_norm_indexed import arrange_naive, naive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)

    parser.add_argument("--warps", type=int, choices=(4, 8, 16), default=4)
    args = parser.parse_args()
    args.output.mkdir()
    hidden = nt.Tensor(shape=(None, 5376), dtype=nt.bfloat16)
    scale = nt.Tensor(shape=(None, 5376), dtype=nt.bfloat16)
    tensors = (hidden, nt.Tensor(shape=(5376,), dtype=nt.bfloat16),
               scale, nt.Tensor(shape=scale.shape, dtype=nt.bfloat16),
               nt.Tensor(shape=(hidden.shape[0],), dtype=nt.int64),
               nt.Tensor(shape=hidden.shape, dtype=nt.bfloat16))
    arrange, app = arrange_naive, naive
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
    command = compile_host(args.output / "wide-norm-indexed-host.so", ("wide_norm_indexed_host.cc",))
    (args.output / "wide-norm-indexed-build.json").write_text(json.dumps(dict(
        arguments=arguments, warps=args.warps, shared=shared, source=str(source),
        mode="naive", hsaco_sha256=digest, host_command=command, experimental=True), indent=2) + "\n")


if __name__ == "__main__":
    main()
