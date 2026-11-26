#!/usr/bin/env python3
"""构建同一动态长度 ABI 下的宽归约与索引调制融合候选。"""

import argparse
import json
import os
from pathlib import Path
import sys

import ninetoothed as nt

from aot_codegen import compile_host, compile_image, lower_kernel
from wide_norm_indexed import arrange_naive, naive, original_serial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)

    parser.add_argument("--warps", type=int, choices=(4, 8, 16), default=4)
    parser.add_argument("--mode", choices=("naive", "original-serial"), default="naive")
    args = parser.parse_args()
    args.output.mkdir()
    if args.mode=="original-serial":
        os.environ["H3_AOT_SAVE_IR"]="1"
    hidden = nt.Tensor(shape=(None, 5376), dtype=nt.bfloat16)
    scale = nt.Tensor(shape=(None, 5376), dtype=nt.bfloat16)
    tensors = (hidden, nt.Tensor(shape=(5376,), dtype=nt.bfloat16),
               scale, nt.Tensor(shape=scale.shape, dtype=nt.bfloat16),
               nt.Tensor(shape=(hidden.shape[0],), dtype=nt.int64),
               nt.Tensor(shape=hidden.shape, dtype=nt.bfloat16))
    arrange, app = arrange_naive, (naive if args.mode=="naive" else original_serial)
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
    if args.mode=="original-serial":
        import hashlib
        import triton
        from triton.backends.compiler import GPUTarget
        ir=args.output/"wide-norm-indexed.ttgir"
        text=ir.read_text()
        previous=f"#blocked = #ttg.blocked<{{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [{args.warps}], order = [0]}}>"
        if text.count(previous)!=1:
            raise RuntimeError("原库归约布局不匹配；禁止跳过布局守卫")
        text=text.replace(previous,previous.replace("sizePerThread = [8]","sizePerThread = [1]"))
        ir.write_text(text)
        compiled=triton.compile(str(ir),target=GPUTarget("hip","gfx936",64),options=dict(num_warps=args.warps,num_stages=1,matrix_instr_nonkdim=0,kpack=1))
        image=compiled.asm["hsaco"]
        (args.output/"wide-norm-indexed.hsaco").write_bytes(image)
        (args.output/"wide-norm-indexed.amdgcn").write_text(compiled.asm["amdgcn"])
        digest=hashlib.sha256(image).hexdigest()
        shared=compiled.metadata.shared
    command = compile_host(args.output / "wide-norm-indexed-host.so", ("wide_norm_indexed_host.cc",))
    (args.output / "wide-norm-indexed-build.json").write_text(json.dumps(dict(
        arguments=arguments, warps=args.warps, shared=shared, source=str(source),
        mode=args.mode, hsaco_sha256=digest, host_command=command, experimental=True), indent=2) + "\n")


if __name__ == "__main__":
    main()
