#!/usr/bin/env python3
"""Compile native dispatch for the existing 7-head H3 NineToothed attention."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path

import ninetoothed as nt
from songshan_h3.attention_kernel import definition

from aot_codegen import compile_host, compile_image, lower_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--fp32-scale", action="store_true")
    parser.add_argument("--fp16-pv", action="store_true", help="Use FP16 probabilities and V in the PV dot; experimental")
    parser.add_argument("--aligned-strides", action="store_true")
    parser.add_argument("--stages", default=1, type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--warps", default=4, type=int, choices=(2, 4, 8, 16))
    parser.add_argument("--query-block", default=64, type=int, choices=(32, 64, 128))
    parser.add_argument("--key-block", default=64, type=int, choices=(32, 64, 128))
    parser.add_argument("--matrix-instr-nonkdim", default=0, type=int, choices=(0, 16, 32))
    parser.add_argument("--kpack", default=1, type=int, choices=(1, 2))
    parser.add_argument("--optimize-long-attention", action="store_true",
                        help="Hoist read-only Q and swizzle PV LDS for the validated gfx936 32x32 preset")
    args = parser.parse_args()
    if args.fp16_pv and not args.fp32_scale:
        parser.error("--fp16-pv requires --fp32-scale")
    if args.optimize_long_attention and not (
            args.fp32_scale and not args.fp16_pv and args.aligned_strides
            and args.query_block == args.key_block == 32 and args.warps == 2
            and args.stages == 1 and args.kpack == 2 and args.matrix_instr_nonkdim == 0):
        parser.error("--optimize-long-attention requires the BF16 PV, FP32 scale, aligned 32x32, "
                     "2-warp, 1-stage, kpack=2, matrix-instr-nonkdim=0 preset")
    build = args.destination.resolve()
    build.mkdir()
    arrangement, application, tensors = definition(head_dim=128, dtype=nt.bfloat16, causal=False)
    if args.query_block != 64 or args.key_block != 64:
        from attention_layout import arrangement as configurable_arrangement
        arrangement = partial(configurable_arrangement, query_block=args.query_block, key_block=args.key_block)
    if args.fp32_scale:
        from accurate_attention import application
    if args.fp16_pv:
        from precise_attention import application
    for tensor in (*tensors[:3], tensors[-1]):
        tensor.shape = (1, 7, tensor.shape[2], 128)
    source, kernel, bindings = lower_kernel(build, arrangement, application, tensors, "dense_attention_aot",
                                          warps=args.warps, hoist_query=args.optimize_long_attention)
    signature, constants, dynamic = {}, {}, []
    for binding in bindings:
        name, tensor, kind, axis = binding.name, binding.source, binding.kind, binding.dim
        if kind == "tensor":
            signature[name] = "*bf16"
        elif kind == "scalar" and tensor == "scale":
            signature[name] = "fp32"
        elif axis == 2 and kind in {"shape", "stride"}:
            signature[name] = "i32"
            dynamic.append((tensor, kind))
        else:
            signature[name] = "constexpr"
            if kind == "stride":
                constants[name] = (0, 128, None, 1)[axis]
            elif kind == "constexpr" and tensor == "is_causal":
                constants[name] = False
            else:
                raise RuntimeError(f"Unexpected attention argument: {binding}")
    if dynamic != [(t, k) for t in ("query", "key", "value", "output") for k in ("shape", "stride")]:
        raise RuntimeError(f"Unexpected attention scalar order: {dynamic}")
    warps = args.warps
    aligned = [b.name for b in bindings if b.kind == "stride" and b.dim == 2]
    runtime = ["*bf16"] * 3 + ["fp32", "*bf16"] + ["i32"] * 8
    digest, shared = compile_image(build / "dense-attention-aot.hsaco", kernel, signature, constants,
                                   runtime, warps, block=1, allow_shared=True,
                                   aligned_scalars=aligned if args.aligned_strides else (), num_stages=args.stages,
                                   matrix_instr_nonkdim=args.matrix_instr_nonkdim, kpack=args.kpack,
                                   attention_lds_swizzle=args.optimize_long_attention)
    host = compile_host(build / "dense-attention-aot.so", ("attention_aot_host.cc",),
                        defines=(("H3_ATTENTION_QUERY_BLOCK", args.query_block),))
    record = {"source": str(source), "hsaco_sha256": digest, "warps": warps,
              "shared": shared, "fp32_scale": args.fp32_scale,
              "fp16_pv": args.fp16_pv,
              "query_block": args.query_block,
              "key_block": args.key_block,
              "aligned_strides": args.aligned_strides, "host_command": host}
    record["num_stages"] = args.stages
    record["matrix_instr_nonkdim"] = args.matrix_instr_nonkdim
    record["kpack"] = args.kpack
    record["optimize_long_attention"] = args.optimize_long_attention
    if args.optimize_long_attention:
        from attention_codegen import LONG_ATTENTION_LDS_SWIZZLE
        record["codegen_passes"] = ["readonly_query_licm", "pv_lds_swizzle"]
        record["pv_lds_swizzle"] = LONG_ATTENTION_LDS_SWIZZLE
        record["codegen_sha256"] = hashlib.sha256(
            Path(__file__).with_name("attention_codegen.py").read_bytes()).hexdigest()
        record["generated_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (build / "dense-attention-aot-build.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
