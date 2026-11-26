#!/usr/bin/env python3
"""对照逐行分块，编译连续扁平分块的九齿索引门控。"""

import argparse
import json
import os
from pathlib import Path
import sys

import ninetoothed as nt
import ninetoothed.language as ntl

sys.path.insert(0, str(Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization/adaln"))
from aot_codegen import WIDTH, compile_image, lower_kernel


def indexed_gate_flat(residual, activation, gate, indices, output):
    element = output.offsets(0).to(ntl.int32)
    token = element // 5376
    column = element % 5376
    table_row = indices.source[token].to(ntl.int32)
    gate_value = gate.source[table_row, column].to(ntl.float32)
    activation_value = activation.source[element].to(ntl.float32)
    residual_value = residual.source[element].to(ntl.float32)
    scaled = (gate_value * activation_value).to(output.dtype)
    output = residual_value + scaled.to(ntl.float32)



def indexed_gate_segmented(residual, activation, gate, indices, output):
    element = output.offsets(0).to(ntl.int32)
    segment = element // 256
    token = segment // 21
    column = (segment % 21) * 256 + element % 256
    table_row = indices.source[token].to(ntl.int32)
    gate_value = gate.source[table_row, column].to(ntl.float32)
    activation_value = activation.source[element].to(ntl.float32)
    residual_value = residual.source[element].to(ntl.float32)
    scaled = (gate_value * activation_value).to(output.dtype)
    output = residual_value + scaled.to(ntl.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tile", type=int, choices=(1024, 2048, 4096, 8192), required=True)
    parser.add_argument("--warps", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--aligned-elements", action="store_true")
    parser.add_argument("--segmented-index", action="store_true")
    parser.add_argument("--buffer-addressing", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True)

    def arrange(residual, activation, gate, indices, output):
        return residual.tile((args.tile,)), activation.tile((args.tile,)), gate.tile((1, WIDTH)), indices.tile((1,)), output.tile((args.tile,))

    elements = nt.Tensor(shape=(None,), dtype=nt.bfloat16)
    table = nt.Tensor(shape=(None, WIDTH), dtype=nt.bfloat16)
    tokens = nt.Tensor(shape=(None,), dtype=nt.int64)
    tensors = (elements, nt.Tensor(shape=elements.shape, dtype=nt.bfloat16), table,
               tokens, nt.Tensor(shape=elements.shape, dtype=nt.bfloat16))
    application = indexed_gate_segmented if args.segmented_index else indexed_gate_flat
    source, kernel, bindings = lower_kernel(args.output, arrange, application, tensors,
                                           "indexed_gate_flat", args.warps)
    signature, constants, arguments = {}, {}, []
    aligned_scalars = []
    shape_names = {"residual": "elements", "activation": "elements", "output": "elements",
                   "indices": "rows", "gate": "table_rows"}
    for binding in bindings:
        name, tensor = binding.name, binding.source
        if binding.kind == "tensor":
            signature[name] = "*i64" if tensor == "indices" else "*bf16"
            arguments.append(("ptr", tensor))
        elif binding.kind == "shape" and binding.dim == 0:
            signature[name] = "i32"
            arguments.append(("i32", shape_names[tensor]))
            if args.aligned_elements and shape_names[tensor] == "elements":
                aligned_scalars.append(name)
        elif binding.kind == "stride":
            signature[name] = "constexpr"
            constants[name] = 6 * WIDTH if tensor == "gate" and binding.dim == 0 else 1
        else:
            raise RuntimeError(f"不支持的绑定 {binding}")

    if [v for k, v in arguments if k == "ptr"] != ["residual", "activation", "gate", "indices", "output"]:
        raise RuntimeError(f"门控指针 ABI 不符: {arguments}")
    runtime = [kind for kind in signature.values() if kind != "constexpr"]
    if args.buffer_addressing:
        os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"
    digest, shared = compile_image(args.output / "indexed_gate_flat.hsaco", kernel, signature,
                                   constants, runtime, args.warps, block=args.tile, aligned_scalars=aligned_scalars,
                                   buffer_addressing=args.buffer_addressing)
    record = dict(hsaco_sha256=digest, shared=shared, warps=args.warps, tile=args.tile,
                  arguments=arguments, source=str(source),
                  aligned_elements=args.aligned_elements, segmented_index=args.segmented_index,
                  buffer_addressing=args.buffer_addressing)
    (args.output / "build.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
