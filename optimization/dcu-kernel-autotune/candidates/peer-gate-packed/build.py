#!/usr/bin/env python3
"""从九齿表达生成双BF16分片门控，记录32位指针ABI及原舍入边界。"""

import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys

import ninetoothed as nt

from kernel import application, arrangement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--warps", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir()
    code = Path(os.environ["H3_AUTOTUNE_CODE_ROOT"])
    sys.path.insert(0, str(code / "optimization/adaln"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rccl-sequence-parallel"))
    from aot_codegen import compile_image, lower_kernel
    from nt_bitcast_compat import enable
    compatibility = enable()
    tensors = tuple(nt.Tensor(1, dtype=nt.uint32) for _ in range(8)) + (
        nt.Tensor(1, dtype=nt.uint8), nt.Tensor(1, dtype=nt.uint32),
        nt.Tensor(2, dtype=nt.uint32), nt.Tensor(1, dtype=nt.int64),
        nt.Tensor(0, dtype=nt.int32), nt.Tensor(0, dtype=nt.int32),
        nt.Tensor(1, dtype=nt.uint32))
    source, kernel, bindings = lower_kernel(
        args.output, partial(arrangement, tile=args.tile), application, tensors, "h3_peer_gate", args.warps)
    signature, constants, runtime, arguments = {}, {}, [], []
    for binding in bindings:
        if binding.kind == "tensor":
            dtype = "*u8" if binding.source == "owners" else "*i64" if binding.source == "indices" else "*u32"
            signature[binding.name] = dtype
            runtime.append(dtype)
            arguments.append(dict(kind="tensor", source=binding.source))
        elif binding.kind in ("shape", "scalar"):
            if binding.kind == "shape" and binding.source == "gate" and binding.dim == 1:
                signature[binding.name] = "constexpr"
                constants[binding.name] = 2688
            else:
                signature[binding.name] = "i32"
                runtime.append("i32")
                arguments.append(dict(kind=binding.kind, source=binding.source, dim=binding.dim))
        elif binding.kind == "stride":
            signature[binding.name] = "constexpr"
            constants[binding.name] = 16128 if binding.source == "gate" and binding.dim == 0 else 1
        else:
            raise ValueError(binding)
    os.environ["H3_AOT_SAVE_IR"] = "1"
    digest, shared = compile_image(args.output / "peer-gate.hsaco", kernel, signature, constants,
                                    runtime, args.warps, block=args.tile)
    metadata = dict(packed_bf16=True, tile=args.tile, warps=args.warps, shared=shared,
                    hsaco_sha256=digest, arguments=arguments, bitcast_compat=compatibility,
                    source_sha256=hashlib.sha256(Path(__file__).with_name("kernel.py").read_bytes()).hexdigest(),
                    lowered=str(source), numerical_verified=False,
                    arithmetic="两个BF16分别逐步RNE、门控RNE、残差RNE；每对共享rank选择")
    (args.output / "peer-gate-build.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
