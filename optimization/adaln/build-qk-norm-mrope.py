#!/usr/bin/env python3
"""构建保留 BF16 中间舍入的变长 Q/K RMSNorm 与 MRoPE 九齿融合。"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path

import ninetoothed as nt

from aot_codegen import compile_host, compile_image, lower_kernel
from qk_norm_mrope import application, arrange


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build = args.destination.resolve()
    build.mkdir()
    query = nt.Tensor(shape=(None, 7, 128), dtype=nt.bfloat16)
    cache = nt.Tensor(shape=(None, 48), dtype=nt.bfloat16)
    tensors = (query, nt.Tensor(shape=query.shape, dtype=nt.bfloat16),
               nt.Tensor(shape=(128,), dtype=nt.bfloat16),
               nt.Tensor(shape=(128,), dtype=nt.bfloat16),
               cache, nt.Tensor(shape=cache.shape, dtype=nt.bfloat16),
               nt.Tensor(shape=(3, query.shape[0]), dtype=nt.int64),
               nt.Tensor(shape=query.shape, dtype=nt.bfloat16),
               nt.Tensor(shape=query.shape, dtype=nt.bfloat16))
    source, kernel, bindings = lower_kernel(
        build, partial(arrange, rows=1), application, tensors, "qk_norm_mrope", 2)
    signature, constants, dynamic, runtime = {}, {}, [], []
    dynamic_bindings = {("shape", "query", 0), ("stride", "query", 0),
                        ("stride", "key", 0), ("shape", "cos", 0),
                        ("stride", "positions", 0)}
    constant_strides = {
        "query": (None, 128, 1), "key": (None, 128, 1),
        "query_weight": (1,), "key_weight": (1,),
        "cos": (96, 1), "sin": (96, 1), "positions": (None, 1),
        "query_out": (896, 128, 1), "key_out": (896, 128, 1),
    }
    for binding in bindings:
        name, kind, tensor, axis = binding.name, binding.kind, binding.source, binding.dim
        key = (kind, tensor, axis)
        if kind == "tensor":
            signature[name] = "*i64" if tensor == "positions" else "*bf16"
            runtime.append(signature[name])
        elif key in dynamic_bindings:
            signature[name] = "i32"
            runtime.append("i32")
            dynamic.append(key)
        elif kind == "stride" and constant_strides[tensor][axis] is not None:
            signature[name] = "constexpr"
            constants[name] = constant_strides[tensor][axis]
        else:
            raise RuntimeError(f"未经支持的融合 ABI: {binding}")
    (build / "launch-abi.json").write_text(json.dumps({
        "dynamic": dynamic, "signature": signature, "constants": constants,
    }, indent=2) + "\n")
    expected = [("shape", "query", 0), ("stride", "query", 0),
                ("stride", "key", 0), ("shape", "cos", 0),
                ("stride", "positions", 0)]
    if dynamic != expected:
        raise RuntimeError(f"融合主机桥 ABI 不一致: {dynamic}")
    digest, shared = compile_image(build / "qk-norm-mrope.hsaco", kernel,
                                   signature, constants, runtime, 2,
                                   block=128, allow_shared=True)
    host = compile_host(build / "qk-norm-mrope-host.so", ("qk_norm_mrope_host.cc",))
    record = dict(source=str(source), hsaco_sha256=digest, warps=2, rows=1,
                  shared=shared, block=128, dynamic=dynamic, host_command=host,
                  application_sha256=hashlib.sha256(
                      Path(__file__).with_name("qk_norm_mrope.py").read_bytes()).hexdigest())
    (build / "qk-norm-mrope-build.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
