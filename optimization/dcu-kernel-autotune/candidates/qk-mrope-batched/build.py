#!/usr/bin/env python3
"""复用验收版九齿数学表达，仅改变每个 CTA 承担的 token 行数。"""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys

import ninetoothed as nt

sys.path.insert(0, str(Path(os.environ['H3_AUTOTUNE_CODE_ROOT']) / 'optimization/adaln'))
from aot_codegen import compile_image, lower_kernel
from qk_norm_mrope import application, arrange


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('--rows', type=int, choices=(1, 2, 4, 8), required=True)
    cli.add_argument('--warps', type=int, choices=(2, 4, 8), required=True)
    a = cli.parse_args()
    if a.rows != 1:
        raise RuntimeError('多行arrangement仍生成一维归约与按行program域，已在语义审查中拒绝；需先实现分段归约')
    a.output.mkdir()
    query = nt.Tensor(shape=(None, 7, 128), dtype=nt.bfloat16)
    cache = nt.Tensor(shape=(None, 48), dtype=nt.bfloat16)
    tensors = (query, nt.Tensor(shape=query.shape, dtype=nt.bfloat16),
               nt.Tensor(shape=(128,), dtype=nt.bfloat16), nt.Tensor(shape=(128,), dtype=nt.bfloat16),
               cache, nt.Tensor(shape=cache.shape, dtype=nt.bfloat16),
               nt.Tensor(shape=(3, query.shape[0]), dtype=nt.int64),
               nt.Tensor(shape=query.shape, dtype=nt.bfloat16), nt.Tensor(shape=query.shape, dtype=nt.bfloat16))
    source, kernel, bindings = lower_kernel(a.output, partial(arrange, rows=a.rows), application,
                                            tensors, 'wide_norm_indexed', a.warps)
    dynamic = {('shape', 'query', 0): 'input_rows', ('stride', 'query', 0): 'query_stride',
               ('stride', 'key', 0): 'key_stride', ('shape', 'cos', 0): 'cache_rows',
               ('stride', 'positions', 0): 'positions_stride'}
    strides = dict(query=(None, 128, 1), key=(None, 128, 1), query_weight=(1,), key_weight=(1,),
                   cos=(96, 1), sin=(96, 1), positions=(None, 1),
                   query_out=(896, 128, 1), key_out=(896, 128, 1))
    signature, constants, runtime, arguments = {}, {}, [], []
    for b in bindings:
        name, source_name = b.name, b.source
        key = (b.kind, source_name, b.dim)
        if b.kind == 'tensor':
            signature[name] = '*i64' if source_name == 'positions' else '*bf16'
            arguments.append(('ptr', source_name))
        elif key in dynamic:
            signature[name] = 'i32'
            arguments.append(('i32', dynamic[key]))
        elif b.kind == 'stride' and strides[source_name][b.dim] is not None:
            signature[name] = 'constexpr'
            constants[name] = strides[source_name][b.dim]
        else:
            raise RuntimeError(f'不支持绑定 {b}')
        if signature[name] != 'constexpr':
            runtime.append(signature[name])
    assert len(arguments) == 14, arguments
    digest, shared = compile_image(a.output / 'wide-norm-indexed.hsaco', kernel, signature,
        constants, runtime, a.warps, block=a.rows * 128, allow_shared=True)
    app = Path(os.environ['H3_AUTOTUNE_CODE_ROOT']) / 'optimization/adaln/qk_norm_mrope.py'
    (a.output / 'build.json').write_text(json.dumps(dict(arguments=arguments, warps=a.warps,
        row_tile=a.rows, shared=shared, hsaco_sha256=digest, source=str(source),
        application_sha256=hashlib.sha256(app.read_bytes()).hexdigest(), mode='qk-mrope-batched'), indent=2) + '\n')


if __name__ == '__main__':
    main()
