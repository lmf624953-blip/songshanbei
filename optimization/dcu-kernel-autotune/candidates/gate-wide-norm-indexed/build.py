#!/usr/bin/env python3
"""以九齿生成门控残差和宽归约调制的双输出探索镜像。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import ninetoothed as nt
sys.path.insert(0, str(Path(os.environ['H3_AUTOTUNE_CODE_ROOT']) / 'optimization/adaln'))
from aot_codegen import compile_image, lower_kernel
from kernel import arrange, application


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('--warps', type=int, default=4, choices=(4, 8))
    args = cli.parse_args()
    args.output.mkdir()
    hidden = nt.Tensor(shape=(None, 5376), dtype=nt.bfloat16)
    table = nt.Tensor(shape=(None, 5376), dtype=nt.bfloat16)
    tensors = (hidden, nt.Tensor(shape=hidden.shape, dtype=nt.bfloat16),
               nt.Tensor(shape=(5376,), dtype=nt.bfloat16), table,
               nt.Tensor(shape=table.shape, dtype=nt.bfloat16),
               nt.Tensor(shape=table.shape, dtype=nt.bfloat16),
               nt.Tensor(shape=(hidden.shape[0],), dtype=nt.int64),
               nt.Tensor(shape=hidden.shape, dtype=nt.bfloat16),
               nt.Tensor(shape=hidden.shape, dtype=nt.bfloat16))
    source, kernel, bindings = lower_kernel(args.output, arrange, application, tensors,
                                           'wide_norm_indexed', args.warps)
    signature, constants, runtime, arguments = {}, {}, [], []
    for binding in bindings:
        name, origin = binding.name, binding.source
        if binding.kind == 'tensor':
            signature[name] = '*i64' if origin == 'indices' else '*bf16'
            arguments.append(('ptr', origin))
        elif binding.kind == 'shape' and binding.dim == 0 and origin in ('hidden', 'gate'):
            signature[name] = 'i32'
            arguments.append(('i32', 'rows' if origin == 'hidden' else 'table_rows'))
        elif binding.kind == 'stride':
            signature[name] = 'constexpr'
            constants[name] = 5376 * (6 if origin in ('gate', 'scale', 'shift') else 1) if binding.dim == 0 and origin not in ('weight', 'indices') else 1
        else:
            raise RuntimeError(f'不支持的绑定 {binding}')
        if signature[name] != 'constexpr':
            runtime.append(signature[name])
    assert len(arguments) == 11, arguments
    digest, shared = compile_image(args.output/'wide-norm-indexed.hsaco', kernel, signature,
                                   constants, runtime, args.warps, block=8192, allow_shared=True)
    (args.output/'build.json').write_text(json.dumps(dict(arguments=arguments, warps=args.warps,
        shared=shared, source=str(source), hsaco_sha256=digest, mode='gate-norm-indexed',
        scope='仅供独立微测，未接入正式模型'), ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
