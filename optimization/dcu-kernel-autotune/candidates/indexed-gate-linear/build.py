#!/usr/bin/env python3
"""仅 CPU 构建线性门控微测镜像；正式模型及默认配置保持原状。"""
import argparse
import hashlib
import json
import os
import sys
import re
from pathlib import Path

import ninetoothed as nt

sys.path.insert(0, str(Path(os.environ['H3_AUTOTUNE_CODE_ROOT']) / 'optimization/adaln'))
from aot_codegen import compile_image, lower_kernel
from kernel import application


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('--tile', type=int, choices=(512, 1024, 2048, 4096, 8192), required=True)
    cli.add_argument('--warps', type=int, choices=(2, 4, 8), default=4)
    cli.add_argument('--unify-layout', action='store_true')
    a = cli.parse_args()
    a.output.mkdir()
    if a.unify_layout:
        os.environ['H3_AOT_SAVE_IR'] = '1'

    def arrange(residual, activation, gate, indices, output):
        return (residual.tile((1, 5376)), activation.tile((1, 5376)),
                gate.tile((1, 5376)), indices.tile((1,)), output.tile((a.tile,)))

    residual = nt.Tensor(shape=(None, 5376), dtype=nt.bfloat16)
    tensors = (residual, nt.Tensor(shape=residual.shape, dtype=nt.bfloat16),
               nt.Tensor(shape=(None, 5376), dtype=nt.bfloat16),
               nt.Tensor(shape=(residual.shape[0],), dtype=nt.int64),
               nt.Tensor(shape=(None,), dtype=nt.bfloat16))
    source, kernel, bindings = lower_kernel(a.output, arrange, application, tensors,
                                            'wide_norm_indexed', a.warps)
    signature, constants, runtime, arguments = {}, {}, [], []
    for binding in bindings:
        name = binding.name
        if binding.kind == 'tensor':
            signature[name] = '*i64' if binding.source == 'indices' else '*bf16'
            arguments.append(('ptr', binding.source))
        elif binding.kind == 'shape' and binding.dim == 0:
            signature[name] = 'i32'
            arguments.append(('i32', 'elements' if binding.source == 'output'
                              else 'table_rows' if binding.source == 'gate' else 'input_rows'))
        elif binding.kind == 'stride':
            signature[name] = 'constexpr'
            constants[name] = (5376 * (6 if binding.source == 'gate' else 1)
                               if binding.dim == 0 and binding.source != 'indices' else 1)
        else:
            raise RuntimeError(f'不支持绑定 {binding}')
        if signature[name] != 'constexpr':
            runtime.append(signature[name])
    digest, shared = compile_image(a.output / 'wide-norm-indexed.hsaco', kernel, signature,
                                    constants, runtime, a.warps, block=a.tile, allow_shared=True)
    if a.unify_layout:
        import triton
        from triton.backends.compiler import GPUTarget
        ir = a.output / 'wide-norm-indexed.ttgir'
        text = ir.read_text()
        layouts = dict(re.findall(r'^(#\w+) = (#ttg.blocked<[^\n]+>)$', text, re.M))
        assert len(layouts) == 2, layouts
        vector = next(name for name, value in layouts.items() if 'sizePerThread = [1]' not in value)
        scalar = next(name for name, value in layouts.items() if 'sizePerThread = [1]' in value)
        vector_size = re.search(r'sizePerThread = \[(\d+)\]', layouts[vector])[1]
        assert int(vector_size) in (2, 4, 8)
        assert layouts[scalar].replace('sizePerThread = [1]', f'sizePerThread = [{vector_size}]') == layouts[vector]
        text = re.sub(r'^' + re.escape(scalar) + r' = [^\n]+\n', '', text, flags=re.M)
        text = re.sub(re.escape(scalar) + r'(?![\w])', vector, text)
        ir.write_text(text)
        compiled = triton.compile(str(ir), target=GPUTarget('hip', 'gfx936', 64), options=dict(
            num_warps=a.warps, num_stages=1, matrix_instr_nonkdim=0, kpack=1))
        assert compiled.metadata.profile_scratch_size == 0
        image = compiled.asm['hsaco']
        (a.output / 'wide-norm-indexed.hsaco').write_bytes(image)
        (a.output / 'wide-norm-indexed.amdgcn').write_text(compiled.asm['amdgcn'])
        digest, shared = hashlib.sha256(image).hexdigest(), compiled.metadata.shared
    (a.output / 'build.json').write_text(json.dumps(dict(
        arguments=arguments, warps=a.warps, tile=a.tile, shared=shared,
        hsaco_sha256=digest, source=str(source), mode='indexed-gate-linear',
        unified_layout=a.unify_layout), indent=2) + '\n')


if __name__ == '__main__':
    main()
