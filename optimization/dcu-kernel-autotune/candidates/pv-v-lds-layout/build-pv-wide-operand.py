#!/usr/bin/env python3
"""仅改变 Q128 packed 的 PV 操作数打包宽度，保留所有矩阵运算。"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil

import triton
from triton.backends.compiler import GPUTarget
from songshan_h3.compiler_compat import configure_hygon_compiler


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('reference', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--wide-p', action='store_true')
    cli.add_argument('--wide-v', action='store_true')
    cli.add_argument('--direct-p', action='store_true')
    a = cli.parse_args()
    root = a.reference.resolve(strict=True)
    record = json.loads((root / 'dense-attention-aot-build.json').read_text())
    assert record['query_block'] == 128 and record['key_block'] == 32
    assert record['warps'] == 4 and record['packed_strides']['strides'] == [896, 896, 2688, 896]
    source = (root / 'dense-attention-packed-pv-swizzle.ttgir').read_text()
    if a.direct_p:
        helper = runpy.run_path(str(Path(__file__).with_name('p_resources.py')))
        source = helper['apply'](source, True, query_block=128)
    assert source.count('kWidth = 4') == 8
    source = source.replace('kWidth = 4', 'kWidth = 8')
    if a.wide_p:
        old = '#shared2 = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 16, order = [0, 1]}>'
        new = '#shared2 = #ttg.swizzled_shared<{vec = 16, perPhase = 1, maxPhase = 8, order = [0, 1]}>'
        assert source.count(old) == 1
        source = source.replace(old, new)
    if a.wide_v:
        old = '#shared3 = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1]}>'
        new = '#shared3 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>'
        assert source.count(old) == 1
        source = source.replace(old, new)
    helper = runpy.run_path(str(Path(__file__).with_name('build-q128-backend.py')))
    source = helper['preserve_scaled_rounding'](source)
    compile_packed(root, a.output, source, dict(
        pv_operand_width=8, wide_p=a.wide_p, wide_v=a.wide_v, direct_p=a.direct_p))


def compile_packed(root, output, source, experiment):
    record = json.loads((root / 'dense-attention-aot-build.json').read_text())
    output.mkdir()
    ir = output / 'dense-attention-packed-pv-swizzle.ttgir'
    ir.write_text(source)
    os.environ['AMDGCN_USE_BUFFER_OPS'] = '1'
    configure_hygon_compiler()
    compiled = triton.compile(str(ir), target=GPUTarget('hip', 'gfx936', 64), options=dict(
        num_warps=4, num_stages=1, matrix_instr_nonkdim=0, kpack=2))
    asm = compiled.asm['amdgcn']
    isa = dict(Counter(re.findall(r'^[ \t]*(ds_\w+|v_mmac_\w+|global_\w+|s_barrier|s_waitcnt)\b', asm, re.M)))
    old_asm = (root / 'dense-attention-packed.amdgcn').read_text()
    old_dot_count = len(re.findall(r'^\s*v_mmac_f32_16x16x16_bf16\b', old_asm, re.M))
    assert isa.get('v_mmac_f32_16x16x16_bf16') == old_dot_count > 0, isa
    assert compiled.metadata.profile_scratch_size == 0
    abi = re.search(r'define amdgpu_kernel void @dense_attention_aot_kernel\(([^\n]*)\)', compiled.asm['llir'])
    assert abi and re.findall(r'(ptr addrspace\(1\)|i32|float) inreg', abi[1]) == (
        ['ptr addrspace(1)'] * 3 + ['float', 'ptr addrspace(1)'] + ['i32'] * 8
        + ['ptr addrspace(1)'] * 2)
    for name in ('dense-attention-aot.so', 'dense-attention-aot.hsaco', 'dense-attention-fallback.hsaco'):
        shutil.copy2(root / name, output / name)
    image = compiled.asm['hsaco']
    (output / 'dense-attention-packed.hsaco').write_bytes(image)
    (output / 'dense-attention-packed.amdgcn').write_text(asm)
    (output / 'dense-attention-packed.llir').write_text(compiled.asm['llir'])
    record['packed_strides'].update(hsaco_sha256=hashlib.sha256(image).hexdigest(), shared=compiled.metadata.shared)
    record['experiment'] = dict(experiment, isa=isa,
        resources={k: int(v) for k, v in re.findall(
            r'^[ \t]*\.(vgpr_count|sgpr_count|private_segment_fixed_size):[ \t]*(\d+)', asm, re.M)},
        reference=str(root), rounding_boundary=True)
    (output / 'dense-attention-aot-build.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record['experiment']), flush=True)


if __name__ == '__main__':
    main()
