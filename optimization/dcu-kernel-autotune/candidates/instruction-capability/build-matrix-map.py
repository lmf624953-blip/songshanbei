#!/usr/bin/env python3
"""构造独立LDS地址标签诊断，复用九齿AOT描述符，不作为模型镜像。"""

import argparse
import hashlib
import json
from pathlib import Path
import re

from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget


NAME="h3_matrix_load_map_probe"
PATTERNS={"zero":[],"stride16":["\tv_lshlrev_b32_e32 v8, 4, v8"],
          "stride512":["\tv_lshlrev_b32_e32 v8, 9, v8"],
          "stride512-plus64":["\tv_lshlrev_b32_e32 v8, 9, v8","\tv_add_u32_e32 v8, 64, v8"],
          "stride512-xor256":["\tv_lshlrev_b32_e32 v8, 9, v8","\tv_xor_b32_e32 v8, 256, v8"]}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference",type=Path)
    cli.add_argument("output",type=Path)
    args=cli.parse_args()
    source=(args.reference/"dense-attention-packed.amdgcn").read_text()
    assert hashlib.sha256((args.reference/"dense-attention-packed.hsaco").read_bytes()).hexdigest()=="a00cc85245e936216531503c3c1dfb57293d00d907dd517b4b1b6c8932f8a3cd"
    desc=re.search(r"\t\.amdhsa_kernel .*?\t\.end_amdhsa_kernel",source,re.S)[0]
    meta=re.search(r"\t\.amdgpu_metadata.*?\t\.end_amdgpu_metadata",source,re.S)[0]
    desc=desc.replace(".amdhsa_next_free_vgpr 219",".amdhsa_next_free_vgpr 32")
    meta=meta.replace(".vgpr_count:     219",".vgpr_count:     32")
    configure_hygon_compiler();backend=HIPBackend(GPUTarget("hip","gfx936",64))
    options=backend.parse_options(dict(num_warps=4,num_stages=1,kpack=2))
    args.output.mkdir();variants=[]
    for opcode in ("ds_read_b128","ds_read_m32x16_b16","ds_read_m32x16_b16_alt"):
        for pattern,operations in PATTERNS.items():
            if opcode=="ds_read_b128" and pattern not in ("zero","stride16","stride512"):
                continue
            lines=['\t.amdgcn_target "amdgcn-amd-amdhsa--gfx936"','\t.text',f'\t.globl {NAME}',
                '\t.p2align 8',f'\t.type {NAME},@function',f'{NAME}:',
                '\tv_mov_b32_e32 v20, v0',
                '\ts_load_dwordx2 s[8:9], s[4:5], 0x0',
                '\ts_load_dwordx2 s[10:11], s[4:5], 0x20',
                '\tv_lshlrev_b32_e32 v1, 4, v20', '\ts_mov_b32 s3, 16',
                '\ts_mov_b32 m0, -1','\ts_waitcnt lgkmcnt(0)', '.Lcopy_lds_tags:',
                '\tglobal_load_dwordx4 v[4:7], v1, s[8:9]', '\ts_waitcnt vmcnt(0)',
                '\tds_write_b128 v1, v[4:7]', '\ts_waitcnt lgkmcnt(0)',
                '\tv_add_u32_e32 v1, 4096, v1', '\ts_sub_u32 s3, s3, 1',
                '\ts_cmp_lg_u32 s3, 0', '\ts_cbranch_scc1 .Lcopy_lds_tags',
                '\ts_barrier','\tv_and_b32_e32 v8, 63, v20']
            if pattern=="zero":
                lines.append('\tv_mov_b32_e32 v8, 0')
            else:
                lines+=operations
            lines += [f'\t{opcode} v[4:7], v8', '\ts_waitcnt lgkmcnt(0)',
                '\tv_lshlrev_b32_e32 v1, 4, v20', '\tglobal_store_dwordx4 v1, v[4:7], s[10:11]',
                '\ts_endpgm',f'.Lprobe_end:',f'\t.size {NAME},.Lprobe_end-{NAME}',
                '\t.section .rodata,#alloc','\t.p2align 6',desc,meta]
            text=('\n'.join(lines)+'\n').replace('dense_attention_aot_kernel',NAME)
            assert 'v_mmac' not in text
            image=backend.make_hsaco(text,{},options)
            stem=opcode+'-'+pattern
            (args.output/(stem+'.amdgcn')).write_text(text)
            (args.output/(stem+'.hsaco')).write_bytes(image)
            variants.append(dict(opcode=opcode,pattern=pattern,image=stem+'.hsaco',sha256=hashlib.sha256(image).hexdigest()))
    (args.output/'build.json').write_text(json.dumps(dict(scope='独立内存布局诊断，未测模型/性能',
        symbol=NAME,threads=256,shared_bytes=65536,input_bytes=65536,output_words=1024,variants=variants),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
