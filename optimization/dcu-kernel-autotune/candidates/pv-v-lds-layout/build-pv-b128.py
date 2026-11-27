#!/usr/bin/env python3
"""用可逆V布局置换把8条read2st64_b64换成read_b128，MMAC及尾部保留。"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget


def permute(value,bankzip):
    value=((value&~1032)|((value&8)<<7)|((value&1024)>>7))
    if bankzip:value=(value&~160)|((value&32)<<2)|((value&128)>>2)
    return value


def proof(bankzip):
    mapping=[permute(i,bankzip) for i in range(8192)]
    assert sorted(mapping)==list(range(8192))
    assert all(mapping[mapping[i]]==i for i in range(8192))
    checks=0
    for tid in range(256):
        old=(((tid&15)<<6)|((tid&14)<<2))^((tid&48)>>1)
        for base in (old,old^32):
            new=permute(base,bankzip)
            assert new%16==0
            for offset in (8192,10240,12288,14336):
                a,b=base+offset,base+offset+1024
                expected=list(range(a,a+8))+list(range(b,b+8))
                actual=[8192+mapping[i-8192] for i in range(new+offset,new+offset+16)]
                assert actual==expected,(tid,base,offset)
                checks+=16
    return dict(bijective_bytes=8192,inverse_is_self=True,matched_read_bytes=checks,threads=256)


def transform(source,bankzip):
    # 逐 lane 地址证明依赖这一版基地址公式；上游汇编变化时必须重新证明。
    expected = (
        '\tv_and_b32_e32 v111, 15, v0', '\tv_and_b32_e32 v112, 48, v0',
        '\tv_lshrrev_b32_e32 v106, 1, v112', '\tv_and_b32_e32 v97, 14, v0',
        '\tv_lshlrev_b32_e32 v14, 2, v97', '\tv_lshl_or_b32 v14, v111, 6, v14',
        '\tv_xor_b32_e32 v135, v14, v106', '\tv_xor_b32_e32 v14, 32, v135',
        '\tv_add_u32_e32 v149, 0, v14',
    )
    assert all(line in source for line in expected), 'V 基地址公式变化，需要重新验证布局证明'
    lines=source.splitlines();entry=lines.index('.LBB0_14:');end=lines.index('.LBB0_20:')
    # 完整循环两组V基地址在0..1023且8字节对齐，已在逐lane证明中覆盖。
    init=['\tv_and_b32_e32 v206, 8, v135','\tv_and_b32_e32 v135, 0xfffffff7, v135',
          '\tv_lshlrev_b32_e32 v206, 7, v206','\tv_or_b32_e32 v135, v135, v206']
    if bankzip:
        init += ['\tv_lshrrev_b32_e32 v206, 2, v135','\tv_xor_b32_e32 v206, v206, v135',
                 '\tv_and_b32_e32 v206, 32, v206','\tv_lshl_or_b32 v207, v206, 2, v206',
                 '\tv_xor_b32_e32 v135, v135, v207']
    init += [f'\tv_xor_b32_e32 v149, {128 if bankzip else 32}, v135']
    changes=0
    for i in range(entry,end):
        m=re.fullmatch(r'\tds_read2st64_b64 (v\[\d+:\d+\]), (v\d+) offset0:(\d+) offset1:(\d+)',lines[i])
        if not m:continue
        assert m[2] in ('v176','v149') and int(m[4])==int(m[3])+2
        lines[i]=f'\tds_read_b128 {m[1]}, {m[2]} offset:{int(m[3])*512}'
        changes+=1
    assert changes==8
    lines[entry:entry]=init
    result='\n'.join(lines)+'\n'
    assert result[result.index('.LBB0_20:'):]==source[source.index('.LBB0_20:'):]
    assert result.count('v_mmac_f32_16x16x16_bf16')==source.count('v_mmac_f32_16x16x16_bf16')
    return result


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('reference',type=Path);cli.add_argument('prepack',type=Path);cli.add_argument('output',type=Path);cli.add_argument('--layout',choices=('zip','bankzip'),required=True);args=cli.parse_args()
    meta=json.loads((args.reference/'dense-attention-aot-build.json').read_text());pmeta=json.loads((args.prepack/'prepack-value-build.json').read_text())
    assert pmeta['layout']==args.layout
    digest=hashlib.sha256((args.reference/'dense-attention-packed.hsaco').read_bytes()).hexdigest();assert digest==meta['packed_strides']['hsaco_sha256']
    sha=hashlib.sha256((args.prepack/'prepack-value.hsaco').read_bytes()).hexdigest();assert sha==pmeta['hsaco_sha256']
    address_proof=proof(args.layout=='bankzip')
    text=transform((args.reference/'dense-attention-packed.amdgcn').read_text(),args.layout=='bankzip')
    configure_hygon_compiler();backend=HIPBackend(GPUTarget('hip','gfx936',64))
    image=backend.make_hsaco(text,{},backend.parse_options(dict(num_warps=4,num_stages=1,kpack=2)))
    shutil.copytree(args.reference,args.output);(args.output/'dense-attention-packed.hsaco').write_bytes(image);(args.output/'dense-attention-packed.amdgcn').write_text(text)
    shutil.copy2(args.prepack/'prepack-value.hsaco',args.output/'prepack-value.hsaco')
    meta['packed_strides']['hsaco_sha256']=hashlib.sha256(image).hexdigest();meta['prepacked_value_dispatch']['prepack']=pmeta
    meta['experiment']['pv_b128']=dict(layout=args.layout,reference_packed_sha256=digest,replaced_reads=8,
        address_proof=address_proof,scope='仅完整块V预排布和LDS地址；P/QK/softmax/MMAC/原始V尾部不变',numerical_verification='尚未完成')
    (args.output/'dense-attention-aot-build.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
