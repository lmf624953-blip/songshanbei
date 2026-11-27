#!/usr/bin/env python3
"""针对当前P行布局验证wave所有权，只缩小P写后读前的同步范围。"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil

from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget


def proof():
    layout=runpy.run_path(str(Path(__file__).with_name('build-p-row-lds.py')))
    layout['verify_mapping']()
    owners={}
    for tid in range(256):
        for item in range(16):
            base=layout['store_base'](tid)
            if item%2:base^=0x110
            if item>=8:base^=0x200
            offsets=(32,32,0,0,4128,4128,4096,4096) if item>=8 else (0,0,32,32,4096,4096,4128,4128)
            address=base+offsets[item%8]
            assert address not in owners
            owners[address]=tid//64
    reads=0
    for tid in range(256):
        base=layout['read_base'](tid)
        for address in (base,base^32,base+4096,(base^32)+4096):
            for offset in range(0,8,2):
                assert owners[address+offset]==tid//64
                reads+=1
    assert set(owners)==set(range(0,8192,2))
    return dict(unique_bf16_stores=len(owners),bf16_reads=reads,all_reads_same_wave_as_writer=True,
                p_region_bytes=[0,8192],value_region_bytes=[8192,16384])


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('reference',type=Path);cli.add_argument('output',type=Path)
    cli.add_argument('--compact-wait',action='store_true');args=cli.parse_args()
    meta=json.loads((args.reference/'dense-attention-aot-build.json').read_text());assert meta['experiment']['prepacked_v']['p_row_layout']
    source=(args.reference/'dense-attention-packed.amdgcn').read_text();lines=source.splitlines()
    digest=hashlib.sha256((args.reference/'dense-attention-packed.hsaco').read_bytes()).hexdigest();assert digest==meta['packed_strides']['hsaco_sha256']
    entry,tail=lines.index('.LBB0_14:'),lines.index('.LBB0_20:')
    assert '\tv_add_u32_e32 v142, 8192, v142' in lines[:entry]
    first=lines.index('\tds_write_b16_d16_hi v132, v122',entry)
    read=lines.index('\tds_read_b64 v[122:123], v133',first)
    triple=['\ts_waitcnt vmcnt(0) lgkmcnt(0)','\ts_barrier','\ts_waitcnt lgkmcnt(0)']
    assert lines[read-3:read]==triple
    prior=max(i for i in range(entry,first) if lines[i]=='\ts_barrier')
    assert lines[prior-1]=='\ts_waitcnt vmcnt(0) lgkmcnt(0)'
    assert any('global_load_dwordx4' in line and line.endswith(' lds') for line in lines[entry:prior])
    # V在未动的前一workgroup barrier已经发布，且P写入不覆盖V。
    assert not any(line.startswith('\tglobal_') for line in lines[prior+1:read])
    address_proof=proof()
    if args.compact_wait:lines[read-3:read]=['\ts_waitcnt lgkmcnt(0)']
    else:del lines[read-2]
    text='\n'.join(lines)+'\n'
    assert text[text.index('.LBB0_20:'):]==source[source.index('.LBB0_20:'):]
    assert text.count('\ts_barrier')==source.count('\ts_barrier')-1
    configure_hygon_compiler();backend=HIPBackend(GPUTarget('hip','gfx936',64))
    image=backend.make_hsaco(text,{},backend.parse_options(dict(num_warps=4,num_stages=1,kpack=2)))
    shutil.copytree(args.reference,args.output);(args.output/'dense-attention-packed.hsaco').write_bytes(image);(args.output/'dense-attention-packed.amdgcn').write_text(text)
    meta['packed_strides']['hsaco_sha256']=hashlib.sha256(image).hexdigest()
    meta['experiment']['p_wave_publication']=dict(reference_packed_sha256=digest,compact_wait=args.compact_wait,address_proof=address_proof,
       scope='当前P行布局只改完整块P写后读前同步；K发布/K读后覆写/V发布/尾部屏障均保留',numerical_verification='尚未完成')
    (args.output/'dense-attention-aot-build.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
