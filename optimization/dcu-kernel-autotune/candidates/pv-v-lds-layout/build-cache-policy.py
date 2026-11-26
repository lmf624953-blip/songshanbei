#!/usr/bin/env python3
"""仅改变既有九齿 Q128 的 K/V 加载缓存属性，保持计算与动态长度。"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import triton
from triton.backends.compiler import GPUTarget
from songshan_h3.compiler_compat import configure_hygon_compiler


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('reference',type=Path)
    cli.add_argument('output',type=Path)
    args=cli.parse_args()
    root=args.reference.resolve(strict=True)
    record=json.loads((root/'dense-attention-aot-build.json').read_text())
    assert record['query_block']==128 and record['experiment']['compact_args']=='common-length-fixed-strides'
    assert hashlib.sha256((root/'dense-attention-aot.hsaco').read_bytes()).hexdigest()==record['hsaco_sha256']
    text=(root/'pv-swizzle.ttgir').read_text()
    assert len([l for l in text.splitlines() if ' = tt.load ' in l])==5
    args.output.mkdir();shutil.copytree(root,args.output/'v0')
    os.environ['AMDGCN_USE_BUFFER_OPS']='1';configure_hygon_compiler()
    for group,policy in [('k','cg'),('v','cg'),('kv','cg'),('kv','cv'),('kv','ca')]:
        lines=text.splitlines();count=0
        for i,line in enumerate(lines):
            if ' = tt.load ' not in line:continue
            is_k='tensor<128x32x!tt.ptr<bf16>, #blocked1>' in line
            is_v='tensor<32x128x!tt.ptr<bf16>, #h3_v_producer>' in line
            if (is_k and 'k' in group) or (is_v and 'v' in group):
                assert line.count(' : tensor<')==1 and 'cacheModifier' not in line
                lines[i]=line.replace(' : tensor<',f' cacheModifier = {policy} : tensor<',1);count+=1
        assert count==(4 if group=='kv' else 2)
        out=args.output/f'v-{group}-{policy}';out.mkdir()
        source=out/'pv-swizzle.ttgir';source.write_text('\n'.join(lines)+'\n')
        compiled=triton.compile(str(source),target=GPUTarget('hip','gfx936',64),options=dict(num_warps=4,num_stages=1,matrix_instr_nonkdim=0,kpack=2))
        asm=compiled.asm['amdgcn'];image=compiled.asm['hsaco']
        assert compiled.metadata.profile_scratch_size==0
        match=re.search(r'define amdgpu_kernel void @dense_attention_aot_kernel\(([^\n]*)\)',compiled.asm['llir'])
        assert match and re.findall(r'(ptr addrspace\(1\)|i32|float) inreg',match[1])==['ptr addrspace(1)']*3+['float','ptr addrspace(1)']+['i32']*8+['ptr addrspace(1)']*2
        for name in ('dense-attention-aot.so','dense-attention-fallback.hsaco'):shutil.copy2(root/name,out/name)
        (out/'dense-attention-aot.hsaco').write_bytes(image);(out/'dense-attention-aot.amdgcn').write_text(asm)
        (out/'dense-attention-aot.llir').write_text(compiled.asm['llir'])
        meta=json.loads(json.dumps(record));meta.update(source=str(source),hsaco_sha256=hashlib.sha256(image).hexdigest(),shared=compiled.metadata.shared)
        isa=dict(Counter(re.findall(r'^\s*(ds_\w+|v_mmac_\w+|global_\w+|s_barrier|s_waitcnt)\b',asm,re.M)))
        resources={k:int(v) for k,v in re.findall(r'^\s*\.(vgpr_count|sgpr_count|private_segment_fixed_size):\s*(\d+)',asm,re.M)}
        meta['experiment'].update(cache_group=group,cache_policy=policy,isa=isa,resources=resources)
        (out/'dense-attention-aot-build.json').write_text(json.dumps(meta,indent=2)+'\n')
        flags=Counter(' '.join(l.split()[-2:]) for l in asm.splitlines() if 'global_load_' in l)
        print(json.dumps(dict(candidate=out.name,resources=resources,load_flags=dict(flags))),flush=True)


if __name__=='__main__':main()
