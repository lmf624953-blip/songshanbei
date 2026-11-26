#!/usr/bin/env python3
"""仅用于微测：由九齿重新生成 Q256，检查更大 Q 复用与 4/8 warps。"""
import argparse
from collections import Counter
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

import ninetoothed as nt
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from compact_args import apply as compact_args
from producer_layout import apply as apply_v
from q_resources_v2 import apply as apply_q


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('reference',type=Path)
    cli.add_argument('output',type=Path)
    args=cli.parse_args()
    code=Path(os.environ['H3_AUTOTUNE_CODE_ROOT'])/'optimization/adaln'
    sys.path.insert(0,str(code))
    from aot_codegen import compile_host,lower_kernel
    from attention_layout import arrangement
    from tail_attention import application
    from songshan_h3.attention_kernel import definition
    record=json.loads((args.reference/'dense-attention-aot-build.json').read_text())
    assert record['query_block']==128 and record['experiment']['compact_args']=='common-length-fixed-strides'
    args.output.mkdir()
    raw=args.output/'source-q256';raw.mkdir()
    _,_,tensors=definition(head_dim=128,dtype=nt.bfloat16,causal=False)
    for tensor in (*tensors[:3],tensors[-1]):tensor.shape=(1,7,tensor.shape[2],128)
    source,kernel,bindings=lower_kernel(raw,partial(arrangement,query_block=256,key_block=32),
        application,tensors,'dense_attention_aot',warps=4,hoist_query=True,query_loads=2)
    signature,constants,attrs={},{},{}
    dynamic=[]
    for b in bindings:
        if b.kind=='tensor':signature[b.name]='*bf16'
        elif b.kind=='scalar' and b.source=='scale':signature[b.name]='fp32'
        elif b.dim==2 and b.kind in ('shape','stride'):
            signature[b.name]='i32';dynamic.append((b.source,b.kind))
        else:
            signature[b.name]='constexpr'
            if b.kind=='stride':constants[b.name]=(0,128,None,1)[b.dim]
            elif b.kind=='constexpr' and b.source=='is_causal':constants[b.name]=False
            else:raise RuntimeError(f'不支持的绑定 {b}')
    assert dynamic==[(t,k) for t in ('query','key','value','output') for k in ('shape','stride')]
    signature['BLOCK']='constexpr';constants['BLOCK']=1
    aligned={b.name for b in bindings if b.kind=='stride' and b.dim==2}
    for i,name in enumerate(signature):
        if signature[name].startswith('*') or name in aligned:attrs[(i,)]=[['tt.divisibility',16]]
        if signature[name].startswith('*'):attrs[(i,)].append(['tt.pointer_range',32])
    os.environ['AMDGCN_USE_BUFFER_OPS']='1'
    target=GPUTarget('hip','gfx936',64)
    options=dict(num_warps=4,num_stages=1,matrix_instr_nonkdim=0,kpack=2)
    # 显式 target 编译，不载入镜像，不调用 GPU。原始镜像仅供取得九齿生成 TTGIR。
    first=triton.compile(ASTSource(kernel,signature,constexprs=constants,attrs=attrs),target=target,options=options)
    text=first.asm['ttgir'];(raw/'initial.ttgir').write_text(text)
    for label,shape,config in [('#shared2','256x32',(8,2,16,[0,1])),('#shared3','32x128',(8,1,16,[1,0]))]:
        v,p,m,order=config
        original=f'{label} = #ttg.swizzled_shared<{{vec = 1, perPhase = 1, maxPhase = 1, order = {order}}}>'
        assert text.count(original)==1 and f'!ttg.memdesc<{shape}xbf16, {label}, #smem>' in text
        text=text.replace(original,f'{label} = #ttg.swizzled_shared<{{vec = {v}, perPhase = {p}, maxPhase = {m}, order = {order}}}>')
    old='#shared3 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>'
    assert text.count(old)==1
    text=apply_v(text.replace(old,'#shared3 = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1]}>'),'k4-n4')
    text=compact_args(apply_q(text,'direct',query_block=256),True)
    shutil.copytree(args.reference,args.output/'v0')
    for num_warps,phases in [(4,16),(4,32),(8,16),(8,32)]:
        out=args.output/f'v-q256-w{num_warps}-p{phases}';out.mkdir()
        candidate=text.replace('#shared2 = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 16, order = [0, 1]}>',f'#shared2 = #ttg.swizzled_shared<{{vec = 8, perPhase = 2, maxPhase = {phases}, order = [0, 1]}}>')
        if num_warps==8:
            assert candidate.count('"ttg.num-warps" = 4 : i32')==1
            candidate=candidate.replace('"ttg.num-warps" = 4 : i32','"ttg.num-warps" = 8 : i32')
            candidate=candidate.replace('warpsPerCTA = [4, 1]','warpsPerCTA = [8, 1]').replace('warpsPerCTA = [1, 4]','warpsPerCTA = [1, 8]')
        ir=out/'pv-swizzle.ttgir';ir.write_text(candidate)
        compiled=triton.compile(str(ir),target=target,options={**options,'num_warps':num_warps})
        asm=compiled.asm['amdgcn'];image=compiled.asm['hsaco']
        match=re.search(r'define amdgpu_kernel void @dense_attention_aot_kernel\(([^\n]*)\)',compiled.asm['llir'])
        assert match and re.findall(r'(ptr addrspace\(1\)|i32|float) inreg',match[1])==['ptr addrspace(1)']*3+['float','ptr addrspace(1)']+['i32']*8+['ptr addrspace(1)']*2
        resources={k:int(v) for k,v in re.findall(r'^\s*\.(vgpr_count|sgpr_count|private_segment_fixed_size):\s*(\d+)',asm,re.M)}
        (out/'dense-attention-aot.hsaco').write_bytes(image)
        (out/'dense-attention-aot.amdgcn').write_text(asm)
        (out/'dense-attention-aot.llir').write_text(compiled.asm['llir'])
        # 此候选严格限定微测的同长度、packed stride 和 <2GiB 缓冲，不作为正式配置。
        shutil.copy2(out/'dense-attention-aot.hsaco',out/'dense-attention-fallback.hsaco')
        command=compile_host(out/'dense-attention-aot.so',('attention_buffer_host.cc',),defines=(('H3_ATTENTION_QUERY_BLOCK',256),))
        meta=json.loads(json.dumps(record));meta.update(query_block=256,warps=num_warps,shared=compiled.metadata.shared,
            hsaco_sha256=hashlib.sha256(image).hexdigest(),source=str(ir),elide_full_kv_load_mask=False,host_command=command,
            fallback=dict(hsaco_sha256=hashlib.sha256(image).hexdigest(),shared=compiled.metadata.shared))
        isa=dict(Counter(re.findall(r'^\s*(ds_\w+|v_mmac_\w+|global_\w+|s_barrier|s_waitcnt)\b',asm,re.M)))
        meta['codegen_passes']=[x for x in meta['codegen_passes'] if x!='elide_full_kv_load_mask']
        meta['pv_lds_swizzle']['P']['maxPhase']=phases
        meta['generated_source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
        meta['experiment'].update(backend_variant=f'q256-w{num_warps}-p{phases}',isa=isa,resources=resources,
            scope='固定七 head 和 packed stride 的微测；fallback 同镜像，不可用于正式推理')
        (out/'dense-attention-aot-build.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(candidate=out.name,resources=resources,shared=compiled.metadata.shared)),flush=True)


if __name__=='__main__':main()
