#!/usr/bin/env python3
"""构建九齿V置换预排布；复用原发射ABI和每次重新预处理的语义。"""

import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
import re

import ninetoothed as nt
from v_zip_prepack_kernel import arrangement, zipped, bank_zipped


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('output',type=Path);cli.add_argument('--layout',choices=('zip','bankzip'),required=True);args=cli.parse_args()
    code=Path(os.environ['H3_AUTOTUNE_CODE_ROOT'])/'optimization';sys.path.insert(0,str(code/'adaln'))
    from aot_codegen import lower_kernel,compile_image
    args.output.mkdir()
    application=zipped if args.layout=='zip' else bank_zipped
    source,kernel,bindings=lower_kernel(args.output,partial(arrangement,tile=2048),application,(
        nt.Tensor(shape=(None,7,128),dtype=nt.bfloat16),nt.Tensor(0,dtype=nt.int32),nt.Tensor(1,dtype=nt.bfloat16)),
        'h3_prepack_value',8)
    signature,constants,runtime,arguments={},{},[],[]
    for binding in bindings:
        if binding.kind=='tensor':
            signature[binding.name]='*bf16';runtime.append('*bf16');arguments.append(dict(kind='tensor',source=binding.source))
        elif binding.kind=='scalar' and binding.source=='blocks':
            signature[binding.name]='i32';runtime.append('i32');arguments.append(dict(kind='scalar',source='blocks'))
        elif binding.kind=='shape':
            if binding.source=='value' and binding.dim in (1,2):
                signature[binding.name]='constexpr';constants[binding.name]=(7,128)[binding.dim-1]
            else:
                signature[binding.name]='i32';runtime.append('i32');arguments.append(dict(kind='shape',source=binding.source,dim=binding.dim))
        elif binding.kind=='stride':
            signature[binding.name]='constexpr';constants[binding.name]=(2688,128,1)[binding.dim] if binding.source=='value' else 1
        else:raise ValueError(binding)
    assert runtime==['*bf16','i32','*bf16','i32','i32']
    os.environ['H3_AOT_SAVE_IR']='1'
    sha,shared=compile_image(args.output/'prepack-value.hsaco',kernel,signature,constants,runtime,8,block=2048,allow_shared=True)
    rewrite=None
    if shared:
        # 1D全局gather无需跨wave通信；统一为输出layout，避免编译器插入LDS重排。
        import triton
        from triton.backends.compiler import GPUTarget
        ir=args.output/'prepack-value.ttgir';text=ir.read_text()
        store=next(line for line in reversed(text.splitlines()) if 'tt.store ' in line)
        target=re.findall(r'(#blocked\w*)',store)[-1]
        aliases=re.findall(r'^(#blocked\w*) = #ttg.blocked<.*$',text,re.M)
        assert target in aliases and len(aliases)>1,(target,aliases)
        previous=text
        text='\n'.join(line for line in text.splitlines() if not any(line.startswith(alias+' =') for alias in aliases if alias!=target))+'\n'
        for alias in sorted(aliases,key=len,reverse=True):
            if alias!=target:text=re.sub(re.escape(alias)+r'\b',target,text)
        (args.output/'prepack-value-before-layout.ttgir').write_text(previous)
        ir.write_text(text)
        compiled=triton.compile(str(ir),target=GPUTarget('hip','gfx936',64),options=dict(num_warps=8,num_stages=1,matrix_instr_nonkdim=0,kpack=1))
        assert compiled.metadata.shared==0 and compiled.metadata.profile_scratch_size==0
        image=compiled.asm['hsaco'];(args.output/'prepack-value.hsaco').write_bytes(image)
        (args.output/'prepack-value.amdgcn').write_text(compiled.asm['amdgcn'])
        sha=hashlib.sha256(image).hexdigest();shared=0
        rewrite=dict(single_layout=target,removed_layouts=[a for a in aliases if a!=target])
    record=dict(source=str(source),hsaco_sha256=sha,shared=shared,warps=8,tile=2048,arguments=arguments,layout=args.layout,layout_rewrite=rewrite)
    (args.output/'prepack-value-build.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
