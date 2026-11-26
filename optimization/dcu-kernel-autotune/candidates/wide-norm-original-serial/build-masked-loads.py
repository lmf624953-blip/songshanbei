#!/usr/bin/env python3
"""在九齿宽归约 IR 中屏蔽最终必被置零的重复加载，数学与舍入顺序不变。"""
import argparse,hashlib,json,re
from pathlib import Path
import triton
from triton.backends.compiler import GPUTarget
from songshan_h3.compiler_compat import configure_hygon_compiler

def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('reference',type=Path);cli.add_argument('output',type=Path);a=cli.parse_args();a.output.mkdir()
    meta=json.loads((a.reference/'build.json').read_text());assert meta['mode'] in ['original-serial','original-serial-grouped'];assert meta.get('linear_spt',1)==1
    source=(a.reference/'wide-norm-indexed.ttgir').read_text()
    assert '#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64]' in source
    predicate=re.search(r'(%\w+) = arith.cmpi slt, %offsets, (%\w+) : tensor<8192xi32, #blocked>',source)
    constant=re.search(r'(%\w+) = arith.constant dense<1024> : tensor<8192xi32, #blocked>',source);assert constant
    predicate=re.search(r'(%\w+) = arith.cmpi slt, %offsets, '+re.escape(constant[1])+r' : tensor<8192xi32, #blocked>',source);assert predicate
    loads=list(re.finditer(r'^(    )(%\w+) = tt.load (%\w+), (%\w+), (%\w+) : tensor<8192x!tt.ptr<bf16>, #blocked>[^\n]*',source,re.M));assert len(loads)==10,len(loads)
    for index,load in reversed(list(enumerate(loads[:6]))):
        indent,result,pointer,mask,other=load.groups();mask_name=f'%h3_reduction_load_mask_{index}'
        prefix=f'{indent}{mask_name} = arith.andi {mask}, {predicate[1]} : tensor<8192xi1, #blocked>\n'
        if index==5:
            prefix+=f'{indent}%h3_column256 = arith.constant dense<256> : tensor<8192xi32, #blocked>\n{indent}%h3_valid_tail = arith.cmpi slt, %offsets, %h3_column256 : tensor<8192xi32, #blocked>\n{indent}%h3_tail_mask = arith.andi {mask_name}, %h3_valid_tail : tensor<8192xi1, #blocked>\n'
            mask_name='%h3_tail_mask'
        text=load[0].replace(f', {mask},',f', {mask_name},',1)
        source=source[:load.start()]+prefix+text+source[load.end():]
    ir=a.output/'wide-norm-indexed.ttgir';ir.write_text(source);configure_hygon_compiler()
    compiled=triton.compile(str(ir),target=GPUTarget('hip','gfx936',64),options=dict(num_warps=meta['warps'],num_stages=1,matrix_instr_nonkdim=0,kpack=1));image=compiled.asm['hsaco']
    (a.output/'wide-norm-indexed.hsaco').write_bytes(image);(a.output/'wide-norm-indexed.amdgcn').write_text(compiled.asm['amdgcn'])
    meta.update(hsaco_sha256=hashlib.sha256(image).hexdigest(),shared=compiled.metadata.shared,source=str(ir),masked_reduction_loads=True,reference=str(a.reference))
    (a.output/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
