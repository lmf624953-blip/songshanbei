#!/usr/bin/env python3
"""先共享分组归约结果，再执行原有16项顺序求和，减少重复跨wave传递。"""
import argparse,hashlib,json,re
from pathlib import Path
import triton
from triton.backends.compiler import GPUTarget
from songshan_h3.compiler_compat import configure_hygon_compiler


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('reference',type=Path);cli.add_argument('output',type=Path);cli.add_argument('--compact',action='store_true');a=cli.parse_args();a.output.mkdir()
    meta=json.loads((a.reference/'build.json').read_text());assert meta['mode']=='original-serial-grouped' and meta['warps']==4 and meta.get('linear_spt',1)==1
    source=(a.reference/'wide-norm-indexed.ttgir').read_text()
    pattern=r'^(    )(%\w+) = tt.gather (%\w+)\[(%\w+)\] \{axis = 0 : i32\} : \(tensor<128xf32, (#ttg.slice<\{dim = 1, parent = #\w+\}>)>, tensor<1xi32, (#\w+)>\) -> tensor<1xf32, (#\w+)>[^\n]*'
    matches=list(re.finditer(pattern,source,re.M));assert len(matches)==16,len(matches)
    group=matches[0][3];layout=matches[0][5];index_layout=matches[0][6]
    assert all((m[3],m[5],m[6],m[7])==(group,layout,index_layout,index_layout) for m in matches)
    if a.compact:
        prefix=f'''    %h3_group_range = tt.make_range {{end = 16 : i32, start = 0 : i32}} : tensor<16xi32, {index_layout}>
    %h3_group_broadcast = tt.gather {group}[%h3_group_range] {{axis = 0 : i32}} : (tensor<128xf32, {layout}>, tensor<16xi32, {index_layout}>) -> tensor<16xf32, {index_layout}>
'''
        group_type=f'tensor<16xf32, {index_layout}>'
    else:
        prefix=f'    %h3_group_broadcast = ttg.convert_layout {group} : tensor<128xf32, {layout}> -> tensor<128xf32, {index_layout}>\n'
        group_type=f'tensor<128xf32, {index_layout}>'
    for i,m in reversed(list(enumerate(matches))):
        text=m[0].replace(f'tt.gather {group}[','tt.gather %h3_group_broadcast[').replace(f'tensor<128xf32, {layout}>',group_type)
        source=source[:m.start()]+(prefix if i==0 else '')+text+source[m.end():]
    ir=a.output/'wide-norm-indexed.ttgir';ir.write_text(source);configure_hygon_compiler()
    compiled=triton.compile(str(ir),target=GPUTarget('hip','gfx936',64),options=dict(num_warps=4,num_stages=1,matrix_instr_nonkdim=0,kpack=1));image=compiled.asm['hsaco']
    (a.output/'wide-norm-indexed.hsaco').write_bytes(image);(a.output/'wide-norm-indexed.amdgcn').write_text(compiled.asm['amdgcn'])
    meta.update(hsaco_sha256=hashlib.sha256(image).hexdigest(),shared=compiled.metadata.shared,source=str(ir),mode='original-serial-group-broadcast',compact_group_broadcast=a.compact,reference=str(a.reference));(a.output/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
