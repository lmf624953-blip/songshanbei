#!/usr/bin/env python3
"""把九齿 16 个按 wave 掩码归约合成一次二维归约，保留 16 项顺序相加。"""
import argparse,hashlib,json,os,re,sys
from pathlib import Path
import triton
from triton.backends.compiler import GPUTarget
from songshan_h3.compiler_compat import configure_hygon_compiler


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('reference',type=Path);cli.add_argument('output',type=Path);cli.add_argument('--warps',type=int,choices=(1,2,4,8),default=4);cli.add_argument('--linear-spt',type=int,choices=(1,8),default=1);a=cli.parse_args();a.output.mkdir(parents=True)
    source=(a.reference/'wide-norm-indexed.ttir').read_text()
    definitions={m.group(1):m.group(2) for m in re.finditer(r'^    (%[\w]+) = (.*)$',source,re.M)}
    pattern=r'^    (%[\w]+) = "tt.reduce"\((%[\w]+)\) <\{axis = 0 : i32\}> \(\{.*?^    \}\) : \(tensor<8192xf32>\) -> f32[^\n]*'
    groups=list(re.finditer(pattern,source,re.M|re.S));assert len(groups)==16,len(groups)
    common,bound,zero=None,None,None
    for group in groups:
        first=re.match(r'arith.select (%\w+), (%\w+), (%\w+)',definitions[group[2]]);assert first
        second=re.match(r'arith.select (%\w+), (%\w+), (%\w+)',definitions[first[2]]);assert second
        if common is None:common,bound,zero=second[2],first[1],first[3]
        assert (second[2],first[1],first[3],second[3])==(common,bound,zero,zero)
    prefix=f'''    %h3_partial = arith.select {bound}, {common}, {zero} : tensor<8192xi1>, tensor<8192xf32>
    %h3_matrix = tt.reshape %h3_partial : tensor<8192xf32> -> tensor<128x64xf32>
    %h3_groups = "tt.reduce"(%h3_matrix) <{{axis = 1 : i32}}> ({{
    ^bb0(%h3_a: f32, %h3_b: f32):
      %h3_c = arith.addf %h3_a, %h3_b : f32
      tt.reduce.return %h3_c : f32
    }}) : (tensor<128x64xf32>) -> tensor<128xf32>
'''
    def replace(match):
        index=next(i for i,g in enumerate(groups) if g.start()==match.start())
        result=f'''    %h3_index_{index} = arith.constant dense<{index}> : tensor<1xi32>
    %h3_selected_{index} = tt.gather %h3_groups[%h3_index_{index}] {{axis = 0 : i32}} : (tensor<128xf32>, tensor<1xi32>) -> tensor<1xf32>
    {match[1]} = "tt.reduce"(%h3_selected_{index}) <{{axis = 0 : i32}}> ({{
    ^bb0(%h3_sa_{index}: f32, %h3_sb_{index}: f32):
      %h3_sc_{index} = arith.addf %h3_sa_{index}, %h3_sb_{index} : f32
      tt.reduce.return %h3_sc_{index} : f32
    }}) : (tensor<1xf32>) -> f32'''
        return (prefix if index==0 else '')+result
    text=re.sub(pattern,replace,source,flags=re.M|re.S);ir=a.output/'grouped.ttir';ir.write_text(text)
    configure_hygon_compiler();options=dict(num_warps=a.warps,num_stages=1,matrix_instr_nonkdim=0,kpack=1)
    first=triton.compile(str(ir),target=GPUTarget('hip','gfx936',64),options=options)
    text=first.asm['ttgir'];(a.output/'grouped-initial.ttgir').write_text(text)
    # 先保存编译器初始布局，再强制 wave 内 64 个元素分给 64 lanes。
    layouts=re.findall(r'(#\w+) = #ttg.blocked<\{sizePerThread = \[([^]]+)\], threadsPerWarp = \[([^]]+)\], warpsPerCTA = \[([^]]+)\], order = \[([^]]+)\]\}>',text)
    for name,spt,tpw,wpc,order in layouts:
        if len(spt.split(','))==2:
            previous=f'{name} = #ttg.blocked<{{sizePerThread = [{spt}], threadsPerWarp = [{tpw}], warpsPerCTA = [{wpc}], order = [{order}]}}>'
            text=text.replace(previous,f'{name} = #ttg.blocked<{{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [{a.warps}, 1], order = [1, 0]}}>')
        elif len(spt.split(','))==1:
            previous=f'{name} = #ttg.blocked<{{sizePerThread = [{spt}], threadsPerWarp = [{tpw}], warpsPerCTA = [{wpc}], order = [{order}]}}>'
            text=text.replace(previous,f'{name} = #ttg.blocked<{{sizePerThread = [{a.linear_spt}], threadsPerWarp = [64], warpsPerCTA = [{a.warps}], order = [0]}}>')
    if a.linear_spt==8:
        # reshape 的编码须与源线性布局一致，显式转换后才使用 wave 内 64-lane 布局。
        text=f"#h3_linear = #ttg.blocked<{{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [{a.warps}], order = [0]}}>\n"+text
        pattern=r"^(    )(%\w+) = tt.reshape (%\w+) : tensor<8192xf32, (#\w+)> -> tensor<128x64xf32, (#\w+)>"
        def layout_bridge(match):
            indent,result,operand,encoding,parent=match.groups()
            return f"{indent}%h3_reshape_input = ttg.convert_layout {operand} : tensor<8192xf32, {encoding}> -> tensor<8192xf32, #h3_linear>\n{indent}{result} = tt.reshape %h3_reshape_input : tensor<8192xf32, #h3_linear> -> tensor<128x64xf32, {parent}>"
        text,count=re.subn(pattern,layout_bridge,text,flags=re.M);assert count==1
    ir=a.output/'wide-norm-indexed.ttgir';ir.write_text(text)
    compiled=triton.compile(str(ir),target=GPUTarget('hip','gfx936',64),options=options)
    image=compiled.asm['hsaco'];(a.output/'wide-norm-indexed.hsaco').write_bytes(image);(a.output/'wide-norm-indexed.amdgcn').write_text(compiled.asm['amdgcn'])
    meta=json.loads((a.reference/'build.json').read_text());meta.update(warps=a.warps,linear_spt=a.linear_spt,hsaco_sha256=hashlib.sha256(image).hexdigest(),shared=compiled.metadata.shared,mode='original-serial-grouped',source=str(ir));(a.output/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
