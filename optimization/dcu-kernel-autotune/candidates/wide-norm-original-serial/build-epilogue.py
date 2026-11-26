#!/usr/bin/env python3
"""保留原库归约布局，仅为归约后的逐元素读写使用连续向量布局。"""
import argparse,hashlib,json,re
from pathlib import Path
import triton
from triton.backends.compiler import GPUTarget
from songshan_h3.compiler_compat import configure_hygon_compiler


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('reference',type=Path);cli.add_argument('output',type=Path);cli.add_argument('--spt',type=int,choices=(2,4,8),required=True);cli.add_argument('--rematerialize-index',action='store_true');a=cli.parse_args();a.output.mkdir()
    meta=json.loads((a.reference/'build.json').read_text());assert meta['mode']=='original-serial' and meta['warps']==4
    source=(a.reference/'wide-norm-indexed.ttgir').read_text()
    header='#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>'
    assert source.count(header)==1
    marker='    %v229 = arith.divsi %offsets, %cst_2 : tensor<8192xi32, #blocked>'
    assert source.count(marker)==1
    cut=source.index(marker);before,after=source[:cut],source[cut:]
    after_body=after[:after.index('    tt.return')]
    local=set(re.findall(r'^    (%\w+) =',after_body,re.M));used=set(re.findall(r'%\w+',after_body));external=used-local
    records=[];inserts=[]
    for name in sorted(external):
        match=re.search(r'^    '+re.escape(name)+r' = ([^\n]+)',before,re.M)
        if not match:continue # scalar function arguments
        expr=match[1];types=re.findall(r'tensor<8192x(.+?), #blocked>',expr)
        if not types:continue
        if expr.startswith('"tt.reduce"'):continue
        element=types[-1]
        if expr.startswith('arith.cmpi'):element='i1'
        if expr.startswith('tt.load'):
            assert element.startswith('!tt.ptr<');element=element[len('!tt.ptr<'):-1]
        replacement='%h3_epilogue_'+name[1:]
        inserts.append(f'    {replacement} = ttg.convert_layout {name} : tensor<8192x{element}, #blocked> -> tensor<8192x{element}, #epilogue>\n')
        after=re.sub(re.escape(name)+r'(?![\w])',replacement,after)
        records.append(dict(value=name,element=element))
    assert len(records)>=8,records
    after=after.replace('#blocked','#epilogue')
    if a.rematerialize_index:
        allowed=('arith.constant','tt.make_range','arith.cmpi','tt.splat','arith.andi')
        clones=[]
        names={r['value']:'%h3_epilogue_'+r['value'][1:] for r in records}
        for line in before.splitlines(True):
            m=re.match(r'    (%\w+) = (.*)',line)
            if not m or m[1] not in names:continue
            assert m[2].startswith(allowed),m[2]
            for name,replacement in names.items():
                line=re.sub(re.escape(name)+r'(?![\w])',replacement,line)
            clones.append(line.replace('#blocked','#epilogue'))
        assert len(clones)==len(records)
        inserts=clones
    source=before+''.join(inserts)+after
    source=source.replace(header,header+'\n#epilogue = '+header.split(' = ',1)[1].replace('sizePerThread = [1]',f'sizePerThread = [{a.spt}]'))
    ir=a.output/'wide-norm-indexed.ttgir';ir.write_text(source);configure_hygon_compiler()
    compiled=triton.compile(str(ir),target=GPUTarget('hip','gfx936',64),options=dict(num_warps=4,num_stages=1,matrix_instr_nonkdim=0,kpack=1));image=compiled.asm['hsaco']
    (a.output/'wide-norm-indexed.hsaco').write_bytes(image);(a.output/'wide-norm-indexed.amdgcn').write_text(compiled.asm['amdgcn'])
    meta.update(hsaco_sha256=hashlib.sha256(image).hexdigest(),shared=compiled.metadata.shared,source=str(ir),mode='original-serial-epilogue',epilogue_spt=a.spt,rematerialize_index=a.rematerialize_index,layout_conversions=records,reference=str(a.reference))
    (a.output/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
