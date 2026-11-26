#!/usr/bin/env python3
"""核对原算法探针的 BF16 输出，再定位 sum、inverse RMS 和最终舍入。"""
import argparse,ctypes as ct,json,os,runpy,sys
from pathlib import Path
import torch
import infinicore as ic
from infinicore.nn.functional import rms_norm


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('build',type=Path);cli.add_argument('--output',type=Path,required=True);args=cli.parse_args()
    helper=runpy.run_path(str(Path(__file__).resolve().parents[1]/'wide-norm-indexed/bench.py'));Fusion,wrap=helper['Fusion'],helper['wrap']
    torch.cuda.set_device(0);ic.set_device(ic.device('cuda',0));stream=ic.get_stream()
    if type(stream).__name__=='PyCapsule':
        get=ct.pythonapi.PyCapsule_GetPointer;get.argtypes,get.restype=[ct.py_object,ct.c_char_p],ct.c_void_p;stream=get(stream,None)
    modules={f.parent.name:Fusion(f.parent) for f in sorted(args.build.glob('*/build.json'))}
    probes={}
    for name in ['default','fast','serial']:
        host=ct.CDLL(str(args.build/f'reference-{name}.so'));fn=host.songshan_probe_rms
        fn.argtypes=[ct.c_int]*2+[ct.c_void_p]*5+[ct.c_size_t];fn.restype=ct.c_int;probes[name]=(host,fn)
    generator=torch.Generator(device='cuda').manual_seed(42);records=[]
    def compare(a,b):
        bits=torch.int16 if a.dtype==torch.bfloat16 else torch.int32
        return dict(exact=torch.equal(a.view(bits),b.view(bits)),mismatches=int((a.view(bits)!=b.view(bits)).sum()),max_abs=float((a.float()-b.float()).abs().max()))
    for rows,magnitude in [(1,1),(33,8),(4096,1)]:
        hidden=torch.randn((rows,5376),dtype=torch.bfloat16,device='cuda',generator=generator)*magnitude
        weight=torch.randn((5376,),dtype=torch.bfloat16,device='cuda',generator=generator)
        original=torch.full_like(hidden,float('nan'))
        rms_norm(wrap(hidden),[5376],wrap(weight),eps=1e-5,out=wrap(original));torch.cuda.synchronize()
        refs={}
        for name,(_,fn) in probes.items():
            output=torch.full_like(hidden,float('nan'));sums=torch.full((rows,1),float('nan'),device='cuda');inverse=torch.full_like(sums,float('nan'))
            rc=fn(rows,5376,*(x.data_ptr() for x in (hidden,weight,output,sums,inverse)),stream)
            if rc:raise RuntimeError(f'probe launch: {rc}')
            torch.cuda.synchronize();refs[name]=(output,sums,inverse)
            record=dict(rows=rows,magnitude=magnitude,probe=name,original_output=compare(original,output),finite=all(bool(torch.isfinite(t).all()) for t in (output,sums,inverse)))
            records.append(record);print(json.dumps(record),flush=True)
        for name,module in modules.items():
            storage=torch.full((rows+2,5376),-19,dtype=hidden.dtype,device='cuda');output=storage[:rows];output.fill_(float('nan'))
            sums=torch.full((rows,1),float('nan'),device='cuda');inverse=torch.full_like(sums,float('nan'))
            module.launch(stream,rows=rows,reduction=hidden,hidden=hidden,weight=weight,normalized=output,sums=sums,inverse=inverse);torch.cuda.synchronize()
            record=dict(rows=rows,magnitude=magnitude,variant=name,original_output=compare(original,output),tail_ok=bool((storage[rows:]==-19).all()),finite=all(bool(torch.isfinite(t).all()) for t in (output,sums,inverse)),probes={})
            for mode,(y,ss,inv) in refs.items():
                record['probes'][mode]=dict(sum=compare(ss,sums),inverse=compare(inv,inverse),normalized=compare(y,output))
                different=(ss.view(torch.int32)!=sums.view(torch.int32))[:,0].nonzero().flatten()[:8]
                record['probes'][mode]['sum_examples']=[dict(row=int(i),reference=float(ss[i,0]),candidate=float(sums[i,0])) for i in different.cpu().tolist()]
            records.append(record);print(json.dumps(record),flush=True)
    args.output.write_text(json.dumps(dict(scope='诊断中间量，不作为性能计时或生产候选',records=records),ensure_ascii=False,indent=2)+'\n')
    assert all(r['finite'] and r.get('tail_ok',True) for r in records)
    assert any(r.get('probe') and r['original_output']['exact'] and r['rows']==4096 for r in records),'没有与原库逐位一致的参考探针，禁止直接归因'


if __name__=='__main__':main()
