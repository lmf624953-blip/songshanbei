#!/usr/bin/env python3
"""同一 stream 对照原库 SwiGLU，保留 sigmoid BF16 舍入和 packed 输入。"""
import argparse,ctypes as ct,json,os,runpy
from pathlib import Path
from statistics import median
import torch,infinicore as ic
from infinicore.nn.functional import swiglu

def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('build',type=Path);cli.add_argument('--output',type=Path,required=True);a=cli.parse_args()
    helper=runpy.run_path(str(Path(__file__).resolve().parents[1]/'wide-norm-indexed/bench.py'));Fusion,wrap=helper['Fusion'],helper['wrap']
    torch.cuda.set_device(0);ic.set_device(ic.device('cuda',0));stream=ic.get_stream()
    if type(stream).__name__=='PyCapsule':
        get=ct.pythonapi.PyCapsule_GetPointer;get.argtypes,get.restype=[ct.py_object,ct.c_char_p],ct.c_void_p;stream=get(stream,None)
    timing_stream=torch.cuda.ExternalStream(stream,device=0)
    modules={f.parent.name:Fusion(f.parent) for f in sorted(a.build.glob('*/build.json'))}
    gen=torch.Generator(device='cuda').manual_seed(42);records=[]
    for rows,magnitude in [(37,'all-finite-bf16'),(1,1),(33,8),(1024,64),(22016,1),(27570,1)]:
        if magnitude=='all-finite-bf16':
            table=torch.ones((rows,3584),dtype=torch.bfloat16,device='cuda')
            bits=torch.arange(65536,dtype=torch.int32,device='cuda').to(torch.int16)
            values=bits.view(torch.bfloat16)
            values=torch.where(torch.isfinite(values),values,torch.zeros_like(values))
            values=torch.cat((values,torch.zeros(rows*1792-values.numel(),dtype=torch.bfloat16,device='cuda')))
            table[:,1792:].copy_(values.reshape(rows,1792))
        else:
            table=torch.randn((rows,3584),dtype=torch.bfloat16,device='cuda',generator=gen)*magnitude
        up,gate=table[:,:1792],table[:,1792:]
        ref=torch.empty((rows,1792),dtype=torch.bfloat16,device='cuda')
        storage=torch.full((rows*1792+512,),-17,dtype=torch.bfloat16,device='cuda');output=storage[:rows*1792]
        u,g,r=map(wrap,(up,gate,ref))
        def baseline():swiglu(u,g,out=r)
        torch.cuda.synchronize();baseline();torch.cuda.synchronize()
        for name,module in modules.items():
            def candidate():module.launch(stream,rows=(rows*module.meta["row_tiles"] if "row_tiles" in module.meta else (output.numel()+module.meta['tile']-1)//module.meta['tile']),input_rows=rows,elements=output.numel(),up=up,gate=gate,output=output)
            output.fill_(float('nan'));torch.cuda.synchronize();candidate();torch.cuda.synchronize()
            diff=output.float()-ref.flatten().float();record=dict(rows=rows,magnitude=magnitude,variant=name,exact=torch.equal(output.view(torch.int16),ref.flatten().view(torch.int16)),mismatches=int((output.view(torch.int16)!=ref.flatten().view(torch.int16)).sum()),finite=bool(torch.isfinite(output).all() and torch.isfinite(ref).all()),tail_ok=bool((storage[output.numel():]==-17).all()),max_abs=float(diff.abs().max()),rmse=float(diff.square().mean().sqrt()))
            if rows>=22016 and record['finite'] and record['tail_ok']:
                samples=[[],[]]
                for trial in range(15):
                    for side in (trial%2,1-trial%2):
                        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record(timing_stream);(baseline,candidate)[side]();end.record(timing_stream);end.synchronize()
                        if trial>=4:samples[side].append(begin.elapsed_time(end))
                record.update(device_ms=[median(x) for x in samples],paired_reduction_percent=100*median(1-b/a for a,b in zip(*samples)),samples_ms=samples)
            print(json.dumps(record),flush=True);records.append(record)
    a.output.write_text(json.dumps(dict(scope='独立单算子，不作为正式模型验收',cases=records),ensure_ascii=False,indent=2)+'\n')
    assert all(r['finite'] and r['tail_ok'] for r in records)
if __name__=='__main__':main()
