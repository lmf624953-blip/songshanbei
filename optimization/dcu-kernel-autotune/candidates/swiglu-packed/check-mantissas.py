#!/usr/bin/env python3
"""扩大逐位验证：全部有限 BF16 gate 与 128 种 up 尾数组合及极小 up。"""
import argparse,ctypes as ct,json,os,runpy,sys
from pathlib import Path
import torch,infinicore as ic
from infinicore.nn.functional import swiglu


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('build',type=Path);cli.add_argument('--output',type=Path,required=True);a=cli.parse_args()
    sys.path.insert(0,str(Path(os.environ['H3_AUTOTUNE_CODE_ROOT'])/'optimization'))
    from adaln.aot_runtime import load_swiglu
    wrap=runpy.run_path(str(Path(__file__).resolve().parents[1]/'wide-norm-indexed/bench.py'))['wrap']
    torch.cuda.set_device(0);ic.set_device(ic.device('cuda',0));stream=ic.get_stream()
    if type(stream).__name__=='PyCapsule':
        get=ct.pythonapi.PyCapsule_GetPointer;get.argtypes,get.restype=[ct.py_object,ct.c_char_p],ct.c_void_p;stream=get(stream,None)
    host=load_swiglu(a.build);launch=host.songshan_bench_swiglu;launch.argtypes,launch.restype=[ct.c_int,ct.c_int]+[ct.c_void_p]*3+[ct.c_size_t],ct.c_char_p
    values=torch.arange(65536,dtype=torch.int32,device='cuda').to(torch.int16).view(torch.bfloat16)
    values=torch.where(torch.isfinite(values),values,torch.zeros_like(values))
    records=[]
    for start,label in [(0x3f00,'positive-0.5-to-1'),(-16640,'negative-0.5-to-1'),(0,'positive-subnormal'),(128,'positive-smallest-normal')]:
        ups=torch.arange(start,start+128,dtype=torch.int32,device='cuda').to(torch.int16).view(torch.bfloat16)
        count=65536*128;rows=(count+1791)//1792
        table=torch.zeros((rows,3584),dtype=torch.bfloat16,device='cuda')
        gate=torch.nn.functional.pad(values.repeat_interleave(128),(0,rows*1792-count)).reshape(rows,1792)
        up=torch.nn.functional.pad(ups.repeat(65536),(0,rows*1792-count)).reshape(rows,1792)
        table[:,:1792].copy_(up);table[:,1792:].copy_(gate);up,gate=table[:,:1792],table[:,1792:]
        ref=torch.empty_like(up);storage=torch.full((rows*1792+512,),-17,dtype=torch.bfloat16,device='cuda');out=storage[:rows*1792]
        torch.cuda.synchronize();swiglu(wrap(up),wrap(gate),out=wrap(ref));error=launch(0,rows,up.data_ptr(),gate.data_ptr(),out.data_ptr(),stream)
        if error:raise RuntimeError(error.decode())
        torch.cuda.synchronize();bad=out.view(torch.int16)!=ref.flatten().view(torch.int16)
        r=dict(label=label,gate_values=65280,up_mantissas=128,compared_elements=rows*1792,exact=not bool(bad.any()),mismatches=int(bad.sum()),finite=bool(torch.isfinite(out).all()),tail_ok=bool((storage[out.numel():]==-17).all()))
        if r['mismatches']:
            indices=torch.nonzero(bad).flatten()[:8];r['first_mismatches']=[dict(up=float(up.flatten()[i]),gate=float(gate.flatten()[i]),reference=float(ref.flatten()[i]),candidate=float(out[i])) for i in indices.tolist()]
        print(json.dumps(r),flush=True);records.append(r)
    a.output.write_text(json.dumps(dict(scope='独立扩大数值覆盖，不作为正式计时',cases=records),ensure_ascii=False,indent=2)+'\n')
    assert all(r[k] for r in records for k in ('exact','finite','tail_ok'))
if __name__=='__main__':main()
