#!/usr/bin/env python3
"""以实际原库输出验证生产主机桥的动态长度、BF16 结果和尾部边界。"""
import argparse,ctypes as ct,json,os,runpy,sys
from pathlib import Path
import torch,infinicore as ic
from infinicore.nn.functional import swiglu


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('build',type=Path);cli.add_argument('--output',type=Path,required=True);a=cli.parse_args()
    sys.path.insert(0,str(Path(os.environ['H3_AUTOTUNE_CODE_ROOT'])/'optimization'))
    from adaln.aot_runtime import load_swiglu
    helper=runpy.run_path(str(Path(__file__).resolve().parents[1]/'wide-norm-indexed/bench.py'));wrap=helper['wrap']
    torch.cuda.set_device(0);ic.set_device(ic.device('cuda',0));stream=ic.get_stream()
    if type(stream).__name__=='PyCapsule':
        get=ct.pythonapi.PyCapsule_GetPointer;get.argtypes,get.restype=[ct.py_object,ct.c_char_p],ct.c_void_p;stream=get(stream,None)
    host=load_swiglu(a.build);launch=host.songshan_bench_swiglu
    launch.argtypes,launch.restype=[ct.c_int,ct.c_int]+[ct.c_void_p]*3+[ct.c_size_t],ct.c_char_p
    gen=torch.Generator(device='cuda').manual_seed(43);records=[]
    for rows,mag in [(37,'all-finite-bf16'),(1,1),(33,8),(1024,64),(22016,1),(27570,1)]:
        if isinstance(mag,str):
            table=torch.ones((rows,3584),dtype=torch.bfloat16,device='cuda')
            values=torch.arange(65536,dtype=torch.int32,device='cuda').to(torch.int16).view(torch.bfloat16)
            values=torch.where(torch.isfinite(values),values,torch.zeros_like(values))
            table[:,1792:].copy_(torch.cat((values,torch.zeros(rows*1792-65536,dtype=torch.bfloat16,device='cuda'))).reshape(rows,1792))
        else:table=torch.randn((rows,3584),dtype=torch.bfloat16,device='cuda',generator=gen)*mag
        up,gate=table[:,:1792],table[:,1792:];ref=torch.empty((rows,1792),dtype=torch.bfloat16,device='cuda')
        storage=torch.full((rows*1792+512,),-17,dtype=torch.bfloat16,device='cuda');out=storage[:rows*1792];out.fill_(float('nan'))
        torch.cuda.synchronize();swiglu(wrap(up),wrap(gate),out=wrap(ref))
        error=launch(0,rows,up.data_ptr(),gate.data_ptr(),out.data_ptr(),stream)
        if error:raise RuntimeError(error.decode())
        torch.cuda.synchronize()
        record=dict(rows=rows,magnitude=mag,exact=torch.equal(out.view(torch.int16),ref.flatten().view(torch.int16)),finite=bool(torch.isfinite(out).all()),tail_ok=bool((storage[out.numel():]==-17).all()))
        print(json.dumps(record),flush=True);records.append(record)
    a.output.write_text(json.dumps(dict(cases=records),indent=2)+'\n')
    assert all(r[k] for r in records for k in ('exact','finite','tail_ok'))
if __name__=='__main__':main()
