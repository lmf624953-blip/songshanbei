#!/usr/bin/env python3
"""对正式变长融合 ABI 验证尾部、步长、位置越界与放大输入。"""
import argparse,ctypes,hashlib,json,os,sys,time
from pathlib import Path
from statistics import median
import torch
import infinicore as ic

CODE=Path(os.environ.get('H3_AUTOTUNE_CODE_ROOT','/public/home/acvduipmem/xsh/songshanbei/submission-xsh'))
sys.path.insert(0,str(CODE/'optimization/adaln'))
from aot_runtime import load_qknorm,load_qknorm_mrope

def wrap(tensor):
    return ic.strided_from_blob(tensor.data_ptr(),list(tensor.shape),list(tensor.stride()),
            dtype=ic.int64 if tensor.dtype==torch.int64 else ic.bfloat16,
            device=ic.device('cuda',tensor.device.index))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('build',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise RuntimeError('禁止覆盖验证记录')
    torch.cuda.set_device(0);ic.set_device(ic.device('cuda',0))
    stream=ic.get_stream();get=ctypes.pythonapi.PyCapsule_GetPointer
    get.argtypes=[ctypes.py_object,ctypes.c_char_p];get.restype=ctypes.c_void_p
    if type(stream).__name__=='PyCapsule':stream=get(stream,None)
    norm=load_qknorm(Path(os.environ['H3_QKNORM_BUILD'])).songshan_bench_qknorm
    norm.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*4;norm.restype=ctypes.c_char_p
    fused=load_qknorm_mrope(a.build).songshan_bench_qknorm_mrope
    fused.argtypes=[ctypes.c_int]*6+[ctypes.c_void_p]*10;fused.restype=ctypes.c_char_p
    generator=torch.Generator(device='cuda').manual_seed(20261125)
    cases=[]
    specs=[(n,1,21*128,21*128,4096,0) for n in (1,2,3,4,5,31,32,33,4096,22016,27570,65535)]
    specs += [(33,scale,qs,ks,17,16) for scale in (1,8) for qs,ks in ((896,896),(2816,2944))]
    specs += [(n, 1, 2688, 2688, n, -1) for n in (1, 33, 22016, 27570)]
    for n,scale,qs,ks,cache_rows,pad in specs:
        q=torch.empty_strided((n,7,128),(qs,128,1),device='cuda',dtype=torch.bfloat16)
        k=torch.empty_strided((n,7,128),(ks,128,1),device='cuda',dtype=torch.bfloat16)
        for t in(q,k):t.copy_(torch.randn(t.shape,device='cuda',dtype=t.dtype,generator=generator)*scale)
        weights=[torch.randn((128,),device='cuda',dtype=torch.bfloat16,generator=generator) for _ in range(2)]
        angle=torch.randn((cache_rows,48),device='cuda',generator=generator)
        cache=torch.cat((angle.cos(),angle.sin()),1).to(torch.bfloat16);cos,sin=cache[:,:48],cache[:,48:]
        if pad == -1:
            # 正式模型使用展开三轴角度的缓存和一维行号。
            pos = torch.arange(n, device='cuda', dtype=torch.int64)
            positions_stride = 0
        else:
            posbase = torch.randint(-3, cache_rows + 3, (3, n + pad), device='cuda', generator=generator)
            pos = posbase[:, :n]
            positions_stride = pos.stride(0)
        normalized=[torch.empty((n,7,128),device='cuda',dtype=torch.bfloat16) for _ in range(2)]
        expected=[torch.empty_like(t) for t in normalized]
        base=[torch.full((n+8,7,128),-11.25,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
        actual=[t[:n] for t in base]
        qi,ki=[wrap(t.flatten(1)) for t in normalized];co,si,po=map(wrap,(cos,sin,pos));oq,ok=[wrap(t.flatten(1)) for t in expected]
        def original():
            for x,w,y in zip((q,k),weights,normalized):
                error=norm(0,n,x.stride(0),x.data_ptr(),w.data_ptr(),y.data_ptr(),stream)
                if error:raise RuntimeError(error.decode())
            ic.mrope(qi,ki,co,si,po,128,96,16,16,16,False,out=(oq,ok))
        def candidate():
            ptrs=[t.data_ptr() for t in(q,k,*weights,cos,sin,pos,*actual)]
            error=fused(0,n,qs,ks,cache_rows,positions_stride,*ptrs,torch.cuda.current_stream().cuda_stream)
            if error:raise RuntimeError(error.decode())
        torch.cuda.synchronize();original();torch.cuda.synchronize();candidate();torch.cuda.synchronize()
        errors=[]
        for x,y,b in zip(actual,expected,base):
            d=x.float()-y.float()
            errors.append(dict(exact=torch.equal(x.view(torch.int16),y.view(torch.int16)),finite=bool(torch.isfinite(x).all()),
                               max_abs=float(d.abs().max()),rmse=float(d.square().mean().sqrt()),tail_guard=bool((b[n:]==-11.25).all())))
        result=dict(length=n,scale=scale,query_stride=qs,key_stride=ks,cache_rows=cache_rows,positions_stride=positions_stride,errors=errors)
        if n in(22016,27570):
            samples=[[],[]]
            for i in range(13):
                for v in(i%2,1-i%2):
                    torch.cuda.synchronize();started=time.perf_counter();(original,candidate)[v]();torch.cuda.synchronize()
                    if i>=2:samples[v].append(1000*(time.perf_counter()-started))
            result['diagnostic_sync_wall_ms']=[median(s) for s in samples]
        cases.append(result);print(json.dumps(result),flush=True)
        passed=all(e['exact'] and e['finite'] and e['tail_guard'] for row in cases for e in row['errors'])
        record=dict(verdict='PASS' if passed else 'FAIL',cases=cases,
                    hsaco_sha256=hashlib.sha256((a.build/'qk-norm-mrope.hsaco').read_bytes()).hexdigest())
        a.output.write_text(json.dumps(record,indent=2)+'\n')
        if not passed:raise RuntimeError('融合数值或尾部验证未通过')
if __name__=='__main__':main()
