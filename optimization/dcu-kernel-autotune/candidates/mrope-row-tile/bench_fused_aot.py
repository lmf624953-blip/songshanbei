#!/usr/bin/env python3
"""比较九齿Q/K RMSNorm+MRoPE融合与原三次设备计算，保留BF16中间舍入。"""
import argparse
import ctypes
import os
from functools import partial
import hashlib
import json
from pathlib import Path
from statistics import median
import time

import infinicore as ic
import ninetoothed as nt
import torch

from adaln.aot_codegen import compile_image, lower_kernel
from fused_norm import application, arrange
from adaln.aot_runtime import load_qknorm
from songshan_h3.compiler_compat import configure_hygon_compiler
from bench import wrap


class HIPModule:
    def __init__(self, path, symbol, grid, warps, shared, tensors):
        self.hip=ctypes.CDLL('libamdhip64.so')
        self.hip.hipGetErrorString.argtypes=[ctypes.c_int]
        self.hip.hipGetErrorString.restype=ctypes.c_char_p
        self.hip.hipModuleLoad.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_char_p]
        self.hip.hipModuleGetFunction.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p,ctypes.c_char_p]
        self.hip.hipModuleLaunchKernel.argtypes=[ctypes.c_void_p]+[ctypes.c_uint]*7+[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p]
        self.module=ctypes.c_void_p();self.function=ctypes.c_void_p()
        self.check(self.hip.hipModuleLoad(ctypes.byref(self.module),str(path).encode()))
        self.check(self.hip.hipModuleGetFunction(ctypes.byref(self.function),self.module,symbol.encode()))
        self.values=[ctypes.c_void_p(t.data_ptr()) for t in tensors]+[ctypes.c_void_p(),ctypes.c_void_p()]
        self.arguments=(ctypes.c_void_p*len(self.values))(*(ctypes.addressof(x) for x in self.values))
        self.grid,self.warps,self.shared=grid,warps,shared
        self.tensors=tensors

    def check(self, code):
        if code:raise RuntimeError(self.hip.hipGetErrorString(code).decode())

    def __call__(self):
        self.check(self.hip.hipModuleLaunchKernel(self.function,self.grid,1,1,self.warps*64,1,1,self.shared,
                   torch.cuda.current_stream().cuda_stream,self.arguments,None))


def build_native(directory, tensors, rows, warps):
    directory.mkdir()
    names=('query','key','query_weight','key_weight','cos','sin','positions','query_out','key_out')
    specs=list(nt.Tensor(shape=tuple(t.shape),dtype=nt.int64 if t.dtype==torch.int64 else nt.bfloat16) for t in tensors)
    # Q/K及两个输出的N共享一个符号，保留变长表达并证明相同归约输出域。
    specs[0]=nt.Tensor(shape=(None,7,128),dtype=nt.bfloat16)
    for index in (1,7,8):specs[index]=nt.Tensor(shape=specs[0].shape,dtype=nt.bfloat16)
    specs=tuple(specs)
    source,kernel,bindings=lower_kernel(directory,partial(arrange,rows=rows),application,specs,'h3_fused_norm_mrope_aot',warps)
    actual=dict(zip(names,tensors));signature={};constants={};runtime=[];pointers=[]
    for binding in bindings:
        name,tensor,kind,axis=binding.name,binding.source,binding.kind,binding.dim
        if kind=='tensor':
            signature[name]='*i64' if actual[tensor].dtype==torch.int64 else '*bf16'
            runtime.append(signature[name]);pointers.append(actual[tensor])
        else:
            signature[name]='constexpr'
            if kind=='shape':constants[name]=actual[tensor].shape[axis]
            elif kind=='stride':constants[name]=actual[tensor].stride(axis)
            else:raise RuntimeError(f'未经支持的MRoPE绑定：{binding}')
    digest,shared=compile_image(directory/'h3-mrope.hsaco',kernel,signature,constants,runtime,warps,block=rows*128,allow_shared=True)
    record=dict(warps=warps,rows=rows,shared=shared,hsaco_sha256=digest,
                block=rows*128,grid=((tensors[0].shape[0]+rows-1)//rows)*7,
                application_sha256=hashlib.sha256(Path(__file__).with_name('fused_norm.py').read_bytes()).hexdigest(),
                signature=signature,constants=constants,input_shapes=[list(t.shape) for t in tensors],input_strides=[list(t.stride()) for t in tensors])
    (directory/'build.json').write_text(json.dumps(record,indent=2)+'\n')
    length=tensors[0].shape[0]
    return HIPModule(directory/'h3-mrope.hsaco','h3_fused_norm_mrope_aot_kernel',((length+rows-1)//rows)*7,warps,shared,pointers)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--lengths',nargs='+',type=int,default=[27570,22016])
    args=parser.parse_args();args.out.mkdir()
    configure_hygon_compiler();ic.set_device(ic.device('cuda',0))
    generator=torch.Generator(device='cuda').manual_seed(42);results=[]
    for length in args.lengths:
        packed=torch.randn((length,21,128),device='cuda',dtype=torch.bfloat16,generator=generator)
        q,k=packed[:,:7],packed[:,7:14]
        qweight,kweight=(torch.randn((128,),device='cuda',dtype=torch.bfloat16,generator=generator) for _ in range(2))
        normalized=[torch.empty((length,7,128),device='cuda',dtype=torch.bfloat16) for _ in range(2)]
        angles=torch.randn((4096,48),device='cuda',dtype=torch.float32,generator=generator)
        cache=torch.cat((angles.cos(),angles.sin()),dim=1).to(torch.bfloat16)
        cos,sin=cache[:,:48],cache[:,48:]
        positions=torch.randint(-1,4098,(3,length),device='cuda',generator=generator)
        expected=[torch.empty_like(t) for t in normalized];actual=[torch.empty_like(t) for t in normalized]
        qi,ki=wrap(normalized[0].flatten(1)),wrap(normalized[1].flatten(1));ci,si,pi=map(wrap,(cos,sin,positions));qo,ko=(wrap(t.flatten(1)) for t in expected)
        host=load_qknorm(Path(os.environ['H3_QKNORM_BUILD']))
        native_norm=host.songshan_bench_qknorm
        native_norm.argtypes=[ctypes.c_int]*3+[ctypes.c_void_p]*4
        native_norm.restype=ctypes.c_char_p
        ic_stream=ic.get_stream()
        if type(ic_stream).__name__=='PyCapsule':
            capsule_pointer=ctypes.pythonapi.PyCapsule_GetPointer
            capsule_pointer.argtypes=[ctypes.py_object,ctypes.c_char_p]
            capsule_pointer.restype=ctypes.c_void_p
            ic_stream=capsule_pointer(ic_stream,None)
        if not isinstance(ic_stream,int):raise RuntimeError(f'未经支持的InfiniCore stream：{type(ic_stream)}')
        def original():
            for input,weight,output in zip((q,k),(qweight,kweight),normalized):
                error=native_norm(0,length,input.stride(0),input.data_ptr(),weight.data_ptr(),output.data_ptr(),ic_stream)
                if error:raise RuntimeError(error.decode())
            ic.mrope(qi,ki,ci,si,pi,128,96,16,16,16,False,out=(qo,ko))
        torch.cuda.synchronize();original();torch.cuda.synchronize()
        for rows,warps in ((1,2),):
            row=dict(sequence=length,rows=rows,warps=warps)
            try:
                native=build_native(args.out/f'n{length}-m{rows}-w{warps}',(q,k,qweight,kweight,cos,sin,positions,*actual),rows,warps)
                for tensor in actual:tensor.fill_(float("nan"))
                native();torch.cuda.synchronize()
                errors=[]
                for a,b in zip(actual,expected):
                    diff=a.float()-b.float();errors.append(dict(finite=bool(torch.isfinite(a).all()),exact=torch.equal(a.view(torch.int16),b.view(torch.int16)),max_abs=float(diff.abs().max()),rmse=float(diff.square().mean().sqrt())))
                row['errors']=errors
                for _ in range(3):original();native()
                torch.cuda.synchronize();samples=[[],[]]
                for i in range(21):
                    for variant in (i%2,1-i%2):
                        started=time.perf_counter();(original,native)[variant]();torch.cuda.synchronize()
                        samples[variant].append(1000*(time.perf_counter()-started))
                row.update(reference_ms=median(samples[0]),candidate_ms=median(samples[1]),samples_ms=samples,
                           paired_reduction_percent=100*median(1-b/a for a,b in zip(*samples)),
                           timing_scope='原生发射后全设备同步墙钟，含两种Python接口开销；设备收益另行trace，不直接作为正式收益')
            except Exception as error:row['error']=repr(error)
            results.append(row);(args.out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n');print(json.dumps(row,ensure_ascii=False),flush=True)
    passed=all('errors'in r and all(e['exact'] and e['finite'] for e in r['errors']) for r in results)
    (args.out/'complete.json').write_text(json.dumps(dict(all_variants_exact=passed,variants=len(results)))+'\n')


if __name__=='__main__':main()
