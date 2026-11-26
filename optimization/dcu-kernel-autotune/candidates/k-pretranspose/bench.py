#!/usr/bin/env python3
"""诊断 K 转置的收益上限，同时计入每次转置；胜出后再实现九齿排布。"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import runpy
from statistics import median
import sys

import torch


def check_case(reference, native, length, unit, profile_candidate):
    generator = torch.Generator(device="cuda").manual_seed(42)
    def random(heads):
        return torch.randn((1,length,heads,128),dtype=torch.bfloat16,device="cuda",generator=generator).transpose(1,2)
    q,k = random(7),random(7)
    v = random(21)[:,-7:]
    pad=(length+31)//32*32
    padded=torch.zeros((1,7,pad,128),dtype=q.dtype,device=q.device)
    padded[:,:,:length].copy_(k)
    source=(padded.view(1,7,pad//32,32,128//unit,unit).permute(0,1,2,4,3,5)
            if unit else k.transpose(2,3))
    packed=(torch.empty_like(source,memory_format=torch.contiguous_format) if unit else
            torch.zeros((1,7,128,pad),dtype=q.dtype,device=q.device))
    storage=[torch.empty((1,length+2,7,128),dtype=q.dtype,device=q.device) for _ in range(2)]
    outputs=[buf[:,:length].transpose(1,2) for buf in storage]
    def transpose():
        if unit:
            padded[:,:,:length].copy_(k)
            packed.copy_(source)
        else:
            packed[:,:,:,:length].copy_(source)
    def candidate():
        err=native(0,length,128**-.5,q.data_ptr(),packed.data_ptr(),v.data_ptr(),outputs[1].data_ptr(),896,896,2688,896,torch.cuda.current_stream().cuda_stream)
        if err: raise RuntimeError(err.decode())
    def baseline():
        reference(q,k,v,128**-.5,outputs[0])
    def combined():
        transpose()
        candidate()
    if profile_candidate:
        transpose()
        for _ in range(4):
            candidate()
        torch.cuda.synchronize()
        finite=bool(torch.isfinite(outputs[1]).all())
        if not finite:
            raise RuntimeError("硬件计数候选产生非有限值")
        return dict(sequence=length,profile_candidate=True,finite=finite,attention_calls=4)
    def compare():
        for buf in storage:
            buf.fill_(float("nan"))
            buf[:,length:].fill_(-19)
        baseline()
        combined()
        torch.cuda.synchronize()
        return dict(exact=torch.equal(outputs[0].view(torch.int16),outputs[1].view(torch.int16)),
                    finite=all(bool(torch.isfinite(o).all()) for o in outputs),
                    tail_ok=all(bool(b[:,length:].eq(-19).all()) for b in storage),
                    max_abs=float((outputs[0].float()-outputs[1].float()).abs().max()))
    normal=compare()
    q.mul_(2);k.mul_(2)
    stress=compare()
    passed=all(check[field] for check in (normal,stress) for field in ("exact","finite","tail_ok"))
    result=dict(sequence=length,normal=normal,stress=stress,passed=passed)
    if passed:
        samples={name:[] for name in ("baseline","combined","attention_only","transpose")}
        functions={"baseline":baseline,"combined":combined,"attention_only":candidate,"transpose":transpose}
        events=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
        for repeat in range(13):
            names=list(functions)
            if repeat%2:names.reverse()
            for name in names:
                events[0].record();functions[name]();events[1].record();events[1].synchronize()
                if repeat>=2:samples[name].append(events[0].elapsed_time(events[1]))
        medians={name:median(values) for name,values in samples.items()}
        result.update(samples_ms=samples,medians_ms=medians,
                      reduction_percent=100*median(1-b/a for a,b in zip(samples["baseline"],samples["combined"])))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference",type=Path)
    parser.add_argument("candidate",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--unit",type=int,choices=(0,8,16),default=0)
    parser.add_argument("--profile-candidate",action="store_true")
    args=parser.parse_args()
    code=Path(os.environ["H3_AUTOTUNE_CODE_ROOT"])/"optimization"
    sys.path.insert(0,str(code))
    reference=runpy.run_path(str(code/"bench-attention.py"))["native_launcher"](args.reference)
    from adaln.aot_runtime import load_attention
    host=load_attention(args.candidate)
    native=host.songshan_bench_attention
    native.argtypes=[ctypes.c_int]*2+[ctypes.c_float]+[ctypes.c_void_p]*4+[ctypes.c_int]*4+[ctypes.c_void_p]
    native.restype=ctypes.c_char_p
    records=[]
    for length in (27570,22016):
        records.append(check_case(reference,native,length,args.unit,args.profile_candidate))
        print(json.dumps(records[-1]),flush=True)
        args.output.write_text(json.dumps(dict(scope="GPU 算子诊断，转置采用 PyTorch copy，正式候选需九齿实现",records=records),indent=2)+"\n")


if __name__=="__main__":
    main()
