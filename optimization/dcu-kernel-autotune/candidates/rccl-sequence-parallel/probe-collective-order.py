#!/usr/bin/env python3
"""先验证AllReduce与RS/AG、分块AllReduce的BF16位顺序，不改正式推理。"""
import argparse
import ctypes as ct
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time
import torch


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('library',type=Path);cli.add_argument('output',type=Path)
    cli.add_argument('--rows',type=int,default=27570);cli.add_argument('--width',type=int,default=5376)
    cli.add_argument('--pairs',type=int,default=5);cli.add_argument('--chunks',type=int,nargs='+',default=[2,4,8])
    cli.add_argument('--modes',nargs='+',choices=['full','rsag','chunked'],default=['full','rsag','chunked'])
    args=cli.parse_args();assert args.rows>0 and args.width>0 and args.pairs>0
    helper=Path(__file__).resolve().parent.parent/'rccl-allreduce/bench-allreduce.py'
    spec=importlib.util.spec_from_file_location('rccl_helper',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    lib=module.load_rccl(args.library.resolve(strict=True))
    for name,signature in {
        'ncclReduceScatter':[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p],
        'ncclAllGather':[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_int,ct.c_void_p,ct.c_void_p],
    }.items():
        f=getattr(lib,name);f.argtypes=signature;f.restype=ct.c_int;f.errcheck=lib.ncclAllReduce.errcheck
    ranks=8;assert torch.cuda.device_count()==ranks
    streams=[torch.cuda.Stream(device=i) for i in range(ranks)]
    comms=(ct.c_void_p*ranks)();devices=(ct.c_int*ranks)(*range(ranks));lib.ncclCommInitAll(comms,ranks,devices)
    records=[]
    def sync():
        for s in streams:s.synchronize()
    def grouped(call):
        lib.ncclGroupStart()
        for rank in range(ranks):
            with torch.cuda.device(rank):call(rank)
        lib.ncclGroupEnd()
    try:
        for padded in [False,True]:
            rows=(args.rows+7)//8*8 if padded else args.rows
            count=rows*args.width;assert count%ranks==0
            sources=[]
            for rank in range(ranks):
                with torch.cuda.device(rank):
                    gen=torch.Generator(device=rank).manual_seed(104729+rank)
                    x=torch.randn(args.rows*args.width,device=rank,generator=gen,dtype=torch.bfloat16)
                    x.mul_(2**(rank%4))
                    if padded:x=torch.nn.functional.pad(x,(0,count-x.numel()))
                    streams[rank].wait_stream(torch.cuda.current_stream(rank));sources.append(x)
            reference=[torch.empty_like(x) for x in sources];candidate=[torch.empty_like(x) for x in sources]
            shards=[torch.empty(count//ranks,device=i,dtype=torch.bfloat16) for i in range(ranks)]
            def reset(buffers):
                for rank in range(ranks):
                    with torch.cuda.stream(streams[rank]):buffers[rank].copy_(sources[rank])
                sync()
            def full(buffers):
                grouped(lambda r:lib.ncclAllReduce(buffers[r].data_ptr(),buffers[r].data_ptr(),count,9,0,comms[r],streams[r].cuda_stream))
            def rsag():
                grouped(lambda r:lib.ncclReduceScatter(sources[r].data_ptr(),shards[r].data_ptr(),count//ranks,9,0,comms[r],streams[r].cuda_stream))
                grouped(lambda r:lib.ncclAllGather(shards[r].data_ptr(),candidate[r].data_ptr(),count//ranks,9,comms[r],streams[r].cuda_stream))
            def chunked(chunks):
                # 用完整行分块，保持输入形状语义；集合调用不跨块重排。
                for c in range(chunks):
                    begin=(rows*c//chunks)*args.width;end=(rows*(c+1)//chunks)*args.width
                    grouped(lambda r:lib.ncclAllReduce(candidate[r].data_ptr()+begin*2,candidate[r].data_ptr()+begin*2,end-begin,9,0,comms[r],streams[r].cuda_stream))
            reset(reference);full(reference);sync()
            starts=[torch.cuda.Event(enable_timing=True) for _ in streams];stops=[torch.cuda.Event(enable_timing=True) for _ in streams]
            modes=([('full',0)] if 'full' in args.modes else [])
            modes+=([('rsag',0)] if 'rsag' in args.modes else [])
            modes+=([('chunked',c) for c in args.chunks] if 'chunked' in args.modes else [])
            for mode,chunks in modes:
                measurements=[];mismatches=None
                for t in range(1+args.pairs):
                    reset(candidate)
                    for e,s in zip(starts,streams):e.record(s)
                    t0=time.perf_counter()
                    if mode=='full':full(candidate)
                    elif mode=='rsag':rsag()
                    else:chunked(chunks)
                    for e,s in zip(stops,streams):e.record(s)
                    sync();wall=(time.perf_counter()-t0)*1000
                    if t:measurements.append(dict(gpu_ms=max(a.elapsed_time(b) for a,b in zip(starts,stops)),wall_ms=wall))
                mismatches=[]
                for rank in range(ranks):
                    with torch.cuda.device(rank):mismatches.append(int(torch.count_nonzero(candidate[rank][:args.rows*args.width].view(torch.int16)!=reference[rank][:args.rows*args.width].view(torch.int16)).item()))
                row=dict(mode=mode,chunks=chunks,padded=padded,rows=rows,original_rows=args.rows,width=args.width,count=count,mismatches=mismatches,exact=not any(mismatches),samples=measurements,median_gpu_ms=statistics.median(x['gpu_ms'] for x in measurements),median_wall_ms=statistics.median(x['wall_ms'] for x in measurements))
                records.append(row);print(json.dumps(row),flush=True)
                args.output.write_text(json.dumps(dict(settings={k:os.environ.get(k) for k in ['NCCL_ALGO','NCCL_PROTO','NCCL_MIN_NCHANNELS','NCCL_MAX_NCHANNELS']},scope='集合同序及分块数值闸门；未加入分片消费者，不是端到端计时',records=records),ensure_ascii=False,indent=2)+'\n')
            del sources,reference,candidate,shards,x
            for rank in range(ranks):
                with torch.cuda.device(rank):torch.cuda.empty_cache()
    finally:
        for rank,comm in enumerate(comms):
            with torch.cuda.device(rank):lib.ncclCommDestroy(comm)


if __name__=='__main__':main()
