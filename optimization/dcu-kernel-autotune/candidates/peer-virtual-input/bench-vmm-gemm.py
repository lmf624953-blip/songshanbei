#!/usr/bin/env python3
"""真实九齿norm+本地发布/GEMM对共享VMM输入/GEMM；独立八卡微测。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time

import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library',type=Path);parser.add_argument('norm',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--rows',nargs='+',type=int,default=[27570,22899,24062]);parser.add_argument('--m',nargs='+',type=int,default=[2688,3584])
    parser.add_argument('--pairs',type=int,default=39);parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--map-granularity',type=int,default=0)
    parser.add_argument('--failure-recheck',action='store_true')
    parser.add_argument('--profile-mode',choices=['local','vmm'])
    args=parser.parse_args();assert not args.output.exists() and torch.cuda.device_count()==8
    assert args.pairs>=3 and args.warmup>=2
    if args.map_granularity:os.environ['H3_VMM_MAPPING_GRANULARITY']=str(args.map_granularity)
    if args.profile_mode:assert len(args.rows)==len(args.m)==1
    lib=ct.CDLL(str(args.library));norm=ct.CDLL(str(args.norm/'wide-norm-indexed-host.so'))
    meta=json.loads((args.norm/'wide-norm-indexed-build.json').read_text())
    assert hashlib.sha256((args.norm/'wide-norm-indexed.hsaco').read_bytes()).hexdigest()==meta['hsaco_sha256']
    norm.songshan_prepare_widenorm_indexed.argtypes=[ct.c_char_p,ct.c_int,ct.c_int];norm.songshan_prepare_widenorm_indexed.restype=ct.c_char_p
    def check(error):
        if error:raise RuntimeError(error.decode())
    check(norm.songshan_prepare_widenorm_indexed(os.fsencode(args.norm/'wide-norm-indexed.hsaco'),meta['warps'],meta['shared']))
    ptrs=ct.POINTER(ct.c_uint64)
    lib.vm_setup.argtypes=[ct.c_size_t,ptrs,ct.c_void_p,ptrs,ct.POINTER(ct.c_size_t),ct.POINTER(ct.c_size_t)]
    lib.vm_norm.argtypes=[ct.c_int]+[ptrs]*7
    lib.vm_gemm.argtypes=[ct.c_int,ct.c_int,ct.c_int]+[ct.c_void_p]*3
    lib.vm_finish.argtypes=[ct.POINTER(ct.c_float)];lib.vm_copy_input.argtypes=[ct.c_int,ct.c_void_p]
    for name in ['vm_setup','vm_norm','vm_gemm','vm_begin','vm_finish','vm_close','vm_copy_input']:getattr(lib,name).restype=ct.c_char_p
    def addresses(tensors):return (ct.c_uint64*8)(*(t.data_ptr() for t in tensors))
    streams=[torch.cuda.Stream(device=r) for r in range(8)];stream_ptrs=(ct.c_uint64*8)(*(s.cuda_stream for s in streams))
    records=[];norm_module=ct.cast(norm.songshan_bench_widenorm_indexed,ct.c_void_p)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for rows in args.rows:
            address=ct.c_uint64();boundaries=(ct.c_size_t*9)();granularity=ct.c_size_t()
            check(lib.vm_setup(rows,stream_ptrs,norm_module,ct.byref(address),boundaries,ct.byref(granularity)))
            try:
                gen=torch.Generator().manual_seed(295037+rows)
                host=torch.randn((rows,5376),generator=gen,dtype=torch.bfloat16)
                host_weight=torch.randn((5376,),generator=gen,dtype=torch.bfloat16)
                host_table=torch.randn((3,6*5376),generator=gen,dtype=torch.bfloat16)
                hidden=[];weight=[];table=[];indices=[];partial=[];local=[]
                for r in range(8):
                    with torch.cuda.device(r),torch.cuda.stream(streams[r]):
                        hidden.append(host.to(device=r));weight.append(host_weight.to(device=r));table.append(host_table.to(device=r))
                        indices.append(torch.arange(rows,device=r,dtype=torch.int64)%3)
                        n=rows*(r+1)//8-rows*r//8
                        partial.append(torch.empty(n*5376+16,device=r,dtype=torch.bfloat16));partial[-1].view(torch.int16).fill_(0x3535)
                        local.append(torch.empty((rows,5376),device=r,dtype=torch.bfloat16));local[-1].view(torch.int16).fill_(0x3535)
                for s in streams:s.synchronize()
                p_hidden,p_weight,p_indices,p_partial,p_local=map(addresses,[hidden,weight,indices,partial,local])
                p_scale=(ct.c_uint64*8)(*(t.data_ptr()+4*5376*2 for t in table));p_shift=(ct.c_uint64*8)(*(t.data_ptr()+3*5376*2 for t in table))
                for m in args.m:
                    matrices=[];outputs=[[],[]]
                    for r in range(8):
                        with torch.cuda.device(r),torch.cuda.stream(streams[r]):
                            torch.cuda.manual_seed(492003+r+m);matrices.append(torch.randn((m,5376),device=r,dtype=torch.bfloat16))
                            for mode in range(2):
                                outputs[mode].append(torch.empty(rows*m+16,device=r,dtype=torch.bfloat16));outputs[mode][-1].view(torch.int16).fill_(0x3535)
                    for s in streams:s.synchronize()
                    matrix_ptrs=[t.data_ptr() for t in matrices];output_ptrs=[[t.data_ptr() for t in part] for part in outputs]
                    def run(mode):
                        start=time.perf_counter();check(lib.vm_begin())
                        check(lib.vm_norm(mode,p_hidden,p_weight,p_scale,p_shift,p_indices,p_partial,p_local))
                        def gemm(rank):check(lib.vm_gemm(mode,rank,m,matrix_ptrs[rank],p_local[rank],output_ptrs[mode][rank]))
                        list(pool.map(gemm,range(8)));device_ms=(ct.c_float*8)();check(lib.vm_finish(device_ms))
                        return dict(wall_ms=(time.perf_counter()-start)*1000,device_ms=list(device_ms))
                    run(0);run(1)
                    valid=[]
                    for r in range(8):
                        with torch.cuda.device(r):
                            copied=torch.empty_like(local[r]);check(lib.vm_copy_input(r,copied.data_ptr()))
                            input_exact=torch.equal(copied.view(torch.int16),local[r].view(torch.int16))
                            output_exact=torch.equal(outputs[0][r][:rows*m].view(torch.int16),outputs[1][r][:rows*m].view(torch.int16))
                            tail=all(bool(torch.all(t[r][-16:].view(torch.int16)==0x3535)) for t in outputs)
                            norm_tail=bool(torch.all(partial[r][-16:].view(torch.int16)==0x3535))
                            valid.append(dict(rank=r,input_exact=input_exact,output_exact=output_exact,output_tail=tail,norm_tail=norm_tail))
                            if not (input_exact and output_exact and tail and norm_tail):
                                a=outputs[0][r][:rows*m];b=outputs[1][r][:rows*m]
                                failure=dict(rows=rows,m=m,rank=r,validation=valid[-1],
                                    mismatches=int(torch.count_nonzero(a.view(torch.int16)!=b.view(torch.int16))),
                                    max_abs=float((a.float()-b.float()).abs().max()))
                                if args.failure_recheck:
                                    previous_a=a.clone();previous_b=b.clone();run(0);run(1)
                                    failure.update(local_repeat_exact=torch.equal(previous_a.view(torch.int16),a.view(torch.int16)),
                                        vmm_repeat_exact=torch.equal(previous_b.view(torch.int16),b.view(torch.int16)),
                                        repeated_cross_exact=torch.equal(a.view(torch.int16),b.view(torch.int16)))
                                args.output.with_suffix('.failure.json').write_text(json.dumps(failure,ensure_ascii=False,indent=2)+'\n')
                                raise AssertionError(failure)
                            del copied
                    for _ in range(args.warmup):run(0);run(1)
                    if args.profile_mode:
                        hip=ct.CDLL('libamdhip64.so');assert hip.hipProfilerStart()==0
                        name='norm-publish-gemm.'+args.profile_mode
                        print('[hipprof.begin]',json.dumps(dict(name=name,epoch_ns=time.time_ns())),flush=True)
                        sample=run(0 if args.profile_mode=='local' else 1)
                        print('[hipprof.end]',json.dumps(dict(name=name,epoch_ns=time.time_ns())),flush=True)
                        time.sleep(2);assert hip.hipProfilerStop()==0
                        args.output.write_text(json.dumps(dict(scope='单次预热后独立PMC诊断，不是正式计时',diagnostic_profiler=True,
                            mode=args.profile_mode,rows=rows,m=m,granularity=granularity.value,validation=valid,
                            sample=sample,library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest()),ensure_ascii=False,indent=2)+'\n')
                        continue
                    pairs=[]
                    for i in range(args.pairs):
                        order=[0,1] if i%2==0 else [1,0];sample={mode:run(mode) for mode in order}
                        pairs.append(dict(order=order,local=sample[0],vmm=sample[1]))
                    a=[p['local']['wall_ms'] for p in pairs];b=[p['vmm']['wall_ms'] for p in pairs]
                    row=dict(rows=rows,m=m,k=5376,granularity=granularity.value,vmm_row_boundaries=list(boundaries),
                        vmm_physical_byte_boundaries=[r*((((rows*5376*2+7)//8)+granularity.value-1)//granularity.value*granularity.value) for r in range(9)],
                        reference_partition='floor(rows*rank/8)',validation=valid,pairs=pairs,
                        local_median_ms=median(a),vmm_median_ms=median(b),paired_reduction_median_percent=median(100*(1-y/x) for x,y in zip(a,b)))
                    records.append(row)
                    report=dict(scope='独立八卡：现用九齿局部norm+主动发布+原rocBLAS，对局部norm写VMM+就绪依赖+相同rocBLAS；包含主机与同步，排除共同前驱gate；非模型成绩',
                        compiler_library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),norm_image_sha256=meta['hsaco_sha256'],
                        rocblas='原TN/batch1/alpha1/beta0/BF16/FP32累加/默认算法，未改计算顺序',records=records)
                    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
                    print(json.dumps({k:v for k,v in row.items() if k in ['rows','m','local_median_ms','vmm_median_ms','paired_reduction_median_percent']}),flush=True)
                    del matrices,outputs
            finally:check(lib.vm_close())
            del hidden,weight,table,indices,partial,local


if __name__=='__main__':main()
