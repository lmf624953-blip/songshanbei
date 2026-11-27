#!/usr/bin/env python3
"""八卡peer两阶段原型与RCCL原地AllReduce配对；数值失败只作可行性诊断。"""
import argparse
import ctypes as ct
import importlib.util
import hashlib
import json
from pathlib import Path
import statistics
import time
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"rccl-sequence-parallel"))
import torch
from ring_order_map import table


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('rccl',type=Path);cli.add_argument('peer',type=Path);cli.add_argument('output',type=Path)
    cli.add_argument('--counts',type=int,nargs='+',default=[148216320])
    cli.add_argument('--vectors',type=int,nargs='+',default=[1,2,4,8]);cli.add_argument('--blocks',type=int,nargs='+',default=[320])
    cli.add_argument('--pairs',type=int,default=9);cli.add_argument('--seed',type=int,default=104729)
    cli.add_argument('--channels',type=int,default=16);cli.add_argument('--chunk',type=int,default=524288);cli.add_argument('--alignment',type=int,default=1024)
    cli.add_argument('--fixed-order',action='store_true',help='只做性能可行性，不使用位置顺序表')
    cli.add_argument('--gather',choices=['read','write'],default='read')
    cli.add_argument('--case',choices=['random','tiny','wide','special'],default='random')
    cli.add_argument('--indexed-build',type=Path,help='提供现用九齿构建，比较AllReduce+indexed_gate与分片融合')
    cli.add_argument('--nt-build',type=Path,help='使用九齿分片镜像及独立transport库')
    cli.add_argument('--bridge-build',type=Path,help='以InfiniCCL路径和八线程主机桥进行接入前验证')
    cli.add_argument('--separate-output',action='store_true',help='模型一致的独立输出，直接发布必需')
    cli.add_argument('--consumer-copy',action='store_true',help='在相同stream紧接着由GPU读取输出，验证事件与可见性')
    cli.add_argument('--reference-peer-build',type=Path,help='使用现用两阶段peer作为对照，复用同一InfiniCCL communicator')
    cli.add_argument('--profile-regions',action='store_true',help='只作独立trace的主机时间边界标记，不作正式计时')
    args=cli.parse_args();assert args.pairs>0
    reference_peer=None
    helper=Path(__file__).resolve().parent.parent/'rccl-allreduce/bench-allreduce.py'
    spec=importlib.util.spec_from_file_location('rccl_helper',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if args.bridge_build:
        from bridge_micro import InfiniCCLProxy,install_adapter as install_bridge
        lib=InfiniCCLProxy(args.rccl);peer=ct.CDLL(str(args.peer.resolve(strict=True)));install_bridge(peer,args.bridge_build,args.indexed_build,lib)
    else:
        lib=module.load_rccl(args.rccl.resolve(strict=True));peer=ct.CDLL(str(args.peer.resolve(strict=True)))
    if args.reference_peer_build:
        assert args.bridge_build and args.indexed_build
        reference_peer=ct.CDLL(str((args.reference_peer_build/'peer-gate-bridge.so').resolve()))
        install_bridge(reference_peer,args.reference_peer_build,args.indexed_build,lib)
    if args.nt_build:
        assert args.indexed_build
        from nt_peer_launcher import install_adapter
        install_adapter(peer,args.nt_build)
    peer.peer_setup.restype=ct.c_int;peer.peer_cleanup.restype=ct.c_int
    pointer=ct.POINTER(ct.c_uint64);peer.peer_execute.argtypes=[pointer,pointer,pointer,pointer,pointer,ct.c_size_t,ct.c_int,ct.c_int];peer.peer_execute.restype=ct.c_int
    if args.gather=='write':
        peer.peer_execute_mode.argtypes=peer.peer_execute.argtypes+[ct.c_int];peer.peer_execute_mode.restype=ct.c_int
    native=None;native_record=None
    if args.indexed_build:
        assert args.gather=='write'
        meta=json.loads((args.indexed_build/'indexed-build.json').read_text())
        for op in ['indexed_adaln','indexed_gate']:
            assert hashlib.sha256((args.indexed_build/(op+'.hsaco')).read_bytes()).hexdigest()==meta[op]['hsaco_sha256']
        native=ct.CDLL(str((args.indexed_build/'indexed-host.so').resolve()))
        native.songshan_prepare_indexed.argtypes=[ct.c_char_p,ct.c_char_p,ct.c_int];native.songshan_prepare_indexed.restype=ct.c_char_p
        error=native.songshan_prepare_indexed(str(args.indexed_build/'indexed_adaln.hsaco').encode(),str(args.indexed_build/'indexed_gate.hsaco').encode(),meta['indexed_gate']['warps'])
        if error:raise RuntimeError(error.decode())
        assert native.songshan_indexed_tile()==meta['indexed_gate']['tile']
        native.songshan_bench_indexed.argtypes=[ct.c_bool,ct.c_int,ct.c_int,ct.c_int]+[ct.c_void_p]*5+[ct.c_size_t];native.songshan_bench_indexed.restype=ct.c_char_p
        peer.peer_execute_gate.argtypes=[pointer]*8+[ct.c_size_t,ct.c_int,ct.c_int];peer.peer_execute_gate.restype=ct.c_int
        native_record=dict(build=str(args.indexed_build),gate_sha256=meta['indexed_gate']['hsaco_sha256'],tile=meta['indexed_gate']['tile'])
    def check(status):
        if status:raise RuntimeError(f'peer HIP status {status}')
    ranks=8;assert torch.cuda.device_count()==ranks
    streams=[torch.cuda.Stream(device=i) for i in range(ranks)];comms=(ct.c_void_p*ranks)();devices=(ct.c_int*ranks)(*range(ranks));lib.ncclCommInitAll(comms,ranks,devices)
    check(peer.peer_setup());records=[]
    if reference_peer:check(reference_peer.peer_setup())
    try:
        for count in args.counts:
            assert count%64==0
            sources=[]
            for rank in range(ranks):
                with torch.cuda.device(rank):
                    gen=torch.Generator(device=rank).manual_seed(args.seed+rank)
                    if args.case=='special':
                        values=[0,0x8000,1,0x8001,0x7f,0x807f,0x80,0x8080,0x3f80,0xbf80,0x7f7f,0xff7f,0x7f80,0xff80,0x7fc1,0x7f81,
                                0,0x8000,0x3f00,0xbf00,0x4000,0xc000,0x3e80,0xbe80,0x100,0x8100,0x7f80,0xff80,0x7fff,0xffff,0xffc1,0xff81]
                        values=values[rank:]+values[:rank]
                        raw=torch.tensor([v if v<32768 else v-65536 for v in values],device=rank,dtype=torch.int16)
                        x=raw.view(torch.bfloat16).repeat(count//32)
                    else:
                        x=torch.randn(count,device=rank,generator=gen,dtype=torch.bfloat16);x.mul_(2**(rank%4))
                        if args.case=='tiny':x.mul_(2.0**-130)
                        if args.case=='wide':x.mul_(2.0**120)
                    sources.append(x)
                    streams[rank].wait_stream(torch.cuda.current_stream(rank))
            reference=[torch.empty_like(x) for x in sources]
            residuals=[];tables=[];indices=[];reference_out=[]
            if native:
                assert count%5376==0
                for rank in range(ranks):
                    with torch.cuda.device(rank):
                        gen=torch.Generator(device=rank).manual_seed(args.seed+30000)
                        residuals.append(torch.randn(count,device=rank,generator=gen,dtype=torch.bfloat16))
                        if args.case=='tiny':residuals[-1].mul_(2.0**-130)
                        if args.case=='wide':residuals[-1].mul_(2.0**120)
                        tables.append(torch.randn((3,6*5376),device=rank,generator=gen,dtype=torch.bfloat16))
                        indices.append(torch.arange(count//5376,device=rank,dtype=torch.int64)%3)
                        reference_out.append(torch.empty_like(reference[rank]))
                        streams[rank].wait_stream(torch.cuda.current_stream(rank))
            candidate=[torch.full((count+8,),0,device=i,dtype=torch.bfloat16) for i in range(ranks)]
            published=[torch.full_like(x,0) for x in candidate] if args.separate_output else candidate
            consumer=[torch.full_like(x,0) for x in published] if args.consumer_copy else None
            reference_consumer=[torch.empty_like(x) for x in reference_out] if args.consumer_copy else None
            shards=[torch.full((count//8+8,),0,device=i,dtype=torch.bfloat16) for i in range(ranks)]
            for rank in range(ranks):
                with torch.cuda.device(rank):
                    candidate[rank][-8:].view(torch.int16).fill_(0x3535);published[rank][-8:].view(torch.int16).fill_(0x3535);shards[rank][-8:].view(torch.int16).fill_(0x3535)
                    if consumer:consumer[rank][-8:].view(torch.int16).fill_(0x3535)
                    streams[rank].wait_stream(torch.cuda.current_stream(rank))
            hostmap=None if args.fixed_order else torch.from_numpy(table(count,channels=args.channels,chunk=args.chunk,alignment=args.alignment))
            maps=None if hostmap is None else [hostmap.to(i) for i in range(ranks)]
            if maps:
                for rank in range(ranks):streams[rank].wait_stream(torch.cuda.current_stream(rank))
            pointers=lambda tensors:(ct.c_uint64*ranks)(*(x.data_ptr() for x in tensors))
            cp,sp,st=pointers(candidate),pointers(shards),(ct.c_uint64*ranks)(*(s.cuda_stream for s in streams));mp=pointers(maps) if maps else None
            if native:
                rp,ip=pointers(residuals),pointers(indices)
                tp=(ct.c_uint64*ranks)(*(x.data_ptr()+2*5376*2 for x in tables))
            def sync():
                for s in streams:s.synchronize()
            def reset(buffers):
                for rank in range(ranks):
                    with torch.cuda.stream(streams[rank]):buffers[rank][:count].copy_(sources[rank])
                if args.consumer_copy:
                    destination=published if buffers is candidate else reference_out
                    dependent=consumer if buffers is candidate else reference_consumer
                    for rank in range(ranks):
                        with torch.cuda.stream(streams[rank]):
                            destination[rank][:count].view(torch.int16).fill_(0x3535)
                            dependent[rank][:count].view(torch.int16).fill_(0x3535)
                sync()
            def baseline():
                if reference_peer:
                    check(reference_peer.peer_execute_gate(pointers(reference),sp,pointers(reference_out),st,mp,rp,tp,ip,count,vec,blocks))
                else:
                    lib.ncclGroupStart()
                    for rank in range(ranks):
                        with torch.cuda.device(rank):lib.ncclAllReduce(reference[rank].data_ptr(),reference[rank].data_ptr(),count,9,0,comms[rank],streams[rank].cuda_stream)
                    lib.ncclGroupEnd()
                    if native:
                        for rank in range(ranks):
                            with torch.cuda.device(rank):
                                error=native.songshan_bench_indexed(True,rank,count//5376,3,residuals[rank].data_ptr(),reference[rank].data_ptr(),tp[rank],indices[rank].data_ptr(),reference_out[rank].data_ptr(),streams[rank].cuda_stream)
                                if error:raise RuntimeError(error.decode())
                if reference_consumer:
                    for rank in range(ranks):
                        with torch.cuda.stream(streams[rank]):torch.bitwise_xor(reference_out[rank].view(torch.int16),0,out=reference_consumer[rank].view(torch.int16))
            starts=[torch.cuda.Event(enable_timing=True) for _ in streams];stops=[torch.cuda.Event(enable_timing=True) for _ in streams]
            profile_counts={}
            def measured(function,buffers):
                reset(buffers)
                for e,s in zip(starts,streams):e.record(s)
                tag=function.__name__+'.'+str(profile_counts.get(function.__name__,0))
                profile_counts[function.__name__]=profile_counts.get(function.__name__,0)+1
                if args.profile_regions:print('[hipprof.begin]',json.dumps(dict(name=tag,epoch_ns=time.time_ns())),flush=True)
                begin=time.perf_counter();function()
                for e,s in zip(stops,streams):e.record(s)
                sync()
                if args.profile_regions:print('[hipprof.end]',json.dumps(dict(name=tag,epoch_ns=time.time_ns())),flush=True)
                return dict(gpu_ms=max(a.elapsed_time(b) for a,b in zip(starts,stops)),wall_ms=(time.perf_counter()-begin)*1000)
            for vec in args.vectors:
                for blocks in args.blocks:
                    def custom():
                        if native:check(peer.peer_execute_gate(cp,sp,pointers(published),st,mp,rp,tp,ip,count,vec,blocks))
                        elif args.gather=='write':check(peer.peer_execute_mode(cp,sp,cp,st,mp,count,vec,blocks,1))
                        else:check(peer.peer_execute(cp,sp,cp,st,mp,count,vec,blocks))
                        if consumer:
                            for rank in range(ranks):
                                with torch.cuda.stream(streams[rank]):torch.bitwise_xor(published[rank][:count].view(torch.int16),0,out=consumer[rank][:count].view(torch.int16))
                    for _ in range(2):measured(baseline,reference);measured(custom,candidate)
                    mismatch=[];tail=[]
                    for rank in range(ranks):
                        with torch.cuda.device(rank):
                            expected=reference_consumer[rank] if reference_consumer else reference_out[rank] if native else reference[rank]
                            observed=consumer[rank][:count] if consumer else published[rank][:count]
                            mismatch.append(int(torch.count_nonzero(expected.view(torch.int16)!=observed.view(torch.int16)).item()))
                            tail.append(all(bool(torch.all(t[rank][-8:].view(torch.int16)==0x3535)) for t in [candidate,published,shards]+([consumer] if consumer else [])))
                    samples=[];consumer_round_exact=[]
                    for pair in range(args.pairs):
                        if pair%2:b=measured(custom,candidate);a=measured(baseline,reference)
                        else:a=measured(baseline,reference);b=measured(custom,candidate)
                        samples.append(dict(reference=a,candidate=b))
                        if consumer:
                            consumer_round_exact.append(all(bool(torch.equal(
                                reference_consumer[r].view(torch.int16),consumer[r][:count].view(torch.int16)))
                                for r in range(ranks)))
                    row=dict(diagnostic_profile_regions=args.profile_regions,consumer_round_exact=consumer_round_exact,bridge_build=str(args.bridge_build) if args.bridge_build else None,reference_peer_build=str(args.reference_peer_build) if args.reference_peer_build else None,separate_output=args.separate_output,consumer_copy=args.consumer_copy,count=count,vec=vec,blocks=blocks,gather=args.gather,case=args.case,indexed_gate=native_record,nt_build=str(args.nt_build) if args.nt_build else None,bridge_stats=json.loads(peer.h3_peer_bridge_stats()) if args.bridge_build else None,fixed_order=args.fixed_order,seed=args.seed,order_model=dict(channels=args.channels,chunk=args.chunk,alignment=args.alignment),mismatches=mismatch,tail_ok=tail,exact=not any(mismatch) and all(tail) and all(consumer_round_exact),samples=samples,reference_median_ms=statistics.median(s['reference']['gpu_ms'] for s in samples),candidate_median_ms=statistics.median(s['candidate']['gpu_ms'] for s in samples),paired_gain_percent=statistics.median(100*(1-s['candidate']['gpu_ms']/s['reference']['gpu_ms']) for s in samples))
                    records.append(row);args.output.write_text(json.dumps(dict(scope='无profiler八卡分片门控微测；对照由reference_peer_build明确，独立输出及可选GPU消费者；不是模型成绩',records=records),ensure_ascii=False,indent=2)+'\n');print(json.dumps(row),flush=True)
            del sources,reference,candidate,published,consumer,reference_consumer,shards,maps,hostmap,x,residuals,tables,indices,reference_out
            for rank in range(ranks):
                with torch.cuda.device(rank):torch.cuda.empty_cache()
    finally:
        check(peer.peer_cleanup())
        if reference_peer:check(reference_peer.peer_cleanup())
        for rank,comm in enumerate(comms):
            with torch.cuda.device(rank):lib.ncclCommDestroy(comm)


if __name__=='__main__':main()
