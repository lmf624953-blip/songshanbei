#!/usr/bin/env python3
"""以重复BF16探针识别采样位置的rank累加链；只是诊断，不能替代完整顺序证明。"""
import argparse
import ctypes as ct
import importlib.util
import itertools
import json
from pathlib import Path
import numpy as np
import torch


def bf16_rne(x):
    bits=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype(np.uint16)


def f32(bits):return (bits.astype(np.uint32)<<16).view(np.float32)


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('library',type=Path);cli.add_argument('output',type=Path)
    cli.add_argument('--count',type=int,default=148216320);cli.add_argument('--patterns',type=int,default=32)
    cli.add_argument('--seed',type=int,default=82371)
    cli.add_argument('--samples',type=int,default=65536);args=cli.parse_args()
    assert args.count>0 and args.count%args.patterns==0 and args.patterns>=16
    args.output.mkdir();ranks=8;assert torch.cuda.device_count()==ranks
    helper=Path(__file__).resolve().parent.parent/'rccl-allreduce/bench-allreduce.py'
    spec=importlib.util.spec_from_file_location('rccl_helper',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    lib=module.load_rccl(args.library.resolve(strict=True));streams=[torch.cuda.Stream(device=i) for i in range(ranks)]
    comms=(ct.c_void_p*ranks)();devices=(ct.c_int*ranks)(*range(ranks));lib.ncclCommInitAll(comms,ranks,devices)
    patterns=(torch.randn((ranks,args.patterns),generator=torch.Generator().manual_seed(args.seed))*torch.tensor([2**(i%4) for i in range(ranks)])[:,None]).to(torch.bfloat16)
    raw_patterns=patterns.view(torch.int16).numpy().view(np.uint16)
    groups=args.count//args.patterns
    sample_ids=np.unique(np.concatenate([np.linspace(0,groups-1,min(args.samples,groups),dtype=np.int64),np.arange(min(4096,groups)),np.arange(max(0,groups-4096),groups),np.clip(np.arange(1025)[:,None]*groups//1024+np.arange(-16,17)[None,:],0,groups-1).ravel()]))
    inputs=[]
    try:
        for rank in range(ranks):
            with torch.cuda.device(rank):
                x=patterns[rank].to(rank).repeat(groups);streams[rank].wait_stream(torch.cuda.current_stream(rank));inputs.append(x)
        for s in streams:s.synchronize()
        lib.ncclGroupStart()
        for rank in range(ranks):
            with torch.cuda.device(rank):lib.ncclAllReduce(inputs[rank].data_ptr(),inputs[rank].data_ptr(),args.count,9,0,comms[rank],streams[rank].cuda_stream)
        lib.ncclGroupEnd()
        for s in streams:s.synchronize()
        with torch.cuda.device(0):
            selected=inputs[0].view(groups,args.patterns)[torch.from_numpy(sample_ids).to('cuda:0')].view(torch.int16).cpu().numpy().view(np.uint16)
        np.savez_compressed(args.output/'samples.npz',raw_patterns=raw_patterns,group_ids=sample_ids,output=selected)
    finally:
        for rank,comm in enumerate(comms):
            with torch.cuda.device(rank):lib.ncclCommDestroy(comm)
    # 第一对加法交换可等价；保留全部匹配排列，不把一个匹配排列当唯一实际指令序。
    orders=np.array(list(itertools.permutations(range(ranks))),dtype=np.int16)
    acc=raw_patterns[orders[:,0]]
    for step in range(1,ranks):acc=bf16_rne(f32(acc)+f32(raw_patterns[orders[:,step]]))
    unique,counts=np.unique(selected,axis=0,return_counts=True)
    results=[]
    for signature,count in zip(unique,counts):
        matched=np.flatnonzero(np.all(acc==signature,axis=1))
        locations=sample_ids[np.all(selected==signature,axis=1)]
        results.append(dict(sampled_groups=int(count),first_group=int(locations[0]),last_group=int(locations[-1]),
                            matching_permutations=orders[matched].tolist(),matches=int(matched.size),signature=[int(v) for v in signature]))
    record=dict(count=args.count,patterns=args.patterns,seed=args.seed,sampled_groups=int(len(sample_ids)),distinct_signatures=len(results),
                completely_matched_groups=sum(r['sampled_groups'] for r in results if r['matches']),
                scope='只识别采样位置在这些探针上的BF16逐步RNE串行排列等价类；边界可能混合多条链，不能当完整索引映射证明',signatures=results)
    (args.output/'inference.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='signatures'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
