#!/usr/bin/env python3
"""不用完整权重的八卡模型桥闸门：真实Tensor接口、分片残差、物化与关闭切换。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'rccl-sequence-parallel'))
from bridge_micro import InfiniCCLProxy,install_adapter


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('build',type=Path);cli.add_argument('output',type=Path)
    cli.add_argument('--reference',type=Path,required=True);cli.add_argument('--indexed',type=Path,required=True)
    cli.add_argument('--norm',type=Path,required=True);cli.add_argument('--rows',type=int,nargs='+',default=[27570,22899,24062])
    cli.add_argument('--mismatch-weight',action='store_true')
    cli.add_argument('--test-abort',action='store_true',help='正向检查后中断仅7个rank到达的gate，验证等待唤醒与拒绝继续使用')
    args=cli.parse_args();assert torch.cuda.device_count()==8
    proxy=InfiniCCLProxy(Path(os.environ['INFINI_ROOT'])/'lib/libinfiniccl.so')
    comms=(ct.c_void_p*8)();proxy.ncclCommInitAll(comms,8,(ct.c_int*8)(*range(8)))
    indexed=ct.CDLL(str(args.indexed/'indexed-host.so'));meta=json.loads((args.indexed/'indexed-build.json').read_text())
    indexed.songshan_prepare_indexed.argtypes=[ct.c_char_p,ct.c_char_p,ct.c_int];indexed.songshan_prepare_indexed.restype=ct.c_char_p
    def check(message):
        if message:raise RuntimeError(message.decode())
    check(indexed.songshan_prepare_indexed(str(args.indexed/'indexed_adaln.hsaco').encode(),str(args.indexed/'indexed_gate.hsaco').encode(),meta['indexed_gate']['warps']))
    reference=ct.CDLL(str(args.reference/'peer-gate-bridge.so'));install_adapter(reference,args.reference,args.indexed,proxy);reference.peer_setup()
    norm=ct.CDLL(str(args.norm/'wide-norm-indexed-host.so'));nmeta=json.loads((args.norm/'wide-norm-indexed-build.json').read_text())
    norm.songshan_prepare_widenorm_indexed.argtypes=[ct.c_char_p,ct.c_int,ct.c_int];norm.songshan_prepare_widenorm_indexed.restype=ct.c_char_p
    check(norm.songshan_prepare_widenorm_indexed(str(args.norm/'wide-norm-indexed.hsaco').encode(),nmeta['warps'],nmeta['shared']))
    norm.songshan_bench_widenorm_indexed.argtypes=[ct.c_int]*3+[ct.c_void_p]*6+[ct.c_uint64];norm.songshan_bench_widenorm_indexed.restype=ct.c_char_p
    model=ct.CDLL(str(args.build/'peer-gate-bridge.so'));bmeta=json.loads((args.build/'peer-runtime-build.json').read_text())
    codes=(ct.c_int*len(bmeta['abi_codes']))(*bmeta['abi_codes'])
    model.h3_peer_bridge_prepare.argtypes=[ct.c_char_p,ct.c_int,ct.c_int,ct.POINTER(ct.c_int),ct.c_int,ct.c_char_p,ct.c_void_p];model.h3_peer_bridge_prepare.restype=ct.c_char_p
    check(model.h3_peer_bridge_prepare(str(args.build/'peer-gate.hsaco').encode(),bmeta['nt']['tile'],bmeta['nt']['warps'],codes,len(codes),str(args.build/'peer-transport.so').encode(),ct.cast(indexed.songshan_bench_indexed,ct.c_void_p)))
    model.h3_model_prepare.argtypes=[ct.c_void_p]*2;model.h3_model_prepare.restype=ct.c_char_p
    check(model.h3_model_prepare(ct.cast(norm.songshan_bench_widenorm_indexed,ct.c_void_p),ct.cast(reference.songshan_launch_peer_gate,ct.c_void_p)))
    model.h3_model_set_enabled.argtypes=[ct.c_int];model.h3_model_set_enabled.restype=ct.c_char_p
    model.h3_model_gate_raw.argtypes=[ct.c_int]+[ct.c_uint64]*5+[ct.c_size_t,ct.c_int,ct.c_uint64,ct.c_uint64];model.h3_model_gate_raw.restype=ct.c_char_p
    model.h3_model_norm_raw.argtypes=[ct.c_int,ct.c_size_t,ct.c_int]+[ct.c_uint64]*7;model.h3_model_norm_raw.restype=ct.c_char_p
    model.h3_model_materialize_raw.argtypes=[ct.c_int,ct.c_size_t]+[ct.c_uint64]*3;model.h3_model_materialize_raw.restype=ct.c_char_p
    model.h3_peer_bridge_stats.restype=ct.c_char_p
    streams=[torch.cuda.Stream(device=r) for r in range(8)];records=[];poison=0x3535
    def sync():
        for stream in streams:stream.synchronize()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for rows in args.rows:
                count=rows*5376;source=[];work=[];tables=[];indices=[];weights=[];initial=[]
                for r in range(8):
                    with torch.cuda.device(r):
                        g=torch.Generator(device=r).manual_seed(92003+r)
                        source.append(torch.randn(count,device=r,dtype=torch.bfloat16,generator=g));work.append(torch.empty_like(source[-1]))
                        g.manual_seed(29101);initial.append(torch.randn(count,device=r,dtype=torch.bfloat16,generator=g));tables.append(torch.randn(3,6*5376,device=r,dtype=torch.bfloat16,generator=g));weights.append(torch.randn(5376,device=r,dtype=torch.bfloat16,generator=g));indices.append(torch.arange(rows,device=r,dtype=torch.int64)%3)
                        if args.mismatch_weight and r==7:weights[-1][0]=weights[-1][0]+1
                        streams[r].wait_stream(torch.cuda.current_stream(r))
                sync()
                for enabled in [1,0,1]:
                    check(model.h3_model_set_enabled(enabled));by=initial;cy=initial
                    for step in range(3):
                        old_by,old_cy=by,cy;by=[];bz=[];cy=[];cz=[]
                        for r in range(8):
                            with torch.cuda.device(r),torch.cuda.stream(streams[r]):
                                for buffers in [by,bz,cy,cz]:
                                    buffers.append(torch.empty(count+8,device=r,dtype=torch.bfloat16));buffers[-1].view(torch.int16).fill_(poison)
                                work[r].copy_(source[r])
                        sync()
                        def ref(r):
                            with torch.cuda.device(r):
                                table=tables[r].data_ptr();stream=streams[r].cuda_stream
                                check(reference.h3_peer_bridge_raw(r,work[r].data_ptr(),by[r].data_ptr(),old_by[r].data_ptr(),table+2*5376*2,indices[r].data_ptr(),count,3,comms[r],stream))
                                check(norm.songshan_bench_widenorm_indexed(r,rows,3,by[r].data_ptr(),weights[r].data_ptr(),table+4*5376*2,table+3*5376*2,indices[r].data_ptr(),bz[r].data_ptr(),stream))
                        list(pool.map(ref,range(8)));sync()
                        for r in range(8):
                            with torch.cuda.stream(streams[r]):work[r].copy_(source[r])
                        sync()
                        def cand(r):
                            with torch.cuda.device(r):
                                table=tables[r].data_ptr();stream=streams[r].cuda_stream
                                check(model.h3_model_gate_raw(r,work[r].data_ptr(),cy[r].data_ptr(),old_cy[r].data_ptr(),table+2*5376*2,indices[r].data_ptr(),count,3,comms[r],stream))
                                check(model.h3_model_norm_raw(r,count,3,cy[r].data_ptr(),weights[r].data_ptr(),table+4*5376*2,table+3*5376*2,indices[r].data_ptr(),cz[r].data_ptr(),stream))
                        list(pool.map(cand,range(8)));sync();state=json.loads(model.h3_peer_bridge_stats());partial=state['pending_ranks']==8;details=[]
                        for r in range(8):
                            first=rows*r//8*5376;last=rows*(r+1)//8*5376
                            with torch.cuda.device(r):
                                exact=torch.equal(bz[r][:count].view(torch.int16),cz[r][:count].view(torch.int16))
                                hidden=torch.equal(by[r][first:last].view(torch.int16),cy[r][first:last].view(torch.int16)) if partial else torch.equal(by[r][:count].view(torch.int16),cy[r][:count].view(torch.int16))
                                untouched=not partial or (bool(torch.all(cy[r][:first].view(torch.int16)==poison)) and bool(torch.all(cy[r][last:].view(torch.int16)==poison)))
                                tails=all(bool(torch.all(buffers[r][-8:].view(torch.int16)==poison)) for buffers in [by,bz,cy,cz]);assert exact and hidden and untouched and tails,(rows,enabled,step,r,exact,hidden,untouched,tails)
                                details.append(dict(rank=r,normalized_exact=exact,hidden_exact=hidden,unowned_untouched=untouched,tails=tails))
                        records.append(dict(rows=rows,enabled=enabled,step=step,partial=partial,validated=details,stats=state))
                    if partial:
                        error=model.h3_model_set_enabled(0);assert error and b'Mode change requires' in error
                    def finish(r):check(model.h3_model_materialize_raw(r,count,cy[r].data_ptr(),comms[r],streams[r].cuda_stream))
                    list(pool.map(finish,range(8)));sync()
                    assert all(torch.equal(by[r][:count].view(torch.int16),cy[r][:count].view(torch.int16)) for r in range(8))
                    assert json.loads(model.h3_peer_bridge_stats())['pending_ranks']==0
                    args.output.write_text(json.dumps(dict(scope='独立八卡模型Tensor桥闸门，尚非真实模型',mismatch_weight=args.mismatch_weight,records=records,final_materialization_exact=True),ensure_ascii=False,indent=2)+'\n')
                print(rows,'on/off/on与局部残差和最终物化通过',flush=True)
            if args.test_abort:
                def waiting(r):
                    with torch.cuda.device(r):
                        return model.h3_model_gate_raw(r,work[r].data_ptr(),cy[r].data_ptr(),old_cy[r].data_ptr(),
                            tables[r].data_ptr()+2*5376*2,indices[r].data_ptr(),count,3,comms[r],streams[r].cuda_stream)
                futures=[pool.submit(waiting,r) for r in range(7)]
                model.h3_model_abort.argtypes=[];model.h3_model_abort.restype=None
                try:
                    try:
                        futures[0].result(timeout=0.25)
                        raise AssertionError('缺少第8个rank时gate不应完成')
                    except TimeoutError:
                        pass
                    assert not json.loads(model.h3_peer_bridge_stats())['failed']
                finally:
                    model.h3_model_abort()
                errors=[future.result(timeout=5) for future in futures]
                assert all(message and b'Native forward aborted' in message for message in errors),errors
                message=model.h3_model_set_enabled(0);assert message and b'Native forward aborted' in message
                result=json.loads(args.output.read_text());result.update(abort_rejected_waiting_calls=True,
                    abort_waiting_call_errors=[x.decode() for x in errors],stats_after_abort=json.loads(model.h3_peer_bridge_stats()))
                args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    finally:
        reference.peer_cleanup()
        for r,comm in enumerate(comms):
            with torch.cuda.device(r):proxy.ncclCommDestroy(comm)


if __name__=='__main__':main()
