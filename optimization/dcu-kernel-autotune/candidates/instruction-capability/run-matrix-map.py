#!/usr/bin/env python3
"""检查真实设备的LDS矩阵读取地址映射；不作吞吐比较。"""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

import torch


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('build',type=Path)
    cli.add_argument('output',type=Path)
    args=cli.parse_args();meta=json.loads((args.build/'build.json').read_text())
    hip=ct.CDLL('libamdhip64.so')
    hip.hipModuleLoad.argtypes=[ct.POINTER(ct.c_void_p),ct.c_char_p]
    hip.hipModuleGetFunction.argtypes=[ct.POINTER(ct.c_void_p),ct.c_void_p,ct.c_char_p]
    hip.hipModuleLaunchKernel.argtypes=[ct.c_void_p]+[ct.c_uint]*7+[ct.c_void_p]*3
    hip.hipModuleUnload.argtypes=[ct.c_void_p]
    def check(status):
        if status:raise RuntimeError(f'HIP status {status}')
    torch.cuda.init()
    tags=torch.arange(32768,dtype=torch.int32,device='cuda').to(torch.int16)
    storage=torch.full((meta['output_words']+16,),-559038737,dtype=torch.int32,device='cuda')
    stream=torch.cuda.current_stream().cuda_stream
    records=[]
    with args.output.open('x')as destination:
        for variant in meta['variants']:
            image=args.build/variant['image']
            assert hashlib.sha256(image.read_bytes()).hexdigest()==variant['sha256']
            module,function=ct.c_void_p(),ct.c_void_p()
            check(hip.hipModuleLoad(ct.byref(module),str(image).encode()))
            check(hip.hipModuleGetFunction(ct.byref(function),module,meta['symbol'].encode()))
            values=[ct.c_uint64(0)for _ in range(15)]
            values[0].value=tags.data_ptr();values[4].value=storage.data_ptr()
            parameters=(ct.c_void_p*15)(*(ct.addressof(x)for x in values))
            repeats=[]
            for _ in range(2):
                storage.fill_(-559038737)
                check(hip.hipModuleLaunchKernel(function,1,1,1,meta['threads'],1,1,
                    meta['shared_bytes'],stream,parameters,None))
                torch.cuda.synchronize()
                assert bool((storage[meta['output_words']:] == -559038737).all())
                raw=storage[:meta['output_words']].view(torch.int16).view(256,8).to(torch.int32).cpu()
                assert bool(((raw>=0)&(raw<32768)).all()),'读取超出已知标签范围'
                repeats.append(raw)
            assert torch.equal(*repeats)
            raw=repeats[0]
            if variant['opcode']=='ds_read_b128':
                stride={'zero':0,'stride16':16,'stride512':512}[variant['pattern']]
                expected=(torch.arange(256)%64)[:,None]*stride//2+torch.arange(8)[None,:]
                assert torch.equal(raw,expected)
            row=dict(**variant,guard_ok=True,repeat_exact=True,all_tags_in_range=True,
                waves_equal=all(torch.equal(raw[:64],raw[i:i+64])for i in (64,128,192)),values=raw.tolist())
            records.append(row);destination.write(json.dumps(row)+'\n');destination.flush()
            print(json.dumps({k:v for k,v in row.items()if k!='values'}),flush=True)
            check(hip.hipModuleUnload(module))
    assert torch.equal(tags.cpu().to(torch.int32),torch.arange(32768,dtype=torch.int32))
    (args.output.parent/(args.output.stem+'-summary.json')).write_text(json.dumps(dict(
        scope='真实设备独立内存布局诊断，不是模型数值/性能成绩',input_unchanged=True,records=records),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
