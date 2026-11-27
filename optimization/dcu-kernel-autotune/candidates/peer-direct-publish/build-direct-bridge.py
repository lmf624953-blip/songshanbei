#!/usr/bin/env python3
"""编译通信适配和主机桥，固定九齿镜像ABI与来源；不替换正式库。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('nt_build',type=Path);cli.add_argument('output',type=Path)
    cli.add_argument('--rotate-outputs',action='store_true',help='按rank轮转发布顺序以分散同时访问目的卡')
    cli.add_argument('--owner-partition',action='store_true',help='按RCCL owner分工，预排输入指针并发射二维channel网格')
    cli.add_argument('--reuse-runtime',type=Path,help='表达/ABI一致时复用已编译桥及transport，只替换九齿几何')
    args=cli.parse_args()
    args.output.mkdir();here=Path(__file__).resolve().parent
    meta=json.loads((args.nt_build/'peer-gate-build.json').read_text());assert hashlib.sha256((args.nt_build/'peer-gate.hsaco').read_bytes()).hexdigest()==meta['hsaco_sha256']
    for p in args.nt_build.iterdir():
        if p.is_file():shutil.copy2(p,args.output/p.name)
    command=['hipcc','-O3','-std=c++17','--offload-arch=gfx936','-shared','-fPIC',str(here/'direct_transport.hip'),'-o',str(args.output/'peer-transport.so')]
    sys.path.insert(0,str(Path(os.environ['H3_AUTOTUNE_CODE_ROOT'])/'optimization/adaln'))
    from aot_codegen import compile_host
    source=(here/'direct_bridge.cc').read_text()
    assert bool(meta.get('owner_partition'))==args.owner_partition
    if args.owner_partition:
        assert meta['order_model_constants']==dict(channels=16,chunk=524288,alignment=1024)
        source=source.replace('if(code<8)value=input[code];',
            'if(code<8)value=input[(e.ringbits>>(3*((r+code+1)&7)))&7];')
        source=source.replace('+uint64_t(r)*(count/8)*2','')
        source=source.replace('if(!e.ok){','if(!e.ok||e.channels!=16||e.chunk!=524288||e.alignment!=1024){')
        source=source.replace('  else{\n   for(int r=0;r<8;++r)',
            '  else{\n   size_t length=count/16,full=length/4194304,rem=length%4194304;size_t owner_capacity=full*524288+(((rem+7)/8+1023)/1024)*1024;\n   for(int r=0;r<8;++r)')
        source=source.replace('case 103:value=count/5376;break;',
            'case 103:value=count/5376;break;case 107:value=count/16;break;')
        source=source.replace('unsigned((count/8+tile-1)/tile),1,1,warps*64',
            '16*unsigned((owner_capacity+tile-1)/tile),1,1,warps*64')
        assert 'owner_capacity' in source and 'case 107:' in source
    if args.rotate_outputs:
        assert source.count('out[code-20]')==1
        source=source.replace('out[code-20]','out[(code-20+r)&7]')
    generated=args.output/'direct_bridge_generated.cc';generated.write_text(source)
    if args.reuse_runtime:
        prior=json.loads((args.reuse_runtime/'peer-runtime-build.json').read_text())
        assert prior.get('rotate_outputs',False)==args.rotate_outputs
        assert prior.get('owner_partition',False)==args.owner_partition
        assert prior['source_sha256']['direct_bridge.cc']==hashlib.sha256((here/'direct_bridge.cc').read_bytes()).hexdigest()
        assert prior['source_sha256']['direct_transport.hip']==hashlib.sha256((here/'direct_transport.hip').read_bytes()).hexdigest()
        old_generated=args.reuse_runtime/'direct_bridge_generated.cc'
        assert (old_generated.read_text() if old_generated.exists() else (here/'direct_bridge.cc').read_text())==source
        for name in ('peer-gate-bridge.so','peer-transport.so'):shutil.copy2(args.reuse_runtime/name,args.output/name)
        command,host=prior['transport_command'],prior['bridge_command']
    else:
        subprocess.run(command,check=True)
        host=compile_host(args.output/'peer-gate-bridge.so',[generated])
    codes=[]
    tensors={f'p{i}':i for i in range(8)}|dict(owners=8,residual=9,gate=10,indices=11,**{f'out{i}':20+i for i in range(8)})
    for arg in meta['arguments']:
        source=arg['source']
        if arg['kind']=='tensor':codes.append(tensors[source])
        elif arg['kind']=='scalar':codes.append(dict(rank=12,ringbits=13)[source])
        elif arg['kind']=='shape':codes.append(107 if args.owner_partition and source.startswith('out') else 101 if source=='owners' or source.startswith('out') else 102 if source=='gate' else 103 if source=='indices' else 100)
        elif arg['kind']=='stride':
            assert args.owner_partition and source.startswith('out') and arg['dim']==0
            codes.append(107)
        else:raise ValueError(arg)
    if args.reuse_runtime:assert codes==prior['abi_codes']
    record=dict(direct_publish=True,owner_partition=args.owner_partition,rotate_outputs=args.rotate_outputs,reused_runtime=str(args.reuse_runtime) if args.reuse_runtime else None,nt=meta,abi_codes=codes,transport_command=command,bridge_command=host,
                sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.is_file()},
                source_sha256={name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in ['direct_transport.hip','direct_bridge.cc','direct_kernel.py']})
    if args.owner_partition:record['source_sha256']['owner_kernel.py']=hashlib.sha256((here/'owner_kernel.py').read_bytes()).hexdigest()
    (args.output/'peer-runtime-build.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
