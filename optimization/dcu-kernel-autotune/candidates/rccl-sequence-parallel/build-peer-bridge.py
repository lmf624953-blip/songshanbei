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
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('nt_build',type=Path);cli.add_argument('output',type=Path);args=cli.parse_args()
    args.output.mkdir();here=Path(__file__).resolve().parent
    meta=json.loads((args.nt_build/'peer-gate-build.json').read_text());assert hashlib.sha256((args.nt_build/'peer-gate.hsaco').read_bytes()).hexdigest()==meta['hsaco_sha256']
    for p in args.nt_build.iterdir():
        if p.is_file():shutil.copy2(p,args.output/p.name)
    command=['hipcc','-O3','-std=c++17','--offload-arch=gfx936','-shared','-fPIC',str(here/'peer_transport.hip'),'-o',str(args.output/'peer-transport.so')]
    subprocess.run(command,check=True)
    sys.path.insert(0,str(Path(os.environ['H3_AUTOTUNE_CODE_ROOT'])/'optimization/adaln'))
    from aot_codegen import compile_host
    host=compile_host(args.output/'peer-gate-bridge.so',[here/'peer_gate_bridge.cc'])
    codes=[]
    tensors={f'p{i}':i for i in range(8)}|dict(owners=8,residual=9,gate=10,indices=11,output=14)
    for arg in meta['arguments']:
        source=arg['source']
        if arg['kind']=='tensor':codes.append(tensors[source])
        elif arg['kind']=='scalar':codes.append(dict(rank=12,ringbits=13)[source])
        elif arg['kind']=='shape':codes.append(101 if source in ('owners','output') else 102 if source=='gate' else 103 if source=='indices' else 100)
        else:raise ValueError(arg)
    record=dict(nt=meta,abi_codes=codes,transport_command=command,bridge_command=host,
                sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.is_file()},
                source_sha256={name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in ['peer_transport.hip','peer_gate_bridge.cc','peer_nt_kernel.py','nt_bitcast_compat.py']})
    (args.output/'peer-runtime-build.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
