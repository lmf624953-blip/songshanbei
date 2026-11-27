#!/usr/bin/env python3
"""同进程比较同源双Q128无scratch候选与正式版本。"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import torch


def launcher(build, reference, fallback):
    meta=json.loads((build/'dual-build.json').read_text())
    assert hashlib.sha256((build/'dual.hsaco').read_bytes()).hexdigest()==meta['hsaco_sha256']
    host=ctypes.CDLL(str(build/'dual-host.so'))
    prepare=host.prepare_dual;prepare.argtypes=[ctypes.c_char_p]*2;prepare.restype=ctypes.c_char_p
    error=prepare(os.fsencode(build/'dual.hsaco'),os.fsencode(reference/'prepack-value.hsaco'))
    if error:raise RuntimeError(error.decode())
    run=host.bench_dual;run.argtypes=[ctypes.c_int,ctypes.c_float]+[ctypes.c_void_p]*5+[ctypes.c_int]*3+[ctypes.c_size_t];run.restype=ctypes.c_char_p
    pack=json.loads((reference/'dense-attention-aot-build.json').read_text())['prepacked_value_dispatch']['prepack']
    buffers={}
    def call(q,k,v,scale,out):
        n=q.shape[2]
        if n<4096:return fallback(q,k,v,scale,out)
        assert [x.stride(2) for x in (q,k,v,out)]==[896,896,2688,896]
        key=(n,q.device.index)
        if key not in buffers:buffers[key]=torch.empty(7*(n//32)*4096,device=q.device,dtype=torch.bfloat16)
        error=run(n,scale,*(x.data_ptr() for x in (q,k,v,out,buffers[key])),pack['tile'],pack['warps']*64,
                  int(meta['warps']==8),torch.cuda.current_stream(q.device).cuda_stream)
        if error:raise RuntimeError(error.decode())
    return call


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('reference',type=Path);cli.add_argument('builds',type=Path);cli.add_argument('output',type=Path)
    cli.add_argument('--variants',nargs='+',default=['single-resource','independent','shared']);cli.add_argument('--lengths',nargs='+',type=int,default=[4096,4097,8193,22016,27570]);cli.add_argument('--pairs',type=int,default=19)
    cli.add_argument('--compare-independent',action='store_true');args=cli.parse_args()
    code=Path(os.environ['H3_AUTOTUNE_CODE_ROOT'])/'optimization';sys.path.insert(0,str(code))
    baseline=runpy.run_path(str(code/'bench-attention.py'))['native_launcher'](args.reference)
    fallback=baseline
    if args.compare_independent:baseline=launcher(args.builds/'independent',args.reference,fallback)
    screen=runpy.run_path(str(Path(__file__).with_name('screen.py')))['screen']
    with args.output.open('x') as output:
        for name in args.variants:
            candidate=launcher(args.builds/name,args.reference,fallback)
            for length in args.lengths:
                result={'variant':name,'reference_variant':'independent' if args.compare_independent else 'selected',**screen(baseline,candidate,length,args.pairs)}
                output.write(json.dumps(result)+'\n');output.flush()
                print(json.dumps({k:v for k,v in result.items() if k!='samples_ms'}),flush=True)
                if not result['passed']:raise RuntimeError(result)


if __name__=='__main__':main()
