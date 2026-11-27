#!/usr/bin/env python3
"""保存九齿表达、lowered源码、IR与ABI，构建分片门控镜像。"""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
import ninetoothed as nt
from peer_nt_kernel import arrangement,application
from nt_bitcast_compat import enable


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('output',type=Path);cli.add_argument('--tile',type=int,default=256);cli.add_argument('--warps',type=int,default=4);args=cli.parse_args()
    sys.path.insert(0,str(Path(os.environ['H3_AUTOTUNE_CODE_ROOT'])/'optimization/adaln'))
    from aot_codegen import lower_kernel,compile_image
    args.output.mkdir()
    compat=enable()
    tensors=tuple(nt.Tensor(1,dtype=nt.bfloat16) for _ in range(8))+(nt.Tensor(1,dtype=nt.uint8),nt.Tensor(1,dtype=nt.bfloat16),nt.Tensor(2,dtype=nt.bfloat16),nt.Tensor(1,dtype=nt.int64),nt.Tensor(0,dtype=nt.int32),nt.Tensor(0,dtype=nt.int32),nt.Tensor(1,dtype=nt.bfloat16))
    source,kernel,bindings=lower_kernel(args.output,partial(arrangement,tile=args.tile),application,tensors,'h3_peer_gate',args.warps)
    signature,constants,runtime,arguments={},{},[],[]
    for binding in bindings:
        if binding.kind=='tensor':
            dtype='*u8' if binding.source=='owners' else '*i64' if binding.source=='indices' else '*bf16'
            signature[binding.name]=dtype;runtime.append(dtype);arguments.append(dict(kind='tensor',source=binding.source))
        elif binding.kind in ('shape','scalar'):
            if binding.kind=='shape' and binding.source=='gate' and binding.dim==1:
                signature[binding.name]='constexpr';constants[binding.name]=5376
            else:
                signature[binding.name]='i32';runtime.append('i32');arguments.append(dict(kind=binding.kind,source=binding.source,dim=binding.dim))
        elif binding.kind=='stride':
            signature[binding.name]='constexpr';constants[binding.name]=32256 if binding.source=='gate' and binding.dim==0 else 1
        else:raise ValueError(binding)
    os.environ['H3_AOT_SAVE_IR']='1'
    sha,shared=compile_image(args.output/'peer-gate.hsaco',kernel,signature,constants,runtime,args.warps,block=args.tile)
    record=dict(bitcast_compat=compat,hsaco_sha256=sha,warps=args.warps,tile=args.tile,shared=shared,arguments=arguments,numerical_verified=False,source_sha256=hashlib.sha256(Path(__file__).with_name('peer_nt_kernel.py').read_bytes()).hexdigest(),lowered=str(source))
    (args.output/'peer-gate-build.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
