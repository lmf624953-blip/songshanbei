#!/usr/bin/env python3
"""构建已逐位校验的 packed BF16 SwiGLU 九齿 AOT 与主机桥。"""
import argparse,json
from pathlib import Path
import ninetoothed as nt
from aot_codegen import lower_kernel,compile_image,compile_host
from swiglu_kernel import application


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    args=cli.parse_args();args.output.mkdir()
    tile,warps=1024,4
    def arrange(up,gate,output):
        return up.tile((1,1792)),gate.tile((1,1792)),output.tile((tile,))
    up=nt.Tensor(shape=(None,1792),dtype=nt.bfloat16)
    tensors=(up,nt.Tensor(shape=up.shape,dtype=nt.bfloat16),nt.Tensor(shape=(None,),dtype=nt.bfloat16))
    source,kernel,bindings=lower_kernel(args.output,arrange,application,tensors,'swiglu',warps)
    signature,constants,runtime,arguments={},{},[],[]
    for b in bindings:
        if b.kind=='tensor':
            signature[b.name]='*bf16';arguments.append(('ptr',b.source))
        elif b.kind=='shape' and b.dim==0:
            signature[b.name]='i32';arguments.append(('i32','input_rows' if b.source=='up' else 'elements'))
        elif b.kind=='stride':
            signature[b.name]='constexpr'
            constants[b.name]=3584 if b.dim==0 and b.source in ('up','gate') else 1
        else:raise RuntimeError(f'不支持绑定 {b}')
        if signature[b.name]!='constexpr':runtime.append(signature[b.name])
    expected=[('ptr','up'),('ptr','gate'),('ptr','output'),('i32','input_rows'),('i32','elements')]
    if arguments!=expected:raise RuntimeError(f'非预期 SwiGLU ABI: {arguments}')
    digest,shared=compile_image(args.output/'swiglu.hsaco',kernel,signature,constants,runtime,warps,block=tile,allow_shared=True)
    command=compile_host(args.output/'swiglu-host.so',('swiglu_host.cc',))
    (args.output/'swiglu-build.json').write_text(json.dumps(dict(arguments=arguments,warps=warps,tile=tile,shared=shared,hsaco_sha256=digest,source=str(source),math='native-fast-exp-rcp-bf16-sigmoid',host_command=command),indent=2)+'\n')
if __name__=='__main__':main()
