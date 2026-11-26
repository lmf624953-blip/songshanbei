#!/usr/bin/env python3
"""构建已逐位校验的 packed BF16 SwiGLU 九齿 AOT 与主机桥。"""
import argparse,json
from pathlib import Path
import ninetoothed as nt
from aot_codegen import lower_kernel,compile_image,compile_host
from swiglu_kernel import application,application_row


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    cli.add_argument('--mode',choices=('linear','row'),default='linear')
    args=cli.parse_args();args.output.mkdir()
    tile,warps=(2048,8) if args.mode=='row' else (1024,4)
    def arrange(up,gate,output):
        if args.mode=='linear':
            return up.tile((1,1792)),gate.tile((1,1792)),output.tile((tile,))
        def tiled(tensor):
            result=tensor.tile((1,tile));result.dtype=result.dtype.squeeze(0);return result
        return tiled(up),tiled(gate),tiled(output)
    up=nt.Tensor(shape=(None,1792),dtype=nt.bfloat16)
    tensors=(up,nt.Tensor(shape=up.shape,dtype=nt.bfloat16),nt.Tensor(shape=(up.shape if args.mode=='row' else (None,)),dtype=nt.bfloat16))
    source,kernel,bindings=lower_kernel(args.output,arrange,(application_row if args.mode=='row' else application),tensors,'swiglu',warps)
    signature,constants,runtime,arguments={},{},[],[]
    for b in bindings:
        if b.kind=='tensor':
            signature[b.name]='*bf16';arguments.append(('ptr',b.source))
        elif b.kind=='shape' and b.dim==0:
            signature[b.name]='i32';arguments.append(('i32','input_rows' if b.source=='up' else 'elements'))
        elif b.kind=='stride':
            signature[b.name]='constexpr'
            constants[b.name]=((3584 if b.source in ('up','gate') else 1792) if b.dim==0 else 1) if args.mode=='row' else (3584 if b.dim==0 and b.source in ('up','gate') else 1)
        else:raise RuntimeError(f'不支持绑定 {b}')
        if signature[b.name]!='constexpr':runtime.append(signature[b.name])
    expected=[('ptr','up'),('ptr','gate'),('ptr','output'),('i32','input_rows'),('i32','elements')]
    if args.mode=='row':expected=expected[:-1]
    if arguments!=expected:raise RuntimeError(f'非预期 SwiGLU ABI: {arguments}')
    digest,shared=compile_image(args.output/'swiglu.hsaco',kernel,signature,constants,runtime,warps,block=tile,allow_shared=True)
    command=compile_host(args.output/'swiglu-host.so',('swiglu_host.cc',),defines=(('H3_SWIGLU_ROW',int(args.mode=='row')),))
    (args.output/'swiglu-build.json').write_text(json.dumps(dict(mode=args.mode,arguments=arguments,warps=warps,tile=tile,shared=shared,hsaco_sha256=digest,source=str(source),math='native-fast-exp-rcp-bf16-sigmoid',host_command=command),indent=2)+'\n')
if __name__=='__main__':main()
