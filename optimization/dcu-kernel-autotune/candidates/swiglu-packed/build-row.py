#!/usr/bin/env python3
"""按 packed 行对齐 SwiGLU 九齿微测，减少线性索引中的除余。"""
import argparse,json,os,sys
from pathlib import Path
import ninetoothed as nt
import ninetoothed.language as ntl
sys.path.insert(0,str(Path(os.environ['H3_AUTOTUNE_CODE_ROOT'])/'optimization/adaln'))
from aot_codegen import lower_kernel,compile_image


def application(up,gate,output):
    g=gate.to(ntl.float32)
    sig=ntl.libdevice.fast_dividef(1.0,1.0+ntl.libdevice.fast_expf(-g)).to(output.dtype).to(ntl.float32)
    output=(g*sig)*up.to(ntl.float32)


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('output',type=Path);cli.add_argument('--tile',type=int,choices=(256,512,1024,2048),required=True);cli.add_argument('--warps',type=int,choices=(2,4,8),default=4);a=cli.parse_args();a.output.mkdir()
    def arrange(up,gate,output):
        def tile(x):
            result=x.tile((1,a.tile));result.dtype=result.dtype.squeeze(0);return result
        return tile(up),tile(gate),tile(output)
    up=nt.Tensor(shape=(None,1792),dtype=nt.bfloat16)
    tensors=(up,nt.Tensor(shape=up.shape,dtype=nt.bfloat16),nt.Tensor(shape=up.shape,dtype=nt.bfloat16))
    source,kernel,bindings=lower_kernel(a.output,arrange,application,tensors,'wide_norm_indexed',a.warps)
    signature,constants,runtime,arguments={},{},[],[]
    for b in bindings:
        if b.kind=='tensor':signature[b.name]='*bf16';arguments.append(('ptr',b.source))
        elif b.kind=='shape' and b.dim==0:signature[b.name]='i32';arguments.append(('i32','input_rows'))
        elif b.kind=='stride':signature[b.name]='constexpr';constants[b.name]=(3584 if b.source in ('up','gate') else 1792) if b.dim==0 else 1
        else:raise RuntimeError(f'不支持绑定 {b}')
        if signature[b.name]!='constexpr':runtime.append(signature[b.name])
    assert len(arguments)==4,arguments
    digest,shared=compile_image(a.output/'wide-norm-indexed.hsaco',kernel,signature,constants,runtime,a.warps,block=a.tile,allow_shared=True)
    (a.output/'build.json').write_text(json.dumps(dict(arguments=arguments,warps=a.warps,tile=a.tile,shared=shared,hsaco_sha256=digest,source=str(source),mode='swiglu-row',row_tiles=(1792+a.tile-1)//a.tile,math='fast-rcp'),indent=2)+'\n')
if __name__=='__main__':main()
