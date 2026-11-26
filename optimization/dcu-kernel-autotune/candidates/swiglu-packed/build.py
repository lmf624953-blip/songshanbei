#!/usr/bin/env python3
"""仅 CPU AOT 构建 SwiGLU 动态长度微测镜像，不接入正式模型。"""
import argparse,json,os,sys
from pathlib import Path
import ninetoothed as nt
sys.path.insert(0,str(Path(os.environ['H3_AUTOTUNE_CODE_ROOT'])/'optimization/adaln'))
from aot_codegen import lower_kernel,compile_image
from kernel import application,application_fast,application_fast_rcp,application_native_exp


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('output',type=Path);cli.add_argument('--tile',type=int,choices=(256,512,1024,2048,4096),required=True);cli.add_argument('--warps',type=int,choices=(2,4,8),default=4);cli.add_argument('--math',choices=('accurate','fast','fast-rcp','native-exp'),default='accurate');a=cli.parse_args();a.output.mkdir()
    def arrange(up,gate,output):return up.tile((1,1792)),gate.tile((1,1792)),output.tile((a.tile,))
    up=nt.Tensor(shape=(None,1792),dtype=nt.bfloat16);tensors=(up,nt.Tensor(shape=up.shape,dtype=nt.bfloat16),nt.Tensor(shape=(None,),dtype=nt.bfloat16))
    source,kernel,bindings=lower_kernel(a.output,arrange,{'accurate':application,'fast':application_fast,'fast-rcp':application_fast_rcp,'native-exp':application_native_exp}[a.math],tensors,'wide_norm_indexed',a.warps)
    signature,constants,runtime,args={},{},[],[]
    for b in bindings:
        if b.kind=='tensor':signature[b.name]='*bf16';args.append(('ptr',b.source))
        elif b.kind=='shape' and b.dim==0:
            signature[b.name]='i32';args.append(('i32','input_rows' if b.source=='up' else 'elements'))
        elif b.kind=='stride':signature[b.name]='constexpr';constants[b.name]=3584 if b.dim==0 and b.source in ('up','gate') else 1
        else:raise RuntimeError(f'不支持绑定 {b}')
        if signature[b.name]!='constexpr':runtime.append(signature[b.name])
    assert len(args)==5,args
    digest,shared=compile_image(a.output/'wide-norm-indexed.hsaco',kernel,signature,constants,runtime,a.warps,block=a.tile,allow_shared=True)
    (a.output/'build.json').write_text(json.dumps(dict(arguments=args,warps=a.warps,tile=a.tile,shared=shared,hsaco_sha256=digest,source=str(source),mode='swiglu-packed',math=a.math),indent=2)+'\n')
if __name__=='__main__':main()
