#!/usr/bin/env python3
"""构建 FP32 归约中间量探针；生产计算仍只使用九齿候选。"""
import argparse,hashlib,json,os,re,sys,subprocess
from pathlib import Path
import ninetoothed as nt
import triton
from triton.backends.compiler import GPUTarget
sys.path.insert(0,str(Path(os.environ['H3_AUTOTUNE_CODE_ROOT'])/'optimization/adaln'))
from aot_codegen import lower_kernel,compile_image
from kernel import arrange,naive,cub_warp,original_serial


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('output',type=Path);a=cli.parse_args();a.output.mkdir()
    compiler=Path(os.environ['DTK_ROOT'])/'bin/hipcc'
    for mode,flags in [('default',[]),('fast',['-ffast-math']),('serial',['-DPROBE_ORIGINAL_REDUCTION'])]:
        subprocess.run([str(compiler),'-std=c++17','--gcc-toolchain=/public/software/compiler/gcc-11.4.0','-shared','-fPIC','-O3','--offload-arch=gfx936',*flags,str(Path(__file__).with_name('reference_probe.hip')),'-o',str(a.output/f'reference-{mode}.so')],check=True)
    for name,app,warps in [('naive',naive,4),('cub-warp',cub_warp,16),('original-serial',original_serial,16)]:
        out=a.output/name;out.mkdir()
        h=nt.Tensor(shape=(None,5376),dtype=nt.bfloat16)
        tensors=(h,nt.Tensor(shape=(5376,),dtype=nt.bfloat16),nt.Tensor(shape=h.shape,dtype=nt.bfloat16),nt.Tensor(shape=(h.shape[0],1),dtype=nt.float32),nt.Tensor(shape=(h.shape[0],1),dtype=nt.float32))
        source,kernel,bindings=lower_kernel(out,arrange,app,tensors,'wide_norm_indexed',warps)
        signature,constants,runtime,args={},{},[],[]
        for b in bindings:
            if b.kind=='tensor':
                signature[b.name]='*fp32' if b.source in ('sums','inverse') else '*bf16';args.append(('ptr',b.source))
            elif b.kind=='shape' and b.source=='hidden' and b.dim==0:signature[b.name]='i32';args.append(('i32','rows'))
            elif b.kind=='stride':
                signature[b.name]='constexpr';constants[b.name]=(5376 if b.dim==0 else 64 if b.dim==1 else 1) if b.source=='reduction' else (5376 if b.dim==0 and b.source in ('hidden','normalized') else 1)
            else:raise RuntimeError(f'不支持绑定 {b}')
            if signature[b.name]!='constexpr':runtime.append(signature[b.name])
        assert len(args)==6,args
        digest,shared=compile_image(out/'wide-norm-indexed.hsaco',kernel,signature,constants,runtime,warps,block=8192,allow_shared=True)
        if name in ('cub-warp','original-serial'):
            f=out/'wide-norm-indexed.ttgir';s=f.read_text();old='#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [16], order = [0]}>';assert s.count(old)==1
            s=s.replace(old,old.replace('sizePerThread = [8]','sizePerThread = [1]'));f.write_text(s)
            c=triton.compile(str(f),target=GPUTarget('hip','gfx936',64),options=dict(num_warps=16,num_stages=1,matrix_instr_nonkdim=0,kpack=1));image=c.asm['hsaco'];(out/'wide-norm-indexed.hsaco').write_bytes(image);(out/'wide-norm-indexed.amdgcn').write_text(c.asm['amdgcn']);digest=hashlib.sha256(image).hexdigest();shared=c.metadata.shared
        (out/'build.json').write_text(json.dumps(dict(arguments=args,warps=warps,shared=shared,mode=name,hsaco_sha256=digest,source=str(source)),indent=2)+'\n')
        print(name,'compiled',flush=True)


if __name__=='__main__':main()
