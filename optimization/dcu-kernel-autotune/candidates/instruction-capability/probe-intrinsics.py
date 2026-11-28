#!/usr/bin/env python3
"""仅检查HCU矩阵intrinsic的目标支持；不执行被汇编器拒绝的位编码。"""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess


def module(cz):
    suffix='.cz' if cz else ''
    ctype='' if cz else ', <4 x float>'
    text='target triple = "amdgcn-amd-amdhsa"\n'
    text+=f'declare <4 x float> @llvm.hcu.mmac.f32.16x16x16.bf16{suffix}(<4 x i16>, <4 x i16>{ctype}, i1, i1)\n'
    for a,b in [(0,0),(1,0),(0,1),(1,1)]:
        name=f'probe_{int(cz)}_{a}_{b}'
        text+=f'define amdgpu_kernel void @{name}(ptr addrspace(1) %A, ptr addrspace(1) %B, ptr addrspace(1) %C) {{\n'
        text+='  %a = load <4 x i16>, ptr addrspace(1) %A, align 8\n  %b = load <4 x i16>, ptr addrspace(1) %B, align 8\n'
        if not cz:text+='  %c = load <4 x float>, ptr addrspace(1) %C, align 16\n'
        carg='' if cz else ', <4 x float> %c'
        text+=f'  %r = call <4 x float> @llvm.hcu.mmac.f32.16x16x16.bf16{suffix}(<4 x i16> %a, <4 x i16> %b{carg}, i1 {"true" if a else "false"}, i1 {"true" if b else "false"})\n'
        text+='  store <4 x float> %r, ptr addrspace(1) %C, align 16\n  ret void\n}\n'
    return text


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    cli.add_argument('--tools',type=Path,default=Path('/public/software/compiler/dtk-26.04/aillvm/bin'))
    args=cli.parse_args();args.output.mkdir();resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    rows=[]
    for cz in [False,True]:
        source=args.output/('cz.ll' if cz else 'normal.ll');source.write_text(module(cz))
        command=[str(args.tools/'llc'),'-march=amdgcn','-mcpu=gfx936','-filetype=asm',str(source),'-o',str(source.with_suffix('.s'))]
        proc=subprocess.run(command,capture_output=True,text=True,timeout=30)
        source.with_suffix('.log').write_text(proc.stdout+proc.stderr)
        row=dict(cz=cz,command=command,exit_code=proc.returncode,error=proc.stderr)
        if proc.returncode==0:
            row['matrix_assembly']=[line for line in source.with_suffix('.s').read_text().splitlines()if 'v_mmac' in line]
        rows.append(row)
    flags=[]
    for modifier in ['lit','lts','lit lts']:
        source=args.output/('flags-'+modifier.replace(' ','-')+'.s')
        source.write_text('.text\n.amdgcn_target "amdgcn-amd-amdhsa--gfx936"\nv_mmac_f32_16x16x16_bf16 v[8:11], v[4:5], v[6:7], v[8:11] '+modifier+'\ns_endpgm\n')
        cmd=[str(args.tools/'llvm-mc'),'-triple=amdgcn-amd-amdhsa','-mcpu=gfx936','-filetype=obj',str(source),'-o',str(source.with_suffix('.o'))]
        proc=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
        source.with_suffix('.log').write_text(proc.stdout+proc.stderr)
        flags.append(dict(modifier=modifier,command=cmd,exit_code=proc.returncode,error=proc.stderr))
    result=dict(scope='仅CPU编译诊断，未执行任何GPU编码',gpu_kernels_launched=0,
        compiler_sha256=hashlib.sha256((args.tools/'llc').read_bytes()).hexdigest(),intrinsics=rows,assembler=flags,
        decision='不能把llc可以打印lit/lts误当gfx936支持；汇编器明确拒绝该目标，未通过原始位补丁绕过限制')
    (args.output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(intrinsic_exit_codes=[row['exit_code']for row in rows],
        flag_exit_codes=[row['exit_code']for row in flags],gpu_kernels_launched=0)),flush=True)


if __name__=='__main__':main()
