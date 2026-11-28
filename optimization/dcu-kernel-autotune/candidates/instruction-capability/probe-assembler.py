#!/usr/bin/env python3
"""只检查gfx936指令编码，不发射任何GPU任务。"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


CASES = {
    "mmac-control": "v_mmac_f32_16x16x16_bf16 v[0:3], v[4:5], v[6:7], v[0:3]",
    "mmac-k32": "v_mmac_f32_16x16x32_bf16 v[0:3], v[4:7], v[8:11], v[0:3]",
    "mmac-m32": "v_mmac_f32_32x32x8_bf16 v[0:15], v[16:17], v[18:19], v[0:15]",
    "mfma-m32": "v_mfma_f32_32x32x8_bf16 v[0:15], v[16:17], v[18:19], v[0:15]",
    "mfma-m32-1k": "v_mfma_f32_32x32x8bf16_1k v[0:15], v[16:19], v[20:23], v[0:15]",
    "mfma-m16-alias": "v_mfma_f32_16x16x16_bf16 v[0:3], v[4:5], v[6:7], v[0:3]",
    "mfma-m16-1k": "v_mfma_f32_16x16x16bf16_1k v[0:3], v[4:7], v[8:11], v[0:3]",
    "wmma-m16": "v_wmma_f32_16x16x16_bf16 v[0:7], v[8:15], v[16:23], v[0:7]",
    "bf16-cvt-single": "v_cvt_bf16_f32 v0, v1",
    "bf16-cvt-single-high": "v_cvt_bf16_f32 v0, v1 op_sel:[0,1]",
    "bf16-cvt-back": "v_cvt_f32_bf16 v0, v1",
    "lds-m32x16": "ds_read_m32x16_b16 v[0:3], v8",
    "lds-m32x16-alt": "ds_read_m32x16_b16_alt v[0:3], v8",
    "lds-matrix-format": "ds_read_matrix_format v[0:3], v8",
    "lds-matrix-trans-format": "ds_read_matrix_trans_format v[0:3], v8",
}
for name,modifier in (("cbsz1","cbsz:1"),("cbsz2","cbsz:2"),
                      ("abid1","abid:1"),("blgp1","blgp:1"),("blgp4","blgp:4")):
    CASES["mmac-"+name] = CASES["mmac-control"] + " " + modifier


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output",type=Path)
    cli.add_argument("--compiler",type=Path,
        default=Path("/public/software/compiler/dtk-26.04/aillvm/bin/llvm-mc"))
    args=cli.parse_args();args.output.mkdir()
    rows=[]
    for name,instruction in CASES.items():
        source=args.output/(name+".s")
        source.write_text('\t.text\n\t.amdgcn_target "amdgcn-amd-amdhsa--gfx936"\n'+instruction+'\ns_endpgm\n')
        command=[str(args.compiler),'-triple=amdgcn-amd-amdhsa','-mcpu=gfx936','-filetype=obj',str(source),'-o',str(source.with_suffix('.o'))]
        proc=subprocess.run(command,capture_output=True,text=True,timeout=20)
        source.with_suffix('.log').write_text(proc.stdout+proc.stderr)
        source.with_suffix('.exit-code').write_text(str(proc.returncode)+'\n')
        row=dict(name=name,instruction=instruction,command=command,exit_code=proc.returncode,
            error=proc.stderr,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        if proc.returncode==0:
            show=subprocess.run([str(args.compiler),'-triple=amdgcn-amd-amdhsa','-mcpu=gfx936','--show-encoding',str(source)],capture_output=True,text=True,timeout=20)
            assert show.returncode==0
            source.with_suffix('.encoding.txt').write_text(show.stdout)
            row['encoding']=show.stdout
            row['object_sha256']=hashlib.sha256(source.with_suffix('.o').read_bytes()).hexdigest()
        rows.append(row);print(name,proc.returncode,flush=True)
    result=dict(scope='仅CPU汇编能力，不是GPU数值或性能验证',gpu_kernels_launched=0,
        compiler=str(args.compiler),compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        reused_negative='gfx936/DTK26.04拒绝v_cvt_pk_bf16_f32，已有p-round-pack-a14r1n11记录，不重复该原型',records=rows)
    (args.output/'capability.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
