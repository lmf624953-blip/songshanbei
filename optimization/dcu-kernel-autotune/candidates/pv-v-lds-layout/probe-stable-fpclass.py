#!/usr/bin/env python3
"""独立原始FP32位模式探针：class非零谓词是否等价于整数忽略符号位后判非零。"""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re


def build(reference, output):
    from songshan_h3.compiler_compat import configure_hygon_compiler
    from triton.backends.amd.compiler import HIPBackend
    from triton.backends.compiler import GPUTarget
    source=(reference/"dense-attention-packed.amdgcn").read_text()
    descriptor=re.search(r"\t\.amdhsa_kernel .*?\t\.end_amdhsa_kernel",source,re.S)[0]
    metadata=re.search(r"\t\.amdgpu_metadata.*?\t\.end_amdgpu_metadata",source,re.S)[0]
    lines=['\t.amdgcn_target "amdgcn-amd-amdhsa--gfx936"',"\t.text","\t.globl dense_attention_aot_kernel",
           "\t.p2align 8","\t.type dense_attention_aot_kernel,@function","dense_attention_aot_kernel:",
           "\ts_load_dwordx2 s[8:9], s[4:5], 0","\ts_load_dwordx2 s[10:11], s[4:5], 0x20",
           "\ts_load_dword s12, s[4:5], 0x28","\tv_mov_b32_e32 v1, s6",
           "\tv_lshlrev_b32_e32 v1, 8, v1","\tv_add_u32_e32 v1, v0, v1","\ts_waitcnt lgkmcnt(0)",
           "\tv_cmp_gt_u32_e32 vcc, s12, v1","\ts_and_saveexec_b64 s[14:15], vcc",
           "\tv_lshlrev_b32_e32 v1, 2, v1","\tglobal_load_dword v2, v1, s[8:9]",
           "\ts_waitcnt vmcnt(0)","\ts_mov_b32 s13, 0x39f","\tv_cmp_class_f32_e64 vcc, v2, s13",
           "\tv_mov_b32_e32 v3, 1","\tv_cndmask_b32_e32 v2, 0, v3, vcc","\tglobal_store_dword v1, v2, s[10:11]",
           "\ts_waitcnt vmcnt(0)","\ts_endpgm",".Lprobe_end:",
           "\t.size dense_attention_aot_kernel,.Lprobe_end-dense_attention_aot_kernel",
           "\t.section .rodata,#alloc","\t.p2align 6",descriptor,metadata]
    text="\n".join(lines)+"\n"
    configure_hygon_compiler();backend=HIPBackend(GPUTarget("hip","gfx936",64))
    image=backend.make_hsaco(text,{},backend.parse_options(dict(num_warps=4,num_stages=1,kpack=2)))
    output.mkdir();(output/"probe.amdgcn").write_text(text);(output/"probe.hsaco").write_bytes(image)
    (output/"build.json").write_text(json.dumps({'sha256':hashlib.sha256(image).hexdigest(),'mask':'0x39f','scope':'仅谓词语义探针，非模型kernel'},ensure_ascii=False,indent=2)+'\n')


def verify(folder, output):
    import torch
    image=folder/'probe.hsaco';meta=json.loads((folder/'build.json').read_text())
    assert hashlib.sha256(image.read_bytes()).hexdigest()==meta['sha256']
    torch.cuda.init()
    hip=ctypes.CDLL('libamdhip64.so');hip.hipModuleLoad.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_char_p]
    hip.hipModuleGetFunction.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p,ctypes.c_char_p]
    hip.hipModuleLaunchKernel.argtypes=[ctypes.c_void_p]+[ctypes.c_uint]*7+[ctypes.c_void_p]*3
    def check(code):
        if code:raise RuntimeError(f'HIP status {code}')
    module,function=ctypes.c_void_p(),ctypes.c_void_p();check(hip.hipModuleLoad(ctypes.byref(module),str(image).encode()));check(hip.hipModuleGetFunction(ctypes.byref(function),module,b'dense_attention_aot_kernel'))
    gen=torch.Generator().manual_seed(20261127)
    hi=torch.arange(65536,dtype=torch.int64)<<16
    cases=torch.cat([hi,hi|1,hi|32768,hi|65535,hi|torch.randint(0,65536,(65536,),generator=gen,dtype=torch.int64),torch.randint(0,2**32,(262144,),generator=gen,dtype=torch.int64),torch.tensor([0x00000000,0x80000000,0x00000001,0x80000001,0x007fffff,0x807fffff,0x7f800000,0xff800000,0x7f800001,0xff800001,0x7fc00000,0xffc00000],dtype=torch.int64)])
    raw=cases.to(torch.int32).cuda();result=torch.full((raw.numel()+32,),-19,dtype=torch.int32,device='cuda')
    values=[ctypes.c_uint64(0) for _ in range(15)];values[0].value=raw.data_ptr();values[4].value=result.data_ptr();values[5].value=raw.numel()
    params=(ctypes.c_void_p*15)(*(ctypes.addressof(v) for v in values))
    check(hip.hipModuleLaunchKernel(function,(raw.numel()+255)//256,1,1,256,1,1,16384,torch.cuda.current_stream().cuda_stream,params,None));torch.cuda.synchronize()
    expected=((cases&0x7fffffff)!=0).to(torch.int32);actual=result[:raw.numel()].cpu();bad=expected!=actual
    record={'raw_patterns':raw.numel(),'mismatches':int(bad.sum()),'tail_ok':bool((result[raw.numel():]==-19).all()),'s_nan_and_q_nan_included':True,'all_65536_high_words_with_five_low_patterns':True,'random_patterns':262144,'zero_expected_only_signed_zero':True,'first_mismatch_bits':[hex(int(x)) for x in cases[bad][:16]],'mask':'0x39f'}
    with output.open('x') as stream:json.dump(record,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)
    assert record['mismatches']==0 and record['tail_ok']


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('mode',choices=('build','verify'));cli.add_argument('source',type=Path);cli.add_argument('output',type=Path);args=cli.parse_args()
    if args.mode=='build':build(args.source,args.output)
    else:verify(args.source,args.output)


if __name__=='__main__':main()
