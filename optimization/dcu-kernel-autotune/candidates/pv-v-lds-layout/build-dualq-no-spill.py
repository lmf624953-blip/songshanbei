#!/usr/bin/env python3
"""两个四wave组复用同源Q128汇编，保留无scratch的独立/共享K/V对照。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys

from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget


def transform(source, shared):
    regs=runpy.run_path(str(Path(__file__).with_name("build-qk-online-order.py")))["registers"]
    assert not any(219 in regs(line) for line in source.splitlines())
    assert not re.search(r"\bs(?:5[4-9]|6[01])\b|s\[(?:5[4-9]|6[01]):",source)
    assert ".amdhsa_private_segment_fixed_size 0" in source
    lines=source.splitlines();begin=next(i for i,l in enumerate(lines) if l.startswith("\ts_load_dword s33,"))
    setup=["\tv_lshrrev_b32_e32 v219, 8, v0","\tv_readfirstlane_b32 s54, v219",
           "\tv_and_b32_e32 v0, 255, v0","\ts_load_dword s55, s[4:5], 0x28","\ts_waitcnt lgkmcnt(0)",
           "\ts_add_u32 s55, s55, 127","\ts_lshr_b32 s55, s55, 7",
           "\ts_lshl_b32 s56, s6, 1","\ts_add_u32 s56, s56, s54",
           "\ts_cmp_ge_u32 s56, s55","\ts_cselect_b64 s[58:59], 0, -1",
           "\ts_sub_u32 s57, s55, 1","\ts_min_u32 s56, s56, s57",
           "\ts_mul_i32 s7, s7, s55","\ts_add_u32 s6, s7, s56",
           "\ts_lshl_b32 s56, s54, 14"]
    # 最后只有一个Q tile时，上半组继续参与所有barrier，复制合法tile但禁止最终写出。
    lines[begin:begin]=setup
    entry=lines.index(".LBB0_14:");end=lines.index(".LBB0_20:")
    p_addresses={132,146,217,218,133,147}
    independent_addresses=p_addresses|{138,139,140,141,142,135,149}
    addresses=sorted(p_addresses if shared else independent_addresses)
    additions=[f"\tv_add_u32_e32 v{r}, s56, v{r}" for r in addresses]
    if not shared:additions.append("\ts_add_u32 s32, s32, s56")
    lines[entry:entry]=additions
    if not shared:
        instruction="\tv_add_u32_e32 v101, 0, v110"
        assert lines.count(instruction)==1
        lines[lines.index(instruction)]="\tv_add_u32_e32 v101, s56, v110"
    else:
        # K/V只由第一组发射/写入，两组QK读取相同K/V；P按组隔离。
        pairs=[
            ("\tglobal_load_dwordx4 v[166:169], v137, s[22:23]",
             "\tglobal_load_dwordx4 v[170:173], v136, s[22:23]"),
            ("\tds_write_b128 v138, v[166:169]","\tds_write_b128 v138, v[170:173] offset:4096")]
        number=0
        for left,right in pairs:
            positions=[i for i in range(lines.index(".LBB0_14:"),lines.index(".LBB0_20:")) if lines[i]==left]
            for i in reversed(positions):
                assert lines[i+1]==right
                label=f".Lh3_dual_skip_k_{number}";number+=1
                lines[i:i+2]=["\ts_cmp_eq_u32 s54, 0",f"\ts_cbranch_scc0 {label}",left,right,label+":"]
        start=lines.index("\ts_andn2_b64 vcc, exec, s[26:27]",lines.index(".LBB0_16:"))
        lines[start:start]=["\ts_cmp_eq_u32 s54, 0","\ts_cbranch_scc0 .Lh3_prepacked_v_done"]
    # 尾部保持每组独立LDS，逐条地址加组偏移；每个Q tile仅一次尾块。
    tail=lines.index(".LBB0_20:");rodata=next(i for i,l in enumerate(lines) if l.startswith("\t.section\t.rodata"))
    for i in range(rodata-1,tail-1,-1):
        line=lines[i]
        if line.startswith("\tds_"):
            if line.startswith("\tds_read"):
                match=re.search(r", (v\d+)(?: |$)",line)
            else:match=re.search(r"\tds_\S+ (v\d+),",line)
            assert match,line
            address=match[1]
            lines[i:i+1]=[f"\tv_add_u32_e32 v219, s56, {address}",line[:match.start(1)]+"v219"+line[match.end(1):]]
    # 独立组和共享组都覆盖奇数Q tile数，不提前退出任何需要同步的wave。
    for i in range(len(lines)-1,-1,-1):
        if lines[i].startswith("\tglobal_store"):
            lines[i:i+1]=["\ts_and_saveexec_b64 s[60:61], s[58:59]",lines[i],"\ts_mov_b64 exec, s[60:61]"]
    text="\n".join(lines)+"\n"
    for a,b in [(".amdhsa_system_sgpr_workgroup_id_y 0",".amdhsa_system_sgpr_workgroup_id_y 1"),
                (".amdhsa_next_free_vgpr 219",".amdhsa_next_free_vgpr 220"),
                (".amdhsa_next_free_sgpr 54",".amdhsa_next_free_sgpr 62"),
                (".max_flat_workgroup_size: 256",".max_flat_workgroup_size: 512"),
                (".vgpr_count:     219",".vgpr_count:     220"),
                (".sgpr_count:     58",".sgpr_count:     66")]:
        assert text.count(a)==1,a;text=text.replace(a,b,1)
    assert text.count("v_mmac_f32_16x16x16_bf16")==source.count("v_mmac_f32_16x16x16_bf16")
    assert not any(line.startswith("\tscratch_") for line in text.splitlines())
    return text


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument("reference",type=Path);cli.add_argument("output",type=Path)
    cli.add_argument("--mode",choices=("single-resource","single-lds32","independent","shared"),required=True);args=cli.parse_args()
    source=(args.reference/"dense-attention-packed.amdgcn").read_text();meta=json.loads((args.reference/"dense-attention-aot-build.json").read_text())
    digest=hashlib.sha256((args.reference/"dense-attention-packed.hsaco").read_bytes()).hexdigest();assert digest==meta["packed_strides"]["hsaco_sha256"]
    single=args.mode.startswith("single-")
    if single:
        text=source.replace(".amdhsa_next_free_sgpr 54",".amdhsa_next_free_sgpr 62").replace(".sgpr_count:     58",".sgpr_count:     66")
    else:text=transform(source,args.mode=="shared")
    configure_hygon_compiler();backend=HIPBackend(GPUTarget("hip","gfx936",64));warps=4 if single else 8
    image=backend.make_hsaco(text,{},backend.parse_options(dict(num_warps=warps,num_stages=1,kpack=2)))
    args.output.mkdir();(args.output/"dual.hsaco").write_bytes(image);(args.output/"dual.amdgcn").write_text(text)
    sys.path.insert(0,str(Path(os.environ["H3_AUTOTUNE_CODE_ROOT"])/"optimization/adaln"));from aot_codegen import compile_host
    host=Path(__file__).with_name("dualq_no_spill_host.cc")
    if args.mode=="single-lds32":
        original=host.read_text();assert original.count("mode?32768:16384")==1
        host=args.output/'lds32-control-host.cc';host.write_text(original.replace("mode?32768:16384","32768"))
    command=compile_host(args.output/"dual-host.so",(host.resolve(),))
    record=dict(mode=args.mode,reference=str(args.reference.resolve()),reference_packed_sha256=digest,warps=warps,
                shared_bytes=32768 if args.mode=="single-lds32" or warps==8 else 16384,hsaco_sha256=hashlib.sha256(image).hexdigest(),host_command=command,
                host_source_sha256=hashlib.sha256(host.read_bytes()).hexdigest(),scope="两个Q128的原计算序列及尾块，各组四wave；奇数Q tile上半组禁写但参与所有同步",numerical_verification="尚未验证",selected=False)
    (args.output/"dual-build.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":main()
