#!/usr/bin/env python3
"""精简稳定max谓词及alpha=1初始化，保留所有非零/非有限delta的慢路径。"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import shutil

from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget


def full_exec_proof(source):
    lines = [line for line in source.splitlines() if not line.lstrip().startswith((".loc", ".file"))]
    entry, end = lines.index(".LBB0_14:"), lines.index(".LBB0_20:")
    saves = []
    for i,line in enumerate(lines[:entry]):
        if not re.match(r"\ts_\S+ (?:exec|s\[\d+:\d+\]),",line) or "exec" not in line:
            continue
        save = re.fullmatch(r"\ts_and_saveexec_b64 (s\[\d+:\d+\]), s\[\d+:\d+\]",line)
        restore = re.fullmatch(r"\ts_or_b64 exec, exec, (s\[\d+:\d+\])",line)
        if save:
            assert not saves, "不支持嵌套exec掩码"
            saves.append((i,save[1]))
        else:
            assert restore and saves and restore[1]==saves[-1][1],line
            begin,pair=saves.pop()
            assert lines[begin+1].startswith("\ts_cbranch_execz ")
            target=lines[begin+1].split()[-1]+":"
            assert lines[i-1]==target
            assert not any(x.startswith("\ts_") for x in lines[begin+2:i-1]), "save/restore间有标量状态变更"
    assert not saves
    # 256线程组的4个完整wave；每次入口掩码原样恢复，完整循环中无exec写入。
    assert ".max_flat_workgroup_size: 256" in source
    assert not any(re.match(r"\ts_\S+ exec,|\ts_\S*saveexec",line) for line in lines[entry:end])
    return dict(prologue_restored_masks=6, full_loop_exec_writes=0, block_threads=256)


def transform(source, reduce_or, uniform_ones, drop_exec_mask, fpclass):
    lines=source.splitlines()
    entry,end=lines.index(".LBB0_14:"),lines.index(".LBB0_20:")
    guard=["\tv_or_b32_e32 v206, v188, v189"]
    guard += [f"\tv_or_b32_e32 v206, v206, v{i}" for i in range(190,196)]
    start=next(i for i in range(entry,end) if lines[i]==guard[0])
    assert lines[start:start+7]==guard
    if reduce_or=="chain":
        replacement=["\tv_or3_b32 v206, v188, v189, v190",
                     "\tv_or3_b32 v206, v206, v191, v192",
                     "\tv_or3_b32 v206, v206, v193, v194",
                     "\tv_or_b32_e32 v206, v206, v195"]
    elif reduce_or=="tree":
        registers=runpy.run_path(str(Path(__file__).with_name("build-qk-online-order.py")))["registers"]
        assert not any(207 in registers(line) for line in lines[:end])
        replacement=["\tv_or3_b32 v206, v188, v189, v190",
                     "\tv_or3_b32 v207, v191, v192, v193",
                     "\tv_or3_b32 v206, v206, v207, v194",
                     "\tv_or_b32_e32 v206, v206, v195"]
    else:
        replacement=guard
    lines[start:start+7]=replacement
    if fpclass:
        pair=["\tv_and_b32_e32 v206, 0x7fffffff, v206","\tv_cmp_ne_u32_e32 vcc, 0, v206"]
        index=lines.index(pair[0],start)
        assert lines[index:index+2]==pair
        lines[index:index+2]=["\tv_cmp_class_f32_e64 vcc, v206, s56"]
    proof=None
    if drop_exec_mask:
        proof=full_exec_proof(source)
        mask="\ts_and_b64 vcc, exec, vcc"
        assert lines.count(mask)==1
        lines.remove(mask)
    if uniform_ones:
        assert not re.search(r"\bs5[4-9]\b|s\[5[4-9]:",source)
        first=lines.index("\tv_mov_b32_e32 v188, 1.0",start)
        ones=["\tv_mov_b32_e32 v188, 1.0","\tv_mov_b32_e32 v189, 1.0"]
        ones += [f"\tv_mov_b64_e32 v[{i}:{i+1}], v[188:189]" for i in (190,192,194)]
        assert lines[first:first+5]==ones
        lines[first:first+5]=[f"\tv_mov_b64_e32 v[{i}:{i+1}], s[54:55]" for i in (188,190,192,194)]
        lines[entry+1:entry+1]=["\ts_mov_b32 s54, 0x3f800000","\ts_mov_b32 s55, 0x3f800000"]
    if fpclass:
        lines[entry+1:entry+1]=["\ts_mov_b32 s56, 0x39f"]
    result="\n".join(lines)+"\n"
    assert result[result.index(".LBB0_20:"):]==source[source.index(".LBB0_20:"):]
    if uniform_ones:
        result=result.replace(".amdhsa_next_free_sgpr 54", ".amdhsa_next_free_sgpr 56")
        result=result.replace(".sgpr_count:     58", ".sgpr_count:     60")
    if fpclass:
        result=re.sub(r"\.amdhsa_next_free_sgpr \d+", ".amdhsa_next_free_sgpr 57", result)
        result=re.sub(r"\.sgpr_count:\s+\d+", ".sgpr_count:     61", result)
    assert result.count("v_mmac_f32_16x16x16_bf16")==source.count("v_mmac_f32_16x16x16_bf16")
    assert result.count("v_exp_f32_e32")==source.count("v_exp_f32_e32")
    return result,proof


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference",type=Path)
    cli.add_argument("output",type=Path)
    cli.add_argument("--reduce-or",choices=("keep","chain","tree"),default="keep")
    cli.add_argument("--uniform-ones",action="store_true")
    cli.add_argument("--drop-exec-mask",action="store_true")
    cli.add_argument("--fpclass",action="store_true")
    args=cli.parse_args()
    meta=json.loads((args.reference/"dense-attention-aot-build.json").read_text())
    digest=hashlib.sha256((args.reference/"dense-attention-packed.hsaco").read_bytes()).hexdigest()
    assert digest==meta["packed_strides"]["hsaco_sha256"]
    assert meta["experiment"]["stable_max_rescale"]["mode"]=="guarded"
    updated,proof=transform((args.reference/"dense-attention-packed.amdgcn").read_text(),args.reduce_or,args.uniform_ones,args.drop_exec_mask,args.fpclass)
    configure_hygon_compiler()
    backend=HIPBackend(GPUTarget("hip","gfx936",64))
    image=backend.make_hsaco(updated,{},backend.parse_options(dict(num_warps=4,num_stages=1,kpack=2)))
    shutil.copytree(args.reference,args.output)
    (args.output/"dense-attention-packed.hsaco").write_bytes(image)
    (args.output/"dense-attention-packed.amdgcn").write_text(updated)
    meta["packed_strides"]["hsaco_sha256"]=hashlib.sha256(image).hexdigest()
    meta["experiment"]["stable_guard_compact"]=dict(reference_packed_sha256=digest,reduce_or=args.reduce_or,
        uniform_ones=args.uniform_ones,drop_exec_mask=args.drop_exec_mask,full_exec_proof=proof,fpclass=args.fpclass,
        numerical_verification="尚未验证；fpclass需要独立原始位模式探针，不得只靠最终输出通过")
    (args.output/"dense-attention-aot-build.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":main()
