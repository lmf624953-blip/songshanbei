#!/usr/bin/env python3
"""将 PV 的 V 分两批装入同一组寄存器，不增加 LDS/全局读写。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import runpy
import shutil

from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget


PINNED = "a00cc85245e936216531503c3c1dfb57293d00d907dd517b4b1b6c8932f8a3cd"


def transform(source):
    helper = runpy.run_path(str(Path(__file__).parents[1] / "attention-query-reload/build.py"))
    rename = helper["rename"]
    lines = source.splitlines()
    for tail in (True, False):
        lower = lines.index(".LBB0_20:") if tail else lines.index(".LBB0_14:")
        upper = lines.index(".Ltmp6:") if tail else lines.index(".Ltmp2:")
        mmac = [i for i in range(lower, upper) if lines[i].startswith("\tv_mmac_f32_16x16x16_bf16 ")][-32:]
        assert len(mmac) == 32
        loads = [i for i in range(lower, mmac[0]) if lines[i].startswith("\tds_read2st64_b64 ")][-8:]
        assert len(loads) == 8
        second_loads = [lines[i] for i in loads[4:]]
        mapping = ({169+j: target for j, target in enumerate([*range(97, 101), *range(157, 169)])}
                   if tail else {166+j: target for j, target in enumerate([*range(97, 105), *range(158, 166)])})
        second = []
        for i in mmac[16:]:
            parts = lines[i].split(", ")
            old = parts[1]
            parts[1] = rename(parts[1], mapping)
            assert rename(parts[1], {v:k for k,v in mapping.items()}) == old
            second.append(", ".join(parts))
        scheduled = [lines[i] for i in mmac[:16]]
        scheduled += [rename(line, mapping) for line in second_loads]
        scheduled += ["\ts_waitcnt lgkmcnt(0)", *second]
        # 原 MMAC 的整体顺序也保持不变，只将后半 V 的装入延至前半已消费。
        mmac_start = mmac[0]
        while lines[mmac_start-1].lstrip().startswith((".loc", "s_waitcnt")):
            mmac_start -= 1
        lines[mmac_start:mmac[-1]+1] = ["\ts_waitcnt lgkmcnt(0)", *scheduled]
        for i in reversed(loads[4:]):
            del lines[i]
        if tail:
            # 先完成缩放，再将16个P半字打包成8个VGPR，最后才装入V。
            first_load = loads[0]
            address = first_load - 1
            while lines[address].lstrip().startswith(".loc"):
                address -= 1
            assert lines[address] == "\tv_add_u32_e32 v101, 0, v137"
            first_mmac = next(i for i in range(first_load, len(lines))
                if lines[i].startswith("\tv_mmac_f32_16x16x16_bf16 "))
            prefix = lines[first_load:first_mmac]
            first_four = [line.replace(", v101 ", ", v137 ") for line in prefix
                          if line.startswith("\tds_read2st64_b64 ")]
            assert len(first_four) == 4 and all(", v101 " not in line for line in first_four)
            scales = [line for line in prefix if line.startswith("\tv_mul_f32_e32 ")]
            perms = [line for line in prefix if line.startswith("\tv_perm_b32 ")]
            assert len(scales) == 64 and len(perms) == 8
            allowed = {"v_mul_f32_e32", "v_perm_b32", "ds_read2st64_b64", "s_waitcnt"}
            assert all(line.lstrip().startswith(".loc") or not line.strip()
                       or line.strip().split()[0] in allowed for line in prefix)
            lines[address:first_mmac] = [*scales, "\ts_waitcnt lgkmcnt(0)", *perms,
                                        *first_four, "\ts_waitcnt lgkmcnt(0)"]
    result = "\n".join(lines) + "\n"
    def counts(text):
        return Counter(line.strip().split()[0] for line in text.splitlines()
                       if line.startswith(("\tds_", "\tglobal_", "\tv_mmac")))
    assert counts(result) == counts(source)
    assert result[result.index(".Ltmp6:"):] == source[source.index(".Ltmp6:"):]
    return result


def early_sum(source):
    lines = source.splitlines()
    first = lines.index(".LBB0_14:")
    guard = lines.index("\tv_or_b32_e32 v206, v188, v189", first)
    done = lines.index(".Lh3_stable_max_done:", guard)
    guard_lines = lines[guard:done+1]
    begin = lines.index(".Lh3_next_k_ready:", done)+1
    end = lines.index("\ts_cbranch_scc1 .Lh3_folded_sum_exit", begin)
    move = [i for i in range(begin,end) if lines[i].startswith("\tv_")
            or lines[i].startswith("\ts_nop ") or lines[i].startswith("\t;;#ASM")]
    scheduled = [lines[i] for i in move]
    assert sum(line.startswith("\tv_fma_f32 ") for line in scheduled) == 8
    assert sum(line.startswith("\tv_add_f32_e32 ") for line in scheduled) == 8
    assert all(line.startswith(("\tv_add_f32", "\tv_fma_f32", "\ts_nop", "\t;;#ASM"))
               for line in scheduled)
    for i in reversed(move):
        del lines[i]
    del lines[guard:done+1]
    target = lines.index("\tds_read_b64 v[122:123], v133", first)
    # alpha计算与原C缩放先完成，再执行原序分母归约；P/V读取尚未覆盖这些临时值。
    lines[target:target] = [*guard_lines, *scheduled]
    result = "\n".join(lines)+"\n"
    assert Counter(line for line in result.splitlines() if line.startswith("\tv_")) == Counter(
        line for line in source.splitlines() if line.startswith("\tv_"))
    assert result[result.index(".LBB0_20:"):] == source[source.index(".LBB0_20:"):]
    return result


def staged_key(source, live_registers=8):
    rename = runpy.run_path(str(Path(__file__).parents[1]/"attention-query-reload/build.py"))["rename"]
    lines = source.splitlines()
    for tail in (True,False):
        lo = lines.index(".LBB0_20:") if tail else lines.index(".LBB0_14:")
        mmac = [i for i in range(lo,len(lines)) if lines[i].startswith("\tv_mmac_f32_16x16x16_bf16 ")][:32]
        loads = [i for i in range(lo,mmac[0]) if lines[i].startswith("\tds_read_b128 ")][-8:]
        assert len(loads)==8
        load_lines = [lines[i] for i in loads]
        bases = (97,150,158,166) if tail else (97,158,166,174)
        assert live_registers in (8,16)
        chunk_stages=live_registers//8
        replacement=[]; count=0
        for i in range(loads[0],mmac[-1]+1):
            line=lines[i]
            if i in loads[2*chunk_stages:] or line.startswith("\ts_waitcnt lgkmcnt("):
                continue
            if line.startswith("\tv_mmac_f32_16x16x16_bf16 "):
                stage=count//8
                target=bases[stage%chunk_stages]
                mapping={bases[stage]+j:target+j for j in range(8)}
                if count%8==0:
                    if stage and stage%chunk_stages==0:
                        for future in range(stage,stage+chunk_stages):
                            future_map={bases[future]+j:bases[future%chunk_stages]+j for j in range(8)}
                            replacement += [rename(x,future_map) for x in load_lines[future*2:future*2+2]]
                    replacement.append(f"\ts_waitcnt lgkmcnt({2*(chunk_stages-1-stage%chunk_stages)})")
                parts=line.split(", ");original=parts[1]
                parts[1]=rename(parts[1],mapping)
                assert rename(parts[1],{v:k for k,v in mapping.items()})==original
                line=", ".join(parts);count+=1
            replacement.append(line)
        assert count==32
        lines[loads[0]:mmac[-1]+1]=replacement
    text="\n".join(lines)+"\n"
    for opcode in ("ds_read_b128", "v_mmac_f32_16x16x16_bf16", "global_load_dwordx4"):
        assert text.count("\t"+opcode+" ")==source.count("\t"+opcode+" ")
    return text


def remat_addresses(source):
    lines=source.splitlines()
    for line in ("\tv_xor_b32_e32 v217, 0x200, v132", "\tv_xor_b32_e32 v218, 0x310, v132"):
        assert lines.count(line)==1
        lines.remove(line)
    first=next(i for i,line in enumerate(lines) if line.startswith("\tds_write_b16_d16_hi v132,"))
    stores=0
    for i,line in enumerate(lines):
        if line.startswith("\tds_write_b16_d16_hi v217,"):
            lines[i]=line.replace("v217,","v97,");stores+=1
        if line.startswith("\tds_write_b16_d16_hi v218,"):
            lines[i]=line.replace("v218,","v98,");stores+=1
    assert stores==8
    lines[first:first]=["\tv_xor_b32_e32 v97, 0x200, v132", "\tv_xor_b32_e32 v98, 0x310, v132"]
    return "\n".join(lines)+"\n"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--early-sum", action="store_true")
    cli.add_argument("--staged-k", action="store_true")
    cli.add_argument("--key-registers",type=int,choices=(8,16),default=8)
    cli.add_argument("--remat-addresses", action="store_true")
    args = cli.parse_args()
    packed = args.reference / "dense-attention-packed.hsaco"
    assert hashlib.sha256(packed.read_bytes()).hexdigest() == PINNED
    source = (args.reference / "dense-attention-packed.amdgcn").read_text()
    updated = transform(source)
    if args.early_sum:
        updated = early_sum(updated)
    if args.staged_k:
        updated = staged_key(updated,args.key_registers)
    if args.remat_addresses:
        updated = remat_addresses(updated)
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    image = backend.make_hsaco(updated, {}, backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2)))
    shutil.copytree(args.reference, args.output)
    (args.output / packed.name).write_bytes(image)
    (args.output / "dense-attention-packed.amdgcn").write_text(updated)
    metadata = json.loads((args.reference / "dense-attention-aot-build.json").read_text())
    metadata["packed_strides"]["hsaco_sha256"] = hashlib.sha256(image).hexdigest()
    metadata["experiment"]["pv_stream"] = dict(reference_packed_sha256=PINNED,
        scope="PV V32→V16分两批；尾块提前P打包；未压缩物理编号",
        mmac_sequence_unchanged=True, LDS_and_global_operation_counts_unchanged=True,
        tail_one_redundant_add_zero_removed=True, numerical_verification="尚未验证")
    metadata["experiment"]["pv_stream"]["early_denominator"] = args.early_sum
    metadata["experiment"]["pv_stream"]["staged_key"] = args.staged_k
    metadata["experiment"]["pv_stream"]["key_live_registers"] = args.key_registers if args.staged_k else 32
    metadata["experiment"]["pv_stream"]["remat_two_P_LDS_addresses"] = args.remat_addresses
    (args.output / "dense-attention-aot-build.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
