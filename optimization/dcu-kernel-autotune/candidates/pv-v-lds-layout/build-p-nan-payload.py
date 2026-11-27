#!/usr/bin/env python3
"""实验性省去内部 P 的 NaN payload 规范化，保留 RNE 与最终输出规范化。"""

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


def transform(source):
    lines = source.splitlines()
    first = lines.index(".LBB0_18:")
    end = lines.index("\tds_write_b16_d16_hi v132, v122", first)
    body = lines[first:end]
    bfes = [re.fullmatch(r"\tv_bfe_u32 (v\d+), (v\d+), 16, 1", line) for line in body]
    bfes = [match for match in bfes if match]
    cmps = [re.fullmatch(r"\tv_cmp_o_f32_e32 vcc, (v\d+), \1", line) for line in body]
    cmps = [match for match in cmps if match]
    masks = [re.fullmatch(r"\tv_cndmask_b32_e32 (v\d+), v123, (v\d+), vcc", line)
             for line in body]
    masks = [match for match in masks if match]
    assert len(bfes) == len(cmps) == len(masks) == 16
    assert [match[2] for match in bfes] == [match[1] for match in cmps]
    assert [match[1] for match in bfes] == [match[2] for match in masks]
    assert all(match[1] == match[2] for match in masks[:-1])
    assert masks[-1].groups() == ("v123", "v172")
    constant = "\tv_mov_b32_e32 v123, 0x7fff0000"
    last_add = "\tv_add3_u32 v172, v205, v172, s31"
    assert body.count(constant) == body.count(last_add) == 1
    # 常量的所有后续读者均为将要删除的 cndmask。
    constant_index = body.index(constant)
    for line in body[constant_index + 1:]:
        if re.search(r"\bv123\b", line):
            assert line.startswith("\tv_cndmask_b32_e32 "), line
    # 删除比较后不能改变后续控制流：此区间 VCC 仅供这些 cndmask 使用，
    # 区间后下一条 VCC 指令完整定义 VCC，且不读取原值。
    assert all(line.startswith(("\tv_cmp_o_f32_e32 ", "\tv_cndmask_b32_e32 "))
               for line in body if "vcc" in line)
    next_vcc = next(line for line in lines[end:] if "vcc" in line)
    assert next_vcc == "\ts_andn2_b64 vcc, exec, s[26:27]", next_vcc
    # 在线分母继续使用原始 FP32 概率，所有 16 个格式化输入都参与求和。
    # 这些指令不在本次修改区域内；NaN 分母会传到未改动的最终归一化。
    sum_start = lines.index(".Lh3_next_k_ready:", end)
    sum_end = lines.index("\t;;#ASMSTART", sum_start)
    pairs = [re.fullmatch(r"\tv_add_f32_e32 v(?:9[7-9]|10[0-4]), (v\d+), (v\d+)", line)
             for line in lines[sum_start:sum_end]]
    pairs = [match.groups() for match in pairs if match]
    assert len(pairs) == 8
    assert sorted(register for pair in pairs for register in pair) == sorted(match[1] for match in cmps)
    helper = runpy.run_path(str(Path(__file__).with_name("build-qk-online-order.py")))
    # v172 在原最后一次格式化之后的首次访问为完整定义，旧临时值没有其他读者。
    for line in lines[end:]:
        if 172 in helper["registers"](line):
            assert line.startswith("\tds_read2st64_b64 v[170:173],"), line
            break
    else:
        raise ValueError("未找到下一次 v172 定义")
    removed = {match[0] for match in cmps + masks} | {constant}
    replaced = [line.replace(last_add, "\tv_add3_u32 v123, v205, v172, s31")
                for line in body if line not in removed]
    assert len(body) - len(replaced) == 33
    lines[first:end] = replaced
    result = "\n".join(lines) + "\n"
    assert result.count("v_mmac_f32_16x16x16_bf16") == source.count("v_mmac_f32_16x16x16_bf16")
    # 最终除法、BF16 输出转换与全部存储必须逐字节不变。
    final = source.index(".LBB0_36:")
    assert result[result.index(".LBB0_36:"):] == source[final:]
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    record = json.loads((args.reference / "dense-attention-aot-build.json").read_text())
    assert record["experiment"]["prepacked_v"]["p_row_layout"]
    packed = args.reference / "dense-attention-packed.hsaco"
    assert hashlib.sha256(packed.read_bytes()).hexdigest() == record["packed_strides"]["hsaco_sha256"]
    source = (args.reference / "dense-attention-packed.amdgcn").read_text()
    updated = transform(source)
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    image = backend.make_hsaco(updated, {}, backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2)))
    shutil.copytree(args.reference, args.output)
    (args.output / packed.name).write_bytes(image)
    (args.output / "dense-attention-packed.amdgcn").write_text(updated)
    record["packed_strides"]["hsaco_sha256"] = hashlib.sha256(image).hexdigest()
    record["experiment"]["p_nan_payload"] = dict(
        scope="完整KV循环的内部P，仅payload规范化；RNE及最终输出转换保留",
        removed_compare=16, removed_mask=16, removed_constant=1,
        last_round_destination="v123", numerical_verification="尚未完成",
        invariants=["原始FP32概率继续用于在线分母", "最终归一化及BF16 NaN规范化不变",
                    "VCC仅供已删除掩码使用，后续先完整重定义", "全部MMAC保留"],
        reference_packed_sha256=hashlib.sha256(packed.read_bytes()).hexdigest())
    (args.output / "dense-attention-aot-build.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
