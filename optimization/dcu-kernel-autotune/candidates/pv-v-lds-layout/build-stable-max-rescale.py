#!/usr/bin/env python3
"""实验：max 未变化的 wave 保留累加器，区分延后 exp、只跳过缩放和两者跳过。"""

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


def transform(source, mode):
    lines = source.splitlines()
    first, end = lines.index(".LBB0_18:"), lines.index(".LBB0_20:")
    assert "\t\t.amdhsa_float_denorm_mode_32 3" in lines
    assert "\t\t.amdhsa_ieee_mode 1" in lines
    helper = runpy.run_path(str(Path(__file__).with_name("build-qk-online-order.py")))
    registers = helper["registers"]
    # v206 只在后续 tail 中由 ds_read_u16 完整定义，不携带循环外活值。
    assert not any(206 in registers(line) for line in lines[:end])
    next_scratch = next(line for line in lines[end:] if 206 in registers(line))
    assert next_scratch.startswith("\tds_read_u16 v206,"), next_scratch
    scale_first = next(i for i in range(first, end) if lines[i].startswith("\tv_pk_mul_f32 "))
    scales = lines[scale_first:scale_first + 32]
    assert all(line.startswith("\tv_pk_mul_f32 ") for line in scales)
    assert sorted(int(lo) for line in scales for lo in re.findall(r"v\[(\d+):\d+\]", line)[:1]) == list(range(1, 65, 2))
    # 旧 VCC 在此后先被重定义，guard 不影响原控制流。
    next_vcc = next(line for line in lines[scale_first + 32:] if "vcc" in line)
    assert next_vcc == "\ts_andn2_b64 vcc, exec, s[26:27]", next_vcc
    next_scalar_flags = next(line for line in lines[scale_first + 32:]
                             if line.startswith(("\ts_cmp", "\ts_cbranch_scc", "\ts_add", "\ts_and", "\ts_or")))
    assert next_scalar_flags == "\ts_add_i32 s46, s46, 1", next_scalar_flags
    alpha_exps, removed = [], []
    for alpha in range(188, 196):
        matches = [(i, re.fullmatch(rf"\tv_exp_f32_e32 v{alpha}, (v\d+)", lines[i]))
                   for i in range(first, scale_first)]
        matches = [(i, match) for i, match in matches if match]
        assert len(matches) == 1
        exp_index, match = matches[0]
        operand = match[1]
        number = int(operand[1:])
        previous = next(i for i in range(exp_index - 1, first, -1) if number in registers(lines[i]))
        sub = re.fullmatch(rf"\tv_sub_f32_e32 {operand}, {operand}, (v\d+)", lines[previous])
        assert sub, lines[previous]
        following = next(line for line in lines[exp_index + 1:] if number in registers(line))
        overwrite = re.fullmatch(rf"\tv_sub_f32_e32 {operand}, (v\d+), (v\d+)", following)
        assert overwrite and operand not in overwrite.groups(), following
        assert not any(alpha in registers(line) for line in lines[exp_index + 1:scale_first])
        lines[previous] = lines[previous].replace(f"v_sub_f32_e32 {operand},", f"v_sub_f32_e32 v{alpha},", 1)
        removed.append(exp_index)
        alpha_exps.append(f"\tv_exp_f32_e32 v{alpha}, v{alpha}")
    if mode == "deferred":
        replacement = alpha_exps + scales
    else:
        guard = ["\tv_or_b32_e32 v206, v188, v189"]
        guard += [f"\tv_or_b32_e32 v206, v206, v{alpha}" for alpha in range(190, 196)]
        guard += ["\tv_and_b32_e32 v206, 0x7fffffff, v206",
                  "\tv_cmp_ne_u32_e32 vcc, 0, v206", "\ts_and_b64 vcc, exec, vcc"]
        # raw delta 的 OR 除符号位外全零，当且仅当全部八项为 ±0。
        # 使用整数判断；NaN、Inf 和最小非零 FP32 均进入原计算分支。
        if mode == "guard-scale":
            replacement = guard + alpha_exps + [
                "\ts_cbranch_vccnz .Lh3_stable_max_slow", "\ts_branch .Lh3_stable_max_done",
                ".Lh3_stable_max_slow:"] + scales + [".Lh3_stable_max_done:"]
        else:
            # 快路径的 alpha 仍供后续原分母 FMA 使用，精确填入 1.0。
            ones = [f"\tv_mov_b32_e32 v{alpha}, 1.0" for alpha in (188, 189)]
            ones += [f"\tv_mov_b64_e32 v[{alpha}:{alpha + 1}], v[188:189]" for alpha in (190, 192, 194)]
            replacement = guard + ["\ts_cbranch_vccnz .Lh3_stable_max_slow"] + ones + [
                "\ts_branch .Lh3_stable_max_done", ".Lh3_stable_max_slow:"] + alpha_exps + scales + [
                ".Lh3_stable_max_done:"]
    lines[scale_first:scale_first + 32] = replacement
    for index in reversed(removed):
        del lines[index]
    result = "\n".join(lines) + "\n"
    assert result.count("v_mmac_f32_16x16x16_bf16") == source.count("v_mmac_f32_16x16x16_bf16")
    assert result[result.index(".LBB0_20:"):] == source[source.index(".LBB0_20:"):]
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--mode", choices=("deferred", "guard-scale", "guarded"), required=True)
    args = cli.parse_args()
    record = json.loads((args.reference / "dense-attention-aot-build.json").read_text())
    assert record["experiment"]["prepacked_v"]["p_row_layout"]
    packed = args.reference / "dense-attention-packed.hsaco"
    reference_digest = hashlib.sha256(packed.read_bytes()).hexdigest()
    assert reference_digest == record["packed_strides"]["hsaco_sha256"]
    source = (args.reference / "dense-attention-packed.amdgcn").read_text()
    updated = transform(source, args.mode)
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    image = backend.make_hsaco(updated, {}, backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2)))
    shutil.copytree(args.reference, args.output)
    (args.output / packed.name).write_bytes(image)
    (args.output / "dense-attention-packed.amdgcn").write_text(updated)
    record["packed_strides"]["hsaco_sha256"] = hashlib.sha256(image).hexdigest()
    record["experiment"]["stable_max_rescale"] = dict(
        mode=args.mode, reference_packed_sha256=reference_digest,
        scope="完整KV循环，无barrier或MMAC变更；wave内全部delta为±0时走恒等缩放",
        scratch="v206仅在tail重新定义，无新增寄存器需求", numerical_verification="尚未完成",
        caution="后续必须验证极小V、带符号零、非有限值及跨长度；未计时、未选用")
    (args.output / "dense-attention-aot-build.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
