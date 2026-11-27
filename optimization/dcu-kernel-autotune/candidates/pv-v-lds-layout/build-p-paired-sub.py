#!/usr/bin/env python3
"""为完整 KV 块连续放置 new_max，并成对计算 P 的 FP32 减法。"""

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


def helpers():
    return runpy.run_path(str(Path(__file__).with_name("build-qk-online-order.py")))


def rename_max(source):
    lines = source.splitlines()
    begin, end = lines.index(".LBB0_18:"), lines.index(".LBB0_20:")
    exit_index = lines.index(".Lh3_folded_sum_exit:", begin)
    assert source.count("s_cbranch_scc1 .Lh3_folded_sum_exit") == 1
    assert source.count(".Lh3_folded_sum_exit") == 2
    registers = helpers()["registers"]
    old = (117, 119, 121, 125, 127, 129, 131, 124)
    mapping = dict(zip(old, range(208, 216)))
    for register in mapping.values():
        assert not any(register in registers(line) for line in lines[:end])
        next_access = next(line for line in lines[end:] if register in registers(line))
        assert next_access.startswith(f"\tds_read_u16 v{register},"), next_access
    definitions = {}
    for index in range(begin, exit_index):
        line = lines[index]
        for register in old:
            if register not in registers(line):
                continue
            assert not any(int(lo) <= register <= int(hi)
                           for lo, hi in re.findall(r"v\[(\d+):(\d+)\]", line)), line
            assert line.startswith(("\tv_max_f32_e32 ", "\tv_sub_f32_e32 ", "\tv_mov_b32_e32 ")), line
            output = re.match(r"\tv_\S+ v(\d+),", line)
            if output and int(output[1]) == register:
                assert line.startswith(f"\tv_max_f32_e32 v{register},"), line
                assert register not in definitions
                assert line.endswith(f", v{register}"), line
                definitions[register] = index
    assert set(definitions) == set(old)
    original = lines[:]
    for index in range(begin, end):
        line = lines[index]
        def replace(match):
            number = int(match[1])
            if number not in mapping or index <= definitions[number]:
                return match[0]
            return f"v{mapping[number]}"
        line = re.sub(r"\bv(\d+)\b", replace, line)
        # 最终 max 的源仍为原来的 DPP 输出，只有定义目标换位置。
        for register, definition in definitions.items():
            if index == definition:
                line = line.replace(f"v_max_f32_e32 v{register},",
                                    f"v_max_f32_e32 v{mapping[register]},", 1)
        lines[index] = line
    restore = [f"\tv_mov_b32_e32 v{register}, v{mapping[register]}" for register in old]
    lines[exit_index + 1:exit_index + 1] = restore
    # 删去仅在退出时执行的恢复，再做逆映射，必须逐字节恢复原程序。
    inverse = {value: key for key, value in mapping.items()}
    prefix = lines[:exit_index + 1] + lines[exit_index + 1 + len(restore):end + len(restore)]
    prefix = [re.sub(r"\bv(\d+)\b", lambda m: f"v{inverse.get(int(m[1]), int(m[1]))}", line)
              for line in prefix]
    assert prefix == original[:end]
    result = "\n".join(lines) + "\n"
    assert result[result.index(".LBB0_20:"):] == source[source.index(".LBB0_20:"):]
    return result


def negative_max(source, negative_state=False):
    lines = source.splitlines()
    first, end = lines.index(".LBB0_18:"), lines.index(".LBB0_20:")
    entry = lines.index(".LBB0_14:")
    exit_index = lines.index(".Lh3_folded_sum_exit:")
    phi = {116, 118, 120, 122, 123, 126, 128, 130}
    changes = dict(maximum=0, subtract=0, restore=0)
    registers = helpers()["registers"]
    for index in range(first, end):
        line = lines[index]
        if registers(line).isdisjoint(range(208, 216)):
            continue
        match = re.fullmatch(r"\tv_max_f32_e32 (v\d+), (v\d+), (v\d+)", line)
        if match:
            old = match[2] if negative_state else "-" + match[2]
            lines[index] = f"\tv_min_f32_e64 {match[1]}, {old}, -{match[3]}"
            changes["maximum"] += 1
            continue
        match = re.fullmatch(r"\tv_sub_f32_e32 (v\d+), (v\d+), (v\d+)", line)
        if match:
            if negative_state and int(match[2][1:]) in phi:
                lines[index] = f"\tv_sub_f32_e32 {match[1]}, {match[3]}, {match[2]}"
            else:
                lines[index] = line.replace("v_sub_f32_e32", "v_add_f32_e32", 1)
            changes["subtract"] += 1
            continue
        match = re.fullmatch(r"\tv_mov_b32_e32 (v\d+), (v\d+)", line)
        assert match, line
        if not negative_state or index > exit_index:
            lines[index] = f"\tv_xor_b32_e32 {match[1]}, 0x80000000, {match[2]}"
        changes["restore"] += 1
    assert changes == dict(maximum=8, subtract=24, restore=16), changes
    if negative_state:
        # 首次进入完整块时翻转一次；回边只到内部 v_guard，不会再执行。
        assert not any(registers(line) & phi for line in source.splitlines()[entry:first])
        assert not any(re.search(r"branch\S* \.LBB0_14$", line) for line in lines)
        assert sum(line.endswith(" .Lh3_v_guard") for line in lines) == 2
        lines[entry + 1:entry + 1] = [f"\tv_xor_b32_e32 v{r}, 0x80000000, v{r}" for r in sorted(phi)]
    result = "\n".join(lines) + "\n"
    assert result[result.index(".LBB0_20:"):] == source[source.index(".LBB0_20:"):]
    return result


def paired_sub(source):
    lines = source.splitlines()
    first, end = lines.index(".LBB0_18:"), lines.index("\tds_write_b16_d16_hi v132, v122")
    helper = helpers()
    registers, uses = helper["registers"], helper["uses"]
    records = {}
    for index in range(first, end):
        match = re.fullmatch(r"\tv_add_f32_e32 (v\d+), v(\d+), v(\d+)", lines[index])
        if not match or not 158 <= int(match[2]) <= 173:
            continue
        temporary, score, maximum = match[1], int(match[2]), int(match[3])
        offset = score - 158
        expected = 208 + (offset % 4) + (4 if offset >= 8 else 0)
        assert maximum == expected, lines[index]
        exp_index = next(i for i in range(index + 1, end) if int(temporary[1:]) in registers(lines[i]))
        exponent = re.fullmatch(rf"\tv_exp_f32_e32 (v\d+), {temporary}", lines[exp_index])
        assert exponent, lines[exp_index]
        if exponent[1] != temporary:
            next_access = next(line for line in lines[exp_index + 1:] if int(temporary[1:]) in registers(line))
            reads, writes = uses(next_access)
            assert int(temporary[1:]) in writes and int(temporary[1:]) not in reads, next_access
        records[score] = (index, temporary, maximum, exp_index, exponent[1])
    assert set(records) == set(range(158, 174))
    drops = []
    for score in range(158, 174, 2):
        left, right = records[score], records[score + 1]
        assert left[0] < right[0] and left[2] + 1 == right[2]
        for item in (left, right):
            maximum = item[2]
            assert any(line.startswith(f"\tv_min_f32_e64 v{maximum},")
                       for line in lines[first:left[0]])
        for actual_score, item in ((score, left), (score + 1, right)):
            for index in range(left[0] + 1, item[3]):
                if not lines[index].startswith("\tv_") or actual_score not in registers(lines[index]):
                    continue
                reads, writes = uses(lines[index])
                assert actual_score not in writes, lines[index]
                assert index == item[0], lines[index]
        lines[left[0]] = (f"\tv_pk_add_f32 v[{score}:{score + 1}], v[{score}:{score + 1}], "
                          f"v[{left[2]}:{right[2]}]")
        drops.append(right[0])
        lines[left[3]] = f"\tv_exp_f32_e32 {left[4]}, v{score}"
        lines[right[3]] = f"\tv_exp_f32_e32 {right[4]}, v{score + 1}"
    for index in reversed(sorted(drops)):
        del lines[index]
    result = "\n".join(lines) + "\n"
    assert len(source.splitlines()) - len(lines) == 8
    assert result[result.index(".LBB0_20:"):] == source[source.index(".LBB0_20:"):]
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--mode", choices=("rename", "neg-scalar", "packed", "state-neg-scalar", "state-packed"), required=True)
    args = cli.parse_args()
    record = json.loads((args.reference / "dense-attention-aot-build.json").read_text())
    packed = args.reference / "dense-attention-packed.hsaco"
    original_digest = hashlib.sha256(packed.read_bytes()).hexdigest()
    assert original_digest == record["packed_strides"]["hsaco_sha256"]
    source = (args.reference / "dense-attention-packed.amdgcn").read_text()
    updated = rename_max(source)
    if args.mode != "rename":
        updated = negative_max(updated, args.mode.startswith("state-"))
    if args.mode.endswith("packed"):
        updated = paired_sub(updated)
    assert updated.count("v_mmac_f32_16x16x16_bf16") == source.count("v_mmac_f32_16x16x16_bf16")
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    image = backend.make_hsaco(updated, {}, backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2)))
    shutil.copytree(args.reference, args.output)
    (args.output / packed.name).write_bytes(image)
    (args.output / "dense-attention-packed.amdgcn").write_text(updated)
    record["packed_strides"]["hsaco_sha256"] = hashlib.sha256(image).hexdigest()
    record["experiment"]["paired_p_sub"] = dict(
        mode=args.mode, reference_packed_sha256=original_digest,
        negative_max=args.mode != "rename",
        negative_phi_state=args.mode.startswith("state-"),
        max_registers="v208:215，尾部入口前恢复原位置", removed_scalar_sub=8 if args.mode.endswith("packed") else 0,
        unchanged="MMAC、SFU数量、RNE、sum和尾部", numerical_verification="尚未完成")
    (args.output / "dense-attention-aot-build.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
