#!/usr/bin/env python3
"""Compare MMAC issue and PV accumulator rescaling in the validated Q128 image."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil

from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget


REFERENCE = "29bb5d483d866616337d33a5eba8544ecd91cee04df2bd007534f6064f8df2c2"
PACKED = "dense-attention-packed"
WAIT = re.compile(r"\ts_waitcnt lgkmcnt\(([0-7])\)")
LDS = re.compile(r"\tds_(read_b128|read2st64_b64) v\[(\d+):(\d+)\], .*")
MMAC = re.compile(
    r"\tv_mmac_f32_16x16x16_bf16 (v\[\d+:\d+\]), v\[(\d+):(\d+)\], (v\[\d+:\d+\]), \1"
)
RESCALE = re.compile(r"\tv_mul_f32_e32 v(\d+), v\1, v(\d+)")
QK_SCALE = re.compile(r"\tv_mul_f32_e32 v(\d+), v114, v(\d+)")
V_REGISTER = re.compile(r"\bv(?:(9[7-9]|10[0-4])\b|\[(9[7-9]|10[0-4]):(9[7-9]|10[0-4])\])")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def chains(operations):
    result = defaultdict(list)
    for operation, _ in operations:
        result[MMAC.fullmatch(operation)[1]].append(operation)
    return dict(result)


def schedule(block, loads, width, issue_ready):
    allowed = {register: 7 - index for index, load in enumerate(loads)
               for register in range(int(load[2]), int(load[3]) + 1)}
    source, anchors = [], defaultdict(list)
    original_waits = []
    for line in block:
        wait = WAIT.fullmatch(line)
        operation = MMAC.fullmatch(line)
        if wait:
            original_waits.append(int(wait[1]))
        elif operation:
            needed = allowed[int(operation[2])]
            assert needed == allowed[int(operation[3])] and needed >= original_waits[-1]
            if not issue_ready:
                needed = original_waits[-1]
            source.append((line, needed))
        else:
            assert line.lstrip().startswith((".loc\t", "v_add_u32_e32 "))
            anchors[len(source)].append(line)
    assert original_waits == list(range(7, -1, -1)) and len(source) == 32
    order = source
    if issue_ready:
        assert not anchors and len(chains(source)) == 16
        # The two halves contribute the first and second operation to each accumulator.
        order = sorted(source[:16], key=lambda op: -op[1]) + sorted(source[16:], key=lambda op: -op[1])
        assert chains(source) == chains(order)
        assert all(MMAC.fullmatch(order[i][0])[1] != MMAC.fullmatch(order[j][0])[1]
                   for i in range(len(order)) for j in range(max(0, i - 7), i))
    result, pending = [], 8
    for index, (operation, needed) in enumerate(order):
        result.extend(anchors[index])
        group_limit = needed // width * width
        if group_limit < pending:
            result.append(f"\ts_waitcnt lgkmcnt({group_limit})")
            pending = group_limit
        result.append(operation)
    assert pending == 0
    return result, sum(bool(WAIT.fullmatch(line)) for line in result)


def pack_rescale(lines, mmac_start):
    start = lines.index("\tv_mul_f32_e32 v61, v61, v188", mmac_start - 85, mmac_start)
    lines_to_pack = lines[start:start + 64]
    packed = []
    for offset in range(0, len(lines_to_pack), 2):
        low, high = (RESCALE.fullmatch(line) for line in lines_to_pack[offset:offset + 2])
        assert low and high
        dest, scale = (int(register) for register in low.groups())
        assert (int(high[1]), int(high[2])) == (dest + 1, scale + 1)
        accumulator = f"v[{dest}:{dest + 1}]"
        packed.append(f"\tv_pk_mul_f32 {accumulator}, {accumulator}, v[{scale}:{scale + 1}]")
    lines[start:start + 64] = packed
    return len(packed)


def pack_qk_scale(lines):
    start = lines.index("\tv_mul_f32_e32 v158, v114, v186")
    # The preceding QK has consumed both registers. The next uses write them.
    packed = ["\tv_mov_b32_e32 v174, v114", "\tv_mov_b32_e32 v175, v114"]
    for offset in range(0, 16, 2):
        low, high = (QK_SCALE.fullmatch(line) for line in lines[start + offset:start + offset + 2])
        assert low and high
        dest, scores = (int(register) for register in low.groups())
        assert (int(high[1]), int(high[2])) == (dest + 1, scores + 1)
        packed.append(f"\tv_pk_mul_f32 v[{dest}:{dest + 1}], v[174:175], v[{scores}:{scores + 1}]")
    lines[start:start + 16] = packed
    return len(packed) - 2


def prefetch_v(lines, mode):
    start = lines.index("\ts_andn2_b64 vcc, exec, s[26:27]", lines.index(".Ltmp1:"))
    end = lines.index(".LBB0_18:", start)
    block = lines[start:end]
    assert sum(line.startswith("\tglobal_load_dwordx2 ") for line in block) == 4
    assert sum(line.startswith("\tv_mov_b32_e32 ") for line in block) == 8
    insert = lines.index(".Ltmp0:") + 1
    sync = lines.index("\ts_waitcnt vmcnt(0) lgkmcnt(0)", end)
    assert all(not re.search(r"(?<!\d)(?:v(?:17[5-9]|18[0-2])\b|v\[(?:17[5-9]|18[0-2]):)", line)
               for line in lines[insert:sync] if line not in block)
    sources = {reg: reg + 78 for reg in range(97, 105)}

    def rename(match):
        if match[1]:
            return f"v{sources[int(match[1])]}"
        low, high = (sources[int(r)] for r in match.groups()[1:])
        assert high - low == int(match[3]) - int(match[2])
        return f"v[{low}:{high}]"

    early = [V_REGISTER.sub(rename, line) for line in block]
    early = [line.replace(".LBB0_18", ".Lh3_v_loaded") for line in early]
    if mode == "copy":
        copies = [f"\tv_mov_b64_e32 v[{reg}:{reg + 1}], v[{reg + 78}:{reg + 79}]"
                  for reg in range(97, 105, 2)]
    else:
        # P's LDS read reuses v182. Move just this half of the last V pair.
        copies = ["\tv_mov_b32_e32 v104, v182"]
        sources[104] = 104
        pack = lines.index("\tv_lshlrev_b32_e32 v122, 16, v99", sync)
        pack_end = lines.index("\tv_add_u32_e32 v122, 0, v135", pack)
        lines[pack:pack_end] = [V_REGISTER.sub(rename, line) for line in lines[pack:pack_end]]
    lines[sync + 1:sync + 1] = copies
    del lines[start:end]
    lines[insert:insert] = early + [".Lh3_v_loaded:"]
    return dict(stage="prefetch_v", mode=mode, loads=4, copy_instructions=len(copies))


def change_kv_zero(lines, mode, v_prefetch):
    v_join = ".Lh3_v_loaded" if v_prefetch else ".LBB0_18"
    for operand, join in (("k", ".LBB0_16"), ("v", v_join)):
        branch = lines.index(f"\ts_cbranch_vccnz {join}")
        start = branch - 8
        zeros = lines[start:branch]
        registers = [re.fullmatch(r"\tv_mov_b32_e32 v(\d+), 0", line) for line in zeros]
        assert all(registers)
        first = int(registers[0][1])
        assert [int(register[1]) for register in registers] == list(range(first, first + 8))
        packed = [f"\tv_mov_b64_e32 v[{reg}:{reg + 1}], 0" for reg in range(first, first + 8, 2)]
        if mode == "miss":
            # vcc is zero only when all active lanes can take the complete load.
            load = f".Lh3_{operand}_load"
            lines[start:branch + 1] = [f"\ts_cbranch_vccz {load}", *packed,
                                       f"\ts_branch {join}", f"{load}:"]
        else:
            lines[start:branch] = packed
    return dict(stage="kv_zero", mode=mode, zero_registers_per_load=8)


def elide_v_operations(lines, mode):
    guard = "\ts_andn2_b64 vcc, exec, s[26:27]"
    first = lines.index(guard, lines.index(".LBB0_14:"))
    second = lines.index(guard, first + 1)
    if mode in ("guard", "both"):
        # The K and V load conditions are identical. Only the first guard writes vcc.
        intervening = lines[first + 1:second]
        assert all("s[26:27]" not in line and not line.startswith(("\tv_cmp", "\ts_cmp"))
                   and ("vcc" not in line or line.startswith("\ts_cbranch_vcc"))
                   for line in intervening)
        del lines[second]
    if mode in ("pack", "both"):
        # Reapplying the same fixed mask to the result leaves it unchanged.
        first = lines.index("\tv_bfi_b32 v122, s42, v175, v122")
        assert lines[first + 1] == "\tv_perm_b32 v177, v177, v175, s44"
        assert lines[first + 2] == lines[first]
        del lines[first + 2]
    return dict(stage="elide_v_operations", mode=mode)


def join_kv_loads(lines, wait):
    k_miss = lines.index("\ts_branch .LBB0_16", lines.index(".LBB0_14:"))
    k_join = lines.index(".LBB0_16:", k_miss)
    k_wait = lines.index("\ts_waitcnt vmcnt(0) lgkmcnt(0)", k_join)
    v_start = lines.index("\ts_andn2_b64 vcc, exec, s[26:27]", lines.index(".Ltmp0:"))
    v_load = lines.index(".Lh3_v_load:", v_start)
    v_join = lines.index(".Lh3_v_loaded:", v_load)
    v_copy = lines.index(".Ltmp1:", v_join) + 1
    assert sum(line.startswith("\tglobal_load_dwordx4 ") for line in lines[k_miss:k_join]) == 2
    old_registers = re.compile(r"\bv\[(17[5-9]|18[0-2]):(17[5-9]|18[0-2])\]")

    def stage_v(match):
        return f"v[{int(match[1]) + 23}:{int(match[2]) + 23}]"

    new_load = [old_registers.sub(stage_v, line) for line in lines[v_load + 1:v_join]]
    assert sum(line.startswith("\tglobal_load_dwordx2 ") for line in new_load) == 4
    assert all(not re.search(r"\bv(?:19[89]|20[0-5])\b|\bv\[(?:19[89]|20[0-5]):", line)
               for line in lines[k_join:v_copy])
    copies = [f"\tv_mov_b64_e32 v[{i}:{i + 1}], v[{i + 23}:{i + 24}]"
              for i in range(175, 183, 2)]
    lines[v_copy:v_copy] = ["\ts_waitcnt vmcnt(0)", *copies]
    del lines[v_start:v_join + 1]
    # All six memory instructions are global loads, with the two K loads first.
    lines[k_wait] = f"\ts_waitcnt vmcnt({wait}) lgkmcnt(0)"
    lines[k_join:k_join] = new_load
    lines[k_miss:k_miss] = [f"\tv_mov_b64_e32 v[{i}:{i + 1}], 0" for i in range(198, 206, 2)]
    return dict(stage="joint_kv_load", k_loads=2, v_loads=4, k_vmcnt=wait, copy_instructions=4)


def transform(assembly, qk, pv, issue_ready, packed_rescale, packed_qk_scale, v_prefetch, kv_zero, elide_v,
              joint_kv_wait):
    normalized = re.sub(r'(^\s*\.file\s+\d+ )"[^"]+"', r'\1"<directory>"', assembly, flags=re.M)
    if digest(normalized.encode()) != REFERENCE:
        raise ValueError("Expected the validated four-row Q128 DPP assembly")
    lines = assembly.splitlines()
    starts = [i for i, line in enumerate(lines) if line == "\ts_waitcnt lgkmcnt(7)"]
    assert len(starts) == 4
    changes = []
    for start in reversed(starts):
        loads = [match for line in lines[:start] if (match := LDS.fullmatch(line))][-8:]
        assert len(loads) == 8 and len({load[1] for load in loads}) == 1
        end = [i for i in range(start, start + 66) if MMAC.fullmatch(lines[i])][31]
        stage = "qk" if loads[0][1] == "read_b128" else "pv"
        width = qk if stage == "qk" else pv
        ready = stage == "pv" and issue_ready
        if width != 1 or ready:
            scheduled, count = schedule(lines[start:end + 1], loads, width, ready)
            lines[start:end + 1] = scheduled
            changes.append(dict(stage=stage, reference_line=start + 1, waits_before=8, waits_after=count))
        if packed_rescale and start == starts[1]:
            # Only the complete block has adjacent FP32 scale factors.
            count = pack_rescale(lines, start)
            assert count == 32
            changes.append(dict(stage="pv_rescale", scalar_multiply=64, packed_multiply=count))
    if packed_qk_scale:
        count = pack_qk_scale(lines)
        changes.append(dict(stage="qk_scale", scalar_multiply=16, packed_multiply=count))
    if v_prefetch:
        changes.append(prefetch_v(lines, v_prefetch))
    if kv_zero:
        changes.append(change_kv_zero(lines, kv_zero, v_prefetch))
    if elide_v:
        changes.append(elide_v_operations(lines, elide_v))
    if joint_kv_wait is not None:
        changes.append(join_kv_loads(lines, joint_kv_wait))
    return "\n".join(lines) + "\n", list(reversed(changes))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--qk-wait-width", type=int, choices=(1, 2, 4, 8), default=1)
    cli.add_argument("--pv-wait-width", type=int, choices=(1, 2, 4, 8), default=1)
    cli.add_argument("--pv-issue-ready", action="store_true")
    cli.add_argument("--pack-pv-rescale", action="store_true")
    cli.add_argument("--pack-qk-scale", action="store_true")
    cli.add_argument("--prefetch-v", choices=("copy", "direct"))
    cli.add_argument("--kv-zero", choices=("packed", "miss"))
    cli.add_argument("--elide-v", choices=("guard", "pack", "both"))
    cli.add_argument("--joint-kv-wait", type=int, choices=(0, 2, 4))
    args = cli.parse_args()
    if args.elide_v and args.prefetch_v != "direct":
        cli.error("--elide-v requires --prefetch-v direct")
    if args.joint_kv_wait is not None and (
        args.prefetch_v != "direct" or args.kv_zero != "miss" or args.elide_v in ("guard", "both")
    ):
        cli.error("--joint-kv-wait requires direct V prefetch and miss-only zeroing")
    asm_name, image_name = PACKED + ".amdgcn", PACKED + ".hsaco"
    source = (args.reference / asm_name).read_text()
    assembly, changes = transform(source, args.qk_wait_width, args.pv_wait_width,
                                  args.pv_issue_ready, args.pack_pv_rescale, args.pack_qk_scale,
                                  args.prefetch_v, args.kv_zero, args.elide_v, args.joint_kv_wait)
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    options = backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2))
    image = backend.make_hsaco(assembly, {}, options)
    original_image = (args.reference / image_name).read_bytes()
    if not changes and image != original_image:
        raise ValueError("Reassembling the reference changed the device image")
    meta_name = "dense-attention-aot-build.json"
    record = json.loads((args.reference / meta_name).read_text())
    record["packed_strides"]["hsaco_sha256"] = digest(image)
    record["experiment"].update(
        mmac_issue=dict(qk_wait_width=args.qk_wait_width, pv_wait_width=args.pv_wait_width,
                        pv_issue_ready=args.pv_issue_ready, sections=changes,
                        packed_pv_rescale=args.pack_pv_rescale, packed_qk_scale=args.pack_qk_scale,
                        v_prefetch=args.prefetch_v,
                        kv_zero=args.kv_zero,
                        elide_v=args.elide_v,
                        joint_kv_wait=args.joint_kv_wait,
                        reference_packed_sha256=digest(original_image))
    )
    args.output.mkdir()
    for name in ("dense-attention-aot.so", "dense-attention-aot.hsaco", "dense-attention-fallback.hsaco"):
        shutil.copy2(args.reference / name, args.output / name)
    (args.output / asm_name).write_text(assembly)
    (args.output / image_name).write_bytes(image)
    (args.output / meta_name).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
