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


def transform(assembly, qk, pv, issue_ready, packed_rescale, packed_qk_scale):
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
    args = cli.parse_args()
    asm_name, image_name = PACKED + ".amdgcn", PACKED + ".hsaco"
    source = (args.reference / asm_name).read_text()
    assembly, changes = transform(source, args.qk_wait_width, args.pv_wait_width,
                                  args.pv_issue_ready, args.pack_pv_rescale, args.pack_qk_scale)
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
