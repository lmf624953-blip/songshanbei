#!/usr/bin/env python3
"""交错完整块的 QK 缩放、最大值 DPP 和在线指数计算。"""

import argparse
from pathlib import Path
import re
import runpy


ASM_START = "\t;;#ASMSTART"
ASM_END = "\t;;#ASMEND"
VGPR = re.compile(r"\bv(\d+)\b|\bv\[(\d+):(\d+)\]")


def registers(operand):
    result = set()
    for match in VGPR.finditer(operand):
        if match[1]:
            result.add(int(match[1]))
        else:
            result.update(range(int(match[2]), int(match[3]) + 1))
    return result


def uses(line):
    opcode, arguments = line.split(maxsplit=1)
    output, operands = arguments.split(",", 1)
    writes = registers(output)
    assert writes, line
    reads = registers(operands)
    if opcode.endswith("_dpp"):
        # A lane disabled by the bank mask retains its old destination.
        reads.update(writes)
    return reads, writes


def max_groups(lines):
    marker = lines.index(".Ltmp0:")
    first = lines.index(ASM_START, marker)
    second = lines.index(ASM_START, lines.index(ASM_END, first) + 1)
    for start in (first, second):
        assert lines[start + 1] == "\ts_nop 1"
        assert all(line.startswith("\tv_max_f32_dpp ") for line in lines[start + 2:start + 22])
        assert lines[start + 22] == ASM_END
    return marker, first, second


def interleave_scale(lines, drop_nop):
    marker, first, second = max_groups(lines)
    maxes = [i for i in range(marker, first) if lines[i].startswith("\tv_max_f32_e32 ")]
    scales = [i for i in range(marker - 16, marker) if lines[i].startswith("\tv_pk_mul_f32 ")]
    assert len(maxes) == len(scales) == 8
    late_scale = [lines[i] for i in scales[4:]]
    late_max = [lines[i] for i in maxes[4:]]
    blocked = registers("\n".join([*(lines[i] for i in maxes[:4]), *lines[first:second]]))
    assert registers("\n".join(late_scale + late_max)).isdisjoint(blocked)
    if drop_nop:
        del lines[second + 1]
    # Keep the second group's producers at least twelve independent DPP
    # instructions away from their first consumer.
    lines[first + 10:first + 10] = late_max
    lines[first + 6:first + 6] = late_scale
    for index in sorted(scales[4:] + maxes[4:], reverse=True):
        del lines[index]


def interleave_online(lines, interleaved):
    second = lines.index(ASM_START, lines.index(ASM_END, lines.index(".Ltmp0:")) + 1)
    end = lines.index(ASM_END, second)
    online = lines.index(".LBB0_18:", end)
    cap = lines.index("\tv_exp_f32_e32 v198, v158", online)
    blocked_read, blocked_write = set(), set()

    def block(line):
        reads, writes = uses(line)
        blocked_read.update(reads)
        blocked_write.update(writes)

    for line in lines[second:online]:
        if line.startswith("\tv_"):
            block(line)
    moved, remaining = [], []
    for line in lines[online + 1:cap]:
        if not line.startswith("\tv_"):
            assert line.lstrip().startswith(".loc"), line
            remaining.append(line)
            continue
        reads, writes = uses(line)
        if reads & blocked_write or writes & (blocked_read | blocked_write):
            block(line)
            remaining.append(line)
        else:
            moved.append(line)
    assert len(moved) == 21
    lines[online + 1:cap] = remaining
    if interleaved:
        # The five DPP phases keep their own order and mask semantics.
        dpp = lines[second + 1:end]
        nop = dpp.pop(0) if dpp[0] == "\ts_nop 1" else None
        assert len(dpp) == 20
        scheduled = [nop] if nop else []
        width = (len(moved) + 4) // 5
        for phase in range(5):
            scheduled.extend(dpp[4 * phase:4 * phase + 4])
            scheduled.extend(moved[phase * width:(phase + 1) * width])
        lines[second + 1:end] = scheduled
    else:
        lines[second:second] = moved
    return len(moved)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--split-scale", action="store_true")
    cli.add_argument("--drop-second-nop", action="store_true")
    cli.add_argument("--online", choices=("before", "interleaved"))
    args = cli.parse_args()
    if args.drop_second_nop and not args.split_scale:
        cli.error("--drop-second-nop requires --split-scale")
    helper = runpy.run_path(str(Path(__file__).with_name("build-disjoint-k-lds.py")))
    lines = helper["verified_assembly"]((args.reference / (helper["STEM"] + ".amdgcn")).read_text())
    if args.split_scale:
        interleave_scale(lines, args.drop_second_nop)
    moved = interleave_online(lines, args.online == "interleaved") if args.online else 0
    helper["write_variant"](args.reference, args.output, "\n".join(lines) + "\n", 16384, "qk_online_order",
                            dict(split_scale=args.split_scale, drop_second_nop=args.drop_second_nop,
                                 online=args.online, moved_online=moved))


if __name__ == "__main__":
    main()
