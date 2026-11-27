#!/usr/bin/env python3
"""配对读取 P 的对齐 dword，用每 lane 字节选择保留原 BF16 半字。"""

import argparse
from pathlib import Path
import re
import runpy


READ = re.compile(r"\tds_read_u16 v(\d+), (v122|v147)(?: offset:(\d+))?")
PERM = re.compile(r"(\tv_perm_b32 v\d+, )v(\d+), v(\d+), s43")


def transform(lines, pending):
    start = lines.index("\tds_read_u16 v182, v122")
    end = start + 16
    original = [READ.fullmatch(line) for line in lines[start:end]]
    assert all(original) and all(match[2] == ("v122" if i < 8 else "v147")
                                 for i, match in enumerate(original))
    assert lines[start - 1] == "\tv_add_u32_e32 v122, 0, v133"
    last_perm = lines.index("\tv_perm_b32 v186, v186, v206, s43", end)
    pattern = lines[end:last_perm + 1]
    permutations = [i for i, line in enumerate(pattern) if PERM.fullmatch(line)]
    assert len(permutations) == 8 and lines[end] == "\ts_waitcnt lgkmcnt(7)"
    registers = ((182, 183), (184, 185), (186, 187), (206, 207),
                 (208, 209), (210, 211), (212, 213), (214, 215))
    # Group by the 4 KiB half so each read2 offset fits its 8-bit dword field.
    pairs = ((0, 1), (6, 7), (2, 3), (4, 5), (8, 9), (14, 15), (10, 11), (12, 13))
    redirected = {}
    reads = []
    for indices, destinations in zip(pairs, registers, strict=True):
        base = "v122" if indices[0] < 8 else "v160"
        offsets = [int(original[i][3] or 0) for i in indices]
        if offsets[0] >= 4096:
            base = "v123" if indices[0] < 8 else "v161"
            offsets = [offset - 4096 for offset in offsets]
        assert all(offset % 4 == 0 and 0 <= offset // 4 < 256 for offset in offsets)
        reads.append(f"\tds_read2_b32 v[{destinations[0]}:{destinations[1]}], {base} "
                     f"offset0:{offsets[0] // 4} offset1:{offsets[1] // 4}")
        redirected.update({int(original[i][1]): reg for i, reg in zip(indices, destinations, strict=True)})
    for index in permutations:
        match = PERM.fullmatch(pattern[index])
        pattern[index] = f"{match[1]}v{redirected[int(match[2])]}, v{redirected[int(match[3])]}, v217"
    pending_permutations = [pattern[index] for index in permutations]
    scheduled = []
    while pending_permutations:
        for i, line in enumerate(pending_permutations):
            dest = int(re.findall(r"\bv(\d+)\b", line)[0])
            other_inputs = {reg for j, other in enumerate(pending_permutations) if j != i
                            for reg in map(int, re.findall(r"\bv(\d+)\b", other)[1:3])}
            if dest not in other_inputs:
                scheduled.append(pending_permutations.pop(i))
                break
        else:
            raise ValueError("P 的置换存在需要额外寄存器的依赖环")
    for index, line in zip(permutations, scheduled, strict=True):
        pattern[index] = line
    pattern[0] = f"\ts_waitcnt lgkmcnt({pending})"
    addresses = [
        "\tv_and_b32_e32 v122, -4, v133",
        "\tv_add_u32_e32 v123, 4096, v122",
        "\tv_and_b32_e32 v160, -4, v147",
        "\tv_add_u32_e32 v161, 4096, v160",
    ]
    lines[start - 1:last_perm + 1] = [*addresses, *reads, *pattern]
    initialization = lines.index("\ts_mov_b32 s45, 0")
    lines[initialization:initialization] = [
        "\tv_and_b32_e32 v217, 1, v0",
        "\tv_cmp_eq_u32_e32 vcc, 0, v217",
        "\tv_mov_b32_e32 v217, s44",
        "\tv_mov_b32_e32 v218, s43",
        "\tv_cndmask_b32_e32 v217, v217, v218, vcc",
    ]
    for field in (".amdhsa_next_free_vgpr", ".vgpr_count:"):
        pattern = re.compile(rf"(^\s*{re.escape(field)}\s+)217(\s*)$")
        (index,) = [i for i, line in enumerate(lines) if pattern.fullmatch(line)]
        lines[index] = pattern.sub(r"\g<1>219\g<2>", lines[index])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--p-pending", type=int, choices=(3, 7), default=7)
    args = cli.parse_args()
    scripts = Path(__file__).parent
    helper = runpy.run_path(str(scripts / "build-disjoint-k-lds.py"))
    lines = helper["verified_assembly"]((args.reference / (helper["STEM"] + ".amdgcn")).read_text())
    transform(lines, args.p_pending)
    helper["write_variant"](args.reference, args.output, "\n".join(lines) + "\n", 16384, "p_read2",
                            dict(pending=args.p_pending, full_block_reads=8, selector_vgpr=217))


if __name__ == "__main__":
    main()
