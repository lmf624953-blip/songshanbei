#!/usr/bin/env python3
"""消除 Q128 尾块唯一的 v216，另测只调整寄存器分配量的对照。"""

import argparse
from pathlib import Path
import re
import runpy


def tail_reuse(lines):
    read = lines.index("\tds_read_u16 v216, v147 offset:128")
    pack = lines.index("\tv_perm_b32 v125, v210, v216, s46", read)
    redefine = lines.index("\tv_perm_b32 v109, v203, v205, s46", pack)
    assert [i for i, line in enumerate(lines) if re.search(r"\bv216\b", line)] == [read, pack]
    # On this straight-line tail path the original v109 is dead through the
    # load and pack. The later permutation restores the original definition.
    block = lines[read:redefine]
    assert all(line.startswith("\t") for line in block)
    assert not any("s_branch" in line or "s_cbranch" in line for line in block)
    assert not any(re.search(r"\bv109\b", line) for line in block)
    pairs = re.findall(r"\bv\[(\d+):(\d+)\]", "\n".join(block))
    assert all(not int(low) <= 109 <= int(high) for low, high in pairs)
    lines[read] = lines[read].replace("v216", "v109")
    lines[pack] = lines[pack].replace("v216", "v109")


def set_allocation(lines, vgprs):
    for field in (".amdhsa_next_free_vgpr", ".vgpr_count:"):
        pattern = re.compile(rf"(^\s*{re.escape(field)}\s+)217(\s*)$")
        indices = [i for i, line in enumerate(lines) if pattern.fullmatch(line)]
        (index,) = indices
        lines[index] = pattern.sub(rf"\g<1>{vgprs}\g<2>", lines[index])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reserve", type=int, choices=(224, 232, 256))
    args = parser.parse_args()
    build = runpy.run_path(str(Path(__file__).with_name("build-disjoint-k-lds.py")))
    lines = build["verified_assembly"]((args.reference / (build["STEM"] + ".amdgcn")).read_text())
    if args.reserve is None:
        tail_reuse(lines)
    vgprs = args.reserve or 216
    set_allocation(lines, vgprs)
    build["write_variant"](args.reference, args.output, "\n".join(lines) + "\n", 16384, "tail_p_vgpr",
                           dict(vgprs=vgprs, allocation_only=args.reserve is not None))


if __name__ == "__main__":
    main()
