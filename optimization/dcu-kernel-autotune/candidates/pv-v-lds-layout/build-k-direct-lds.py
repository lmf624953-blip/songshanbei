#!/usr/bin/env python3
"""让完整 K 块直接写入原 Q128 LDS 布局，保留掩码及两处同步。"""

import argparse
from pathlib import Path
import re
import runpy


K_FIRST = ".Lh3_first_k_load"
K_NEXT = ".Lh3_next_k_load"
K_READY = ".Lh3_next_k_ready"
V_GUARD = ".Lh3_v_guard"
K_WRITES = [
    "\tds_write_b128 v138, v[166:169]",
    "\tds_write_b128 v138, v[170:173] offset:4096",
]
K_GLOBAL = [
    "\tglobal_load_dwordx4 v[166:169], v137, s[22:23]",
    "\tglobal_load_dwordx4 v[170:173], v136, s[22:23]",
]
K_DIRECT = [
    "\ts_mov_b32 m0, s32",
    K_GLOBAL[0] + " lds",
    "\ts_add_u32 m0, s32, 4096",
    K_GLOBAL[1] + " lds",
    "\ts_mov_b32 m0, -1",
]
K_ZERO = [f"\tv_mov_b64_e32 v[{i}:{i + 1}], 0" for i in range(166, 174, 2)]
FENCE = ["\ts_waitcnt lgkmcnt(0)", "\ts_barrier", "\ts_waitcnt lgkmcnt(0)"]
MASK = "\ts_andn2_b64 vcc, exec, s[26:27]"


def producer_layout(lines):
    assembly = "\n".join(lines)
    assert not re.search(r"\bs32\b", assembly)
    pairs = re.findall(r"\bs\[(\d+):(\d+)\]", assembly)
    assert all(not int(lo) <= 32 <= int(hi) for lo, hi in pairs)
    first = lines.index("\tv_or_b32_e32 v2, v2, v108")
    second = lines.index("\tv_or_b32_e32 v1, v1, v108", first)
    assert not any(re.search(r"\bv3\b", line) for line in lines[first:second + 1])
    # For source lane t, the original LDS lane is t XOR (t >> 4).
    # The map is involutive across the 256 lanes in this block.
    lines[first] = lines[first].replace("v108", "v3")
    lines[second] = lines[second].replace("v108", "v3")
    lines[first:first] = ["\tv_lshlrev_b32_e32 v3, 3, v115", "\tv_xor_b32_e32 v3, v3, v108"]
    start = lines.index("\tv_add_u32_e32 v138, 0, v3")
    lines[start + 1:start + 1] = [
        "\tv_and_b32_e32 v139, 0xc00, v138",
        "\tv_readfirstlane_b32 s32, v139",
    ]


def original_site(lines, label, target):
    start = lines.index(f"\ts_cbranch_vccz {label}")
    end = lines.index(f"{target}:", start)
    assert lines[start - 1] == MASK
    assert [line for line in lines[start:end] if not line.lstrip().startswith(".loc")] == [
        f"\ts_cbranch_vccz {label}", *K_ZERO, f"\ts_branch {target}", f"{label}:", *K_GLOBAL,
    ]
    return start, end


def guarded_writes(label, target):
    return [f"\ts_cbranch_vccz {label}", *K_ZERO, *K_WRITES, f"\ts_branch {target}",
            f"{label}:", *K_DIRECT]


def replace_transport(lines, prefetched):
    if prefetched:
        for label, target in ((K_NEXT, K_READY), (K_FIRST, V_GUARD)):
            start, end = original_site(lines, label, target)
            lines[start:end] = [*FENCE, *guarded_writes(label, target)]
    else:
        for label, target in ((K_NEXT, K_READY), (K_FIRST, V_GUARD)):
            start, end = original_site(lines, label, target)
            del lines[start:end]
    first = lines.index(K_WRITES[0], lines.index(".LBB0_16:"))
    assert lines[first:first + 2] == K_WRITES
    pre = "\ts_waitcnt vmcnt(4) lgkmcnt(0)"
    assert lines[first - 3:first] == [pre, *FENCE[1:]]
    post = lines.index(FENCE[0], first)
    assert lines[post:post + 3] == FENCE
    if prefetched:
        lines[first - 3:first + 2] = [pre]
    else:
        lines[post] = "\ts_waitcnt vmcnt(0) lgkmcnt(0)"
        target = ".Lh3_local_k_done"
        lines[first - 3:first + 2] = [
            *FENCE, *guarded_writes(".Lh3_local_k_direct", target), f"{target}:",
        ]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--prefetch", action="store_true")
    cli.add_argument("--v-wait-fence", action="store_true")
    args = cli.parse_args()
    scripts = Path(__file__).parent
    helper = runpy.run_path(str(scripts / "build-disjoint-k-lds.py"))
    lines = helper["verified_assembly"]((args.reference / (helper["STEM"] + ".amdgcn")).read_text())
    if args.v_wait_fence:
        v_pipe = runpy.run_path(str(scripts / "build-v-wait-pipeline.py"))
        v_pipe["register_pipeline"](lines, "p-fence", 206)
    producer_layout(lines)
    replace_transport(lines, args.prefetch)
    helper["write_variant"](args.reference, args.output, "\n".join(lines) + "\n", 16384, "k_direct_lds",
                            dict(prefetch=args.prefetch, v_wait_fence=args.v_wait_fence))


if __name__ == "__main__":
    main()
