#!/usr/bin/env python3
"""在 Q128 求和交错版中延后 V 等待并比较跨块预取。"""

import argparse
from pathlib import Path
import re
import runpy


VGPR = re.compile(r"\bv(\d+)\b|\bv\[(\d+):(\d+)\]")
V_ADDRESS = [
    "\ts_ashr_i32 s25, s24, 31",
    "\ts_lshl_b64 s[2:3], s[24:25], 1",
    "\ts_add_u32 s2, s39, s2",
    "\ts_addc_u32 s3, s40, s3",
]


def register_range(match):
    if match[1]:
        return (int(match[1]),)
    return range(int(match[2]), int(match[3]) + 1)


def rename(lines, mapping):
    def replace(match):
        registers = list(register_range(match))
        updated = [mapping.get(i, i) for i in registers]
        if len(registers) == 1:
            return f"v{updated[0]}"
        assert updated == list(range(updated[0], updated[0] + len(registers))), match[0]
        return f"v[{updated[0]}:{updated[-1]}]"

    return [VGPR.sub(replace, line) for line in lines]


def references(lines):
    return {reg for line in lines for match in VGPR.finditer(line) for reg in register_range(match)}


def register_pipeline(lines, wait_at, register_base):
    guard = lines.index(".Lh3_v_guard:")
    entry = lines.index(".LBB0_16:", guard)
    max_end = lines.index(".Ltmp1:", entry)
    online = lines.index(".LBB0_18:", max_end)
    pack = lines.index("\tv_lshlrev_b32_e32 v122, 16, v177", online)
    pack_end = lines.index("\tds_write2_b64 v172, v[122:123], v[176:177] offset1:8", pack) + 1
    waiting_registers = range(register_base, register_base + 8)
    assert not references(lines[guard:pack_end]).intersection(waiting_registers)
    assert lines[max_end + 1:max_end + 6] == [
        "\ts_waitcnt vmcnt(0)",
        *(f"\tv_mov_b64_e32 v[{175 + i}:{176 + i}], v[{198 + i}:{199 + i}]" for i in range(0, 8, 2)),
    ]
    load_map = dict(zip(range(198, 206), waiting_registers, strict=True))
    pack_map = dict(zip(range(175, 183), waiting_registers, strict=True))
    pack_map[104] = pack_map[182]
    lines[pack:pack_end] = rename(lines[pack:pack_end], pack_map)
    assert lines.count("\tv_mov_b32_e32 v104, v182") == 1
    alias = lines.index("\tv_mov_b32_e32 v104, v182", online, pack)
    del lines[alias]
    if wait_at == "pack":
        if lines[alias - 1] == "\ts_waitcnt vmcnt(0) lgkmcnt(0)":
            lines[alias - 1] = "\ts_waitcnt lgkmcnt(0)"
            lines.insert(pack - 1, "\ts_waitcnt vmcnt(0)")
        else:
            assert lines[alias - 1] == "\ts_waitcnt vmcnt(0)"
    first_removed = max_end + (2 if wait_at == "early" else 1)
    del lines[first_removed:max_end + 6]
    lines[guard:entry] = rename(lines[guard:entry], load_map)


def address_induction(lines):
    init = lines.index("\ts_mov_b32 s24, 0")
    lines[init:init + 1] = ["\ts_mov_b32 s24, s39", "\ts_mov_b32 s25, s40"]
    guard = lines.index(".Lh3_v_guard:")
    begin = lines.index(V_ADDRESS[0], guard)
    assert lines[begin:begin + 4] == V_ADDRESS
    end = lines.index(".LBB0_16:", begin)
    loads = lines[begin + 4:end]
    assert len([line for line in loads if line.startswith("\tglobal_load_dwordx2 ")]) == 4
    lines[begin:end] = [line.replace("s[2:3]", "s[24:25]") for line in loads]
    step = lines.index("\ts_add_i32 s24, s24, 0x15000", end)
    lines[step:step + 1] = ["\ts_add_u32 s24, s24, 0x2a000", "\ts_addc_u32 s25, s25, 0"]


def preload_next(lines, register_base):
    guard = lines.index(".Lh3_v_guard:")
    load = lines.index(".Lh3_v_load:", guard)
    entry = lines.index(".LBB0_16:", load)
    zeros = [x for x in lines[guard:load] if x.startswith("\tv_mov_b64_e32 ")]
    instructions = [x for x in lines[load + 1:entry] if not x.lstrip().startswith(".loc")]
    assert len(zeros) == 4 and len(instructions) in (4, 8)
    ready = lines.index(".Lh3_next_k_ready:", entry)
    next_load = lines.index(".Lh3_next_k_load:", entry, ready)
    backedge = lines.index("\ts_branch .Lh3_v_guard", ready)
    assert lines[next_load - 1] == "\ts_branch .Lh3_next_k_ready"
    # The consumer path resets the address scratch before use.
    assert re.search(r"\bs[23]\b|s\[2:3\]", "\n".join(lines[ready + 1:backedge])) is None
    assert references(lines[ready + 1:backedge]).isdisjoint(range(register_base, register_base + 8))
    lines[backedge] = "\ts_branch .LBB0_16"
    lines[ready:ready] = instructions
    lines[next_load - 1:next_load - 1] = zeros


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--wait-at", choices=("early", "p-fence", "pack"), required=True)
    cli.add_argument("--address-induction", action="store_true")
    cli.add_argument("--preload-next", action="store_true")
    cli.add_argument("--register-base", type=int, choices=(206, 208), default=206)
    extra = cli.add_mutually_exclusive_group()
    extra.add_argument("--early-p-stores", action="store_true")
    extra.add_argument("--qk-order", action="store_true")
    args = cli.parse_args()
    if args.early_p_stores and args.wait_at != "pack":
        cli.error("--early-p-stores requires --wait-at pack")
    scripts = Path(__file__).parent
    helper = runpy.run_path(str(scripts / "build-disjoint-k-lds.py"))
    lines = helper["verified_assembly"]((args.reference / (helper["STEM"] + ".amdgcn")).read_text())
    if args.early_p_stores:
        p_order = runpy.run_path(str(scripts / "build-p-lds-pipeline.py"))
        p_order["advance_p"](lines, 4, True, False)
    if args.qk_order:
        qk_order = runpy.run_path(str(scripts / "build-qk-online-order.py"))
        qk_order["interleave_scale"](lines, True)
        qk_order["interleave_online"](lines, True)
    register_pipeline(lines, args.wait_at, args.register_base)
    if args.address_induction:
        address_induction(lines)
    if args.preload_next:
        preload_next(lines, args.register_base)
    helper["write_variant"](args.reference, args.output, "\n".join(lines) + "\n", 16384, "v_wait_pipeline",
                            dict(wait_at=args.wait_at, address_induction=args.address_induction,
                                 preload_next=args.preload_next, register_base=args.register_base,
                                 early_p_stores=args.early_p_stores, qk_order=args.qk_order))


if __name__ == "__main__":
    main()
