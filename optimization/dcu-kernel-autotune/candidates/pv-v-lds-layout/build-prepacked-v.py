#!/usr/bin/env python3
"""从求和交错镜像构建直接读取已排布 V 的实验版。"""

import argparse
import json
from pathlib import Path
import re
import runpy
import sys


def transform(lines, position):
    maximum = lines.index(".Ltmp1:")
    assert lines[maximum + 1:maximum + 6] == [
        "\ts_waitcnt vmcnt(0)",
        *(f"\tv_mov_b64_e32 v[{175+i}:{176+i}], v[{198+i}:{199+i}]" for i in range(0, 8, 2)),
    ]
    del lines[maximum + 1:maximum + 6]
    lines.remove("\tv_mov_b32_e32 v104, v182")
    guard = lines.index(".Lh3_v_guard:")
    entry = lines.index(".LBB0_16:", guard)
    del lines[guard + 1:entry]
    wait = lines.index("\ts_waitcnt vmcnt(4) lgkmcnt(0)", guard)
    lines[wait] = "\ts_waitcnt vmcnt(0) lgkmcnt(0)"

    start = lines.index("\tv_lshlrev_b32_e32 v122, 16, v177")
    end = lines.index("\tds_write2_b64 v172, v[122:123], v[176:177] offset1:8", start) + 1
    loads = [
        "\ts_andn2_b64 vcc, exec, s[26:27]",
        "\ts_cbranch_vccz .Lh3_prepacked_v_load",
        *(f"\tv_mov_b64_e32 v[{i}:{i+1}], 0" for i in range(198, 206, 2)),
        "\ts_mov_b32 m0, -1",
        "\tds_write_b128 v142, v[198:201]",
        "\tds_write_b128 v142, v[202:205] offset:4096",
        "\ts_branch .Lh3_prepacked_v_done",
        ".Lh3_prepacked_v_load:",
        "\ts_mov_b32 m0, s32",
        "\tglobal_load_dwordx4 v[198:201], v145, s[50:51] lds",
        "\ts_add_u32 m0, s32, 4096",
        "\tglobal_load_dwordx4 v[202:205], v144, s[50:51] lds",
        "\ts_mov_b32 m0, -1",
        ".Lh3_prepacked_v_done:",
    ]
    if position != "late":
        del lines[start:end]
        anchors = {
            "before-k-write": ("\tds_write_b128 v138, v[166:169]", 0),
            "after-k-write": ("\tds_write_b128 v138, v[170:173] offset:4096", 1),
            "after-k-read": ("\tds_read_b128 v[178:181], v141 offset:4096", 1),
            "after-qk": (".Ltmp0:", 0),
            "after-max": (".LBB0_18:", 1),
        }
        anchor, offset = anchors[position]
        insert = lines.index(anchor) + offset
        lines[insert:insert] = loads
    else:
        lines[start:end] = loads
    publication = lines.index("\tds_read_u16 v182, v122")
    fence = lines.index("\ts_waitcnt lgkmcnt(0)", lines.index("\tds_write_b16_d16_hi v146, v123 offset:4104"))
    assert fence < publication
    lines[fence] = "\ts_waitcnt vmcnt(0) lgkmcnt(0)"

    step = lines.index("\ts_add_i32 s24, s24, 0x15000")
    lines[step:step + 1] = ["\ts_add_u32 s50, s50, 8192", "\ts_addc_u32 s51, s51, 0"]
    init = lines.index("\ts_mov_b32 s45, 0")
    lines[init:init] = [
        "\ts_load_dwordx2 s[50:51], s[4:5], 0x48",
        "\ts_lshr_b32 s52, s16, 8",
        "\ts_mul_i32 s52, s52, s41",
        "\ts_mov_b32 s53, 0",
        "\ts_lshl_b64 s[52:53], s[52:53], 13",
        "\ts_waitcnt lgkmcnt(0)",
        "\ts_add_u32 s50, s50, s52",
        "\ts_addc_u32 s51, s51, s53",
        "\tv_lshlrev_b32_e32 v145, 4, v0",
        "\tv_add_u32_e32 v144, 4096, v145",
        "\tv_and_b32_e32 v142, 0xc00, v145",
        "\tv_add_u32_e32 v142, 8192, v142",
        "\tv_readfirstlane_b32 s32, v142",
    ]
    # Four new ordinary SGPRs fit the original 64-SGPR allocation granule.
    for field, original, updated in (
        (".amdhsa_next_free_sgpr", 50, 54), (".sgpr_count:", 54, 58),
    ):
        pattern = re.compile(rf"(^\s*{re.escape(field)}\s+){original}(\s*)$")
        (index,) = [i for i, line in enumerate(lines) if pattern.fullmatch(line)]
        lines[index] = pattern.sub(rf"\g<1>{updated}\g<2>", lines[index])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--early", action="store_true")
    mode.add_argument("--position", choices=("before-k-write", "after-k-write", "after-k-read",
                                          "after-qk", "after-max", "late"))
    args = cli.parse_args()
    scripts = Path(__file__).parent
    helper = runpy.run_path(str(scripts / "build-disjoint-k-lds.py"))
    lines = helper["verified_assembly"]((args.reference / (helper["STEM"] + ".amdgcn")).read_text())
    position = args.position or ("before-k-write" if args.early else "late")
    transform(lines, position)
    helper["write_variant"](args.reference, args.output, "\n".join(lines) + "\n", 16384, "prepacked_v",
                            dict(position=position, packed_pointer_argument_offset=72))
    code = scripts.parents[2]
    sys.path.insert(0, str(code / "adaln"))
    from aot_codegen import compile_host
    command = compile_host(args.output / "prepacked-host.so", (scripts / "prepacked_attention_host.cc",))
    path = args.output / "dense-attention-aot-build.json"
    metadata = json.loads(path.read_text())
    metadata["experiment"]["resources"]["sgpr_count"] = 58
    metadata["prepacked_host_command"] = command
    path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
