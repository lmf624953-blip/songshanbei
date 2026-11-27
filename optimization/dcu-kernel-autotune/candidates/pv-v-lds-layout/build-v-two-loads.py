#!/usr/bin/env python3
"""将完整 V 块改为每线程两行、每行八列，使用两条 128 位读取。"""

import argparse
from pathlib import Path
import runpy


def producer_addresses(lines):
    # These four addresses have no remaining consumers before the tail.
    index = lines.index("\ts_mov_b32 s45, 0")
    lines[index:index] = [
        "\tv_lshrrev_b32_e32 v142, 4, v0",
        "\tv_mul_u32_u24_e32 v145, 0x2a00, v142",
        "\tv_and_b32_e32 v143, 15, v0",
        "\tv_lshlrev_b32_e32 v144, 4, v143",
        "\tv_add_u32_e32 v145, v145, v144",
        "\tv_add_u32_e32 v144, 0x1500, v145",
        "\tv_and_b32_e32 v148, 1, v143",
        "\tv_lshlrev_b32_e32 v148, 3, v148",
        "\tv_xor_b32_e32 v134, v142, v148",
        "\tv_lshlrev_b32_e32 v134, 2, v134",
        "\tv_lshl_or_b32 v134, v143, 9, v134",
        "\tv_xor_b32_e32 v148, 0x88, v134",
        "\tv_xor_b32_e32 v142, 0x110, v134",
        "\tv_xor_b32_e32 v143, 0x198, v134",
    ]


def replace_loads(lines):
    first = lines.index("\tglobal_load_dwordx2 v[198:199], v145, s[2:3]")
    expected = [f"\tglobal_load_dwordx2 v[{198 + 2*i}:{199 + 2*i}], v{145-i}, s[2:3]" for i in range(4)]
    assert lines[first:first + 4] == expected
    lines[first:first + 4] = [
        "\tglobal_load_dwordx4 v[198:201], v145, s[2:3]",
        "\tglobal_load_dwordx4 v[202:205], v144, s[2:3]",
    ]
    wait = lines.index("\ts_waitcnt vmcnt(4) lgkmcnt(0)", first)
    # Two V requests remain after the preceding two K requests complete.
    lines[wait] = "\ts_waitcnt vmcnt(2) lgkmcnt(0)"


def pack_stores(lines, wait_at):
    first = "\tv_lshlrev_b32_e32 v122, 16, v177"
    last = "\tds_write2_b64 v172, v[122:123], v[176:177] offset1:8"
    delayed = bool(wait_at)
    if delayed:
        first = first.replace("v177", "v208")
        last = last.replace("v[176:177]", "v[207:208]")
    start = lines.index(first)
    end = lines.index(last, start) + 1
    words = [175, 176, 177, 178, 179, 180, 181, 104]
    if delayed:
        words = list(range(206, 214))
    out = []
    for colpair, address in enumerate((134, 148, 142, 143)):
        low = words[colpair]
        high = words[colpair + 4]
        out += [
            f"\tv_perm_b32 v122, v{high}, v{low}, s43",
            f"\tv_perm_b32 v123, v{high}, v{low}, s44",
            f"\tv_add_u32_e32 v172, 8192, v{address}",
            "\tds_write2_b32 v172, v122, v123 offset1:16",
        ]
    lines[start:end] = out


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--wait-at", choices=("early", "p-fence", "pack"))
    args = cli.parse_args()
    scripts = Path(__file__).parent
    helper = runpy.run_path(str(scripts / "build-disjoint-k-lds.py"))
    lines = helper["verified_assembly"]((args.reference / (helper["STEM"] + ".amdgcn")).read_text())
    producer_addresses(lines)
    replace_loads(lines)
    if args.wait_at:
        pipe = runpy.run_path(str(scripts / "build-v-wait-pipeline.py"))
        pipe["register_pipeline"](lines, args.wait_at, 206)
    pack_stores(lines, args.wait_at)
    helper["write_variant"](args.reference, args.output, "\n".join(lines) + "\n", 16384, "v_two_loads",
                            dict(wait_at=args.wait_at, rows_per_thread=2, columns_per_thread=8))


if __name__ == "__main__":
    main()
