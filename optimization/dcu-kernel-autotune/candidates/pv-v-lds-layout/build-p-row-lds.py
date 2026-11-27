#!/usr/bin/env python3
"""将完整块 P 改为 row-v4-phase2 LDS 布局，直接读回 PV 的寄存器对。"""

import argparse
from pathlib import Path
import re
import runpy


OFFSETS = (0, 8, 4224, 4232, 128, 136, 4096, 4104)
READ_OFFSETS = (0, 384, 4480, 4352, 4224, 4096, 256, 128)
STORE = re.compile(r"\tds_write_b16_d16_hi (v172|v146), (v\d+)(?: offset:(\d+))?")


def row_address(old):
    key, swizzled_query = divmod(old // 2, 128)
    query = ((swizzled_query // 8) ^ ((key // 2) % 16)) * 8 + swizzled_query % 8
    return 2 * (query * 32 + ((key // 4) ^ ((query // 2) % 8)) * 4 + key % 4)


def store_base(tid):
    base = ((tid & 15) << 1) | ((tid & 48) << 2) | ((tid & 192) << 4)
    return base ^ ((tid & 32) >> 2)


def read_base(tid):
    high = ((tid & 15) << 6) | ((tid & 192) << 4)
    low = ((tid & 14) << 2) ^ ((tid & 48) >> 1)
    return high | low


def verify_mapping():
    addresses = set()
    for tid in range(256):
        old = (((tid & 15) << 8) | ((tid & 14) << 3)) ^ (((tid & 192) >> 1) | ((tid & 48) >> 3))
        for index in range(16):
            base = store_base(tid)
            if index % 2:
                base ^= 0x110
            if index >= 8:
                base ^= 0x200
            offsets = (0, 0, 32, 32, 4096, 4096, 4128, 4128)
            if index >= 8:
                offsets = (32, 32, 0, 0, 4128, 4128, 4096, 4096)
            actual = base + offsets[index % 8]
            expected = row_address((old ^ (16 if index >= 8 else 0)) + OFFSETS[index % 8])
            assert actual == expected
            assert actual not in addresses
            addresses.add(actual)
        old = (((tid & 48) << 6) | ((tid << 1) & 126)) ^ ((tid & 192) >> 1)
        current = [row_address((old ^ (528 if i >= 8 else 0)) + READ_OFFSETS[i % 8])
                   for i in range(16)]
        order = ((0, 6), (8, 14), (4, 2), (12, 10), (7, 1), (15, 9), (5, 3), (13, 11))
        packed_words = []
        base = read_base(tid)
        for address in (base, base ^ 32, base + 4096, (base ^ 32) + 4096):
            packed_words.extend(((address, address + 2), (address + 4, address + 6)))
        assert [(current[a], current[b]) for a, b in order] == packed_words
    assert addresses == set(range(0, 8192, 2))


def transform(lines):
    verify_mapping()
    first = lines.index("\tds_write_b16_d16_hi v172, v122")
    stores = [STORE.fullmatch(line) for line in lines[first:first + 16]]
    assert all(stores)
    for i, match in enumerate(stores):
        address = (132, 146, 217, 218)[(2 if i >= 8 else 0) + i % 2]
        offsets = (0, 0, 32, 32, 4096, 4096, 4128, 4128)
        if i >= 8:
            offsets = (32, 32, 0, 0, 4128, 4128, 4096, 4096)
        offset = offsets[i % 8]
        suffix = f" offset:{offset}" if offset else ""
        lines[first + i] = f"\tds_write_b16_d16_hi v{address}, {match[2]}{suffix}"
    copy = lines.index("\tv_add_u32_e32 v172, 0, v132", 0, first)
    del lines[copy]

    read = lines.index("\tds_read_u16 v182, v122")
    assert lines[read - 1] == "\tv_add_u32_e32 v122, 0, v133"
    assert sum(line.startswith("\tds_read_u16 ") for line in lines[read:read + 16]) == 16
    last_perm = lines.index("\tv_perm_b32 v186, v186, v206, s43", read)
    region = lines[read + 16:last_perm + 1]
    assert region[0] == "\ts_waitcnt lgkmcnt(7)"
    assert sum(line.startswith("\tv_perm_b32 ") for line in region) == 8
    # v122 and v123 now hold P. Use a V destination which is not written
    # until the second set of V reads, after the first address is consumed.
    address = region.index("\tv_add_u32_e32 v122, 0, v135")
    region[address] = "\tv_add_u32_e32 v176, 0, v135"
    reads = [i for i, line in enumerate(region) if line.startswith("\tds_read2st64_b64 ")]
    assert len(reads) == 8
    for i in reads[:4]:
        assert ", v122 " in region[i]
        region[i] = region[i].replace(", v122 ", ", v176 ")
    region = [line for line in region if not line.startswith("\tv_perm_b32 ")]
    lines[read - 1:last_perm + 1] = [
        "\tds_read_b64 v[122:123], v133",
        "\tds_read_b64 v[182:183], v147",
        "\tds_read_b64 v[184:185], v133 offset:4096",
        "\tds_read_b64 v[186:187], v147 offset:4096",
        *region,
    ]
    init = lines.index("\ts_mov_b32 s45, 0")
    lines[init:init] = [
        "\tv_and_b32_e32 v132, 15, v0",
        "\tv_lshlrev_b32_e32 v132, 1, v132",
        "\tv_and_b32_e32 v146, 48, v0",
        "\tv_lshl_or_b32 v132, v146, 2, v132",
        "\tv_and_b32_e32 v217, 0xc0, v0",
        "\tv_lshl_or_b32 v132, v217, 4, v132",
        "\tv_and_b32_e32 v217, 32, v0",
        "\tv_lshrrev_b32_e32 v217, 2, v217",
        "\tv_xor_b32_e32 v132, v132, v217",
        "\tv_xor_b32_e32 v146, 0x110, v132",
        "\tv_xor_b32_e32 v217, 0x200, v132",
        "\tv_and_b32_e32 v133, 15, v0",
        "\tv_lshlrev_b32_e32 v133, 6, v133",
        "\tv_and_b32_e32 v147, 0xc0, v0",
        "\tv_lshl_or_b32 v133, v147, 4, v133",
        "\tv_and_b32_e32 v147, 14, v0",
        "\tv_lshlrev_b32_e32 v147, 2, v147",
        "\tv_and_b32_e32 v218, 48, v0",
        "\tv_lshrrev_b32_e32 v218, 1, v218",
        "\tv_xor_b32_e32 v147, v147, v218",
        "\tv_or_b32_e32 v133, v133, v147",
        "\tv_xor_b32_e32 v147, 32, v133",
        "\tv_xor_b32_e32 v218, 0x310, v132",
    ]
    for field in (".amdhsa_next_free_vgpr", ".vgpr_count:"):
        pattern = re.compile(rf"(^\s*{re.escape(field)}\s+)217(\s*)$")
        (index,) = [i for i, line in enumerate(lines) if pattern.fullmatch(line)]
        lines[index] = pattern.sub(r"\g<1>219\g<2>", lines[index])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    helper = runpy.run_path(str(Path(__file__).with_name("build-disjoint-k-lds.py")))
    lines = helper["verified_assembly"]((args.reference / (helper["STEM"] + ".amdgcn")).read_text())
    transform(lines)
    helper["write_variant"](args.reference, args.output, "\n".join(lines) + "\n", 16384, "p_row_lds",
                            dict(layout="row-v4-phase2", full_block_reads=4))


if __name__ == "__main__":
    main()
