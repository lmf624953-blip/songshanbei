#!/usr/bin/env python3
"""在原有 16 KiB LDS 中提前同步和发射完整块的 P 写入。"""

import argparse
from pathlib import Path
import re
import runpy


STORE = re.compile(r"\tds_write_b16_d16_hi (v172|v146), (v\d+)(?: offset:\d+)?")
DEST = re.compile(r"\tv_\w+ (v\d+), ")
WAIT = "\ts_waitcnt lgkmcnt(0)"
BARRIER = "\ts_barrier"


def advance_p(lines, batch, move_fence, keep_p_pending):
    start = lines.index(".LBB0_18:") + 1
    first_write = lines.index("\tds_write_b16_d16_hi v172, v122", start)
    end = first_write + 16
    expected = [
        "\tv_add_u32_e32 v172, 0, v132",
        "\ts_waitcnt vmcnt(0) lgkmcnt(0)",
        "\tv_mov_b32_e32 v104, v182",
        BARRIER,
        WAIT,
    ]
    assert lines[first_write - len(expected):first_write] == expected
    stores = [(line, STORE.fullmatch(line)) for line in lines[first_write:end]]
    assert all(match for _, match in stores)
    body = lines[start:first_write - len(expected)]
    # No LDS operation or control-flow change occurs between QK and the P stores.
    assert all(line.lstrip().startswith((".loc", "v_")) for line in body)
    last_write = {match[1]: i for i, line in enumerate(body) if (match := DEST.match(line))}
    assert "v132" not in last_write and "v146" not in last_write
    issued = 0
    scheduled = []
    for i, line in enumerate(body):
        scheduled.append(line)
        while batch and issued < len(stores):
            group = stores[issued:issued + batch]
            if any(last_write[match[2]] > i for _, match in group):
                break
            for store, match in group:
                scheduled.append(store.replace("v172,", "v132,", 1) if match[1] == "v172" else store)
            issued += len(group)
    if batch:
        assert issued == len(stores)
        # The address-copy destination is next defined for the V LDS write.
        assert next(line for line in lines[end:] if re.search(r"\bv172\b", line)) == (
            "\tv_add_u32_e32 v172, 8192, v148")
        remaining = []
    else:
        remaining = [expected[0], *(line for line, _ in stores)]
    if move_fence:
        lines[start:end] = [WAIT, BARRIER, WAIT, *scheduled, "\ts_waitcnt vmcnt(0)", expected[2], *remaining]
    elif keep_p_pending:
        # QK has consumed all K reads at this point. P writes can finish
        # together with V writes at the later P/V publication fence.
        lines[start:end] = [*scheduled, "\ts_waitcnt vmcnt(0)", expected[2], BARRIER]
    else:
        lines[start:end] = [*scheduled, *expected[1:]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--move-p-fence", action="store_true")
    parser.add_argument("--store-batch", type=int, choices=(1, 4))
    parser.add_argument("--drop-trailing-waits", action="store_true")
    parser.add_argument("--k-in-v-region", action="store_true")
    parser.add_argument("--keep-p-pending", action="store_true")
    args = parser.parse_args()
    if args.store_batch and not (args.move_p_fence or args.k_in_v_region):
        parser.error("--store-batch requires --move-p-fence or --k-in-v-region")
    if args.k_in_v_region and args.move_p_fence:
        parser.error("K 与 V 共用 LDS 时保留覆盖前的同步位置")
    if args.keep_p_pending and not (args.k_in_v_region and args.store_batch):
        parser.error("--keep-p-pending requires --k-in-v-region and --store-batch")
    if args.keep_p_pending and args.drop_trailing_waits:
        parser.error("请分别测量两种等待调整")
    build = runpy.run_path(str(Path(__file__).with_name("build-disjoint-k-lds.py")))
    lines = build["verified_assembly"]((args.reference / (build["STEM"] + ".amdgcn")).read_text())
    if args.k_in_v_region:
        build["relocate_k"](lines, 8192)
    if args.move_p_fence or args.store_batch:
        advance_p(lines, args.store_batch, args.move_p_fence, args.keep_p_pending)
    if args.drop_trailing_waits:
        waits = [i + 1 for i, line in enumerate(lines) if line == BARRIER]
        assert len(waits) == 8 and all(lines[i] == WAIT for i in waits)
        for index in reversed(waits):
            del lines[index]
    build["write_variant"](args.reference, args.output, "\n".join(lines) + "\n", 16384, "p_lds_pipeline",
                           dict(early_fence=args.move_p_fence, store_batch=args.store_batch,
                                drop_trailing_waits=args.drop_trailing_waits, k_in_v_region=args.k_in_v_region,
                                keep_p_pending=args.keep_p_pending))


if __name__ == "__main__":
    main()
