#!/usr/bin/env python3
"""Change only the workgroup enumeration of Q128 packed attention."""

import argparse
import json
from pathlib import Path
import re
import runpy


def permute(source, group):
    pid = "%h3_invariant_query_0_43"
    tiles = "%h3_invariant_query_0_46"
    size = "%h3_invariant_query_0_47"
    marker = f"    {size} = arith.muli {tiles}, %c7_i32 : i32 loc(#loc384)\n"
    assert source.count(marker) == 1
    assert source.count(f"{pid} = tt.get_program_id x : i32") == 1
    before, after = source.split(marker, 1)
    assert len(re.findall(re.escape(pid) + r"\b", after)) == 4
    if group == 1:
        order = f"""    %h3_order_head = arith.remui {pid}, %c7_i32 : i32
    %h3_order_q = arith.divui {pid}, %c7_i32 : i32
"""
    else:
        order = f"""    %h3_order_group_size = arith.constant {group} : i32
    %h3_order_group_ctas = arith.constant {group * 7} : i32
    %h3_order_group = arith.divui {pid}, %h3_order_group_ctas : i32
    %h3_order_base = arith.muli %h3_order_group, %h3_order_group_size : i32
    %h3_order_left = arith.subi {tiles}, %h3_order_base : i32
    %h3_order_width = arith.minui %h3_order_left, %h3_order_group_size : i32
    %h3_order_local = arith.remui {pid}, %h3_order_group_ctas : i32
    %h3_order_head = arith.divui %h3_order_local, %h3_order_width : i32
    %h3_order_offset = arith.remui %h3_order_local, %h3_order_width : i32
    %h3_order_q = arith.addi %h3_order_base, %h3_order_offset : i32
"""
    order += f"""    %h3_order_head_base = arith.muli %h3_order_head, {tiles} : i32
    %h3_order_pid = arith.addi %h3_order_head_base, %h3_order_q : i32
"""
    return before + marker + order + re.sub(re.escape(pid) + r"\b", "%h3_order_pid", after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--query-group", choices=(0, 1, 4, 8), type=int, required=True)
    parser.add_argument("--rounding-fence", action="store_true")
    args = parser.parse_args()
    root = args.reference.resolve(strict=True)
    config = json.loads((root / "dense-attention-aot-build.json").read_text())
    assert config["query_block"] == 128 and config["key_block"] == 32 and config["warps"] == 4
    assert config["packed_strides"]["strides"] == [896, 896, 2688, 896]
    source = (root / "dense-attention-packed-pv-swizzle.ttgir").read_text()
    directory = Path(__file__).parent
    compile_packed = runpy.run_path(str(directory / "build-pv-wide-operand.py"))["compile_packed"]
    if args.query_group:
        source = permute(source, args.query_group)
    if args.rounding_fence:
        rounding = runpy.run_path(str(directory / "build-q128-backend.py"))["preserve_scaled_rounding"]
        source = rounding(source)
    order = "query group, head, query within group" if args.query_group else "original"
    compile_packed(root, args.output, source, dict(query_group=args.query_group, enumeration=order),
                   rounding_boundary=args.rounding_fence)


if __name__ == "__main__":
    main()
