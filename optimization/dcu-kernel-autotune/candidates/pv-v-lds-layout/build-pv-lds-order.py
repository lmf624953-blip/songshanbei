#!/usr/bin/env python3
"""Reorder V staging relative to softmax and P staging in Q128 packed IR."""

import argparse
import json
from pathlib import Path
import re
import runpy

from producer_layout import LINE


def reorder(source, position):
    lines = source.splitlines()
    allocations = [i for i, line in enumerate(lines)
                   if " = ttg.local_alloc " in line and "#shared3, #smem>" in line]
    assert len(allocations) == 2
    for index in reversed(allocations):
        alloc = LINE.fullmatch(lines[index])
        load = LINE.fullmatch(lines[index + 1])
        assert alloc and load and load[3].startswith(f"ttg.local_load {alloc[2]} :")
        assert " = tt.dot " in lines[index + 2] and f", {load[2]}, " in lines[index + 2]
        assert " = ttg.local_alloc " in lines[index - 2] and "#shared2, #smem>" in lines[index - 2]
        start = max(i for i in range(index) if " = scf.for " in lines[i])
        if position == "swap":
            anchor = index - 2
        else:
            operation = " = tt.dot " if position == "after-qk" else " = arith.truncf "
            anchors = [i for i in range(start, index) if operation in lines[i]]
            assert len(anchors) == 1
            anchor = anchors[0] + (position == "after-qk")
        v_input = alloc[3].split()[1]
        assert re.fullmatch(r"%[-.\w]+", v_input)
        assert any(line.startswith(f"      {v_input} = tt.load ") for line in lines[start:anchor])
        pair = lines[index:index + 2]
        del lines[index:index + 2]
        lines[anchor:anchor] = pair
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("position", choices=("swap", "after-qk", "before-probability-cast", "prefetch-only"))
    parser.add_argument("--rounding-fence", action="store_true")
    args = parser.parse_args()
    root = args.reference.resolve(strict=True)
    record = json.loads((root / "dense-attention-aot-build.json").read_text())
    assert record["query_block"] == 128 and record["key_block"] == 32 and record["warps"] == 4
    assert record["packed_strides"]["strides"] == [896, 896, 2688, 896]
    source = (root / "dense-attention-packed-pv-swizzle.ttgir").read_text()
    directory = Path(__file__).parent
    prefetch = args.position != "swap"
    if prefetch or args.rounding_fence:
        backend = runpy.run_path(str(directory / "build-q128-backend.py"))
        if prefetch:
            source = backend["prefetch_value"](source)
        if args.rounding_fence:
            source = backend["preserve_scaled_rounding"](source)
    if args.position != "prefetch-only":
        source = reorder(source, args.position)
    compile_packed = runpy.run_path(str(directory / "build-pv-wide-operand.py"))["compile_packed"]
    compile_packed(root, args.output, source,
                   dict(v_lds_position=args.position, v_global_prefetch=prefetch),
                   rounding_boundary=args.rounding_fence)


if __name__ == "__main__":
    main()
