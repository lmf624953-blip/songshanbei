#!/usr/bin/env python3
"""Test the alternate contiguous axis for P staging in Q128 packed attention."""

import argparse
import json
from pathlib import Path
import runpy


ORIGINAL = "#shared2 = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 16, order = [0, 1]}>"
PRESETS = {
    "row-v2": (2, 1, 16),
    "row-v4": (4, 1, 8),
    "row-v4-phase2": (4, 2, 8),
    "row-v8": (8, 1, 4),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("preset", choices=PRESETS)
    parser.add_argument("--rounding-fence", action="store_true")
    args = parser.parse_args()
    root = args.reference.resolve(strict=True)
    config = json.loads((root / "dense-attention-aot-build.json").read_text())
    assert config["query_block"] == 128 and config["key_block"] == 32 and config["warps"] == 4
    assert config["packed_strides"]["strides"] == [896, 896, 2688, 896]
    source = (root / "dense-attention-packed-pv-swizzle.ttgir").read_text()
    assert source.count(ORIGINAL) == 1
    assert source.count("!ttg.memdesc<128x32xbf16, #shared2, #smem>") == 4
    vec, phase, phases = PRESETS[args.preset]
    target = (f"#shared2 = #ttg.swizzled_shared<{{vec = {vec}, perPhase = {phase}, "
              f"maxPhase = {phases}, order = [1, 0]}}>")
    source = source.replace(ORIGINAL, target)
    directory = Path(__file__).parent
    if args.rounding_fence:
        helper = runpy.run_path(str(directory / "build-q128-backend.py"))
        source = helper["preserve_scaled_rounding"](source)
    compile_packed = runpy.run_path(str(directory / "build-pv-wide-operand.py"))["compile_packed"]
    compile_packed(root, args.output, source,
                   dict(p_lds_preset=args.preset, p_lds_swizzle=[vec, phase, phases], p_lds_order=[1, 0]),
                   rounding_boundary=args.rounding_fence)


if __name__ == "__main__":
    main()
