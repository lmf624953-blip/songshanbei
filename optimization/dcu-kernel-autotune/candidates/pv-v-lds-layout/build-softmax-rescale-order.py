#!/usr/bin/env python3
"""Move the FP32 accumulator rescale within the original online softmax."""

import argparse
import json
from pathlib import Path
import runpy

from producer_layout import LINE, SSA


def reschedule(source, placement):
    lines = source.splitlines()
    scales = [i for i, line in enumerate(lines)
              if " = arith.mulf " in line and " : tensor<128x128xf32, #mma>" in line]
    assert len(scales) == 2
    for scale in reversed(scales):
        begin = scale - (0 if placement == "multiply-before-pv" else 4)
        chain = lines[begin:scale + 1]
        expected = ("arith.mulf",) if begin == scale else (
            "arith.subf", "math.exp2", "tt.expand_dims", "tt.broadcast", "arith.mulf")
        parsed = [LINE.fullmatch(line) for line in chain]
        assert all(parsed)
        assert tuple(match[3].split()[0] for match in parsed) == expected
        assert all(previous[2] in SSA.findall(current[3])
                   for previous, current in zip(parsed, parsed[1:]))
        end = next(i for i in range(scale + 1, len(lines)) if lines[i].startswith("      scf.yield "))
        pv = [i for i in range(scale + 1, end)
              if " = tt.dot " in lines[i] and parsed[-1][2] in SSA.findall(lines[i])]
        assert len(pv) == 1
        if placement == "chain-after-p":
            casts = [i for i in range(scale + 1, pv[0]) if " = arith.truncf " in lines[i]]
            assert len(casts) == 1
            anchor = casts[0] + 1
        else:
            anchor = pv[0]
        moved = {match[2] for match in parsed}
        assert not any(moved.intersection(SSA.findall(line)) for line in lines[scale + 1:anchor])
        lines[anchor:anchor] = chain
        del lines[begin:scale + 1]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("placement", choices=("chain-after-p", "multiply-before-pv", "chain-before-pv"))
    parser.add_argument("--rounding-fence", action="store_true")
    args = parser.parse_args()
    root = args.reference.resolve(strict=True)
    config = json.loads((root / "dense-attention-aot-build.json").read_text())
    assert (config["query_block"], config["key_block"], config["warps"]) == (128, 32, 4)
    assert config["packed_strides"]["strides"] == [896, 896, 2688, 896]
    source = (root / "dense-attention-packed-pv-swizzle.ttgir").read_text()
    source = reschedule(source, args.placement)
    directory = Path(__file__).parent
    if args.rounding_fence:
        fence = runpy.run_path(str(directory / "build-q128-backend.py"))["preserve_scaled_rounding"]
        source = fence(source)
    compile_packed = runpy.run_path(str(directory / "build-pv-wide-operand.py"))["compile_packed"]
    compile_packed(root, args.output, source, dict(rescale_placement=args.placement),
                   rounding_boundary=args.rounding_fence)


if __name__ == "__main__":
    main()
