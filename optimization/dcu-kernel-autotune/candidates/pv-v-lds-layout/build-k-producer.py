#!/usr/bin/env python3
"""Screen K producer layouts with the original matrix operand mapping."""

import argparse
import json
from pathlib import Path
import re
import runpy

from producer_layout import ALLOWED, LINE, SSA


PRESETS = {
    "d4-n1": ((4, 1), (32, 2), (1, 4)),
    "d8-n2": ((8, 2), (16, 4), (1, 4)),
    "d16-n1": ((16, 1), (8, 8), (1, 4)),
    "d8-n2-wide": ((8, 2), (8, 8), (2, 2)),
}
K_LAYOUT = re.compile(r"#blocked1\b")
K_PRODUCER = "#h3_k_producer"
K_LDS = "#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>"


def change_producer(source, preset):
    lines = source.splitlines()
    definitions = {match[2]: (i, match[3]) for i, line in enumerate(lines)
                   if (match := LINE.fullmatch(line))}
    allocs = [i for i, line in enumerate(lines)
              if "ttg.local_alloc" in line and "!ttg.memdesc<128x32xbf16, #shared1, #smem>" in line]
    assert len(allocs) == 2
    for block, index in reversed(list(enumerate(allocs))):
        alloc = LINE.fullmatch(lines[index])
        value = SSA.search(alloc[3])[0]
        load_index, load = definitions[value]
        assert load_index == index - 1 and load.startswith("tt.load ")
        assert f"tensor<128x32xbf16, #blocked1>" in lines[index]
        emitted, remapped = [], {}

        def clone(name):
            if name in remapped:
                return remapped[name]
            if name not in definitions:
                return name
            position, rhs = definitions[name]
            if not K_LAYOUT.search(rhs):
                return name
            operation = rhs.split(" ", 1)[0]
            if position > load_index or operation not in ALLOWED or (operation == "tt.load" and name != value):
                raise ValueError(f"Unexpected K producer dependency: {name} {operation}")
            rewritten = SSA.sub(lambda match: clone(match[0]), rhs)
            label = f"%h3_k_producer{block}_{len(emitted)}"
            remapped[name] = label
            emitted.append(f"      {label} = " + K_LAYOUT.sub(K_PRODUCER, rewritten))
            return label

        packed = clone(value)
        assert packed != value and 10 < len(emitted) < 120
        lines[index] = K_LAYOUT.sub(K_PRODUCER, lines[index].replace(value, packed))
        lines[load_index:index] = emitted
    per_thread, lanes, warps = map(list, PRESETS[preset])
    header = (f"{K_PRODUCER} = #ttg.blocked<{{sizePerThread = {per_thread}, threadsPerWarp = {lanes}, "
              f"warpsPerCTA = {warps}, order = [0, 1]}}>")
    return header + "\n" + "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("preset", choices=(*PRESETS, "original"))
    parser.add_argument("--row-lds", action="store_true")
    parser.add_argument("--rounding-fence", action="store_true")
    args = parser.parse_args()
    root = args.reference.resolve(strict=True)
    config = json.loads((root / "dense-attention-aot-build.json").read_text())
    assert (config["query_block"], config["key_block"], config["warps"]) == (128, 32, 4)
    assert config["packed_strides"]["strides"] == [896, 896, 2688, 896]
    source = (root / "dense-attention-packed-pv-swizzle.ttgir").read_text()
    if args.preset != "original":
        source = change_producer(source, args.preset)
    if args.row_lds:
        assert source.count(K_LDS) == 1
        source = source.replace(K_LDS, K_LDS.replace(
            "vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]",
            "vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0]"))
    directory = Path(__file__).parent
    if args.rounding_fence:
        fence = runpy.run_path(str(directory / "build-q128-backend.py"))["preserve_scaled_rounding"]
        source = fence(source)
    compile_packed = runpy.run_path(str(directory / "build-pv-wide-operand.py"))["compile_packed"]
    compile_packed(root, args.output, source, dict(k_producer=args.preset, k_row_lds=args.row_lds),
                   rounding_boundary=args.rounding_fence)


if __name__ == "__main__":
    main()
