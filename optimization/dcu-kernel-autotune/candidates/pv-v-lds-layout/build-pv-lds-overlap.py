#!/usr/bin/env python3
"""Make the P and V LDS allocations live concurrently before the PV dot."""

import argparse
import hashlib
from pathlib import Path
import runpy

from producer_layout import LINE, SSA


REFERENCE_SHA256 = "8ebfae6b584b253ff0d5e9128ef04765b6f5b5b47c742fedebc4c56ba7110310"


def overlap(source, full_only, v_first):
    if hashlib.sha256(source.encode()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Use the verified packed Q128 IR")
    lines = source.splitlines()
    stages = [index for index, line in enumerate(lines)
              if "ttg.local_alloc" in line and "!ttg.memdesc<128x32xbf16, #shared2, #smem>" in line]
    assert len(stages) == 2
    for index in reversed(stages[:1] if full_only else stages):
        p_alloc, p_load, v_alloc, v_load, dot = lines[index:index + 5]
        definitions = [LINE.fullmatch(line) for line in (p_alloc, p_load, v_alloc, v_load, dot)]
        assert all(definitions)
        assert tuple(match[3].split()[0] for match in definitions) == (
            "ttg.local_alloc", "ttg.local_load", "ttg.local_alloc", "ttg.local_load", "tt.dot")
        assert definitions[0][2] in SSA.findall(p_load)
        assert definitions[2][2] in SSA.findall(v_load)
        assert all(definitions[i][2] in SSA.findall(dot) for i in (1, 3))
        assert "!ttg.memdesc<32x128xbf16, #shared3, #smem>" in v_alloc
        reads = [v_load, p_load] if v_first else [p_load, v_load]
        lines[index:index + 4] = [p_alloc, v_alloc, *reads]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--v-first", action="store_true")
    args = parser.parse_args()
    root = args.reference.resolve(strict=True)
    source = (root / "dense-attention-packed-pv-swizzle.ttgir").read_text()
    compile_packed = runpy.run_path(str(Path(__file__).with_name("build-pv-wide-operand.py")))["compile_packed"]
    compile_packed(root, args.output, overlap(source, args.full_only, args.v_first),
                   dict(pv_concurrent_allocations=True, full_only=args.full_only, v_first=args.v_first),
                   rounding_boundary=False)


if __name__ == "__main__":
    main()
