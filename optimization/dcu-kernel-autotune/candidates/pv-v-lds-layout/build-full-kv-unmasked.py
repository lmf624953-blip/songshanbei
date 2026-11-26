#!/usr/bin/env python3
"""Remove redundant load predicates in the complete KV tiles of Q128 packed."""

import argparse
import hashlib
from pathlib import Path
import re
import runpy


SOURCE_SHA256 = "8ebfae6b584b253ff0d5e9128ef04765b6f5b5b47c742fedebc4c56ba7110310"
LOAD_TYPES = {
    "k": "tensor<128x32x!tt.ptr<bf16>, #blocked1>",
    "v": "tensor<32x128x!tt.ptr<bf16>, #h3_v_producer>",
}


def unmask(source, operands):
    # This fingerprint fixes both the launch mapping and all original predicates.
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Use the verified Q128 packed reference IR")
    start = source.index(" = scf.for ")
    end = source.index("      scf.yield ", start)
    loop = source[start:end]
    if "to %v18_loop_body step %c1_i32" not in loop:
        raise ValueError("Expected the complete-tile loop ending at floor(rows / 32)")
    for operand in operands:
        pattern = (r"(tt\.load %[-.\w]+), %[-.\w]+, %[-.\w]+"
                   r"( : " + re.escape(LOAD_TYPES[operand]) + r")")
        loop, count = re.subn(pattern, r"\1\2", loop)
        if count != 1:
            raise ValueError(f"Expected one complete-tile load for {operand}")
    return source[:start] + loop + source[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("operands", choices=("k", "v", "kv"))
    parser.add_argument("--rounding-fence", action="store_true")
    args = parser.parse_args()
    root = args.reference.resolve(strict=True)
    source = (root / "dense-attention-packed-pv-swizzle.ttgir").read_text()
    source = unmask(source, args.operands)
    directory = Path(__file__).parent
    if args.rounding_fence:
        fence = runpy.run_path(str(directory / "build-q128-backend.py"))["preserve_scaled_rounding"]
        source = fence(source)
    compile_packed = runpy.run_path(str(directory / "build-pv-wide-operand.py"))["compile_packed"]
    compile_packed(root, args.output, source,
                   dict(full_tile_unmasked=args.operands, reference_ir_sha256=SOURCE_SHA256),
                   rounding_boundary=args.rounding_fence)


if __name__ == "__main__":
    main()
