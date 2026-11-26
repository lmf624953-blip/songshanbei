#!/usr/bin/env python3
"""以每线程一个归约元素复核 CUB 1024 线程的宽 RMSNorm 加和顺序。"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import triton
from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.compiler import GPUTarget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.reference.resolve(strict=True)
    out = args.output.resolve()
    record = json.loads((root / "build.json").read_text())
    assert record["mode"] == "cub-padded" and record["warps"] == 16
    source = (root / "wide-norm-indexed.ttgir").read_text()
    previous = "#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [16], order = [0]}>"
    assert source.count(previous) == 1
    source = source.replace(previous, previous.replace("sizePerThread = [8]", "sizePerThread = [1]"))
    out.mkdir()
    ir = out / "wide-norm-indexed.ttgir"
    ir.write_text(source)
    configure_hygon_compiler()
    compiled = triton.compile(str(ir), target=GPUTarget("hip", "gfx936", 64),
                              options={"num_warps": 16, "num_stages": 1,
                                       "matrix_instr_nonkdim": 0, "kpack": 1})
    assert compiled.metadata.profile_scratch_size == 0
    image = compiled.asm["hsaco"]
    (out / "wide-norm-indexed.hsaco").write_bytes(image)
    (out / "wide-norm-indexed.amdgcn").write_text(compiled.asm["amdgcn"])
    record.update(hsaco_sha256=hashlib.sha256(image).hexdigest(), shared=compiled.metadata.shared,
                  source=str(ir), experiment="sizePerThread=1")
    (out / "build.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
