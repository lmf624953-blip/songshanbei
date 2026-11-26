#!/usr/bin/env python3
"""在 Q128 packed 镜像上筛选 P 相位、QK 操作数宽度与矩阵线程布局。"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import triton
from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.compiler import GPUTarget


P_LAYOUT = "#shared2 = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 16, order = [0, 1]}>"
VARIANTS = {
    "p-phase1": (8, 1, 16, 8, [4, 1]),
    "p-max8": (8, 2, 8, 8, [4, 1]),
    "p-phase4": (8, 4, 16, 8, [4, 1]),
    "p-vec16": (16, 1, 8, 8, [4, 1]),
    "qk-width4": (8, 2, 16, 4, [4, 1]),
    "mma-2x2": (8, 2, 16, 8, [2, 2]),
    "warps8": (8, 2, 16, 8, [8, 1]),
    "warps8-pmax8": (8, 2, 8, 8, [8, 1]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("variant", choices=VARIANTS)
    args = parser.parse_args()
    root = args.reference.resolve(strict=True)
    output = args.output.resolve()
    record = json.loads((root / "dense-attention-aot-build.json").read_text())
    assert record["query_block"] == 128 and record["shared"] == 8192
    assert record["experiment"]["compact_args"] == "common-length-fixed-strides"
    source = (root / "pv-swizzle.ttgir").read_text()
    assert source.count(P_LAYOUT) == 1
    vec, phase, phases, width, warps = VARIANTS[args.variant]
    source = source.replace(P_LAYOUT, f"#shared2 = #ttg.swizzled_shared<{{vec = {vec}, perPhase = {phase}, maxPhase = {phases}, order = [0, 1]}}>")
    if width == 4:
        assert source.count("kWidth = 8") == 7
        source = source.replace("kWidth = 8", "kWidth = 4")
    if warps == [2, 2]:
        original = "#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>"
        assert source.count(original) == 1
        source = source.replace(original, original.replace("[4, 1]", "[2, 2]"))
    num_warps = 8 if warps == [8, 1] else 4
    if num_warps == 8:
        assert source.count('"ttg.num-warps" = 4 : i32') == 1
        source = source.replace('"ttg.num-warps" = 4 : i32', '"ttg.num-warps" = 8 : i32')
        source = source.replace("warpsPerCTA = [4, 1]", "warpsPerCTA = [8, 1]")
        source = source.replace("warpsPerCTA = [1, 4]", "warpsPerCTA = [1, 8]")
    output.mkdir()
    ir = output / "pv-swizzle.ttgir"
    ir.write_text(source)
    os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"
    configure_hygon_compiler()
    compiled = triton.compile(str(ir), target=GPUTarget("hip", "gfx936", 64), options={
        "num_warps": num_warps, "num_stages": 1, "matrix_instr_nonkdim": 0, "kpack": 2,
    })
    asm = compiled.asm["amdgcn"]
    resources = {key: int(value) for key, value in re.findall(
        r"^[ \t]*\.(vgpr_count|sgpr_count|private_segment_fixed_size):[ \t]*(\d+)", asm, re.M)}
    assert compiled.metadata.profile_scratch_size == 0
    isa = dict(Counter(re.findall(r"^[ \t]*(ds_\w+|v_mmac_\w+|global_\w+|s_barrier|s_waitcnt)\b", asm, re.M)))
    match = re.search(r"define amdgpu_kernel void @dense_attention_aot_kernel\(([^\n]*)\)", compiled.asm["llir"])
    assert match and re.findall(r"(ptr addrspace\(1\)|i32|float) inreg", match[1]) == (
        ["ptr addrspace(1)"] * 3 + ["float", "ptr addrspace(1)"] + ["i32"] * 8 + ["ptr addrspace(1)"] * 2)
    image = compiled.asm["hsaco"]
    (output / "dense-attention-aot.hsaco").write_bytes(image)
    (output / "dense-attention-aot.amdgcn").write_text(asm)
    for name in ("dense-attention-aot.so", "dense-attention-fallback.hsaco"):
        shutil.copy2(root / name, output / name)
    record.update(hsaco_sha256=hashlib.sha256(image).hexdigest(), shared=compiled.metadata.shared,
                  source=str(ir), warps=num_warps)
    record["experiment"].update(backend_variant=args.variant, isa=isa, resources=resources)
    (output / "dense-attention-aot-build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(name=args.variant, resources=resources, shared=record["shared"], isa=isa)), flush=True)


if __name__ == "__main__":
    main()
