#!/usr/bin/env python3
"""Use equivalent wave lane maps in the frozen Q128 softmax reductions."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget


REFERENCE_SHA256 = "170beed8a92797444a6525b06d5f0bd43935b89b2cac83ee0652f02dc7d07e9c"
REFERENCE_NORMALIZED_SHA256 = "eb73de5d8905d1b637b84ff9700f71a0ffdef3031b5314cd1db2b043ba0d9611"
LOCATIONS = {58: "max", 89: "max", 72: "sum", 107: "sum"}
UPDATE = re.compile(
    r"  (%o\d+) = tail call i32 @llvm\.amdgcn\.update\.dpp\.i32\("
    r"i32 (%o\d+), i32 (%o\d+), i32 (\d+), i32 15, i32 (\d+), i1 false\), !dbg !(\d+)"
)


def verify_lane_maps():
    for lane in range(64):
        row, column = lane & ~15, lane & 15
        assert (row | ((column + 8) & 15)) == lane ^ 8
        for delta in (4, 8):
            subgroup, local = lane & ~31, lane & 31
            assert (subgroup | ((local & 31) ^ delta)) == lane ^ delta


def transform(source, row8, row4, scope, mov_dpp):
    normalized = re.sub(r'^(\s*!\d+ = !DIFile\(filename: "[^"]*", directory: )"[^"]*"',
                        r'\1""', source, flags=re.M)
    if hashlib.sha256(normalized.encode()).hexdigest() != REFERENCE_NORMALIZED_SHA256:
        raise ValueError("Use the verified Q128 packed LLVM IR")
    verify_lane_maps()
    source = re.sub(r"%(\d+)\b", r"%o\1", source)
    source = re.sub(r"^(\d+):", r"o\1:", source, flags=re.M)
    lines = source.splitlines()
    entry = next(i for i, line in enumerate(lines) if line.startswith("define amdgpu_kernel "))
    lines.insert(entry + 1, "o15:")
    replacements = Counter()
    for index in reversed(range(len(lines) - 1)):
        first = UPDATE.fullmatch(lines[index])
        if not first:
            continue
        if first[4] in ("78", "177") and mov_dpp:
            assert first[2] == first[3] and first[5] == "15"
            stage = LOCATIONS[int(first[6])]
            if scope in ("both", stage):
                lines[index] = (f"  {first[1]} = tail call i32 @llvm.amdgcn.mov.dpp.i32(i32 {first[3]}, "
                                f"i32 {first[4]}, i32 15, i32 15, i1 false), !dbg !{first[6]}")
                replacements[f"{stage}_quad_mov"] += 1
            continue
        if first[4] not in ("276", "280"):
            continue
        second = UPDATE.fullmatch(lines[index + 1])
        assert second and first[2] == first[3] == second[3] and second[2] == first[1]
        delta = int(first[4]) - 272
        assert (int(second[4]), int(first[5]), int(second[5]), first[6]) == (
            256 + delta, 10 if delta == 4 else 12, 5 if delta == 4 else 3, second[6])
        stage = LOCATIONS[int(first[6])]
        mode = row4 if delta == 4 else row8
        if mode == "original" or scope not in ("both", stage):
            continue
        if mode == "rotate" and mov_dpp:
            instruction = (f"tail call i32 @llvm.amdgcn.mov.dpp.i32("
                           f"i32 {first[3]}, i32 296, i32 15, i32 15, i1 false)")
        elif mode == "rotate":
            instruction = (f"tail call i32 @llvm.amdgcn.update.dpp.i32("
                           f"i32 {first[3]}, i32 {first[3]}, i32 296, i32 15, i32 15, i1 false)")
        else:
            mask = 31 | (delta << 10)
            instruction = f"tail call i32 @llvm.amdgcn.ds.swizzle(i32 {first[3]}, i32 {mask})"
        lines[index:index + 2] = [f"  {second[1]} = {instruction}, !dbg !{second[6]}"]
        replacements[f"{stage}_xor{delta}"] += 1
    expected = {f"{stage}_xor{delta}": 16 for delta, mode in ((4, row4), (8, row8))
                for stage in ("max", "sum") if mode != "original" and scope in ("both", stage)}
    if mov_dpp:
        expected.update({f"{stage}_quad_mov": 32 for stage in ("max", "sum") if scope in ("both", stage)})
        lines.append("declare i32 @llvm.amdgcn.mov.dpp.i32(i32, i32 immarg, i32 immarg, i32 immarg, i1 immarg)")
    assert replacements == expected
    if "ds" in (row8, row4):
        lines.append("declare i32 @llvm.amdgcn.ds.swizzle(i32, i32 immarg)")
    return "\n".join(lines) + "\n", replacements


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--row8", choices=("original", "rotate", "ds"), default="rotate")
    cli.add_argument("--row4", choices=("original", "ds"), default="original")
    cli.add_argument("--scope", choices=("both", "max", "sum"), default="both")
    cli.add_argument("--mov-dpp", action="store_true")
    cli.add_argument("--dpp-combine", action="store_true")
    args = cli.parse_args()
    reference = args.reference.resolve(strict=True)
    source, replacements = transform((reference / "dense-attention-packed.llir").read_text(),
                                     args.row8, args.row4, args.scope, args.mov_dpp)
    os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    options = backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2))
    metadata = {}
    if args.dpp_combine:
        from triton._C.libtriton import amd, llvm

        metadata["name"] = "dense_attention_aot_kernel"
        compiler_source = source
        if os.environ.get("DPA4_TRITON_LLVM17_COMPAT") == "1":
            compiler_source = re.sub(r"(getelementptr(?: inbounds)?) (?:nuw|nusw)\b", r"\1", source)
        assembly = llvm.translate_to_asm(compiler_source, amd.TARGET_TRIPLE, "gfx936", "", ["amdgpu-dpp-combine"],
                                         options.enable_fp_fusion, False)
    else:
        assembly = backend.make_amdgcn(source, metadata, options)
    image = backend.make_hsaco(assembly, metadata, options)
    isa = Counter(re.findall(r"^[ \t]*(v_\w+_dpp|v_mmac_\w+|ds_swizzle_b32|s_nop|s_barrier)\b", assembly, re.M))
    resources = {key: int(value) for key, value in re.findall(
        r"^\s*\.(vgpr_count|sgpr_count|private_segment_fixed_size):\s*(\d+)", assembly, re.M)}
    assert isa["v_mmac_f32_16x16x16_bf16"] == 128 and isa["s_barrier"] == 12
    args.output.mkdir()
    for name in ("dense-attention-aot.so", "dense-attention-aot.hsaco", "dense-attention-fallback.hsaco"):
        shutil.copy2(reference / name, args.output / name)
    (args.output / "dense-attention-packed.hsaco").write_bytes(image)
    (args.output / "dense-attention-packed.amdgcn").write_text(assembly)
    (args.output / "dense-attention-packed.llir").write_text(source)
    record = json.loads((reference / "dense-attention-aot-build.json").read_text())
    record["packed_strides"]["hsaco_sha256"] = hashlib.sha256(image).hexdigest()
    record["experiment"] = dict(row8=args.row8, row4=args.row4, scope=args.scope, replacements=replacements,
                                mov_dpp=args.mov_dpp, dpp_combine=args.dpp_combine,
                                isa=isa, resources=resources)
    (args.output / "dense-attention-aot-build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["experiment"]), flush=True)


if __name__ == "__main__":
    main()
