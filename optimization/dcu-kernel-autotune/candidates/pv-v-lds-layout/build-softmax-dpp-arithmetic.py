#!/usr/bin/env python3
"""Measure explicitly fused DPP arithmetic in the Q128 softmax tree."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil

from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget


CONTROLS = {296: "row_ror:8", 78: "quad_perm:[2,3,0,1]", 177: "quad_perm:[1,0,3,2]"}
SCOPES = {"row": (296,), "quad-last": (177,), "quad": (78, 177), "all": (296, 78, 177)}


def fuse(source, scope):
    helpers = runpy.run_path(str(Path(__file__).with_name("build-softmax-xor.py")))
    source, _ = helpers["transform"](source, "rotate", "original", "both", False)
    update = helpers["UPDATE"]
    lines = source.splitlines()
    counts = Counter()
    for index in reversed(range(1, len(lines) - 2)):
        dpp = update.fullmatch(lines[index])
        if not dpp or int(dpp[4]) not in SCOPES[scope]:
            continue
        assert dpp[2] == dpp[3] and dpp[5] == "15"
        control = int(dpp[4])
        stage = helpers["LOCATIONS"][int(dpp[6])]
        bitcast = re.fullmatch(rf"  {dpp[3]} = bitcast float (%o\d+) to i32, !dbg !\d+", lines[index - 1])
        assert bitcast
        number = bitcast[1]
        shuffled = re.fullmatch(rf"  (%o\d+) = bitcast i32 {dpp[1]} to float, !dbg !\d+", lines[index + 1])
        assert shuffled
        if stage == "max":
            pattern = rf"  (%o\d+) = tail call float @llvm\.maxnum\.f32\(float {number}, float {shuffled[1]}\), (!dbg !\d+)"
            opcode = "v_max_f32_dpp"
        else:
            pattern = rf"  (%o\d+) = fadd float {number}, {shuffled[1]}, (!dbg !\d+)"
            opcode = "v_add_f32_dpp"
        operation = re.fullmatch(pattern, lines[index + 2])
        assert operation
        # The backend cannot see the DPP in the inline instruction. Pad its input
        # hazard, and its result when it precedes another exchange in the tree.
        instructions = rf"s_nop 1\0A\09{opcode} $0, $1, $1 {CONTROLS[control]} row_mask:0xf bank_mask:0xf"
        if control != 177:
            instructions += r"\0A\09s_nop 1"
        lines[index - 1:index + 3] = [
            f'  {operation[1]} = call float asm sideeffect "{instructions}", "=v,v"(float {number}), {operation[2]}'
        ]
        counts[f"{stage}_{control}"] += 1
    assert counts == {f"{stage}_{control}": 16 for control in SCOPES[scope] for stage in ("max", "sum")}
    return "\n".join(lines) + "\n", counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("scope", choices=SCOPES)
    args = parser.parse_args()
    ref = args.reference.resolve(strict=True)
    source, counts = fuse((ref / "dense-attention-packed.llir").read_text(), args.scope)
    os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    options = backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2))
    metadata = {}
    assembly = backend.make_amdgcn(source, metadata, options)
    image = backend.make_hsaco(assembly, metadata, options)
    isa = Counter(re.findall(r"^[ \t]*(v_\w+_dpp|v_mmac_\w+|s_nop|s_barrier)\b", assembly, re.M))
    assert isa["v_mmac_f32_16x16x16_bf16"] == 128 and isa["s_barrier"] == 12
    assert isa["v_max_f32_dpp"] == isa["v_add_f32_dpp"] == 16 * len(SCOPES[args.scope])
    resources = {key: int(value) for key, value in re.findall(
        r"^\s*\.(vgpr_count|sgpr_count|private_segment_fixed_size):\s*(\d+)", assembly, re.M)}
    args.output.mkdir()
    for name in ("dense-attention-aot.so", "dense-attention-aot.hsaco", "dense-attention-fallback.hsaco"):
        shutil.copy2(ref / name, args.output / name)
    (args.output / "dense-attention-packed.hsaco").write_bytes(image)
    (args.output / "dense-attention-packed.amdgcn").write_text(assembly)
    (args.output / "dense-attention-packed.llir").write_text(source)
    record = json.loads((ref / "dense-attention-aot-build.json").read_text())
    record["packed_strides"]["hsaco_sha256"] = hashlib.sha256(image).hexdigest()
    record["experiment"] = dict(scope=args.scope, fused=counts, resources=resources, isa=isa)
    (args.output / "dense-attention-aot-build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["experiment"]), flush=True)


if __name__ == "__main__":
    main()
