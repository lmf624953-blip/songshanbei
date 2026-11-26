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
FUSED = re.compile(r'  (%o\d+) = call float asm sideeffect "([^"]*)", "=&?v,v"\(float (%o\d+)\), (!dbg !\d+)')


def fuse(source, scope, xor4=None):
    helpers = runpy.run_path(str(Path(__file__).with_name("build-softmax-xor.py")))
    source, _ = helpers["transform"](source, "rotate", "original", "both", False)
    update = helpers["UPDATE"]
    lines = source.splitlines()
    counts = Counter()
    for index in reversed(range(1, len(lines) - 2)):
        dpp = update.fullmatch(lines[index])
        if not dpp:
            continue
        control = int(dpp[4])
        stage = helpers["LOCATIONS"][int(dpp[6])]
        start = index - 1
        if control == 260 and xor4 in (stage, "both"):
            first = update.fullmatch(lines[index - 1])
            assert first and first[1] == dpp[2] and first[2] == first[3] == dpp[3]
            assert (first[4], first[5], first[6], dpp[5]) == ("276", "10", dpp[6], "5")
            start -= 1
        elif control in SCOPES[scope]:
            assert dpp[2] == dpp[3] and dpp[5] == "15"
        else:
            continue
        bitcast = re.fullmatch(rf"  {dpp[3]} = bitcast float (%o\d+) to i32, !dbg !\d+", lines[start])
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
        constraints = "=v,v"
        if control == 260:
            instructions = (rf"s_nop 1\0A\09{opcode} $0, $1, $1 row_shr:4 row_mask:0xf bank_mask:0xa"
                            rf"\0A\09s_nop 1\0A\09{opcode} $0, $1, $1 row_shl:4 row_mask:0xf bank_mask:0x5")
            constraints = "=&v,v"
        else:
            instructions = rf"s_nop 1\0A\09{opcode} $0, $1, $1 {CONTROLS[control]} row_mask:0xf bank_mask:0xf"
        if control != 177:
            instructions += r"\0A\09s_nop 1"
        lines[start:index + 3] = [
            f'  {operation[1]} = call float asm sideeffect "{instructions}", "{constraints}"(float {number}), {operation[2]}'
        ]
        counts[f"{stage}_{control}"] += 1
    expected = {f"{stage}_{control}": 16 for control in SCOPES[scope] for stage in ("max", "sum")}
    expected.update({f"{stage}_260": 16 for stage in ("max", "sum") if xor4 in (stage, "both")})
    assert counts == expected
    return "\n".join(lines) + "\n", counts


def bundle_reductions(source, scope, width):
    lines = source.splitlines()
    generated = []
    counts = Counter()
    index = 0
    while index < len(lines):
        first = FUSED.fullmatch(lines[index])
        if not first or "row_ror:8 " not in first[2]:
            generated.append(lines[index])
            index += 1
            continue
        stage = "max" if "v_max_f32_dpp" in first[2] else "sum"
        if scope not in (stage, "both"):
            generated.append(lines[index])
            index += 1
            continue
        opcode = "v_max_f32_dpp" if stage == "max" else "v_add_f32_dpp"
        inputs, results = [], []
        for row in range(width):
            begin = index + 4 * row
            start, second, third, last = map(FUSED.fullmatch, lines[begin:begin + 4])
            assert all((start, second, third, last)) and opcode in start[2]
            assert (second[3], third[3], last[3]) == (start[1], second[1], third[1])
            assert all(control in match[2] for control, match in (
                ("row_ror:8 ", start), ("row_shr:4 ", second),
                ("quad_perm:[2,3,0,1]", third), ("quad_perm:[1,0,3,2]", last)))
            inputs.append(start[3])
            results.append(last)
        assert not set(inputs).intersection(result[1] for result in results)
        exchanges = (
            (0, 2, CONTROLS[296], "0xf"),
            (1, 0, "row_shr:4", "0xa"),
            (1, 0, "row_shl:4", "0x5"),
            (0, 1, CONTROLS[78], "0xf"),
            (0, 0, CONTROLS[177], "0xf"),
        )
        assembly = ["s_nop 1"]
        for phase, (dest, operand, control, bank) in enumerate(exchanges):
            # Keep two issue slots between an output and the next DPP read of it.
            if phase and width < 4:
                assembly.append(f"s_nop {2 - width}")
            for row in range(width):
                a, b = f"${dest * width + row}", f"${operand * width + row}"
                assembly.append(f"{opcode} {a}, {b}, {b} {control} row_mask:0xf bank_mask:{bank}")
        instructions = r"\0A\09".join(assembly)
        outputs = "{" + ", ".join(["float"] * (2 * width)) + "}"
        constraints = ",".join(["=&v"] * (2 * width) + ["v"] * width)
        arguments = ", ".join(f"float {number}" for number in inputs)
        tmp = results[0][1] + "_bundle"
        generated.append(f'  {tmp} = call {outputs} asm sideeffect "{instructions}", "{constraints}"({arguments}), {results[0][4]}')
        generated.extend(f"  {match[1]} = extractvalue {outputs} {tmp}, {row}, {match[4]}"
                         for row, match in enumerate(results))
        index += 4 * width
        counts[stage] += width
    assert counts == {stage: 16 for stage in ("max", "sum") if scope in (stage, "both")}
    return "\n".join(generated) + "\n", counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("scope", choices=SCOPES)
    parser.add_argument("--xor4", choices=("max", "sum", "both"))
    parser.add_argument("--bundle", choices=("max", "sum", "both"))
    parser.add_argument("--bundle-width", type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    if args.bundle and (args.scope != "all" or args.xor4 != "both"):
        parser.error("--bundle requires all --xor4 both")
    ref = args.reference.resolve(strict=True)
    source, counts = fuse((ref / "dense-attention-packed.llir").read_text(), args.scope, args.xor4)
    bundles = {}
    if args.bundle:
        source, bundles = bundle_reductions(source, args.bundle, args.bundle_width)
    os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    options = backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2))
    metadata = {}
    assembly = backend.make_amdgcn(source, metadata, options)
    image = backend.make_hsaco(assembly, metadata, options)
    isa = Counter(re.findall(r"^[ \t]*(v_\w+_dpp|v_mmac_\w+|s_nop|s_barrier)\b", assembly, re.M))
    assert isa["v_mmac_f32_16x16x16_bf16"] == 128 and isa["s_barrier"] == 12
    for stage, opcode in (("max", "v_max_f32_dpp"), ("sum", "v_add_f32_dpp")):
        assert isa[opcode] == 16 * len(SCOPES[args.scope]) + 32 * (args.xor4 in (stage, "both"))
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
    record["experiment"] = dict(scope=args.scope, xor4=args.xor4, fused=counts, bundles=bundles,
                                bundle_width=args.bundle_width,
                                resources=resources, isa=isa)
    (args.output / "dense-attention-aot-build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["experiment"]), flush=True)


if __name__ == "__main__":
    main()
