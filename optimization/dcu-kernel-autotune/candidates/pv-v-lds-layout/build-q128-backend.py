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
from producer_layout import ALLOWED, LINE, SSA
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
    "lse-early": (8, 2, 16, 8, [4, 1]),
    "v-prefetch": (8, 2, 16, 8, [4, 1]),
    "lse-v-prefetch": (8, 2, 16, 8, [4, 1]),
    "lse-waves4": (8, 2, 16, 8, [4, 1]),
    "lse-v-waves4": (8, 2, 16, 8, [4, 1]),
    "lse-vgpr192": (8, 2, 16, 8, [4, 1]),
    "lse-v-vgpr192": (8, 2, 16, 8, [4, 1]),
    "lse-attention": (8, 2, 16, 8, [4, 1]),
    "lse-v-attention": (8, 2, 16, 8, [4, 1]),
    "sum-early": (8, 2, 16, 8, [4, 1]),
    "sum-v-prefetch": (8, 2, 16, 8, [4, 1]),
}


K_LAYOUT = "#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>"
K_VARIANTS = {
    "k-phase2": (8, 2, 8),
    "k-phase4": (8, 4, 4),
    "k-max8": (8, 1, 8),
    "k-vec4": (4, 2, 16),
    "k-vec16": (16, 1, 8),
    "k-phase2-max16": (8, 2, 16),
}
for variant in K_VARIANTS:
    VARIANTS[variant] = (8, 2, 16, 8, [4, 1])


ROUNDING_VARIANTS = {
    "lse-rounding": "lse-early",
    "v-rounding": "v-prefetch",
    "lse-v-rounding": "lse-v-prefetch",
    "sum-rounding": "sum-early",
    "sum-v-rounding": "sum-v-prefetch",
    "lse-rounding-vgpr192": "lse-vgpr192",
    "lse-v-rounding-vgpr192": "lse-v-vgpr192",
}
for variant, original in ROUNDING_VARIANTS.items():
    VARIANTS[variant] = VARIANTS[original]


def preserve_scaled_rounding(source):
    lines = source.splitlines()
    found = 0
    for i in reversed(range(len(lines))):
        line = lines[i]
        if " = arith.mulf " not in line or " : tensor<128x32xf32, #mma>" not in line:
            continue
        match = LINE.fullmatch(line)
        assert match
        label = match[2]
        temporary = label + "_h3_rounded"
        lines[i] = line.replace(label + " = ", temporary + " = ", 1)
        lines.insert(i + 1, '      ' + label + ' = tt.elementwise_inline_asm "" '
                     '{constraints = "=v,0", packed_element = 1 : i32, pure = true} ' + temporary
                     + ' : tensor<128x32xf32, #mma> -> tensor<128x32xf32, #mma>')
        found += 1
    assert found == 2, f"缩放舍入边界数量异常 {found}"
    return "\n".join(lines) + "\n"


def prefetch_value(source):
    lines = source.splitlines()
    loads = [i for i, line in enumerate(lines)
             if " = tt.load " in line and "tensor<32x128x!tt.ptr<bf16>, #h3_v_producer>" in line]
    assert len(loads) == 2
    for number, load_index in reversed(list(enumerate(loads))):
        start = max(i for i in range(load_index) if " = scf.for " in lines[i])
        anchor = max(i for i in range(start, load_index)
                     if " = tt.load " in lines[i] and "tensor<128x32x!tt.ptr<bf16>, #blocked1>" in lines[i])
        definitions = {}
        for i, line in enumerate(lines[:load_index + 1]):
            match = LINE.fullmatch(line)
            if match:
                definitions[match[2]] = (i, match[3])
        original = LINE.fullmatch(lines[load_index])[2]
        cloned = {}
        operations = []

        def clone(value):
            if value in cloned:
                return cloned[value]
            definition = definitions.get(value)
            if definition is None or definition[0] < start:
                return value
            _, rhs = definition
            op = rhs.split(" ", 1)[0]
            if op not in ALLOWED or op == "tt.load" and value != original:
                raise RuntimeError(f"V 预取依赖不允许跨越 QK：{op} {value}")
            rhs = SSA.sub(lambda m: clone(m[0]), rhs)
            label = original if value == original else f"%h3_prefetch{number}_{len(operations)}"
            cloned[value] = label
            operations.append("      " + label + " = " + rhs)
            return label

        assert clone(original) == original and 10 < len(operations) < 160
        del lines[load_index]
        lines[anchor:anchor] = operations
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("variant", choices=VARIANTS)
    args = parser.parse_args()
    mode = ROUNDING_VARIANTS.get(args.variant, args.variant)
    root = args.reference.resolve(strict=True)
    output = args.output.resolve()
    record = json.loads((root / "dense-attention-aot-build.json").read_text())
    assert record["query_block"] == 128 and record["shared"] == 8192
    assert record["experiment"]["compact_args"] == "common-length-fixed-strides"
    source = (root / "pv-swizzle.ttgir").read_text()
    if mode in ("lse-early", "lse-v-prefetch", "lse-waves4", "lse-v-waves4",
                        "lse-vgpr192", "lse-v-vgpr192", "lse-attention", "lse-v-attention"):
        lines = source.splitlines()
        yields = [i for i, line in enumerate(lines) if line.startswith("      scf.yield ")]
        assert len(yields) == 2
        for end in reversed(yields):
            dot = max(i for i in range(end) if lines[i].startswith("      %") and " = tt.dot " in lines[i])
            block = lines[dot + 1:end]
            assert len(block) == 7 and "arith.mulf" in block[0] and '"tt.reduce"' in block[1]
            assert "arith.addf" in block[-1]
            anchor = max(i for i in range(dot) if "arith.truncf" in lines[i]
                         and "tensor<128x32xf32, #mma>" in lines[i])
            assert anchor < dot
            lines[dot + 1:end] = []
            lines[anchor:anchor] = block
        source = "\n".join(lines) + "\n"
    if mode in ("sum-early", "sum-v-prefetch"):
        lines = source.splitlines()
        yields = [i for i, line in enumerate(lines) if line.startswith("      scf.yield ")]
        assert len(yields) == 2
        for end in reversed(yields):
            dot = max(i for i in range(end) if lines[i].startswith("      %") and " = tt.dot " in lines[i])
            block = lines[dot + 2:end - 1]
            assert len(block) == 5 and '\"tt.reduce\"' in block[0] and "tt.reduce.return" in block[3]
            anchor = max(i for i in range(dot) if "arith.truncf" in lines[i]
                         and "tensor<128x32xf32, #mma>" in lines[i])
            del lines[dot + 2:end - 1]
            lines[anchor:anchor] = block
        source = "\n".join(lines) + "\n"
    if mode in ("v-prefetch", "lse-v-prefetch", "lse-v-waves4", "lse-v-vgpr192", "lse-v-attention", "sum-v-prefetch"):
        source = prefetch_value(source)
    if args.variant in ROUNDING_VARIANTS or args.variant in K_VARIANTS:
        source = preserve_scaled_rounding(source)
    if args.variant in K_VARIANTS:
        assert source.count(K_LAYOUT) == 1
        kv, kp, km = K_VARIANTS[args.variant]
        source = source.replace(K_LAYOUT, f"#shared1 = #ttg.swizzled_shared<{{vec = {kv}, perPhase = {kp}, maxPhase = {km}, order = [0, 1]}}>")
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
    source += "// H3 backend experiment: " + args.variant + "\n"
    ir.write_text(source)
    os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"
    configure_hygon_compiler()
    options = {
        "num_warps": num_warps, "num_stages": 1, "matrix_instr_nonkdim": 0, "kpack": 2,
    }
    if mode in ("lse-waves4", "lse-v-waves4"):
        options["waves_per_eu"] = 4
    if mode.endswith("attention"):
        options["schedule_hint"] = "attention"
    if mode.endswith("vgpr192"):
        from triton.backends.amd.compiler import HIPBackend
        original_make = HIPBackend.make_amdgcn

        def capped_amdgcn(llvm_source, metadata, compiler_options):
            pattern = r'"amdgpu-waves-per-eu"="[0-9,]+"'
            assert len(re.findall(pattern, llvm_source)) == 1
            llvm_source = re.sub(pattern, '"amdgpu-waves-per-eu"="1" "amdgpu-num-vgpr"="192"', llvm_source)
            (output / "register-cap.llir").write_text(llvm_source)
            return original_make(llvm_source, metadata, compiler_options)

        HIPBackend.make_amdgcn = staticmethod(capped_amdgcn)
    compiled = triton.compile(str(ir), target=GPUTarget("hip", "gfx936", 64), options=options)
    (output / "dense-attention-aot.llir").write_text(compiled.asm["llir"])
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
    record["experiment"].update(backend_variant=args.variant, isa=isa, resources=resources, options=options)
    (output / "dense-attention-aot-build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(name=args.variant, resources=resources, shared=record["shared"], isa=isa)), flush=True)


if __name__ == "__main__":
    main()
