#!/usr/bin/env python3
"""Prototype the P operand exchange in registers on the frozen gfx936 LLVM IR."""

import argparse
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
STORE_OFFSETS = (0, 8, 4224, 4232, 128, 136, 4096, 4104)
LOAD_OFFSETS = (0, 256, 4224, 4480, 128, 384, 4096, 4352)
STORES = (*range(200, 208), *range(210, 218))
LOADS = (*range(224, 232), *range(234, 242))
PAIR_BASES = [i for i in range(16) if not i & 2]


def verify_permutation():
    addresses = {}
    for thread in range(256):
        base = (((thread & 15) << 8) | ((thread & 14) << 3)) ^ (((thread & 192) >> 1) | ((thread & 48) >> 3))
        for reg, offset in enumerate(STORE_OFFSETS * 2):
            addr = (base ^ (16 if reg >= 8 else 0)) + offset
            assert addr not in addresses
            addresses[addr] = (thread, reg)
    assert len(addresses) == 4096
    for thread in range(256):
        base = (((thread & 48) << 6) | ((thread << 1) & 126)) ^ ((thread & 192) >> 1)
        for reg, offset in enumerate(LOAD_OFFSETS * 2):
            addr = (base ^ (528 if reg >= 8 else 0)) + offset
            stage_lane = ((thread & 3) << 4) | ((thread & 48) >> 2) | ((thread & 12) >> 2)
            producer = (thread & 192) | (stage_lane & 60) | (reg & 1) | ((reg & 8) >> 2)
            register = (reg & 6) | (stage_lane & 1) | ((stage_lane & 2) << 2)
            assert addresses[addr] == (producer, register)


def replace_stage(lines, start, end, prefix, pointer_offset, quad_dpp):
    region = lines[start:end]
    stores, loads = {}, {}
    scalars = {}
    for line in region:
        match = re.match(r"  %(\w+) = insertelement <1 x bfloat> poison, bfloat %(\w+), i64 0\b", line)
        if match:
            scalars[match[1]] = match[2]
        match = re.fullmatch(r"  store <1 x bfloat> %(\w+), ptr addrspace\(3\) %o(\d+), align 2, !dbg !\d+", line)
        if match and int(match[2]) - pointer_offset in STORES:
            stores[int(match[2]) - pointer_offset] = scalars[match[1]]
        match = re.fullmatch(r"  %(\w+) = load <1 x i16>, ptr addrspace\(3\) %o(\d+), align 2, !dbg !\d+", line)
        if match and int(match[2]) - pointer_offset in LOADS:
            loads[int(match[2]) - pointer_offset] = match[1]
    assert list(stores) == list(STORES) and list(loads) == list(LOADS)
    assert sum("s.barrier()" in line for line in region) == 2
    code = []

    def emit(expr, label=None):
        label = f"{prefix}_{len(code)}" if label is None else label
        code.append(f"  %{label} = {expr}")
        return "%" + label

    values = {}
    for reg in PAIR_BASES:
        low = emit(f"bitcast bfloat %{stores[STORES[reg]]} to i16")
        high = emit(f"bitcast bfloat %{stores[STORES[reg | 2]]} to i16")
        lo = emit(f"zext i16 {low} to i32")
        hi = emit(f"zext i16 {high} to i32")
        shifted = emit(f"shl i32 {hi}, 16")
        values[reg] = emit(f"or i32 {lo}, {shifted}")

    lane = emit("and i32 %o20, 63")
    address = emit(f"shl i32 {lane}, 2")
    for reg_bit, lane_bit in ((1, 1), (8, 2)):
        bit = emit(f"and i32 {lane}, {lane_bit}")
        high = emit(f"icmp ne i32 {bit}, 0")
        remote = emit(f"xor i32 {address}, {4 * lane_bit}")
        for reg in PAIR_BASES:
            if reg & reg_bit:
                continue
            low_pack, high_pack = values[reg], values[reg | reg_bit]
            send = emit(f"select i1 {high}, i32 {low_pack}, i32 {high_pack}")
            if quad_dpp:
                control = sum((lane ^ lane_bit) << (2 * lane) for lane in range(4))
                recv = emit(f"call i32 @llvm.amdgcn.update.dpp.i32("
                            f"i32 poison, i32 {send}, i32 {control}, i32 15, i32 15, i1 true)")
            else:
                recv = emit(f"call i32 @llvm.amdgcn.ds.bpermute(i32 {remote}, i32 {send})")
            values[reg] = emit(f"select i1 {high}, i32 {recv}, i32 {low_pack}")
            values[reg | reg_bit] = emit(f"select i1 {high}, i32 {high_pack}, i32 {recv}")
    lo = emit(f"and i32 {lane}, 3")
    hi = emit(f"and i32 {lane}, 60")
    lo_shl = emit(f"shl i32 {lo}, 4")
    hi_shr = emit(f"lshr i32 {hi}, 2")
    selector = emit(f"or i32 {lo_shl}, {hi_shr}")
    final_address = emit(f"shl i32 {selector}, 2")
    for reg in PAIR_BASES:
        packed = emit(f"call i32 @llvm.amdgcn.ds.bpermute(i32 {final_address}, i32 {values[reg]})")
        low = emit(f"trunc i32 {packed} to i16")
        shifted = emit(f"lshr i32 {packed}, 16")
        high = emit(f"trunc i32 {shifted} to i16")
        emit(f"insertelement <1 x i16> poison, i16 {low}, i32 0", loads[LOADS[reg]])
        emit(f"insertelement <1 x i16> poison, i16 {high}, i32 0", loads[LOADS[reg | 2]])
    # Keep remaining uses such as the tail's vector shuffles in their original order.
    for line in region:
        if "shufflevector <1 x i16>" in line:
            code.append(line)
    lines[start:end] = code


def transform(source, full_only, quad_dpp=False):
    if hashlib.sha256(source.encode()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Use the verified Q128 packed LLVM IR")
    verify_permutation()
    source = re.sub(r"%(\d+)\b", r"%o\1", source)
    source = re.sub(r"^(\d+):", r"o\1:", source, flags=re.M)
    lines = source.splitlines()
    entry = next(i for i, line in enumerate(lines) if line.startswith("define amdgpu_kernel "))
    lines.insert(entry + 1, "o15:")
    first_stores = [(i, int(match[1]) - 200) for i, line in enumerate(lines)
                    if (match := re.search(r"store <1 x bfloat>.*%o(200|431),", line))]
    assert len(first_stores) == 2
    for n, (i, offset) in reversed(list(enumerate(first_stores[:1] if full_only else first_stores))):
        start = i - 4
        assert "fence syncscope" in lines[start] and "s.barrier()" in lines[start+1]
        marker = f"load <1 x i16>, ptr addrspace(3) %o{241 + offset},"
        last_load = next(j for j in range(i, len(lines)) if marker in lines[j])
        end = next(j for j in range(last_load + 1, len(lines)) if "fence syncscope" in lines[j])
        replace_stage(lines, start, end, f"p_exchange{n}", offset, quad_dpp)
    declaration = "declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)"
    return "\n".join([*lines, declaration, ""])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--full-only", action="store_true")
    cli.add_argument("--quad-dpp", action="store_true",
                     help="Use DPP quad permutations for the two register/lane exchanges")
    args = cli.parse_args()
    ref = args.reference.resolve(strict=True)
    image_source = transform((ref / "dense-attention-packed.llir").read_text(), args.full_only, args.quad_dpp)
    os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    options = backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2))
    meta = {}
    asm = backend.make_amdgcn(image_source, meta, options)
    image = backend.make_hsaco(asm, meta, options)
    resources = {key:int(value) for key,value in re.findall(
        r"^\s*\.(vgpr_count|sgpr_count|private_segment_fixed_size):\s*(\d+)", asm, re.M)}
    assert len(re.findall(r"^\s*v_mmac_f32_16x16x16_bf16\b", asm, re.M)) == 128
    assert len(re.findall(r"^\s*s_barrier\b", asm, re.M)) == (10 if args.full_only else 8)
    args.output.mkdir()
    for name in ("dense-attention-aot.so", "dense-attention-aot.hsaco", "dense-attention-fallback.hsaco"):
        shutil.copy2(ref / name, args.output / name)
    (args.output / "dense-attention-packed.hsaco").write_bytes(image)
    (args.output / "dense-attention-packed.amdgcn").write_text(asm)
    (args.output / "dense-attention-packed.llir").write_text(image_source)
    metadata = json.loads((ref / "dense-attention-aot-build.json").read_text())
    metadata["packed_strides"]["hsaco_sha256"] = hashlib.sha256(image).hexdigest()
    metadata["experiment"] = dict(p_wave_exchange=True, full_only=args.full_only,
                                  quad_dpp=args.quad_dpp, resources=resources)
    (args.output / "dense-attention-aot-build.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["experiment"]))


if __name__ == "__main__":
    main()
