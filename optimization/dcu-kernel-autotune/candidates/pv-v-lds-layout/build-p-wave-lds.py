#!/usr/bin/env python3
"""Check wave-local P publication on the frozen gfx936 Q128 LLVM IR."""

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


BASE_LDS = 8192
PADDED_STRIDES = {"padded": 65, "wide-read": 66}
P_ROOTS = ((200, 210, 224, 234), (431, 441, 455, 465))
WAIT_LDS = '  call void asm sideeffect "s_waitcnt lgkmcnt(0)", "~{memory}"()'


def replace_barrier(lines, index, replacement):
    assert 'fence syncscope("workgroup") release' in lines[index]
    assert 'call void @llvm.amdgcn.s.barrier()' in lines[index + 1]
    assert 'fence syncscope("workgroup") acquire' in lines[index + 2]
    lines[index:index + 3] = replacement


def packed_slot(register, items):
    if items == 4:
        return (register & 1) | ((register & 8) >> 2)
    return (register & 1) | ((register & 12) >> 1)


def padded_wave_bytes(stride):
    return (3 * stride + 64) * 8


def verify_packed_layout(registers, layout, items):
    stride = PADDED_STRIDES.get(layout)

    def address(lane, register):
        slot = packed_slot(register, items)
        if stride:
            return slot * stride + (lane ^ ((lane & 48) >> 2))
        if layout == "swizzled":
            bank_lane_mask = 48 if items == 4 else 32
            lane ^= ((lane & bank_lane_mask) >> 2) | (register & 1) | ((register & 8) >> 2)
        return slot * 64 + lane

    def reader(lane, register):
        stage = ((lane & 3) << 4) | ((lane & 60) >> 2)
        producer = (stage & 60) | (register & 1) | ((register & 8) >> 2)
        source = (register & 6) | (stage & 1) | ((stage & 2) << 2)
        return address(producer, source)

    stores = {address(lane, register) for lane in range(64) for register in registers}
    reads = {reader(lane, register) for lane in range(64) for register in registers}
    assert stores == reads and len(stores) == 1024 // items
    assert max(stores) < (padded_wave_bytes(stride) // 8 if stride else 1024 // items)
    if layout != "linear":
        bank_words = 64 // items
        for register in registers:
            for start in range(0, 64, bank_words):
                for mapping in ((address,) if layout == "wide-read" else (address, reader)):
                    assert len({mapping(lane, register) % bank_words
                                for lane in range(start, start + bank_words)}) == bank_words
    if layout == "wide-read":
        for register in (0, 8):
            for start in range(0, 64, 8):
                words = [reader(lane, register) for lane in range(start, start + 8)]
                assert all(word % 2 == 0 and reader(lane, register | 1) == word + 1
                           for lane, word in enumerate(words, start))
                assert len({word // 2 % 8 for word in words}) == 8


def pack_accesses(lines, first_store, first_load, roots, exchange, layout, dedicated, items):
    start = first_store - 1
    last_load = next(i for i in range(first_load, len(lines))
                     if re.search(rf"load <1 x i16>.*%o{roots[3] + 7},", lines[i]))
    end = last_load + 2
    assert "shufflevector <1 x i16>" in lines[end - 1]
    delta = roots[0] - 200
    stores, loads = {}, {}
    for i in range(start + 1, first_load - 3, 2):
        vector = re.fullmatch(r"  %(o\d+) = insertelement <1 x bfloat> poison, bfloat %(o\d+), i64 0, !dbg !\d+", lines[i - 1])
        store = re.fullmatch(r"  store <1 x bfloat> %(o\d+), ptr addrspace\(3\) %o(\d+), align 2, !dbg !\d+", lines[i])
        assert vector[1] == store[1]
        stores[int(store[2]) - delta] = f"%{vector[2]}"
    for line in lines[first_load:end]:
        match = re.fullmatch(r"  %(o\d+) = load <1 x i16>, ptr addrspace\(3\) %o(\d+), align 2, !dbg !\d+", line)
        if match:
            loads[int(match[2]) - delta] = match[1]
    assert list(stores) == list(exchange["STORES"]) and list(loads) == list(exchange["LOADS"])
    code = []

    def emit(expr, name=None):
        output = name or f"p{delta}_{len(code)}"
        code.append(f"  %{output} = {expr}")
        return f"%{output}"

    lane = emit("and i32 %o20, 63")
    stride = PADDED_STRIDES.get(layout)
    if stride:
        wave_id = emit("lshr i32 %o21, 6")
        wave = emit(f"mul nuw i32 {wave_id}, {padded_wave_bytes(stride)}")
    else:
        wave = emit("shl i32 %o21, 5")
    plane = "%p_lds_base" if dedicated else "@global_smem"
    subregisters = (0, 2, 4, 6) if items == 4 else (0, 2)
    registers = [reg for reg in range(16) if not reg & subregisters[-1]]
    byte_shift = 3 if items == 4 else 2
    vector_type = f"<{items} x i16>"
    verify_packed_layout(registers, layout, items)

    def address(lane_word, slot, fixed_offset=0):
        op = "add nuw" if stride else "or disjoint"
        word = emit(f"{op} i32 {lane_word}, {slot}")
        bytes_in_wave = emit(f"shl i32 {word}, {byte_shift}")
        offset = emit(f"{op} i32 {wave}, {bytes_in_wave}")
        ptr = emit(f"getelementptr i8, ptr addrspace(3) {plane}, i32 {offset}")
        return emit(f"getelementptr i8, ptr addrspace(3) {ptr}, i32 {fixed_offset}") if fixed_offset else ptr

    store_lane = lane
    if layout != "linear":
        bank_lane_mask = 48 if items == 4 else 32
        bank_lane = emit(f"and i32 {lane}, {bank_lane_mask}")
        bank = emit(f"lshr i32 {bank_lane}, 2")
        store_lane = emit(f"xor i32 {lane}, {bank}")
    for register in registers:
        slot = packed_slot(register, items)
        bank = (register & 1) | ((register & 8) >> 2) if layout == "swizzled" else 0
        writer = emit(f"xor i32 {store_lane}, {bank}") if bank else store_lane
        if stride:
            ptr = address(writer, 0, slot * stride * 8)
        else:
            ptr = address(writer, slot * 64)
        packed = "poison"
        for position, subregister in enumerate(subregisters):
            bf16 = stores[exchange["STORES"][register | subregister]]
            value = emit(f"bitcast bfloat {bf16} to i16")
            packed = emit(f"insertelement {vector_type} {packed}, i16 {value}, i32 {position}")
        code.append(f"  store {vector_type} {packed}, ptr addrspace(3) {ptr}, align {items * 2}")

    low = emit(f"and i32 {lane}, 3")
    high = emit(f"and i32 {lane}, 48")
    low = emit(f"shl i32 {low}, 4")
    high = emit(f"lshr i32 {high}, 2")
    read_lane = emit(f"or disjoint i32 {low}, {high}")
    if stride:
        slot_bits = emit(f"and i32 {lane}, 12")
        slot_id = emit(f"lshr i32 {slot_bits}, 2")
        read_slot = emit(f"mul nuw i32 {slot_id}, {stride}")
    else:
        read_slot0 = emit(f"and i32 {lane}, 4")
        read_slot2 = emit(f"and i32 {lane}, 8")
        read_slot0 = emit(f"shl i32 {read_slot0}, 4")
        read_slot2 = emit(f"shl i32 {read_slot2}, {4 if items == 4 else 5}")
        read_slot = emit(f"or disjoint i32 {read_slot0}, {read_slot2}")
    if layout != "linear":
        lane_bits = emit(f"and i32 {lane}, {3 if items == 4 else 2}")
        bank = emit(f"shl i32 {lane_bits}, 2")
        if layout == "swizzled":
            bank23 = emit(f"and i32 {lane}, 12")
            b01 = emit(f"lshr i32 {bank23}, 2")
            bank = emit(f"or disjoint i32 {bank}, {b01}")
        read_lane = emit(f"xor i32 {read_lane}, {bank}")

    code.extend(lines[first_load - 3:first_load])
    read_registers = (0, 8) if layout == "wide-read" else registers
    for register in read_registers:
        lane_bits = (register & 1) | ((register & 8) >> 2)
        if stride:
            ptr = address(read_lane, read_slot, lane_bits * 8)
        else:
            reader = emit(f"xor i32 {read_lane}, {lane_bits}") if lane_bits else read_lane
            slot = emit(f"or disjoint i32 {read_slot}, {(register & 4) * 32}") if register & 4 else read_slot
            ptr = address(reader, slot)
        read_indices = (*subregisters, *(reg | 1 for reg in subregisters)) if layout == "wide-read" else subregisters
        read_type = f"<{len(read_indices)} x i16>"
        packed = emit(f"load {read_type}, ptr addrspace(3) {ptr}, align {len(read_indices) * 2}")
        for position, subregister in enumerate(read_indices):
            value = emit(f"extractelement {read_type} {packed}, i32 {position}")
            reg = register | subregister
            emit(f"insertelement <1 x i16> poison, i16 {value}, i32 0", loads[exchange["LOADS"][reg]])
    code.extend(line for line in lines[first_load:end] if "shufflevector <1 x i16>" in line)
    lines[start:end] = code


def transform(source, full_only, dedicated, packed, workgroup_publication, items):
    exchange = runpy.run_path(str(Path(__file__).with_name("build-p-wave-exchange.py")))
    if hashlib.sha256(source.encode()).hexdigest() != exchange["REFERENCE_SHA256"]:
        raise ValueError("Use the verified Q128 packed LLVM IR")
    exchange["verify_permutation"]()
    source = re.sub(r"%(\d+)\b", r"%o\1", source)
    source = re.sub(r"^(\d+):", r"o\1:", source, flags=re.M)
    lines = source.splitlines()
    entry = next(i for i, line in enumerate(lines) if line.startswith("define amdgpu_kernel "))
    lines.insert(entry + 1, "o15:")
    for roots in reversed(P_ROOTS[:1] if full_only else P_ROOTS):
        first_store = next(i for i, line in enumerate(lines)
                           if re.search(rf"store <1 x bfloat>.*%o{roots[0]},", line))
        first_load = next(i for i in range(first_store, len(lines))
                          if re.search(rf"load <1 x i16>.*%o{roots[2]},", lines[i]))
        assert sum("store <1 x bfloat>" in line for line in lines[first_store:first_load]) == 16
        if packed:
            pack_accesses(lines, first_store, first_load, roots, exchange, packed, dedicated, items)
        if not workgroup_publication:
            load_items = 8 if packed == "wide-read" else (items if packed else 1)
            first_read = next(i for i in range(first_store, len(lines))
                              if f" = load <{load_items} x i16>" in lines[i])
            barrier = next(i for i in range(first_store, first_read)
                           if 'fence syncscope("workgroup") release' in lines[i])
            replace_barrier(lines, barrier, [WAIT_LDS])
        if dedicated:
            # K readers are still protected by the later barrier before the V writes.
            replace_barrier(lines, first_store - 4, [])
            for root in (() if packed else roots):
                index = next(i for i, line in enumerate(lines) if line.startswith(f"  %o{root} = "))
                previous = lines[index]
                assert re.fullmatch(rf"  %o{root} = getelementptr inbounds nuw i8, ptr addrspace\(3\) "
                                    r"@global_smem, i64 %o\d+", previous)
                lines[index] = previous.replace("@global_smem", "%p_lds_base")
    if dedicated:
        lines.insert(entry + 2, f"  %p_lds_base = getelementptr i8, ptr addrspace(3) @global_smem, i64 {BASE_LDS}")
    return "\n".join(lines) + "\n"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--full-only", action="store_true")
    cli.add_argument("--dedicated", action="store_true", help="Reserve a separate P LDS plane")
    cli.add_argument("--packed", choices=("linear", "swizzled", *PADDED_STRIDES))
    cli.add_argument("--items", type=int, choices=(2, 4), default=2, help="BF16 values per packed LDS access")
    cli.add_argument("--workgroup-publication", action="store_true", help="Keep the P store-to-load barrier")
    args = cli.parse_args()
    if args.packed in PADDED_STRIDES and args.items != 4:
        cli.error("The padded layouts require --items 4")
    ref = args.reference.resolve(strict=True)
    image_source = transform((ref / "dense-attention-packed.llir").read_text(), args.full_only, args.dedicated,
                             args.packed, args.workgroup_publication, args.items)
    os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    options = backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2))
    meta = {}
    asm = backend.make_amdgcn(image_source, meta, options)
    image = backend.make_hsaco(asm, meta, options)
    resources = {key: int(value) for key, value in re.findall(
        r"^\s*\.(vgpr_count|sgpr_count|private_segment_fixed_size):\s*(\d+)", asm, re.M)}
    isa = Counter(re.findall(r"^[ \t]*(ds_\w+|v_mmac_\w+|s_barrier|s_waitcnt)\b", asm, re.M))
    assert isa["v_mmac_f32_16x16x16_bf16"] == 128
    removed = int(args.dedicated) + int(not args.workgroup_publication)
    assert isa["s_barrier"] == 12 - removed * (1 if args.full_only else 2)
    args.output.mkdir()
    for name in ("dense-attention-aot.so", "dense-attention-aot.hsaco", "dense-attention-fallback.hsaco"):
        shutil.copy2(ref / name, args.output / name)
    (args.output / "dense-attention-packed.hsaco").write_bytes(image)
    (args.output / "dense-attention-packed.amdgcn").write_text(asm)
    (args.output / "dense-attention-packed.llir").write_text(image_source)
    record = json.loads((ref / "dense-attention-aot-build.json").read_text())
    assert record["packed_strides"]["shared"] == BASE_LDS
    stride = PADDED_STRIDES.get(args.packed)
    p_lds_bytes = padded_wave_bytes(stride) * 4 if stride else BASE_LDS
    shared = BASE_LDS + p_lds_bytes if args.dedicated else max(BASE_LDS, p_lds_bytes)
    record["packed_strides"].update(hsaco_sha256=hashlib.sha256(image).hexdigest(), shared=shared)
    record["experiment"] = dict(p_wave_lds=True, full_only=args.full_only, dedicated=args.dedicated,
                                packed=args.packed, items=args.items, workgroup_publication=args.workgroup_publication,
                                resources=resources, isa=isa)
    (args.output / "dense-attention-aot-build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["experiment"]))


if __name__ == "__main__":
    main()
