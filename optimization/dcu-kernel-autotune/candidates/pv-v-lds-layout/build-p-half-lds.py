#!/usr/bin/env python3
"""Test halfword LDS instructions on the qualified joint K/V attention image."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget


REFERENCE = "5d18866b7f2ca268278df74acdabef735828a6201efe0798f63151d40d25dec8"
SHIFT = re.compile(r"\tv_lshrrev_b32_e32 (v\d+), 16, \1")
SELECT = re.compile(r"\tv_cndmask_b32_e32 (v\d+), (v\d+), (v\d+), vcc")
STORE = re.compile(r"\tds_write_b16 v\d+, (v\d+)(?: offset:\d+)?")
LOAD = re.compile(r"\tds_read_u16 (v\d+), (v\d+(?: offset:\d+)?)")
PERM = re.compile(r"\tv_perm_b32 (v\d+), (v\d+), (v\d+), s(?:43|46)")
MATRIX = re.compile(r"(\tv_mmac_f32_16x16x16_bf16 v\[\d+:\d+\], v\[\d+:\d+\], )v\[(\d+):(\d+)\](, v\[\d+:\d+\])")
STEM = "dense-attention-packed"
FENCE = ["\ts_waitcnt lgkmcnt(0)", "\ts_barrier", "\ts_waitcnt lgkmcnt(0)"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def high_store(lines, stores, region_start):
    prefix = lines[region_start:stores[0]]
    shifts = [m[1] for line in prefix if (m := SHIFT.fullmatch(line))]
    selects = [m.groups() for line in prefix if (m := SELECT.fullmatch(line))]
    assert len(shifts) == len(selects) == len(stores) == 16
    assert shifts == [row[2] for row in selects]
    nan, = {row[1] for row in selects}
    assert set(row[0] for row in selects) == {STORE.fullmatch(lines[i])[1] for i in stores}
    literal = f"\tv_mov_b32_e32 {nan}, 0x7fff"
    assert prefix.count(literal) == 1
    for i in stores:
        lines[i] = lines[i].replace("ds_write_b16", "ds_write_b16_d16_hi", 1)
    lines[region_start:stores[0]] = [
        line.replace(literal, f"\tv_mov_b32_e32 {nan}, 0x7fff0000")
        for line in prefix if not SHIFT.fullmatch(line)
    ]


def packed_read(lines, reads, pv_end, first_temp, read_order):
    matrix = [(i, m) for i in range(reads[-1] + 1, pv_end) if (m := MATRIX.fullmatch(lines[i]))]
    first_mmac = matrix[0][0]
    perms = [(i, PERM.fullmatch(lines[i])) for i in range(first_mmac - 10, first_mmac)
             if PERM.fullmatch(lines[i])]
    assert len(perms) == 8 and len(matrix) == 32
    old_pairs = sorted({(int(m[2]), int(m[3])) for _, m in matrix})
    assert len(old_pairs) == 4 and all(high == low + 1 for low, high in old_pairs)
    destinations = {f"v{reg}": first_temp + 2 * pair + offset
                    for pair, registers in enumerate(old_pairs) for offset, reg in enumerate(registers)}
    assert set(destinations) == {m[1] for _, m in perms}
    # Reuse eight of the scalar LDS destinations without extending the live register range.
    used = set(range(first_temp, first_temp + 8))
    for line in lines[reads[-1] + 1:perms[0][0]] + lines[first_mmac:pv_end]:
        for low, high in re.findall(r"\bv(\d+)\b|\bv\[(\d+:\d+)\]", line):
            live = [int(low)] if low else range(int(high.split(":")[0]), int(high.split(":")[1]) + 1)
            assert used.isdisjoint(live), line
    halves = {source: (destinations[m[1]], suffix) for _, m in perms
              for source, suffix in ((m[2], "_hi"), (m[3], ""))}
    assert len(halves) == len(reads) == 16
    packed_reads = []
    for i in reads:
        source, address = LOAD.fullmatch(lines[i]).groups()
        register, suffix = halves[source]
        if read_order == "swap":
            suffix = "" if suffix else "_hi"
        packed_reads.append((suffix, f"\tds_read_u16_d16{suffix} v{register}, {address}"))
    for i, m in matrix:
        pair = destinations[f"v{m[2]}"]
        lines[i] = f"{m[1]}v[{pair}:{pair + 1}]{m[4]}"
    for i, _ in reversed(perms):
        del lines[i]
    assert reads == list(range(reads[0], reads[0] + 16))
    if read_order == "split":
        replacement = [line for suffix, line in packed_reads if not suffix]
        replacement += ["\ts_waitcnt lgkmcnt(0)"]
        replacement += [line for suffix, line in packed_reads if suffix]
    else:
        replacement = [line for _, line in packed_reads]
    lines[reads[0]:reads[-1] + 1] = replacement


def overlap_lds(lines, stores, reads, end, part, order, p_pending):
    p_address = stores[-1] + 1
    assert lines[p_address + 1:reads[0]] == FENCE
    assert reads == list(range(reads[0], reads[0] + 16))
    v_stores = [i for i in range(reads[-1] + 1, end) if lines[i].startswith("\tds_write2_b64 ")]
    v_reads = [i for i in range(v_stores[-1] + 1, end) if lines[i].startswith("\tds_read2st64_b64 ")]
    assert len(v_stores) == 2 and len(v_reads) == 8
    assert v_reads == list(range(v_reads[0], v_reads[0] + 8))
    assert lines[v_stores[0] - 3:v_stores[0]] == FENCE
    assert lines[v_reads[0] - 3:v_reads[0]] == FENCE
    # Append the V region at 8192 B. Read2st64 offsets use 512 B units.
    original_v_reads = lines[v_reads[0]:v_reads[-1] + 1]
    relocated_reads = []
    for line in original_v_reads:
        if "offset0:" not in line:
            line = line.replace("offset1:", "offset0:0 offset1:")
        relocated_reads.append(re.sub(r"(offset[01]:)(\d+)", lambda m: f"{m[1]}{16 + int(m[2])}", line))
    p_loads = [lines[p_address], *lines[reads[0]:reads[-1] + 1]]
    if p_pending is not None:
        p_loads.append(f"\ts_waitcnt lgkmcnt({p_pending})")
    v_address_index = v_stores[-1] + (1 if part == "full" else -1)
    v_address = lines[v_address_index]
    assert v_address.startswith("\tv_add_u32_e32 ")
    v_loads = [v_address, *relocated_reads]
    v_pack = lines[reads[-1] + 1:v_stores[0] - 3] + lines[v_stores[0]:v_reads[0] - 3]
    v_pack.remove(v_address)
    p_first = order == "p-first"
    first_base = re.fullmatch(r"\tds_write2_b64 (v\d+), .*", lines[v_stores[0]])[1]
    address = next(line for line in v_pack if line.startswith(f"\tv_add_u32_e32 {first_base}, 0, "))
    v_pack[v_pack.index(address)] = address.replace(", 0, ", ", 8192, ")
    last_store = v_pack.index(lines[v_stores[-1]])
    assert lines[v_stores[-1]].startswith("\tds_write2_b64 v148, ")
    scratch = "v172" if part == "full" else "v171"
    assert all(not re.search(rf"\b{scratch}\b", line) for line in v_pack)
    v_pack[last_store:last_store + 1] = [
        f"\tv_add_u32_e32 {scratch}, 8192, v148",
        lines[v_stores[-1]].replace("v148,", f"{scratch},", 1),
    ]
    # LDS reads may still be outstanding while rescaling the PV accumulator.
    final_perm = next(i for i in range(v_reads[-1] + 1, end) if PERM.fullmatch(lines[i]))
    lines.insert(final_perm, f"\ts_waitcnt lgkmcnt({8 if p_first else 0})")
    lines[p_address:v_reads[-1] + 1] = v_pack + FENCE + (
        p_loads + v_loads if p_first else v_loads + p_loads
    )


def transform(source, mode, scope, read_order, overlap, p_pending):
    normalized = re.sub(r'(^\s*\.file\s+\d+ )"[^"]+"', r'\1"<directory>"', source, flags=re.M)
    # LLVM also records the compilation directory in the DWARF string table.
    normalized = re.sub(r'(^\.Linfo_string2:\n[ \t]*\.asciz[ \t]+)"/[^"\n]*"',
                        r'\1"<directory>"', normalized, flags=re.M)
    if digest(normalized.encode()) != REFERENCE:
        raise ValueError("Expected the qualified joint K/V assembly")
    lines = source.splitlines()
    for part, start_tag, end_tag, first_temp in (
        ("tail", ".Ltmp5:", ".Ltmp6:", 209),
        ("full", ".Ltmp1:", ".Ltmp2:", 208),
    ):
        if scope not in (part, "both"):
            continue
        start, end = lines.index(start_tag), lines.index(end_tag)
        stores = [i for i in range(start, end) if STORE.fullmatch(lines[i])]
        reads = [i for i in range(start, end) if LOAD.fullmatch(lines[i])]
        assert len(stores) == len(reads) == 16
        if overlap:
            overlap_lds(lines, stores, reads, end, part, overlap, p_pending)
        if mode in ("read", "both"):
            packed_read(lines, reads, end, first_temp, read_order)
        if mode in ("store", "both"):
            high_store(lines, stores, start)
    return "\n".join(lines) + "\n"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("mode", choices=("read", "store", "both", "unchanged"))
    cli.add_argument("--scope", choices=("full", "tail", "both"), default="both")
    cli.add_argument("--read-order", choices=("interleaved", "split", "swap"), default="interleaved")
    cli.add_argument("--overlap", choices=("p-first", "v-first"))
    cli.add_argument("--p-pending", type=int, choices=(0, 4, 7))
    args = cli.parse_args()
    if args.overlap and args.mode in ("read", "both"):
        cli.error("--overlap supports store or unchanged")
    if not args.overlap and args.mode == "unchanged":
        cli.error("unchanged requires --overlap")
    if args.p_pending is not None and args.overlap != "p-first":
        cli.error("--p-pending requires --overlap p-first")
    source = (args.reference / (STEM + ".amdgcn")).read_text()
    assembly = transform(source, args.mode, args.scope, args.read_order, args.overlap, args.p_pending)
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    options = backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2))
    image = backend.make_hsaco(assembly, {}, options)
    meta = "dense-attention-aot-build.json"
    record = json.loads((args.reference / meta).read_text())
    original = record["packed_strides"]["hsaco_sha256"]
    record["packed_strides"]["hsaco_sha256"] = digest(image)
    if args.overlap:
        record["packed_strides"]["shared"] = 16384
    record["experiment"]["p_half_lds"] = dict(mode=args.mode, scope=args.scope, read_order=args.read_order,
                                              overlap=args.overlap,
                                              p_pending=args.p_pending,
                                              reference_packed_sha256=original)
    args.output.mkdir()
    for name in ("dense-attention-aot.so", "dense-attention-aot.hsaco", "dense-attention-fallback.hsaco"):
        shutil.copy2(args.reference / name, args.output / name)
    (args.output / (STEM + ".amdgcn")).write_text(assembly)
    (args.output / (STEM + ".hsaco")).write_bytes(image)
    (args.output / meta).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
