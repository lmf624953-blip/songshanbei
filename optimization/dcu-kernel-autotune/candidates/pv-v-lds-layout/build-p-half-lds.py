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


REFERENCE = "3581fe29fb7046ee58cdfc6cc972e15967ef529d15bd1f89d4c04e957f98e331"
SHIFT = re.compile(r"\tv_lshrrev_b32_e32 (v\d+), 16, \1")
SELECT = re.compile(r"\tv_cndmask_b32_e32 (v\d+), (v\d+), (v\d+), vcc")
STORE = re.compile(r"\tds_write_b16 v\d+, (v\d+)(?: offset:\d+)?")
LOAD = re.compile(r"\tds_read_u16 (v\d+), (v\d+(?: offset:\d+)?)")
PERM = re.compile(r"\tv_perm_b32 (v\d+), (v\d+), (v\d+), s(?:43|46)")
MATRIX = re.compile(r"(\tv_mmac_f32_16x16x16_bf16 v\[\d+:\d+\], v\[\d+:\d+\], )v\[(\d+):(\d+)\](, v\[\d+:\d+\])")
STEM = "dense-attention-packed"


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


def transform(source, mode, scope, read_order):
    normalized = re.sub(r'(^\s*\.file\s+\d+ )"[^"]+"', r'\1"<directory>"', source, flags=re.M)
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
        if mode in ("read", "both"):
            packed_read(lines, reads, end, first_temp, read_order)
        if mode in ("store", "both"):
            high_store(lines, stores, start)
    return "\n".join(lines) + "\n"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("mode", choices=("read", "store", "both"))
    cli.add_argument("--scope", choices=("full", "tail", "both"), default="both")
    cli.add_argument("--read-order", choices=("interleaved", "split", "swap"), default="interleaved")
    args = cli.parse_args()
    source = (args.reference / (STEM + ".amdgcn")).read_text()
    assembly = transform(source, args.mode, args.scope, args.read_order)
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    options = backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2))
    image = backend.make_hsaco(assembly, {}, options)
    meta = "dense-attention-aot-build.json"
    record = json.loads((args.reference / meta).read_text())
    original = record["packed_strides"]["hsaco_sha256"]
    record["packed_strides"]["hsaco_sha256"] = digest(image)
    record["experiment"]["p_half_lds"] = dict(mode=args.mode, scope=args.scope, read_order=args.read_order,
                                              reference_packed_sha256=original)
    args.output.mkdir()
    for name in ("dense-attention-aot.so", "dense-attention-aot.hsaco", "dense-attention-fallback.hsaco"):
        shutil.copy2(args.reference / name, args.output / name)
    (args.output / (STEM + ".amdgcn")).write_text(assembly)
    (args.output / (STEM + ".hsaco")).write_bytes(image)
    (args.output / meta).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
