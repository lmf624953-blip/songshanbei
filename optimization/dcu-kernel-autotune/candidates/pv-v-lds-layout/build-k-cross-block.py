#!/usr/bin/env python3
"""Test next-block K prefetch on the qualified joint K/V assembly."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget


STEM = "dense-attention-packed"
REFERENCE = "5d18866b7f2ca268278df74acdabef735828a6201efe0798f63151d40d25dec8"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def address_induction(init, k_reads, v_reads, scope):
    for operand, low, high, base, end, loads in (
        ("k", 22, 23, 37, 38, k_reads), ("v", 24, 25, 39, 40, v_reads),
    ):
        if scope not in (operand, "both"):
            continue
        index = init.index(f"\ts_mov_b32 s{low}, 0")
        init[index:index + 1] = [f"\ts_mov_b32 s{low}, s{base}", f"\ts_mov_b32 s{high}, s{end}"]
        first = loads.index(f"\ts_ashr_i32 s{high}, s{low}, 31")
        assert loads[first:first + 4] == [
            f"\ts_ashr_i32 s{high}, s{low}, 31",
            f"\ts_lshl_b64 s[2:3], s[{low}:{high}], 1",
            f"\ts_add_u32 s2, s{base}, s2",
            f"\ts_addc_u32 s3, s{end}, s3",
        ]
        del loads[first:first + 4]
        for i, line in enumerate(loads):
            if line.startswith("\tglobal_load_"):
                assert line.endswith("s[2:3]")
                loads[i] = line.removesuffix("s[2:3]") + f"s[{low}:{high}]"


def transform(source, placement, include_v, pair_with_lds, loop_mask=False, addresses=False):
    normalized = re.sub(r'(^\s*\.file\s+\d+ )"[^"]+"', r'\1"<directory>"', source, flags=re.M)
    # LLVM also records the compilation directory in the DWARF string table.
    normalized = re.sub(r'(^\.Linfo_string2:\n[ \t]*\.asciz[ \t]+)"/[^"\n]*"',
                        r'\1"<directory>"', normalized, flags=re.M)
    if digest(normalized.encode()) != REFERENCE:
        raise ValueError("Expected the qualified joint K/V assembly")
    if pair_with_lds:
        import runpy
        build = runpy.run_path(str(Path(__file__).with_name("build-p-half-lds.py")))
        source = build["transform"](source, "store", "both", "interleaved", "p-first", 7)
    lines = source.splitlines()
    head = lines.index(".LBB0_14:")
    miss = lines.index("\ts_cbranch_vccz .Lh3_k_load", head)
    k_load = lines.index(".Lh3_k_load:")
    v_load = lines.index("\ts_ashr_i32 s25, s24, 31", k_load) - 1
    join = lines.index(".LBB0_16:", k_load)
    # V loads retain the exact original predicate independently of the K pipeline.
    guard = lines[head + 1:miss]
    assert guard[-1] == "\ts_andn2_b64 vcc, exec, s[26:27]"
    k_reads = [line.replace("v[97:100]", "v[166:169]").replace("v[101:104]", "v[170:173]")
               for line in lines[k_load + 1:v_load]]
    assert sum(line.startswith("\tglobal_load_dwordx4 ") for line in k_reads) == 2
    k_zeros = [f"\tv_mov_b64_e32 v[{reg}:{reg + 1}], 0" for reg in range(166, 174, 2)]
    v_zeros = lines[miss + 5:k_load - 1]
    assert len(v_zeros) == 4 and all("], 0" in line for line in v_zeros)
    v_reads = lines[v_load:join]
    init = lines[:head]
    if addresses:
        address_induction(init, k_reads, v_reads, addresses)
    if loop_mask:
        # n > 0, full block indices are below floor(n/32), and the packed
        # host checks the signed 32-bit byte span before launching.
        index = init.index("\ts_mov_b32 s45, 0")
        init[index:index + 1] = [
            "\ts_cmp_gt_i32 s30, -1",
            "\ts_cselect_b64 s[26:27], -1, 0",
            "\ts_and_b64 s[26:27], s[10:11], s[26:27]",
        ]
        guard = guard[-1:]
    if include_v:
        old_v = {f"v[{i}:{i + 1}]": f"v[{i + 8}:{i + 9}]" for i in range(198, 206, 2)}
        for i in range(join, lines.index(".Ltmp2:")):
            if lines[i].startswith("\tv_mov_b64_e32 ") and lines[i].split(", ")[-1] in old_v:
                prefix, source_v = lines[i].rsplit(", ", 1)
                lines[i] = prefix + ", " + old_v[source_v]
        for before, after in old_v.items():
            v_reads = [line.replace(before, after) for line in v_reads]
            v_zeros = [line.replace(before, after) for line in v_zeros]
    for old, new in (("v[97:100]", "v[166:169]"), ("v[101:104]", "v[170:173]")):
        index = next(i for i in range(join, join + 30) if lines[i].startswith("\tds_write_b128 ") and old in lines[i])
        lines[index] = lines[index].replace(old, new)
    next_head = ".LBB0_16" if include_v else ".Lh3_v_guard"
    zeros = [*k_zeros, *v_zeros] if include_v else k_zeros
    loads = [*k_reads, *v_reads] if include_v else k_reads
    first = ["\ts_cbranch_vccz .Lh3_first_k_load", *zeros, f"\ts_branch {next_head}",
             ".Lh3_first_k_load:", *loads]
    if not include_v:
        first.extend([".Lh3_v_guard:", "\ts_cbranch_vccz .Lh3_v_load", *v_zeros, "\ts_branch .LBB0_16",
                      ".Lh3_v_load:", *v_reads])
    lines = init + [lines[head], *guard, *first, *lines[join:]]

    # The predicate and K address use the already advanced loop counters.
    loop_end = lines.index("\ts_branch .LBB0_14", lines.index(".Ltmp3:"))
    # These are the original four increments. No instruction between PV and the backedge
    # reads their values except the existing loop termination compare.
    original_advance = [
        "\ts_add_i32 s46, s46, 1", "\ts_add_i32 s24, s24, 0x15000",
        "\ts_addk_i32 s22, 0x7000", "\ts_add_i32 s45, s45, 32",
    ]
    advancing = list(original_advance)
    if addresses in ("k", "both"):
        advancing[2:3] = ["\ts_add_u32 s22, s22, 0xe000", "\ts_addc_u32 s23, s23, 0"]
    if addresses in ("v", "both"):
        advancing[1:2] = ["\ts_add_u32 s24, s24, 0x2a000", "\ts_addc_u32 s25, s25, 0"]
    if loop_mask:
        del advancing[-1]
    prefetch = [
        *guard, "\ts_cbranch_vccz .Lh3_next_k_load", *zeros,
        "\ts_branch .Lh3_next_k_ready", ".Lh3_next_k_load:", *loads, ".Lh3_next_k_ready:",
    ]
    lines[loop_end] = f"\ts_branch {next_head}"
    update = lines.index(original_advance[0], lines.index(".Ltmp3:"))
    assert lines[update:update + len(original_advance)] == original_advance
    if placement == "backedge":
        lines[loop_end:loop_end] = prefetch
        lines[update:update + len(original_advance)] = advancing
    else:
        del lines[update:update + len(original_advance)]
        start = lines.index(".Ltmp2:") + 1
        if placement == "before-dpp":
            start = lines.index("\t;;#ASMSTART", start)
        live_region = lines[start:lines.index(f"\ts_branch {next_head}", start)]
        # None of the K staging registers are touched between prefetch and the loop header.
        assert not any(re.search(r"\bv(?:16[6-9]|17[0-3])\b", line) for line in live_region)
        if include_v:
            assert not any(re.search(r"\bv(?:20[6-9]|21[0-3])\b", line) for line in live_region)
        lines[start:start] = [
            *advancing, "\ts_cmp_eq_u32 s41, s46", "\ts_cbranch_scc1 .Lh3_next_k_ready", *prefetch,
        ]
    return "\n".join(lines) + "\n"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("placement", choices=("backedge", "after-pv", "before-dpp"))
    cli.add_argument("--include-v", action="store_true")
    cli.add_argument("--pair-with-lds", action="store_true")
    cli.add_argument("--loop-mask", action="store_true")
    cli.add_argument("--address-induction", nargs="?", const="both", choices=("k", "v", "both"))
    args = cli.parse_args()
    asm = STEM + ".amdgcn"
    image_name = STEM + ".hsaco"
    assembly = transform((args.reference / asm).read_text(), args.placement, args.include_v,
                         args.pair_with_lds, args.loop_mask, args.address_induction)
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    options = backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2))
    image = backend.make_hsaco(assembly, {}, options)
    meta = "dense-attention-aot-build.json"
    record = json.loads((args.reference / meta).read_text())
    previous = record["packed_strides"]["hsaco_sha256"]
    record["packed_strides"]["hsaco_sha256"] = digest(image)
    if args.pair_with_lds:
        record["packed_strides"]["shared"] = 16384
    record["experiment"]["k_cross_block"] = dict(
        placement=args.placement, temporary_vgprs="166:173", include_v=args.include_v,
        pair_with_lds=args.pair_with_lds, loop_mask=args.loop_mask, address_induction=args.address_induction,
        reference_packed_sha256=previous
    )
    args.output.mkdir()
    for name in ("dense-attention-aot.so", "dense-attention-aot.hsaco", "dense-attention-fallback.hsaco"):
        shutil.copy2(args.reference / name, args.output / name)
    (args.output / asm).write_text(assembly)
    (args.output / image_name).write_bytes(image)
    (args.output / meta).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
