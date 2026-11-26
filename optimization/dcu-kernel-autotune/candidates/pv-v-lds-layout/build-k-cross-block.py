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
REFERENCE = "3581fe29fb7046ee58cdfc6cc972e15967ef529d15bd1f89d4c04e957f98e331"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def transform(source, placement, include_v, pair_with_lds):
    normalized = re.sub(r'(^\s*\.file\s+\d+ )"[^"]+"', r'\1"<directory>"', source, flags=re.M)
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
    lines[miss:join] = first

    # The predicate and K address use the already advanced loop counters.
    loop_end = lines.index("\ts_branch .LBB0_14", lines.index(".Ltmp3:"))
    # These are the original four increments. No instruction between PV and the backedge
    # reads their values except the existing loop termination compare.
    advancing = [
        "\ts_add_i32 s46, s46, 1", "\ts_add_i32 s24, s24, 0x15000",
        "\ts_addk_i32 s22, 0x7000", "\ts_add_i32 s45, s45, 32",
    ]
    prefetch = [
        *guard, "\ts_cbranch_vccz .Lh3_next_k_load", *zeros,
        "\ts_branch .Lh3_next_k_ready", ".Lh3_next_k_load:", *loads, ".Lh3_next_k_ready:",
    ]
    lines[loop_end] = f"\ts_branch {next_head}"
    if placement == "backedge":
        lines[loop_end:loop_end] = prefetch
    else:
        update = lines.index(advancing[0], lines.index(".Ltmp3:"))
        assert lines[update:update + len(advancing)] == advancing
        del lines[update:update + len(advancing)]
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
    args = cli.parse_args()
    asm = STEM + ".amdgcn"
    image_name = STEM + ".hsaco"
    assembly = transform((args.reference / asm).read_text(), args.placement, args.include_v, args.pair_with_lds)
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
        pair_with_lds=args.pair_with_lds, reference_packed_sha256=previous
    )
    args.output.mkdir()
    for name in ("dense-attention-aot.so", "dense-attention-aot.hsaco", "dense-attention-fallback.hsaco"):
        shutil.copy2(args.reference / name, args.output / name)
    (args.output / asm).write_text(assembly)
    (args.output / image_name).write_bytes(image)
    (args.output / meta).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
