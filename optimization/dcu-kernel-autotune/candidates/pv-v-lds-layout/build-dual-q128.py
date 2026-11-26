#!/usr/bin/env python3
"""Build two four-wave Q128 tiles per CTA from the frozen packed LLVM IR."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget


REFERENCE_SHA256 = "170beed8a92797444a6525b06d5f0bd43935b89b2cac83ee0652f02dc7d07e9c"
P_BASES = (200, 210, 224, 234, 431, 441, 455, 465)
KV_LOCATIONS = {51: (2, 2), 67: (4, 4), 83: (2, 2), 98: (4, 4)}
RELEASE = '  fence syncscope("workgroup") release'


def gate_kv_loads(lines, location, expected):
    blocks = []
    for line_index, line in enumerate(lines):
        if re.search(rf" = load <\d+ x bfloat>, ptr addrspace\(1\).* !dbg !{location}$", line):
            block = next(line for line in reversed(lines[:line_index]) if re.match(r"^[\w.]+:", line))
            blocks.append(block.split(":", 1)[0])
    assert len(set(blocks)) == len(blocks) == expected
    for block in blocks:
        branch = re.compile(rf"  br i1 (%\S+), label %{block}, label (%\S+?)(, !dbg !\d+)?$")
        matches = [(i, match) for i, line in enumerate(lines) if (match := branch.fullmatch(line))]
        assert len(matches) == 1, block
        index, match = matches[0]
        flag = f"%h3_load_{block}"
        lines[index:index + 1] = [f"  {flag} = and i1 {match[1]}, %h3_primary",
                                  f"  br i1 {flag}, label %{block}, label {match[2]}{match[3] or ''}"]


def gate_kv_stores(lines, location, expected):
    stores = [i for i, line in enumerate(lines)
              if re.search(rf"^  store <\d+ x bfloat> %\w+, ptr addrspace\(3\).* !dbg !{location}$", line)]
    assert len(stores) == expected
    first, last = stores[0], stores[-1]
    before = next(i for i in reversed(range(first)) if lines[i].startswith(RELEASE))
    after = next(i for i in range(last + 1, len(lines)) if lines[i].startswith(RELEASE))
    block = next(line for line in reversed(lines[:before]) if re.match(r"^[\w.]+:", line))
    block = block.split(":", 1)[0]
    assert "s.barrier()" in lines[before + 1] and "s.barrier()" in lines[after + 1]
    assert all(not re.match(r"^[\w.]+:|^  br ", line) for line in lines[before + 3:after])
    write, done = f"h3_write_{location}", f"h3_done_{location}"
    lines[after:after] = [f"  br label %{done}", "", f"{done}:"]
    lines[before + 3:before + 3] = [f"  br i1 %h3_primary, label %{write}, label %{done}", "", f"{write}:"]
    for i, line in enumerate(lines):
        if " = phi " in line:
            lines[i] = re.sub(rf", %{block} \]", f", %{done} ]", line)


def transform(source, share_kv):
    if hashlib.sha256(source.encode()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Use the verified Q128 packed LLVM IR")
    source = re.sub(r"%(\d+)\b", r"%o\1", source)
    source = re.sub(r"^(\d+):", r"o\1:", source, flags=re.M)
    lines = source.splitlines()
    entry = next(i for i, line in enumerate(lines) if line.startswith("define amdgpu_kernel "))
    lines.insert(entry + 1, "o15:")
    tid = next(i for i, line in enumerate(lines) if line.startswith("  %o20 = "))
    assert "workitem.id.x()" in lines[tid]
    lines[tid:tid + 1] = [
        lines[tid].replace("%o20", "%h3_tid"),
        "  %o20 = and i32 %h3_tid, 255",
        "  %h3_tile = lshr i32 %h3_tid, 8",
        "  %h3_primary = icmp eq i32 %h3_tile, 0",
        "  %h3_lds_offset = shl nuw nsw i32 %h3_tile, 13",
        "  %h3_tile_lds = getelementptr i8, ptr addrspace(3) @global_smem, i32 %h3_lds_offset",
    ]
    grid = next(i for i, line in enumerate(lines) if line.startswith("  %o32 = "))
    assert "%o31, 7" in lines[grid]
    lines[grid:grid + 1] = ["  %h3_groups = add i32 %o31, 1",
                            "  %h3_group_count = lshr i32 %h3_groups, 1",
                            lines[grid].replace("%o31, 7", "%h3_group_count, 7")]
    for i, line in enumerate(lines):
        if line.startswith("  %.frozen846 = "):
            lines[i] = line.replace("%o31", "%h3_group_count")
        if line.startswith("  %.decomposed847 = "):
            lines[i:i + 1] = [line.replace("%.decomposed847", "%h3_group"),
                               "  %h3_query_base = shl i32 %h3_group, 1",
                               "  %.decomposed847 = or disjoint i32 %h3_query_base, %h3_tile"]
            break
    roots = [i for i, line in enumerate(lines) if "getelementptr inbounds nuw i8, ptr addrspace(3) @global_smem" in line]
    assert len(roots) == 26
    for index in roots:
        root = re.match(r"  %o(\d+) =", lines[index])
        if not share_kv or int(root[1]) in P_BASES:
            lines[index] = lines[index].replace("@global_smem", "%h3_tile_lds")
    if share_kv:
        for location, (loads, stores) in KV_LOCATIONS.items():
            gate_kv_loads(lines, location, loads)
            gate_kv_stores(lines, location, stores)
    source = "\n".join(lines) + "\n"
    attr = '"amdgpu-flat-work-group-size"="1,256"'
    assert source.count(attr) == 1
    return source.replace(attr, '"amdgpu-flat-work-group-size"="512,512"')


def build_host(output):
    adaln = Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization/adaln"
    sys.path.insert(0, str(adaln))
    from aot_codegen import compile_host

    source = (adaln / "attention_buffer_host.cc").read_text()
    launch = """    check(hipModuleLaunchKernel(function, heads * ((rows + query_block - 1) / query_block),
                               1, 1, threads, 1, 1, selected_shared, stream, args, nullptr));"""
    assert source.count(launch) == 1
    modified = """    const int tile_rows = kind == 2 ? query_block * 2 : query_block;
    const int block_threads = kind == 2 ? threads * 2 : threads;
    check(hipModuleLaunchKernel(function, heads * ((rows + tile_rows - 1) / tile_rows),
                               1, 1, block_threads, 1, 1, selected_shared, stream, args, nullptr));"""
    copy = output / "attention_dual_q128_host.cc"
    copy.write_text(source.replace(launch, modified))
    return compile_host(output / "dense-attention-aot.so", (copy,), defines=(("H3_ATTENTION_QUERY_BLOCK", 128),))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--shared-kv", action="store_true")
    args = cli.parse_args()
    ref = args.reference.resolve(strict=True)
    output = args.output.resolve()
    source = transform((ref / "dense-attention-packed.llir").read_text(), args.shared_kv)
    os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    options = backend.parse_options(dict(num_warps=8, num_stages=1, kpack=2))
    metadata = {}
    assembly = backend.make_amdgcn(source, metadata, options)
    image = backend.make_hsaco(assembly, metadata, options)
    isa = Counter(re.findall(r"^[ \t]*(ds_\w+|v_mmac_\w+|s_barrier|buffer_load_\w+|global_load_\w+)\b", assembly, re.M))
    resources = {key: int(value) for key, value in re.findall(
        r"^\s*\.(vgpr_count|sgpr_count|private_segment_fixed_size|group_segment_fixed_size):\s*(\d+)", assembly, re.M)}
    assert isa["v_mmac_f32_16x16x16_bf16"] == 128 and isa["s_barrier"] == 12
    output.mkdir()
    record = json.loads((ref / "dense-attention-aot-build.json").read_text())
    assert record["query_block"] == 128 and record["warps"] == 4 and record["packed_strides"]["shared"] == 8192
    for name in ("dense-attention-aot.hsaco", "dense-attention-fallback.hsaco"):
        shutil.copy2(ref / name, output / name)
    (output / "dense-attention-packed.hsaco").write_bytes(image)
    (output / "dense-attention-packed.amdgcn").write_text(assembly)
    (output / "dense-attention-packed.llir").write_text(source)
    record["host_command"] = build_host(output)
    record["packed_strides"].update(hsaco_sha256=hashlib.sha256(image).hexdigest(), shared=16384)
    record["experiment"] = dict(shared_kv=args.shared_kv, packed_query_block=256, packed_warps=8,
                                resources=resources, isa=isa)
    (output / "dense-attention-aot-build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["experiment"]), flush=True)


if __name__ == "__main__":
    main()
