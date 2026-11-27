#!/usr/bin/env python3
"""把 K 放到独立 LDS 分区，测量去掉每块两处覆盖前同步的影响。"""

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
REFERENCE = "897dfc11810646ba1cf8d15ecbd6e768e6ae0fa0063445e809607fde26324f90"
K_REGION = 16384
K_ACCESS = re.compile(r"(\tds_(write|read)_b128 [^\n]+?)(?: offset:(\d+))?")
FENCE = ["\ts_barrier", "\ts_waitcnt lgkmcnt(0)"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verified_assembly(source):
    normalized = re.sub(r'(^\s*\.file\s+\d+ )"[^"]+"', r'\1"<directory>"', source, flags=re.M)
    normalized = re.sub(r'(^\.Linfo_string2:\n[ \t]*\.asciz[ \t]+)"/[^"\n]*"',
                        r'\1"<directory>"', normalized, flags=re.M)
    if digest(normalized.encode()) != REFERENCE:
        raise ValueError("需要已核验的 Q128 packed 求和交错汇编")
    return source.splitlines()


def relocate_k(lines, base):
    k_stores, k_reads = [], []
    for index, line in enumerate(lines):
        match = K_ACCESS.fullmatch(line)
        if match:
            operation, kind, original_offset = match.groups()
            offset = int(original_offset or 0)
            assert offset in (0, 4096)
            lines[index] = f"{operation} offset:{base + offset}"
            (k_stores if kind == "write" else k_reads).append(index)
    assert len(k_stores) == 4 and len(k_reads) == 16
    return k_stores


def transform(source, remove_k_fence, remove_p_fence):
    lines = verified_assembly(source)
    k_stores = relocate_k(lines, K_REGION)
    p_stores = [i for i, line in enumerate(lines) if line.startswith("\tds_write_b16_d16_hi ")]
    assert len(p_stores) == 32

    drops = []
    # 每个 K 块的读取在上一个 P/V 发布屏障之前已经完成。
    if remove_k_fence:
        drops += [k_stores[0] - 2, k_stores[2] - 2]
    # 本块的 K 发布屏障已覆盖所有 wave 上一块的 P/V 读取。
    if remove_p_fence:
        drops += [p_stores[0] - 2, p_stores[16] - 2]
    for index in sorted(drops, reverse=True):
        assert lines[index:index + 2] == FENCE
        del lines[index:index + 2]
    return "\n".join(lines) + "\n"


def write_variant(reference, output, assembly, shared, name, details):
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    options = backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2))
    image = backend.make_hsaco(assembly, {}, options)
    meta = "dense-attention-aot-build.json"
    record = json.loads((reference / meta).read_text())
    packed = record["packed_strides"]
    original = packed["hsaco_sha256"]
    assert packed["shared"] == K_REGION
    packed.update(hsaco_sha256=digest(image), shared=shared)
    record["experiment"]["resources"]["vgpr_count"] = int(
        re.search(r"^\s*\.vgpr_count:\s*(\d+)", assembly, re.M)[1])
    record["experiment"][name] = dict(details, reference_packed_sha256=original,
                                       static_barriers=assembly.count("\ts_barrier\n"))
    output.mkdir()
    for filename in ("dense-attention-aot.so", "dense-attention-aot.hsaco", "dense-attention-fallback.hsaco"):
        shutil.copy2(reference / filename, output / filename)
    (output / (STEM + ".amdgcn")).write_text(assembly)
    (output / (STEM + ".hsaco")).write_bytes(image)
    (output / meta).write_text(json.dumps(record, indent=2) + "\n")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--allocation-only", action="store_true")
    cli.add_argument("--remove-k-fence", action="store_true")
    cli.add_argument("--remove-p-fence", action="store_true")
    args = cli.parse_args()
    if args.allocation_only and (args.remove_k_fence or args.remove_p_fence):
        cli.error("--allocation-only 保留原同步")
    source = (args.reference / (STEM + ".amdgcn")).read_text()
    if args.allocation_only:
        verified_assembly(source)
        assembly = source
    else:
        assembly = transform(source, args.remove_k_fence, args.remove_p_fence)
    write_variant(args.reference, args.output, assembly, K_REGION + 8192, "disjoint_k_lds",
                  dict(k_region_bytes=0 if args.allocation_only else K_REGION, remove_k_fence=args.remove_k_fence,
                       remove_p_fence=args.remove_p_fence))


if __name__ == "__main__":
    main()
