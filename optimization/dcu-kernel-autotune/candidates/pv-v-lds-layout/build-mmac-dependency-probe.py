#!/usr/bin/env python3
"""构建独立 MMAC 依赖诊断，禁止作为模型注意力镜像使用。"""

import argparse
import hashlib
import json
from pathlib import Path
import re

from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget


def assembly(source, chains, iterations):
    descriptor = re.search(r"\t\.amdhsa_kernel .*?\t\.end_amdhsa_kernel", source, re.S)[0]
    metadata = re.search(r"\t\.amdgpu_metadata.*?\t\.end_amdgpu_metadata", source, re.S)[0]
    assert ".amdhsa_next_free_vgpr 219" in descriptor
    assert ".amdhsa_kernarg_size 88" in descriptor
    assert chains in (1, 2, 4, 8, 16) and 64 % chains == 0
    lines = [
        '\t.amdgcn_target "amdgcn-amd-amdhsa--gfx936"',
        "\t.text", "\t.globl dense_attention_aot_kernel", "\t.p2align 8",
        "\t.type dense_attention_aot_kernel,@function", "dense_attention_aot_kernel:",
        "\tv_mov_b32_e32 v68, v0",
        "\ts_load_dwordx2 s[8:9], s[4:5], 0x20",
    ]
    lines += [f"\tv_mov_b32_e32 v{register}, 0" for register in range(64)]
    lines += [f"\tv_mov_b32_e32 v{register}, 0x3f803f80" for register in range(64, 68)]
    # VALU 初始化后的 MMAC 源寄存器显式等待；不把未就绪的输入当作依赖吞吐。
    lines += [f"\ts_mov_b32 s0, {iterations}", "\ts_nop 15", ".Ldependency_loop:"]
    for index in range(64):
        first = (index % chains) * 4
        accumulator = f"v[{first}:{first + 3}]"
        lines.append(f"\tv_mmac_f32_16x16x16_bf16 {accumulator}, v[64:65], v[66:67], {accumulator}")
    lines += [
        "\ts_sub_u32 s0, s0, 1", "\ts_cmp_lg_u32 s0, 0", "\ts_cbranch_scc1 .Ldependency_loop",
        "\ts_nop 15", "\ts_waitcnt lgkmcnt(0)",
        "\tv_mov_b32_e32 v69, s6", "\tv_lshlrev_b32_e32 v69, 8, v69",
        "\tv_add_u32_e32 v69, v69, v68", "\tv_lshlrev_b32_e32 v69, 2, v69",
        "\tglobal_store_dword v69, v0, s[8:9]", "\ts_endpgm", ".Ldependency_end:",
        "\t.size dense_attention_aot_kernel,.Ldependency_end-dense_attention_aot_kernel",
        '\t.section .rodata,#alloc', "\t.p2align 6", descriptor, metadata,
    ]
    result = "\n".join(lines) + "\n"
    assert result.count("\tv_mmac_f32_16x16x16_bf16 ") == 64
    assert not any(op in result for op in ("ds_read", "ds_write", "s_barrier", "global_load"))
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--iterations", type=int, default=2048)
    args = cli.parse_args()
    if not 1 <= args.iterations <= 16384:
        cli.error("iterations 必须在 1–16384")
    source = (args.reference / "dense-attention-packed.amdgcn").read_text()
    record = json.loads((args.reference / "dense-attention-aot-build.json").read_text())
    assert record["experiment"]["prepacked_v"]["p_row_layout"]
    reference_image = args.reference / "dense-attention-packed.hsaco"
    assert hashlib.sha256(reference_image.read_bytes()).hexdigest() == record["packed_strides"]["hsaco_sha256"]
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    options = backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2))
    args.output.mkdir()
    variants = []
    for chains in (1, 2, 4, 8, 16):
        text = assembly(source, chains, args.iterations)
        image = backend.make_hsaco(text, {}, options)
        stem = f"mmac-chains-{chains}"
        (args.output / f"{stem}.amdgcn").write_text(text)
        (args.output / f"{stem}.hsaco").write_bytes(image)
        variants.append(dict(chains=chains, image=f"{stem}.hsaco",
                             sha256=hashlib.sha256(image).hexdigest(),
                             expected_output=16 * 64 * args.iterations // chains))
    metadata = dict(scope="独立硬件诊断，未测量；不包含真实QK/PV，不可作为模型镜像",
                    loop_iterations=args.iterations, static_mmacs=64,
                    block_threads=256, dynamic_lds_bytes=16384,
                    initial_operand_wait_cycles=16,
                    descriptor="沿用候选的219静态VGPR和54静态SGPR，保持所有探针资源分配相同",
                    variants=variants, original_packed_sha256=record["packed_strides"]["hsaco_sha256"])
    (args.output / "probe-build.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
