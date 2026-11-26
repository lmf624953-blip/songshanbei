#!/usr/bin/env python3
"""基于 Q128 的 K32 生成代码筛选 guard、V 宽加载、Q 直接加载的组合。"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import triton
from producer_layout import apply as apply_v
from q_resources_v2 import apply as apply_q
from triton.backends.compiler import GPUTarget


RECORD = "dense-attention-aot-build.json"
V_BASE = "#shared3 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>"
V_WIDE = "#shared3 = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1]}>"


def build_reference(code, output):
    script = code / "optimization/adaln/build-attention-aot.py"
    build = output / "v2-q128-guarded"
    subprocess.run([
        sys.executable, "-u", str(script), str(build), "--fp32-scale", "--aligned-strides",
        "--query-block", "128", "--key-block", "32", "--warps", "4", "--stages", "1", "--kpack", "2",
        "--optimize-long-attention", "--split-kv-tail", "--elide-full-kv-load-mask", "--guarded-buffer-ops",
    ], check=True, env={**os.environ, "H3_AOT_SAVE_IR": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return build


def tag_reference(reference):
    path = reference / RECORD
    record = json.loads(path.read_text())
    asm = (reference / "dense-attention-aot.amdgcn").read_text()
    record["experiment"] = dict(v_layout=[8, 1, 16, [1, 0]], isa=isa_counts(asm))
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def isa_counts(asm):
    return dict(Counter(re.findall(r"^[ \t]*(ds_\w+|v_mmac_\w+|buffer_\w+|s_barrier|s_waitcnt)\b", asm, re.M)))


def build_variant(reference, output, mode):
    from songshan_h3.compiler_compat import configure_hygon_compiler

    record = json.loads((reference / RECORD).read_text())
    assert record["query_block"] == 128 and record["key_block"] == 32 and record["warps"] == 4
    assert record["elide_full_kv_load_mask"] and record["buffer_ops_limit_bytes"] == 2**31
    text = (reference / "pv-swizzle.ttgir").read_text()
    assert text.count(V_BASE) == 1 and len(re.findall(r"#shared3\b", text)) == 5
    if mode in ("vwide", "both"):
        text = apply_v(text.replace(V_BASE, V_WIDE), "k4-n4")
    if mode in ("qdirect", "both"):
        text = apply_q(text, "direct", query_block=128)
    output.mkdir()
    source = output / "pv-swizzle.ttgir"
    source.write_text(text)
    os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"
    configure_hygon_compiler()
    compiled = triton.compile(str(source), target=GPUTarget("hip", "gfx936", 64), options={
        "num_warps": 4, "num_stages": 1, "matrix_instr_nonkdim": 0, "kpack": 2,
    })
    match = re.search(r"define amdgpu_kernel void @dense_attention_aot_kernel\(([^\n]*)\)", compiled.asm["llir"])
    assert match and re.findall(r"(ptr addrspace\(1\)|i32|float) inreg", match[1]) == (
        ["ptr addrspace(1)"] * 3 + ["float", "ptr addrspace(1)"] + ["i32"] * 8 + ["ptr addrspace(1)"] * 2)
    image = compiled.asm["hsaco"]
    asm = compiled.asm["amdgcn"]
    counts = isa_counts(asm)
    dot = "v_mmac_f32_16x16x16_bf16"
    assert counts[dot] == record["experiment"]["isa"][dot]
    assert compiled.metadata.profile_scratch_size == 0
    for name in ("dense-attention-aot.so", "dense-attention-fallback.hsaco"):
        shutil.copy2(reference / name, output / name)
    (output / "dense-attention-aot.hsaco").write_bytes(image)
    (output / "dense-attention-aot.amdgcn").write_text(asm)
    record.update(hsaco_sha256=hashlib.sha256(image).hexdigest(), source=str(source),
                  shared=compiled.metadata.shared,
                  experiment=dict(mode=mode, isa=counts,
                                  v_layout=[4, 2, 8, [0, 1]] if mode != "qdirect" else [8, 1, 16, [1, 0]]))
    (output / RECORD).write_text(json.dumps(record, indent=2) + "\n")
    resources = dict(re.findall(r"^[ \t]*\.(vgpr_count|sgpr_count|private_segment_fixed_size):[ \t]*(\d+)", asm, re.M))
    print(json.dumps(dict(mode=mode, shared=record["shared"], resources=resources, isa=counts)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("q64", type=Path)
    parser.add_argument("q128", type=Path)
    args = parser.parse_args()
    code = Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]).resolve(strict=True)
    sys.path.insert(0, str(code / "optimization"))
    output = args.output.resolve()
    output.mkdir()
    shutil.copytree(args.q64, output / "v0")
    old = output / "v1-q128-old"
    shutil.copytree(args.q128, old)
    tag_reference(old)
    ref = build_reference(code, output)
    tag_reference(ref)
    for name, mode in (("v3-q128-vwide", "vwide"), ("v4-q128-qdirect", "qdirect"), ("v5-q128-both", "both")):
        build_variant(ref, output / name, mode)


if __name__ == "__main__":
    main()
