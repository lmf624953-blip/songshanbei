#!/usr/bin/env python3
"""在冻结的第 51 轮九齿 lowering 上独立筛选 PV 的 V 片上布局。"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

import triton
from producer_layout import LAYOUTS, apply
from q_resources_v2 import apply as direct_query
from triton.backends.compiler import GPUTarget

sys.path.insert(0, str(Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization"))
from songshan_h3.compiler_compat import configure_hygon_compiler

ORIGINAL = "#shared3 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>"
PRESETS = [
    (8, 1, 16, (1, 0)),  # 重编对照
    (16, 1, 8, (1, 0)),
    (4, 1, 16, (1, 0)),
    (4, 1, 8, (0, 1)),
    (4, 2, 8, (0, 1)),
    (8, 1, 4, (0, 1)),
    (8, 2, 4, (0, 1)),
    (16, 1, 2, (0, 1)),
    (2, 1, 16, (0, 1)),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("index", type=int, choices=range(len(PRESETS)))
    parser.add_argument("--producer", choices=LAYOUTS, required=True)
    args = parser.parse_args()
    build, out = args.reference.resolve(strict=True), args.output.resolve()
    record_file = "dense-attention-aot-build.json"
    record = json.loads((build / record_file).read_text())
    assert record["query_block"] == 64 and record["key_block"] == 32
    assert record["warps"] == 4 and record["kpack"] == 2 and record["num_stages"] == 1
    assert record["elide_full_kv_load_mask"] and record["buffer_ops_limit_bytes"] == 2**31
    original = (build / "pv-swizzle.ttgir").read_text()
    assert original.count(ORIGINAL) == 1
    # 共享布局只用于完整和尾块两处 V 写入/读取。
    assert original.count("!ttg.memdesc<32x128xbf16, #shared3, #smem>") == 4
    assert len(re.findall(r"#shared3\b", original)) == 5
    vec, phase, phases, order = PRESETS[args.index]
    order = list(order)
    assert vec * phases <= [32, 128][order[0]]
    revised = (f"#shared3 = #ttg.swizzled_shared<{{vec = {vec}, perPhase = {phase}, "
               f"maxPhase = {phases}, order = {order}}}>")
    source = direct_query(apply(original.replace(ORIGINAL, revised), args.producer), "direct")
    out.mkdir()
    (out / "pv-swizzle.ttgir").write_text(source)
    os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"
    configure_hygon_compiler()
    compiled = triton.compile(str(out / "pv-swizzle.ttgir"), target=GPUTarget("hip", "gfx936", 64),
                              options={"num_warps": 4, "num_stages": 1,
                                       "matrix_instr_nonkdim": 0, "kpack": 2})
    image = compiled.asm["hsaco"]
    (out / "dense-attention-aot.hsaco").write_bytes(image)
    (out / "dense-attention-aot.amdgcn").write_text(compiled.asm["amdgcn"])
    for name in ("dense-attention-aot.so", "dense-attention-fallback.hsaco"):
        shutil.copy2(build / name, out / name)
    assert compiled.metadata.profile_scratch_size == 0
    asm = compiled.asm["amdgcn"]
    counts = Counter(re.findall(r"^[ \t]*(ds_[\w]+|v_mmac_[\w]+|global_[\w]+|s_barrier|s_waitcnt)\b", asm, re.M))
    ir = compiled.asm["llir"]
    match = re.search(r"define amdgpu_kernel void @dense_attention_aot_kernel\(([^\n]*)\)", ir)
    assert match is not None
    args_abi = re.findall(r"(ptr addrspace\(1\)|i32|float) inreg", match[1])
    assert args_abi == ["ptr addrspace(1)"]*3+["float","ptr addrspace(1)"]+["i32"]*8+["ptr addrspace(1)"]*2
    assert counts["v_mmac_f32_16x16x16_bf16"] == 64
    record.update(hsaco_sha256=sha(image), shared=compiled.metadata.shared, source=str(out / "pv-swizzle.ttgir"),
                  experiment=dict(index=args.index, v_layout=[vec, phase, phases, order],
                                  source_ttgir_sha256=sha(original.encode()), producer=args.producer, q_method="direct", isa=dict(counts)))
    (out / record_file).write_text(json.dumps(record, indent=2) + "\n")
    props = {k: int(v) for k,v in re.findall(r"^[ \t]*\.(vgpr_count|sgpr_count|private_segment_fixed_size):[ \t]*(\d+)", asm, re.M)}
    print(json.dumps(dict(index=args.index, producer=args.producer, layout=record["experiment"]["v_layout"],
                          hsaco_sha256=record["hsaco_sha256"], shared=record["shared"],
                          isa=dict(counts), resources=props)), flush=True)


if __name__ == "__main__":
    main()
