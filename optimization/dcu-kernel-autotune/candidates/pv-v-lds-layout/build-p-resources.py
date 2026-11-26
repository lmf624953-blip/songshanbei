#!/usr/bin/env python3
"""在已筛选的 V 与 Q 方案上测试 P 的直接布局转换。"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from p_resources import apply
from songshan_h3.compiler_compat import configure_hygon_compiler
import triton
from triton.backends.compiler import GPUTarget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--tail",action="store_true")
    args=parser.parse_args()
    root=args.reference.resolve(strict=True)
    out=args.output.resolve()
    out.mkdir()
    metadata="dense-attention-aot-build.json"
    record=json.loads((root/metadata).read_text())
    assert record["shared"]==8192 and record["query_block"]==64 and record["key_block"]==32
    assert record["warps"]==4 and record["num_stages"]==1 and record["kpack"]==2
    assert record["experiment"]["q_method"]=="direct"
    source=out/"pv-swizzle.ttgir"
    source.write_text(apply((root/"pv-swizzle.ttgir").read_text(),args.tail))
    os.environ["AMDGCN_USE_BUFFER_OPS"]="1"
    configure_hygon_compiler()
    device=triton.compile(str(source),target=GPUTarget("hip","gfx936",64),
                          options={"num_warps":4,"num_stages":1,"matrix_instr_nonkdim":0,"kpack":2})
    ir=device.asm["llir"]
    match=re.search(r"define amdgpu_kernel void @dense_attention_aot_kernel\(([^\n]*)\)",ir)
    assert match and re.findall(r"(ptr addrspace\(1\)|i32|float) inreg",match[1])==(
        ["ptr addrspace(1)"]*3+["float","ptr addrspace(1)"]+["i32"]*8+["ptr addrspace(1)"]*2)
    assert device.metadata.profile_scratch_size==0
    asm=device.asm["amdgcn"]
    isa=Counter(re.findall(r"^[ \t]*(ds_\w+|v_mmac_\w+|global_\w+|s_barrier|s_waitcnt)\b",asm,re.M))
    assert isa["v_mmac_f32_16x16x16_bf16"]==64
    (out/"dense-attention-aot.amdgcn").write_text(asm)
    image=device.asm["hsaco"]
    (out/"dense-attention-aot.hsaco").write_bytes(image)
    for name in ("dense-attention-aot.so","dense-attention-fallback.hsaco"):
        shutil.copy2(root/name,out/name)
    record.update(source=str(source),shared=device.metadata.shared,hsaco_sha256=hashlib.sha256(image).hexdigest())
    record["experiment"].update(isa=dict(isa),p_convert="full-and-tail" if args.tail else "full-only")
    (out/metadata).write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(dict(shared=record["shared"],isa=dict(isa)),ensure_ascii=False))


if __name__=="__main__":
    main()
