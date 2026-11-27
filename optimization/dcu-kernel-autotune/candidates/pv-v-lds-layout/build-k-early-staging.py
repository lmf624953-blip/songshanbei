#!/usr/bin/env python3
"""将K暂存移到独立8个VGPR，区分资源代价与更早跨块预取收益。"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import shutil

from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget


def transform(source, placement):
    lines = source.splitlines()
    first, middle, end = (lines.index(x) for x in (".LBB0_14:", ".Ltmp2:", ".LBB0_20:"))
    registers = runpy.run_path(str(Path(__file__).with_name("build-qk-online-order.py")))["registers"]
    assert not any(registers(line) & set(range(220,228)) for line in lines)
    def rename(line):
        line = re.sub(r"\bv(16[6-9]|17[0-3])\b", lambda m:f"v{int(m[1])+54}",line)
        return re.sub(r"v\[(\d+):(\d+)\]", lambda m:
                      f"v[{int(m[1])+54}:{int(m[2])+54}]" if 166<=int(m[1])<=int(m[2])<=173 else m[0],line)
    # 首块读取、受屏蔽清零及每块K的LDS写入统一使用独立暂存寄存器。
    header_end = lines.index("\ts_andn2_b64 vcc, exec, s[26:27]",lines.index(".LBB0_16:"))
    lines[first:header_end] = [rename(line) for line in lines[first:header_end]]
    ready = lines.index(".Lh3_next_k_ready:",middle)
    block = lines[middle+1:ready+1]
    block = [rename(line) for line in block]
    assert sum(line.startswith("\tglobal_load_dwordx4 ") for line in block)==2
    if placement == "control":
        lines[middle+1:ready+1] = block
    else:
        if placement == "after-qk":
            target = lines.index(".Ltmp0:")
        else:
            target = lines.index("\tds_read_b64 v[122:123], v133",first)
        # 此区间不使用提早推进的标量地址/计数/掩码；V已经发射。
        forbidden = {1,2,3,22,23,26,27,45,46,50,51}
        for line in lines[target:middle]:
            refs = set(int(x) for x in re.findall(r"\bs(\d+)\b",line))
            for lo,hi in re.findall(r"s\[(\d+):(\d+)\]",line):refs.update(range(int(lo),int(hi)+1))
            assert refs.isdisjoint(forbidden),line
        del lines[middle+1:ready+1]
        lines[target:target] = block
    text = "\n".join(lines)+"\n"
    text = text.replace(".amdhsa_next_free_vgpr 219", ".amdhsa_next_free_vgpr 228")
    text = text.replace(".vgpr_count:     219", ".vgpr_count:     228")
    assert text[text.index(".LBB0_20:"):].replace(".amdhsa_next_free_vgpr 228", ".amdhsa_next_free_vgpr 219").replace(".vgpr_count:     228", ".vgpr_count:     219") == source[source.index(".LBB0_20:"):]
    assert text.count("v_mmac_f32_16x16x16_bf16")==source.count("v_mmac_f32_16x16x16_bf16")
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--placement",choices=("control","after-qk","before-pv"),required=True)
    args=parser.parse_args()
    source=(args.reference/"dense-attention-packed.amdgcn").read_text()
    meta=json.loads((args.reference/"dense-attention-aot-build.json").read_text())
    digest=hashlib.sha256((args.reference/"dense-attention-packed.hsaco").read_bytes()).hexdigest()
    assert digest==meta["packed_strides"]["hsaco_sha256"]
    updated=transform(source,args.placement)
    configure_hygon_compiler()
    backend=HIPBackend(GPUTarget("hip","gfx936",64))
    image=backend.make_hsaco(updated,{},backend.parse_options(dict(num_warps=4,num_stages=1,kpack=2)))
    shutil.copytree(args.reference,args.output)
    (args.output/"dense-attention-packed.hsaco").write_bytes(image)
    (args.output/"dense-attention-packed.amdgcn").write_text(updated)
    meta["packed_strides"]["hsaco_sha256"]=hashlib.sha256(image).hexdigest()
    meta["experiment"]["k_early_staging"]=dict(placement=args.placement,reference_packed_sha256=digest,
        staging="v220:227",scope="仅完整块K暂存和发射时序；不新增MOV、不改MMAC/softmax/尾部",numerical_verification="尚未完成")
    (args.output/"dense-attention-aot-build.json").write_text(json.dumps(meta,indent=2)+"\n")


if __name__=="__main__":main()
