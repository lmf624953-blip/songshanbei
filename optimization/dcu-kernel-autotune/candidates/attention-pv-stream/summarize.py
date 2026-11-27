#!/usr/bin/env python3
"""复用六组已完成逐位/配对结果与原生PMC，不重新运行GPU。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import socket


def pmc(path):
    rows=[]
    for block in path.read_text().split('kernel-name:"')[1:]:
        lines=block.splitlines();row=dict(kernel_name=lines[0].rstrip('"'))
        for line in lines[1:]:
            match=re.match(r"\s*(\S.*?)\s{2,}(\S.*)$",line)
            if match:
                row[match[1]]=match[2]
        rows.append(row)
    assert len(rows)==2 and all(x["scratch size"]=="0" for x in rows)
    return rows


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("experiment",type=Path)
    cli.add_argument("output",type=Path)
    args=cli.parse_args();root=args.experiment
    assert (root/"pmc-exit-code").read_text().strip()=="0"
    assert (root/"extra-tail-exit-code").read_text().strip()=="0"
    variants=[]
    memory=None
    for name in ("stream","staged","staged-compact-v2","k16","k16-compact","k32"):
        prefix="verify" if name=="stream" else "verify-"+name
        assert (root/(prefix+"-exit-code")).read_text().strip()=="0"
        rows=[json.loads(line) for line in (root/(prefix+".jsonl")).read_text().splitlines()]
        bits=[row for row in rows if row["kind"]=="bitwise"]
        pairs=[row for row in rows if row["kind"]=="paired"]
        assert len(bits)==95 and len(pairs)==2
        assert all(row["exact"] and row["tail_ok"] and row["mismatches"]==0 for row in bits)
        assert all(row["passed"] and all(len(s)==19 for s in row["samples_ms"]) for row in pairs)
        text=(root/name/"dense-attention-packed.amdgcn").read_text()
        counts=Counter(line.strip().split()[0] for line in text.splitlines()
            if line.startswith(("\tds_","\tglobal_","\tv_mmac")))
        if memory is None:
            memory=counts
        assert counts==memory
        meta=json.loads((root/name/"dense-attention-aot-build.json").read_text())
        packed=root/name/"dense-attention-packed.hsaco"
        digest=hashlib.sha256(packed.read_bytes()).hexdigest()
        assert meta["packed_strides"]["hsaco_sha256"]==digest
        variants.append(dict(variant=name,records=rows,packed_sha256=digest,
            static_memory_and_mmac_counts=dict(counts),VGPR_descriptor=meta["experiment"]["resources"]["vgpr_count"],
            build_metadata_sha256=hashlib.sha256((root/name/"dense-attention-aot-build.json").read_bytes()).hexdigest()))
    extra=[json.loads(line) for line in (root/"extra-tails.jsonl").read_text().splitlines()]
    assert len(extra)==30 and all(row["passed"] for row in extra)
    pmcs={name:pmc(root/f"pmc-{name}.txt") for name in ("k16","k16-compact")}
    assert all(row["vgpr count"]=="192" for row in pmcs["k16-compact"])
    old=root.parents[0]/"attention-register-pipeline-f13r1n16-1795816098852363496/pmc-default-reference.txt"
    invalid=[json.loads(line) for line in (root/"verify-staged-compact.jsonl").read_text().splitlines()]
    report=dict(node=socket.gethostname(),experiment=str(root),selected=False,
        scope="真实长注意力单卡微测与独立PMC；没有新八卡/完整成片成绩",variants=variants,
        extra_tails=extra,pmc=pmcs,reference_pmc_reused=dict(path=str(old),sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
            note="复用紧邻上一轮相同冻结镜像的默认PMC，不冒充本轮同期采样"),
        occupancy_API=json.loads((root/"occupancy-all.json").read_text()),
        source_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest()for x in Path(__file__).parent.iterdir()if x.is_file()},
        failed_compact_v1=dict(exit_code=134,passed_records_before_failure=invalid,
            failure="首个实际packed长度4097地址错误；只有48条原短长度回退记录，不计作packed数值验证通过",
            identified_bug="区域着色漏掉仅穿过区域、未在区域内显式读写的活跃值，使后续K地址被临时值覆盖",
            fix="used集合包含live-in/live-out，所有活跃值加入物理占位并校验映射完整性；v2全部逐位及额外尾块通过"),
        CPU_failures=["初版试图线性跟踪复杂输出区EXEC合并而断言；最终保留输出区全部原指令并保守保留v0:113边界",
            "粗粒度区域的192/200寄存器搜索没有找到解；后续改用5阶段区域",
            "K32本轮192寄存器有限搜索没有找到解，不宣称无解"],
        decision="最接近基准的是K16/192VGPR，两个生产长度仍慢约1.14%/1.30%，不采用；正式配置不变",
        limits="本轮证明这些分段调度组合无收益；没有证明全局最优。资源API上限4 blocks/CU不等于动态occupancy或吞吐提升。短长度回退不作为新packed内核数值证据。")
    with args.output.open("x")as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2);stream.write("\n")


if __name__=="__main__":
    main()
