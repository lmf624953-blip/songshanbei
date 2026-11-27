#!/usr/bin/env python3
"""仅复用已完成实验，归档原始配对样本、PMC 与资源门槛。"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import socket


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("experiment", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    root = args.experiment
    assert (root / "screen-exit-code").read_text().strip() == "0"
    assert (root / "pmc-default-exit-code").read_text().strip() == "0"
    screen = json.loads((root / "screen-summary.json").read_text())
    checks = [x for x in screen["records"] if x["kind"] == "normal-stress-tail"]
    pairs = [x for x in screen["records"] if x["kind"] == "formal-paired"]
    assert len(checks) == 36 and len(pairs) == 6
    assert all(x["passed"] for x in checks + pairs)
    assert all(len(series) == 39 for row in pairs for series in row["samples_ms"])
    pmc = {}
    for name in ("reference", "reload-compact"):
        text = (root / f"pmc-default-{name}.txt").read_text()
        rows = []
        for block in text.split('kernel-name:"')[1:]:
            lines = block.splitlines()
            fields = dict(kernel_name=lines[0].rstrip('"'))
            for line in lines[1:]:
                match = re.match(r"\s*(\S.*?)\s{2,}(\S.*)$", line)
                if match:
                    fields[match[1]] = match[2]
            assert fields["scratch size"] == "0"
            rows.append(fields)
        assert len(rows) == 2
        pmc[name] = rows
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in Path(__file__).parent.glob("*.py")}
    result = dict(node=socket.gethostname(), experiment=str(root), selected=False,
        scope="注意力微测及独立硬件诊断；没有新整模型成绩", source_sha256=sources,
        source_note="screen在正式任务终态后仅增加occupancy函数的可选kernel/shared参数，默认路径未改变；当前归档哈希不冒充启动时快照。build的最终CPU构建源未改变。",
        screen=screen, pmc=pmc,
        resource_probe=json.loads((root / "resource-probe/resources.json").read_text()),
        agpr_assembler={name:int((root / (name + ".exit-code")).read_text())
            for name in ("control", "agpr-move", "agpr-mmac")},
        failures=[dict(stage="first-CPU-static-assert", reason="标签子串误匹配，修复为完整标签；无GPU失败"),
            dict(stage="first-PMC-wrapper", exit_code=1,
                reason="--pmc-type 3生成原始CSV，外层误检查txt；reference采样本身完成，CSV保留。后续默认--pmc为不同指标采样，双方txt均通过。")],
        decision="分段重载即使压缩至198VGPR也未跨192资源门槛，新增Q读取和同步导致约两倍退化；不接入模型。保持已验收116.975680189秒稳态A配置。",
        limits="仅否定当前分数物化拆分及Q分段重载实现，不否定所有拆分/寄存器/流水线方案；API资源上限不是实测动态occupancy。负候选未进行完整8卡/成片资格。")
    with args.output.open("x") as destination:
        json.dump(result, destination, ensure_ascii=False, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
