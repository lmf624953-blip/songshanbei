#!/usr/bin/env python3
"""在 CPU 上核对精简 softmax 发布重建、冻结来源及旧计时协议回归。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("tested_attention", type=Path)
    cli.add_argument("rebuilt_release", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--prior-sequence", type=Path)
    cli.add_argument("--prior-summary", type=Path)
    args = cli.parse_args()
    if bool(args.prior_sequence) != bool(args.prior_summary):
        cli.error("旧序列与旧汇总必须同时提供")
    here = Path(__file__).resolve().parent
    code = Path(os.environ.get("H3_AUTOTUNE_CODE_ROOT", here.parents[3]))
    elf = runpy.run_path(str(code / "optimization/dcu-kernel-autotune/adapters/compare-h3-device.py"))["elf_loads"]
    summary = runpy.run_path(str(here / "summarize-public-a.py"))
    release = json.loads((args.rebuilt_release / "release.json").read_text())
    assert release["kind"] in ("p-row-prepacked-v-softmax-v1",
                               "p-row-prepacked-v-softmax-stable-paired-v1")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    reference = Path(release["reference_release"])
    assert digest(reference / "release.json") == release["reference_manifest_sha256"]
    for name, expected in release["source_files"].items():
        assert digest(code / name) == expected, name
    candidate = args.rebuilt_release / "attention"
    identity = summary["images"](reference / "attention", candidate)
    artifacts = {}
    for name, expected in release["artifacts"].items():
        left = args.tested_attention / name
        right = candidate / name
        assert digest(right) == expected, name
        original, rebuilt = elf(left), elf(right)
        same_sections = original["sections"] == rebuilt["sections"]
        same_mapping = all(original[key] == rebuilt[key] for key in ("segments", "elf_target"))
        assert same_sections and same_mapping, name
        byte_equal = left.read_bytes() == right.read_bytes()
        if name.endswith(".so"):
            assert byte_equal, "主机库必须逐字节一致"
        artifacts[name] = dict(byte_equal=byte_equal, loaded_sections_equal=same_sections,
                               mapping_equal=same_mapping, original_sha256=original["sha256"],
                               rebuilt_sha256=rebuilt["sha256"])
    result = dict(scope="CPU重建一致性；不代替GPU数值或完整模型资格验收",
                  artifacts=artifacts, images=identity, source_files_match=True,
                  reference_manifest_matches=True)
    if args.prior_sequence:
        previous = json.loads(args.prior_summary.read_text())
        # report 在内存中含 tuple，JSON 协议将其编码为 array，须按协议比较。
        current = json.loads(json.dumps(summary["report"](args.prior_sequence)))
        assert previous == current, "旧计时结果的字段或样本发生变化"
        result["prior_summary_all_fields_equal"] = True
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(dict(artifacts=len(artifacts), same_loaded_code=True,
                         prior_summary_all_fields_equal=result.get("prior_summary_all_fields_equal")),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
