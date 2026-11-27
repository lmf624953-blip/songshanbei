#!/usr/bin/env python3
"""利用已完成的 xor8 对称性，将完整块的 xor4 两条掩码 DPP 合成一条旋转。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import runpy
import shutil

from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget


FIRST = re.compile(r"\tv_(max|add)_f32_dpp (v\d+), (v\d+), \3 row_shr:4 row_mask:0xf bank_mask:0xa")


def verify_lane_maps():
    for lane in range(64):
        row, column = lane & ~15, lane & 15
        original = lane ^ 4
        rotated = row | ((column - 4) & 15)
        # 对称二元运算得到的 prefix[j] 与 prefix[j xor 8] 数值相同。
        pair = lambda value: sorted((value, value ^ 8))
        assert pair(original) == pair(rotated)


def transform(source):
    verify_lane_maps()
    lines = source.splitlines()
    first, end = lines.index(".LBB0_14:"), lines.index(".LBB0_20:")
    helper = runpy.run_path(str(Path(__file__).with_name("build-qk-online-order.py")))
    drops, changes = [], Counter()
    for index in range(first, end):
        match = FIRST.fullmatch(lines[index])
        if not match:
            continue
        stage, destination, operand = match.groups()
        assert destination != operand
        second = lines[index].replace("row_shr:4", "row_shl:4").replace("bank_mask:0xa", "bank_mask:0x5")
        assert lines[index + 4] == second
        register = int(operand[1:])
        producer = next(line for line in reversed(lines[first:index])
                        if line.startswith("\tv_") and register in helper["uses"](line)[1])
        assert re.fullmatch(rf"\tv_{stage}_f32_dpp {operand}, (v\d+), \1 row_ror:8 row_mask:0xf bank_mask:0xf", producer)
        for line in lines[index + 1:index + 4]:
            if line.startswith("\tv_"):
                reads, writes = helper["uses"](line)
                assert int(destination[1:]) not in reads and register not in writes
        lines[index] = lines[index].replace("row_shr:4", "row_ror:4").replace("bank_mask:0xa", "bank_mask:0xf")
        drops.append(index + 4)
        changes[stage] += 1
    assert changes == {"max": 8, "add": 8}
    for index in reversed(drops):
        del lines[index]
    result = "\n".join(lines) + "\n"
    assert result[result.index(".LBB0_20:"):] == source[source.index(".LBB0_20:"):]
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    record = json.loads((args.reference / "dense-attention-aot-build.json").read_text())
    assert record["experiment"]["prepacked_v"]["p_row_layout"]
    packed = args.reference / "dense-attention-packed.hsaco"
    assert hashlib.sha256(packed.read_bytes()).hexdigest() == record["packed_strides"]["hsaco_sha256"]
    source = (args.reference / "dense-attention-packed.amdgcn").read_text()
    updated = transform(source)
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    image = backend.make_hsaco(updated, {}, backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2)))
    shutil.copytree(args.reference, args.output)
    (args.output / packed.name).write_bytes(image)
    (args.output / "dense-attention-packed.amdgcn").write_text(updated)
    record["packed_strides"]["hsaco_sha256"] = hashlib.sha256(image).hexdigest()
    record["experiment"]["softmax_ror4"] = dict(
        scope="完整KV循环", max_pairs=8, sum_pairs=8, removed_instructions=16,
        invariant="xor8后两个相距8的prefix由交换同一对操作数得到；xor4与ror4的源数值相同",
        reference_packed_sha256=hashlib.sha256(packed.read_bytes()).hexdigest())
    (args.output / "dense-attention-aot-build.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
