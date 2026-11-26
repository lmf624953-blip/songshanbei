#!/usr/bin/env python3
"""为 Q128 packed 诊断镜像预转置 K，保留完整 attention 计算。"""
import argparse
import json
from pathlib import Path
import re
import runpy
import sys


def rewrite(source, unit):
    layout = "#h3_k_direct_dot"
    lines = source.splitlines()
    loads = [i for i, line in enumerate(lines) if " = tt.load " in line
             and f"tensor<128x32x!tt.ptr<bf16>, {layout}>" in line]
    assert len(loads) == 2
    for block, index in reversed(list(enumerate(loads))):
        loop = max(i for i in range(index) if " = scf.for " in lines[i])
        iv = re.search(r"scf.for (%[-.\w]+) =", lines[loop])[1]
        prefix = f"%h3_transpose_k{block}"
        row = f"tensor<128x1xi32, {layout}>"
        col = f"tensor<1x32xi32, {layout}>"
        matrix = f"tensor<128x32xi32, {layout}>"
        steps = [
            f"{prefix}_ceil = arith.addi %ninetoothed_ninetoothed_tensor_0_size_2, %c31_pack : i32",
            f"{prefix}_blocks = arith.divui {prefix}_ceil, %c32_i32 : i32",
            f"{prefix}_pad = arith.muli {prefix}_blocks, %c32_i32 : i32",
            f"{prefix}_headsize = arith.muli {prefix}_pad, %c128_i32 : i32",
            f"{prefix}_head = arith.muli %h3_invariant_query_0_56, {prefix}_headsize : i32",
            f"{prefix}_base = tt.addptr %key, {prefix}_head : !tt.ptr<bf16>, i32",
            f"{prefix}_d = tt.make_range {{end = 128 : i32, start = 0 : i32}} : tensor<128xi32, #ttg.slice<{{dim = 1, parent = {layout}}}>>",
            f"{prefix}_drow = tt.expand_dims {prefix}_d {{axis = 1 : i32}} : tensor<128xi32, #ttg.slice<{{dim = 1, parent = {layout}}}>> -> {row}",
            f"{prefix}_pitch = tt.splat {prefix}_pad : i32 -> {row}",
            f"{prefix}_offsetd = arith.muli {prefix}_drow, {prefix}_pitch : {row}",
            f"{prefix}_n = tt.make_range {{end = 32 : i32, start = 0 : i32}} : tensor<32xi32, #ttg.slice<{{dim = 0, parent = {layout}}}>>",
            f"{prefix}_ncol = tt.expand_dims {prefix}_n {{axis = 0 : i32}} : tensor<32xi32, #ttg.slice<{{dim = 0, parent = {layout}}}>> -> {col}",
            f"{prefix}_start = arith.muli {iv}, %c32_i32 : i32",
            f"{prefix}_startcol = tt.splat {prefix}_start : i32 -> {col}",
            f"{prefix}_offsetn = arith.addi {prefix}_ncol, {prefix}_startcol : {col}",
            f"{prefix}_dr = tt.broadcast {prefix}_offsetd : {row} -> {matrix}",
            f"{prefix}_nc = tt.broadcast {prefix}_offsetn : {col} -> {matrix}",
            f"{prefix}_offset = arith.addi {prefix}_dr, {prefix}_nc : {matrix}",
            f"{prefix}_ptrs = tt.splat {prefix}_base : !tt.ptr<bf16> -> tensor<128x32x!tt.ptr<bf16>, {layout}>",
            f"{prefix}_ptr = tt.addptr {prefix}_ptrs, {prefix}_offset : tensor<128x32x!tt.ptr<bf16>, {layout}>, {matrix}",
        ]
        if unit:
            # 在原地址计算前定义打包块，替换每个矩阵元素的线性偏移。
            steps = steps[:1] + steps[1:3] + [
                f"{prefix}_headsize = arith.muli {prefix}_pad, %c128_i32 : i32",
                f"{prefix}_head = arith.muli %h3_invariant_query_0_56, {prefix}_headsize : i32",
                f"{prefix}_base = tt.addptr %key, {prefix}_head : !tt.ptr<bf16>, i32",
            ] + steps[6:8] + steps[10:12] + [
                f"{prefix}_unit = arith.constant dense<{unit}> : {row}",
                f"{prefix}_quot = arith.divui {prefix}_drow, {prefix}_unit : {row}",
                f"{prefix}_rem = arith.remui {prefix}_drow, {prefix}_unit : {row}",
                f"{prefix}_chunk = arith.constant dense<{32*unit}> : {row}",
                f"{prefix}_chunkoff = arith.muli {prefix}_quot, {prefix}_chunk : {row}",
                f"{prefix}_offsetd = arith.addi {prefix}_chunkoff, {prefix}_rem : {row}",
                f"{prefix}_nunit = arith.constant dense<{unit}> : {col}",
                f"{prefix}_offsetn = arith.muli {prefix}_ncol, {prefix}_nunit : {col}",
                f"{prefix}_blocksize = arith.constant 4096 : i32",
                f"{prefix}_blockoff = arith.muli {iv}, {prefix}_blocksize : i32",
                f"{prefix}_blockbase = tt.addptr {prefix}_base, {prefix}_blockoff : !tt.ptr<bf16>, i32",
                f"{prefix}_dr = tt.broadcast {prefix}_offsetd : {row} -> {matrix}",
                f"{prefix}_nc = tt.broadcast {prefix}_offsetn : {col} -> {matrix}",
                f"{prefix}_offset = arith.addi {prefix}_dr, {prefix}_nc : {matrix}",
                f"{prefix}_ptrs = tt.splat {prefix}_blockbase : !tt.ptr<bf16> -> tensor<128x32x!tt.ptr<bf16>, {layout}>",
                f"{prefix}_ptr = tt.addptr {prefix}_ptrs, {prefix}_offset : tensor<128x32x!tt.ptr<bf16>, {layout}>, {matrix}",
            ]
        line = re.sub(r"(tt.load )%[-.\w]+", r"\g<1>" + prefix + "_ptr", lines[index])
        lines[index:index+1] = ["      " + step for step in steps] + [line]
    entry = next(i for i, line in enumerate(lines) if line.startswith("  tt.func public "))
    lines.insert(entry+1, "    %c31_pack = arith.constant 31 : i32")
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--unit", type=int, choices=(0,8,16), default=0)
    args = parser.parse_args()
    framework = Path(__file__).resolve().parents[2]
    helpers = framework / "candidates/pv-v-lds-layout"
    sys.path.insert(0, str(helpers))
    redirect = runpy.run_path(str(helpers / "k_resources.py"))["apply"]
    compile_packed = runpy.run_path(str(helpers / "build-pv-wide-operand.py"))["compile_packed"]
    config = json.loads((args.reference / "dense-attention-aot-build.json").read_text())
    assert config["query_block"] == 128 and config["key_block"] == 32 and config["warps"] == 4
    source = (args.reference / "dense-attention-packed-pv-swizzle.ttgir").read_text()
    source = redirect(source, True).replace("#h3_k_dot", "#h3_k_direct_dot")
    compile_packed(args.reference, args.output, rewrite(source,args.unit),
                   dict(k_pretranspose=True, unit=args.unit, diagnostic_only=True),
                   rounding_boundary=False)


if __name__ == "__main__":
    main()
