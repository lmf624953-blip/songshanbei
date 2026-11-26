#!/usr/bin/env python3
"""Carry one prefetched K block through the Q128 packed full-block loop."""

import argparse
import json
from pathlib import Path
import re
import runpy

from producer_layout import ALLOWED, LINE, SSA


K_TILE = "tensor<128x32xbf16, #blocked1>"
K_MASK = "tensor<128x32xi1, #blocked1>"
K_LOAD = re.compile(
    r"tt.load (%[-.\w]+), (%[-.\w]+), (%[-.\w]+)"
    r" : tensor<128x32x!tt.ptr<bf16>, #blocked1>"
)


def move_k_load(source, position):
    lines = source.splitlines()
    loops = [i for i, line in enumerate(lines) if " = scf.for " in line]
    assert len(loops) == 2
    start = loops[0]
    end = next(i for i in range(start, loops[1]) if lines[i].startswith("      scf.yield "))
    loop = re.fullmatch(
        r"(    %[-.\w]+):3 = scf.for (%[-.\w]+) = (%[-.\w]+) to (%[-.\w]+)"
        r" step (%[-.\w]+) iter_args\((.*)\) -> \((.*)\) +: i32 \{",
        lines[start],
    )
    assert loop, lines[start]
    result, iv, lower, upper, step, args, types = loop.groups()
    assert lower == "%c0_i32" and step == "%c1_i32"
    body = lines[start + 1:end]
    definitions = {match[2]: match[3] for line in body if (match := LINE.fullmatch(line))}
    loads = [(value, match) for value, rhs in definitions.items() if (match := K_LOAD.match(rhs))]
    assert len(loads) == 1
    k, load = loads[0]
    ptr, mask, zero = load.groups()
    assert zero not in definitions and zero == "%cst_4"
    assert source.count(k + " : (" + K_TILE + ")") == 1

    def prefetch(tag, induction, indent):
        prefix = "%h3_k_" + tag
        code = []
        cloned = {iv: induction}

        def emit(name, expr):
            code.append(f"{indent}{name} = {expr}")
            return name

        def clone(value):
            if value in cloned:
                return cloned[value]
            if value not in definitions:
                return value
            rhs = definitions[value]
            op = rhs.split(" ", 1)[0]
            assert op in ALLOWED - {"tt.load"}, rhs
            expr = SSA.sub(lambda match: clone(match[0]), rhs)
            cloned[value] = emit(f"{prefix}_{len(cloned)}", expr)
            return cloned[value]

        address, valid = clone(ptr), clone(mask)
        active = emit(prefix + "_active", f"arith.cmpi slt, {induction}, {upper} : i32")
        broadcast = emit(prefix + "_all", f"tt.splat {active} : i1 -> {K_MASK}")
        guarded = emit(prefix + "_mask", f"arith.andi {valid}, {broadcast} : {K_MASK}")
        payload = emit(prefix + "_tile", f"tt.load {address}, {guarded}, {zero} : {load[0].split(' : ', 1)[1]}")
        return code, payload

    first, initial = prefetch("first", lower, "    ")
    next_iv = "%h3_k_next_iter"
    lookahead, next_tile = prefetch("next", next_iv, "      ")
    lookahead.insert(0, f"      {next_iv} = arith.addi {iv}, {step} : i32")

    body = [line for line in body if not line.startswith(f"      {k} = ")]
    dots = [i for i, line in enumerate(body) if " = tt.dot " in line]
    assert len(dots) == 2
    anchor = dots[0] + 1 if position == "after-qk" else dots[1]
    body[anchor:anchor] = lookahead
    header = (f"{result}:4 = scf.for {iv} = {lower} to {upper} step {step} "
              f"iter_args({args}, {k} = {initial}) -> ({types}, {K_TILE}) : i32 {{")
    values, yield_types = lines[end].split(" : ", 1)
    yield_types, location = yield_types.rsplit(" loc(", 1)
    updated_yield = f"{values}, {next_tile} : {yield_types}, {K_TILE} loc({location}"
    return "\n".join(lines[:start] + first + [header] + body + [updated_yield] + lines[end + 1:]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("position", choices=("after-qk", "before-pv"))
    args = parser.parse_args()
    root = args.reference.resolve(strict=True)
    record = json.loads((root / "dense-attention-aot-build.json").read_text())
    assert record["query_block"] == 128 and record["key_block"] == 32 and record["warps"] == 4
    assert record["packed_strides"]["strides"] == [896, 896, 2688, 896]
    source = (root / "dense-attention-packed-pv-swizzle.ttgir").read_text()
    source = move_k_load(source, args.position)
    directory = Path(__file__).parent
    rounding = runpy.run_path(str(directory / "build-q128-backend.py"))["preserve_scaled_rounding"]
    compile_packed = runpy.run_path(str(directory / "build-pv-wide-operand.py"))["compile_packed"]
    compile_packed(root, args.output, rounding(source), dict(
        k_lookahead=args.position, scope="full K blocks only, one iteration ahead"))


if __name__ == "__main__":
    main()
