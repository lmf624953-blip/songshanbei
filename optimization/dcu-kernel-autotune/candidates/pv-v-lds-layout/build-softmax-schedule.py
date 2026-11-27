#!/usr/bin/env python3
"""Schedule the per-row online softmax arithmetic of two verified gfx936 builds."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget


STEM = "dense-attention-packed"
REFERENCES = {
    "5d18866b7f2ca268278df74acdabef735828a6201efe0798f63151d40d25dec8",
    "e61aaa7dacdf81adefec52d56e24b22b3e21b01b1848e9fdbfdbf95b9c853fcf",
}
FOLDED_REFERENCE = "51c3d00a93fa9cf8d6c238b6d502647ffb530970bbd2ba0b19c90a1c536de1f0"
OPERATIONS = {
    "v_max_f32_e32", "v_sub_f32_e32", "v_exp_f32_e32", "v_add_u32_e32",
    "v_add3_u32", "v_bfe_u32", "v_lshrrev_b32_e32", "v_cmp_o_f32_e32",
    "v_cndmask_b32_e32", "v_mov_b32_e32",
}
REGISTER = re.compile(r"\b(?:[vs]\d+|vcc)\b")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inputs_and_output(line):
    opcode, operands = line.strip().split(maxsplit=1)
    if opcode not in OPERATIONS:
        raise ValueError(f"Unexpected opcode in the scheduling range: {opcode}")
    output, inputs = operands.split(", ", 1)
    if not REGISTER.fullmatch(output) or "[" in inputs:
        raise ValueError(f"Unexpected operands: {line}")
    return opcode, set(REGISTER.findall(inputs)), output


def schedule(lines, exp_latency, favor_order):
    parsed = [inputs_and_output(line) for line in lines]
    # Record true dependencies separately from ordering constraints on reused registers.
    parents = []
    readers = {}
    writers = {}
    for index, (_, inputs, output) in enumerate(parsed):
        dependencies = {writers[reg]: 1 for reg in inputs if reg in writers}
        for reg in inputs:
            readers.setdefault(reg, set()).add(index)
        for reader in readers.pop(output, set()) - {index}:
            dependencies.setdefault(reader, 0)
        if output in writers:
            dependencies.setdefault(writers[output], 0)
        writers[output] = index
        parents.append(dependencies)
    latency = [exp_latency if opcode == "v_exp_f32_e32" else 4 for opcode, _, _ in parsed]
    height = latency.copy()
    for index in range(len(lines) - 1, -1, -1):
        for parent, true_dep in parents[index].items():
            height[parent] = max(height[parent], height[index] + (latency[parent] if true_dep else 1))
    issued = {}
    pending = list(range(len(lines)))
    while pending:
        candidates = []
        for index in pending:
            dependencies = parents[index]
            if dependencies.keys() <= issued.keys():
                earliest = max((issued[parent] + (latency[parent] if true_dep else 1)
                                for parent, true_dep in dependencies.items()), default=0)
                score = -index if favor_order else height[index]
                candidates.append((max(len(issued), earliest), -score, index))
        _, _, selected = min(candidates)
        issued[selected] = len(issued)
        pending.remove(selected)
    assert all(issued[parent] < issued[index] for index, deps in enumerate(parents) for parent in deps)
    return [lines[index] for index in issued]


def fold_self_max(lines, temporary):
    # In these fixed images, both operands come from a max producer or the
    # initialized -inf online state. Arbitrary signaling NaNs cannot be folded.
    max_op = re.compile(r"\tv_max_f32_e32 (v\d+), (v\d+), (v\d+)")
    definitions = {}
    remove = set()
    for i, line in enumerate(lines):
        opcode, _, dest = inputs_and_output(line)
        if opcode == "v_max_f32_e32":
            target, left, right = max_op.fullmatch(line).groups()
            if left == temporary and target == right:
                a = definitions[left]
                b = definitions[right]
                _, source, equal = max_op.fullmatch(lines[a]).groups()
                assert source == equal and lines[b] == f"\tv_max_f32_e32 {right}, {right}, {right}"
                assert definitions.get(source, -1) < a
                remove.update((a, b))
                lines[i] = f"\tv_max_f32_e32 {target}, {source}, {right}"
        definitions[dest] = i
    assert len(remove) == 16
    return [line for i, line in enumerate(lines) if i not in remove]


def update_row_sum(lines, mode):
    begin = lines.index(".Ltmp3:")
    exit_edge = lines.index("\ts_cbranch_scc1 .LBB0_20", begin)
    tail = lines.index(".LBB0_20:", exit_edge)
    match_fma = re.compile(r"\tv_fmac_f32_e32 (v\d+), (v\d+), (v\d+)")
    updates = [(i, m.groups()) for i in range(begin, exit_edge)
               if (m := match_fma.fullmatch(lines[i]))]
    assert len(updates) == 8
    keep = {f"\tv_mov_b32_e32 {state}, {acc}" for _, (acc, state, _) in updates}
    assert keep <= set(lines[exit_edge + 1:tail])
    assert {acc for _, (acc, _, _) in updates}.isdisjoint({state for _, (_, state, _) in updates})
    fmas = [f"\tv_fma_f32 {state}, {state}, {scale}, {acc}" for _, (acc, state, scale) in updates]
    end_copies = [f"\tv_mov_b32_e32 {acc}, {state}" for _, (acc, state, _) in updates]
    lines[tail:tail] = [".Lh3_folded_sum_exit:", *end_copies]
    lines[exit_edge + 1:tail] = [line for line in lines[exit_edge + 1:tail] if line not in keep]
    lines[exit_edge] = "\ts_cbranch_scc1 .Lh3_folded_sum_exit"
    if mode == "direct":
        for (i, _), fma in zip(updates, fmas, strict=True):
            lines[i] = fma
    else:
        for i, _ in reversed(updates[:4]):
            del lines[i]
        for (i, _), fma in zip(updates[4:], fmas[4:], strict=True):
            lines[i - 4] = fma
        # The first four addends are ready before the second four-row DPP group.
        assert lines[begin - 1] == "\t;;#ASMEND"
        dpp_start = max(i for i in range(begin - 24, begin) if lines[i] == "\t;;#ASMSTART")
        first_end = dpp_start - 1
        assert lines[first_end] == "\t;;#ASMEND"
        assert lines[dpp_start + 1] == "\ts_nop 1"
        lines[first_end + 1:first_end + 1] = fmas[:4]
        if mode == "interleave":
            del lines[dpp_start + len(fmas[:4]) + 1]
    return "\n".join(lines) + "\n"


def transform(source, scope, exp_latency, favor_order, fold_max, fold_only, row_sum=None):
    normalized = re.sub(r'(^\s*\.file\s+\d+ )"[^"]+"', r'\1"<directory>"', source, flags=re.M)
    # LLVM also records the compilation directory in the DWARF string table.
    normalized = re.sub(r'(^\.Linfo_string2:\n[ \t]*\.asciz[ \t]+)"/[^"\n]*"',
                        r'\1"<directory>"', normalized, flags=re.M)
    source_id = digest(normalized.encode())
    if row_sum:
        if source_id not in (FOLDED_REFERENCE, "5d18866b7f2ca268278df74acdabef735828a6201efe0798f63151d40d25dec8"):
            raise ValueError("Expected the verified folded or joint K/V image for row sum updates")
        return update_row_sum(source.splitlines(), row_sum), {"row_sum": 8}
    if source_id not in REFERENCES:
        raise ValueError("Expected a verified joint K/V or K-induction build")
    lines = source.splitlines()
    counts = {}
    for part, label, temporary in (("tail", ".LBB0_34:", "v171"), ("full", ".LBB0_18:", "v174")):
        if scope not in (part, "both"):
            continue
        start = next(i for i in range(lines.index(label) + 1, len(lines))
                     if lines[i].startswith("\tv_max_f32_e32 "))
        end = lines.index("\ts_waitcnt vmcnt(0) lgkmcnt(0)", start)
        instructions = [line for line in lines[start:end] if not line.lstrip().startswith(".loc\t")]
        assert len(instructions) in (126, 138, 142, 154), len(instructions)
        selected = fold_self_max(instructions, temporary) if fold_max else instructions
        if not fold_only:
            selected = schedule(selected, exp_latency, favor_order)
        counts[part] = len(selected)
        if fold_max:
            lines[start:end] = selected
        else:
            reordered = iter(selected)
            for i in range(start, end):
                if not lines[i].lstrip().startswith(".loc\t"):
                    lines[i] = next(reordered)
    return "\n".join(lines) + "\n", counts


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--scope", choices=("full", "tail", "both"), default="both")
    cli.add_argument("--exp-latency", type=int, choices=(8, 16, 32), default=16)
    cli.add_argument("--favor-order", action="store_true")
    cli.add_argument("--fold-max", choices=("only", "scheduled"))
    cli.add_argument("--row-sum", choices=("direct", "interleave", "interleave-nop"))
    args = cli.parse_args()
    asm = STEM + ".amdgcn"
    image_name = STEM + ".hsaco"
    assembly, counts = transform((args.reference / asm).read_text(), args.scope, args.exp_latency,
                                 args.favor_order, bool(args.fold_max), args.fold_max == "only", args.row_sum)
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    options = backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2))
    image = backend.make_hsaco(assembly, {}, options)
    meta = "dense-attention-aot-build.json"
    record = json.loads((args.reference / meta).read_text())
    original = record["packed_strides"]["hsaco_sha256"]
    record["packed_strides"]["hsaco_sha256"] = digest(image)
    record["experiment"]["softmax_schedule"] = dict(
        scope=args.scope, groups=counts, exp_latency=args.exp_latency, favor_order=args.favor_order,
        reference_packed_sha256=original, fold_max=args.fold_max, row_sum=args.row_sum
    )
    args.output.mkdir()
    for name in ("dense-attention-aot.so", "dense-attention-aot.hsaco", "dense-attention-fallback.hsaco"):
        shutil.copy2(args.reference / name, args.output / name)
    (args.output / asm).write_text(assembly)
    (args.output / image_name).write_bytes(image)
    (args.output / meta).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
