#!/usr/bin/env python3
"""解释实际 V 打包汇编，核对全部 4096 个 BF16 元素的 LDS 映射。"""

import argparse
import json
from pathlib import Path
import random
import re


def permute(high, low, selector):
    sources = low.to_bytes(4, "little") + high.to_bytes(4, "little")
    indices = [(selector >> (8 * i)) & 255 for i in range(4)]
    assert all(i < 8 for i in indices)
    return int.from_bytes(bytes(sources[i] for i in indices), "little")


def region(build):
    lines = (build / "dense-attention-packed.amdgcn").read_text().splitlines()
    start = lines.index(".LBB0_18:")
    p_stores = [i for i in range(start, len(lines)) if lines[i].startswith("\tds_write_b16_d16_hi ")]
    assert len(p_stores) == 32
    begin = p_stores[15] + 1
    end = lines.index("\ts_waitcnt lgkmcnt(0)", begin)
    return [line.strip() for line in lines[begin:end] if not line.lstrip().startswith(".loc")]


def emulate(lines, tid, elements, two_loads, delayed):
    registers = {"v0": tid, "s42": 0xffff, "s43": 0x05040100, "s44": 0x07060302}
    source = []
    if two_loads:
        row, column = (tid >> 4) * 2, (tid & 15) * 8
        for r in (row, row + 1):
            source += [elements[r * 128 + column + i] | (elements[r * 128 + column + i + 1] << 16)
                       for i in range(0, 8, 2)]
        base = ((tid & 15) << 9) | (((tid >> 4) ^ ((tid & 1) << 3)) << 2)
        registers.update(v134=base, v148=base ^ 0x88, v142=base ^ 0x110, v143=base ^ 0x198)
    else:
        row, column = (tid >> 5) * 4, (tid & 31) * 4
        for r in range(row, row + 4):
            source += [elements[r * 128 + column + i] | (elements[r * 128 + column + i + 1] << 16)
                       for i in (0, 2)]
        base = ((tid & 31) << 8) | (((tid << 4) & 48) ^ ((tid & 224) >> 2))
        registers.update(v134=base, v148=base ^ 0x88)
    labels = list(range(206, 214)) if delayed else [*range(175, 182), 104]
    registers.update({f"v{index}": value for index, value in zip(labels, source, strict=True)})
    memory = {}
    for line in lines:
        if line == "s_waitcnt vmcnt(0)":
            continue
        opcode, operands = line.split(maxsplit=1)
        fields = operands.split(", ")
        if opcode.startswith("ds_write2_"):
            width = 2 if opcode == "ds_write2_b64" else 1
            assert opcode in ("ds_write2_b64", "ds_write2_b32")
            offset = re.fullmatch(r"(.+) offset1:(\d+)", fields[-1])
            assert offset
            data = [fields[1], offset[1]]
            for operand, byte_offset in zip(data, (0, int(offset[2]) * width * 4), strict=True):
                if width == 2:
                    match = re.fullmatch(r"v\[(\d+):(\d+)\]", operand)
                    assert match and int(match[2]) == int(match[1]) + 1
                    words = [registers[f"v{i}"] for i in range(int(match[1]), int(match[2]) + 1)]
                else:
                    words = [registers[operand]]
                for i, word in enumerate(words):
                    address = registers[fields[0]] + byte_offset + i * 4
                    assert address not in memory
                    memory[address] = word
            continue
        destination = fields[0]
        def value(operand):
            return registers[operand] if operand in registers else int(operand, 0)
        arguments = [value(operand) for operand in fields[1:]]
        if opcode == "v_lshlrev_b32_e32":
            shift, word = arguments
            result = word << shift
        elif opcode == "v_bfi_b32":
            mask, low, high = arguments
            result = (low & mask) | (high & ~mask)
        elif opcode == "v_perm_b32":
            result = permute(*arguments)
        elif opcode == "v_add_u32_e32":
            result = sum(arguments)
        else:
            raise ValueError(f"Unexpected V pack instruction {line}")
        registers[destination] = result & 0xffffffff
    return memory


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("builds", type=Path, nargs="+")
    args = cli.parse_args()
    generator = random.Random(42)
    inputs = [list(range(4096)), [generator.randrange(65536) for _ in range(4096)]]
    for build in args.builds:
        metadata = json.loads((build / "dense-attention-aot-build.json").read_text())
        experiment = metadata["experiment"]
        two_loads = "v_two_loads" in experiment
        options = experiment.get("v_two_loads", experiment.get("v_pack_permute", {}))
        lines = region(build)
        for elements in inputs:
            memory = {}
            for tid in range(256):
                thread = emulate(lines, tid, elements, two_loads, bool(options.get("wait_at")))
                assert memory.keys().isdisjoint(thread)
                memory.update(thread)
            assert sorted(memory) == list(range(8192, 16384, 4))
            for address, word in memory.items():
                index = (address - 8192) // 4
                column = index >> 4
                row = 2 * ((index & 15) ^ (((column >> 1) & 7) << 1))
                expected = elements[row * 128 + column] | (elements[(row + 1) * 128 + column] << 16)
                assert word == expected, (build, address, word, expected)
        print(json.dumps(dict(build=str(build), elements=4096, trials=len(inputs),
                              bitwise_equal=True, two_loads=two_loads)))


if __name__ == "__main__":
    main()
