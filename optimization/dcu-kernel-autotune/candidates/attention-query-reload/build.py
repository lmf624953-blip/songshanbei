#!/usr/bin/env python3
"""对已冻结九齿 ISA 分段重载 Q，隔离访存代价与 VGPR 压缩收益。"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget


VGPR = re.compile(r"\bv(?:(\d+)\b|\[(\d+):(\d+)\])")
PINNED = "a00cc85245e936216531503c3c1dfb57293d00d907dd517b4b1b6c8932f8a3cd"


def registers(line):
    result = set()
    for match in VGPR.finditer(line):
        result.update([int(match[1])] if match[1] else range(int(match[2]), int(match[3]) + 1))
    return result


def rename(line, mapping):
    def replace(match):
        values = ([int(match[1])] if match[1] else list(range(int(match[2]), int(match[3]) + 1)))
        values = [mapping.get(v, v) for v in values]
        assert values == list(range(values[0], values[0] + len(values))), match[0]
        return f"v{values[0]}" if match[1] else f"v[{values[0]}:{values[-1]}]"
    return VGPR.sub(replace, line)


def resources(text, vgpr):
    for field, old, new in ((".amdhsa_next_free_vgpr", 219, vgpr),
                            (".vgpr_count:", 219, vgpr),
                            (".amdhsa_next_free_sgpr", 54, 66),
                            (".sgpr_count:", 58, 70)):
        text, count = re.subn(rf"(^\s*{re.escape(field)}\s+){old}(\s*)$",
                              rf"\g<1>{new}\g<2>", text, flags=re.M)
        assert count == 1, field
    return text


def transform(source, mode):
    lines = source.splitlines()
    if mode == "resource-control":
        return resources(source, 221), dict(q_load_groups=0, vgpr=221)
    start = lines.index("\tv_mov_b32_e32 v65, 0")
    end = lines.index(".LBB0_12:") + 3
    assert lines[end - 1] == "\ts_or_b64 exec, exec, s[0:1]"
    before, after = lines[:start], lines[end:]
    removed = lines[start:end]
    assert sum("global_load_dwordx4" in line for line in removed) == 8
    for label in (".LBB0_2", ".LBB0_4", ".LBB0_6", ".LBB0_8", ".LBB0_10", ".LBB0_12"):
        assert not any(re.search(re.escape(label) + r"(?!\d)", line) for line in before + after), label
    header = [
        "\ts_mov_b64 s[58:59], s[0:1]",
        "\ts_mov_b64 s[60:61], s[22:23]",
        "\tv_or_b32_e32 v219, v4, v106",
        "\tv_lshlrev_b32_e32 v219, 1, v219",
        "\tv_add_u32_e32 v220, 0x1c000, v219",
        "\ts_load_dwordx2 s[22:23], s[4:5], 0x10",
        "\tv_or3_b32 v2, v3, s34, 64",
        "\tv_cmp_gt_i32_e32 vcc, s33, v2",
        "\ts_and_b64 s[24:25], s[24:25], vcc",
        "\ts_mov_b64 s[62:63], s[24:25]",
    ]
    epilogue = after.index("\tv_lshrrev_b32_e32 v65, 4, v112")
    qregs = set(range(65, 97))
    for line in after[:epilogue]:
        if registers(line) & qregs:
            assert line.startswith("\tv_mmac_f32_16x16x16_bf16 "), line
            assert registers(line.split(", ")[2]) <= qregs, line
    assert not any(registers(line) & set(range(198, 221)) for line in after[epilogue:])
    # 每个 QK 保留四个累加链及原 32 条 MMAC，Q 每次只装入 8 个 VGPR。
    stages = ((69, 77), (65, 89), (73, 85), (81, 93))
    transformed = []
    count = 0
    for line in after[:epilogue]:
        if line.startswith("\tv_mmac_f32_16x16x16_bf16 ") and registers(line) & qregs:
            stage = (count // 8) % 4
            if count % 8 == 0:
                label = f".Lh3_query_reload_{count // 8}"
                transformed += [f"\tv_mov_b64_e32 v[{lo}:{lo+1}], 0" for lo in (65, 67, 69, 71)]
                for half, mask, address, destination in ((0, 60, 219, 65), (1, 62, 220, 69)):
                    transformed += [
                        f"\ts_and_saveexec_b64 s[64:65], s[{mask}:{mask+1}]",
                        f"\ts_cbranch_execz {label}_{half}",
                        f"\tglobal_load_dwordx4 v[{destination}:{destination+3}], v{address}, s[58:59] offset:{stage*64}",
                        f"{label}_{half}:",
                        "\ts_mov_b64 exec, s[64:65]",
                    ]
                transformed += ["\ts_waitcnt vmcnt(0)"]
            parts = line.split(", ")
            mapping = {stages[stage][half] + j: 65 + half*4 + j for half in (0, 1) for j in range(4)}
            assert registers(parts[2]) <= set(mapping), line
            original = parts[2]
            parts[2] = rename(parts[2], mapping)
            assert rename(parts[2], {v: k for k, v in mapping.items()}) == original
            line = ", ".join(parts)
            count += 1
        transformed.append(line)
    assert count == 64
    live = before + header + transformed
    assert not any(registers(line) & set(range(73, 97)) for line in live)
    if mode == "reload-compact":
        mapping = {old: old - 125 for old in range(198, 221)}
        compact = [rename(line, mapping) for line in live]
        assert [rename(line, {v: k for k, v in mapping.items()}) for line in compact] == live
        live = compact
        count_vgpr = 198
    else:
        count_vgpr = 221
    # 输出归一化、写回和所有 masked tail 逻辑维持原指令；只改变 Q 供数。
    result = "\n".join(live + after[epilogue:]) + "\n"
    assert result.count("v_mmac_f32_16x16x16_bf16") == 128
    assert max(set().union(*(registers(line) for line in result.splitlines()))) + 1 == count_vgpr
    return resources(result, count_vgpr), dict(q_load_groups=8, vgpr=count_vgpr,
        qk_mmac_count=64, pv_mmac_count=64, query_accumulation_order_unchanged=True,
        epilogue_instructions_unchanged=True, shared_bytes=16384)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--mode", choices=("resource-control", "reload-full", "reload-compact"), required=True)
    args = cli.parse_args()
    packed = args.reference / "dense-attention-packed.hsaco"
    assert hashlib.sha256(packed.read_bytes()).hexdigest() == PINNED
    record = json.loads((args.reference / "dense-attention-aot-build.json").read_text())
    assert record["packed_strides"]["hsaco_sha256"] == PINNED
    source = (args.reference / "dense-attention-packed.amdgcn").read_text()
    transformed, changes = transform(source, args.mode)
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    image = backend.make_hsaco(transformed, {}, backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2)))
    shutil.copytree(args.reference, args.output)
    (args.output / packed.name).write_bytes(image)
    (args.output / "dense-attention-packed.amdgcn").write_text(transformed)
    record["packed_strides"]["hsaco_sha256"] = hashlib.sha256(image).hexdigest()
    record["experiment"]["resources"].update(vgpr_count=changes["vgpr"], sgpr_count=70)
    record["experiment"]["query_reload"] = dict(mode=args.mode, reference_packed_sha256=PINNED,
        numerical_verification="尚未完成", sgpr_count=70, **changes)
    (args.output / "dense-attention-aot-build.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
