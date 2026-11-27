#!/usr/bin/env python3
"""将旧 running max 并入局部 max3，实测是否能省去归约后的八条 max。"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil

from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget


def transform(source, negative_logits=False):
    lines = source.splitlines()
    first, middle, end = (lines.index(name) for name in
                          (".LBB0_14:", ".LBB0_18:", ".LBB0_20:"))
    phi = (116, 118, 120, 122, 123, 126, 128, 130)
    block_max = (117, 119, 121, 125, 127, 129, 131, 124)
    registers = runpy.run_path(str(Path(__file__).with_name("build-qk-online-order.py")))["registers"]
    phi_accesses = [line for line in lines[first:middle] if registers(line) & set(phi)]
    if negative_logits:
        assert phi_accesses == [f"\tv_xor_b32_e32 v{r}, 0x80000000, v{r}" for r in sorted(phi)]
    else:
        assert not phi_accesses
    replacements, drops = 0, []
    for row, (old_max, new_max) in enumerate(zip(phi, block_max, strict=True)):
        left = 158 + row if row < 4 else 162 + row
        right = left + 4
        operation = "min" if negative_logits else "max"
        original = f"\tv_{operation}_f32_e32 v{97 + row}, v{left}, v{right}"
        positions = [i for i in range(first, middle) if lines[i] == original]
        assert len(positions) == 1, original
        # 旧 max 在 query 对应的 DPP 子组内广播，不改变 QK/PV 或求和顺序。
        lines[positions[0]] = f"\tv_{operation}3_f32 v{97 + row}, v{left}, v{right}, v{old_max}"
        target = 208 + row if negative_logits else new_max
        late = f"\tv_{operation}_f32_e32 v{target}, v{old_max}, v{new_max}"
        positions = [i for i in range(middle, end) if lines[i] == late]
        assert len(positions) == 1, late
        drops.append(positions[0])
        if negative_logits:
            # 负 logit 版本已将 new_max 放在连续寄存器；最后一条 DPP 直接写该位置。
            final = max(i for i in range(first, middle)
                        if lines[i].startswith(f"\tv_min_f32_dpp v{new_max},"))
            assert not any(new_max in registers(line) for line in lines[final + 1:middle])
            lines[final] = lines[final].replace(f"v_min_f32_dpp v{new_max},",
                                              f"v_min_f32_dpp v{target},", 1)
        replacements += 1
    assert replacements == 8
    for index in reversed(sorted(drops)):
        del lines[index]
    result = "\n".join(lines) + "\n"
    assert result[result.index(".LBB0_20:"):] == source[source.index(".LBB0_20:"):]
    assert result.count("v_mmac_f32_16x16x16_bf16") == source.count("v_mmac_f32_16x16x16_bf16")
    assert result.count("v_exp_f32_e32") == source.count("v_exp_f32_e32")
    assert len(lines) == len(source.splitlines()) - 8
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--negative-logits", action="store_true")
    args = cli.parse_args()
    record = json.loads((args.reference / "dense-attention-aot-build.json").read_text())
    packed = args.reference / "dense-attention-packed.hsaco"
    digest = hashlib.sha256(packed.read_bytes()).hexdigest()
    assert digest == record["packed_strides"]["hsaco_sha256"]
    source = (args.reference / "dense-attention-packed.amdgcn").read_text()
    if args.negative_logits:
        assert record["experiment"]["paired_p_sub"]["negative_logits"]
    updated = transform(source, args.negative_logits)
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    image = backend.make_hsaco(updated, {}, backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2)))
    shutil.copytree(args.reference, args.output)
    (args.output / packed.name).write_bytes(image)
    (args.output / "dense-attention-packed.amdgcn").write_text(updated)
    record["packed_strides"]["hsaco_sha256"] = hashlib.sha256(image).hexdigest()
    record["experiment"]["running_max_fold"] = dict(
        reference_packed_sha256=digest, local_max3=8, removed_late_max=8,
        negative_logits=args.negative_logits,
        unchanged="QK/PV MMAC、FP32求和、RNE、完整KV遍历及尾部",
        numerical_verification="尚未完成，特别核对NaN和带符号零")
    (args.output / "dense-attention-aot-build.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
