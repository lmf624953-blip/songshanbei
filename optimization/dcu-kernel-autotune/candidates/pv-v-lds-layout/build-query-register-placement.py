#!/usr/bin/env python3
"""仅重命名 VGPR，检查 Q 常驻寄存器位置对 MMAC 的影响。"""

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


def rename(source, mapping):
    def replace(match):
        if match[1] is not None:
            return f"v{mapping[int(match[1])]}"
        first, last = int(match[2]), int(match[3])
        selected = [mapping[value] for value in range(first, last + 1)]
        if selected != list(range(selected[0], selected[0] + last - first + 1)):
            raise ValueError(f"重命名破坏连续寄存器组 {match[0]}")
        return f"v[{selected[0]}:{selected[-1]}]"
    return VGPR.sub(replace, source)


def transform(source, target):
    if target not in (187, 188, 189, 190):
        raise ValueError("仅支持已检查的四种 Q 起始寄存器")
    mapping = {value: value if value < 65 else
               target + value - 65 if value <= 96 else value - 32
               for value in range(219)}
    assert len(set(mapping.values())) == 219 and mapping[0] == 0
    transformed = rename(source, mapping)
    inverse = {value: key for key, value in mapping.items()}
    assert rename(transformed, inverse) == source
    assert transformed.count("\tv_mmac_f32_16x16x16_bf16 ") == 128
    count = max(mapping.values()) + 1
    for name in (".amdhsa_next_free_vgpr", ".vgpr_count:"):
        transformed, matches = re.subn(rf"(^\s*{re.escape(name)}\s+)219(\s*)$",
                                       rf"\g<1>{count}\g<2>", transformed, flags=re.M)
        assert matches == 1
    return transformed, mapping, count


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--query-register", type=int, choices=(187, 188, 189, 190), required=True)
    args = cli.parse_args()
    record = json.loads((args.reference / "dense-attention-aot-build.json").read_text())
    assert record["prepacked_value_dispatch"]["version"] == 1
    assert record["experiment"]["prepacked_v"]["p_row_layout"]
    packed = args.reference / "dense-attention-packed.hsaco"
    assert hashlib.sha256(packed.read_bytes()).hexdigest() == record["packed_strides"]["hsaco_sha256"]
    source = (args.reference / "dense-attention-packed.amdgcn").read_text()
    transformed, mapping, count = transform(source, args.query_register)
    configure_hygon_compiler()
    backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
    options = backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2))
    image = backend.make_hsaco(transformed, {}, options)
    shutil.copytree(args.reference, args.output)
    (args.output / packed.name).write_bytes(image)
    (args.output / "dense-attention-packed.amdgcn").write_text(transformed)
    record["packed_strides"]["hsaco_sha256"] = hashlib.sha256(image).hexdigest()
    record["experiment"]["resources"]["vgpr_count"] = count
    record["experiment"]["query_register_placement"] = dict(
        query_first=args.query_register, query_original=[65, 96],
        inverse_exact=True, added_instructions=0, mathematical_reordering=False,
        reference_packed_sha256=hashlib.sha256(packed.read_bytes()).hexdigest(),
        map={str(key): value for key, value in mapping.items()})
    (args.output / "dense-attention-aot-build.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
