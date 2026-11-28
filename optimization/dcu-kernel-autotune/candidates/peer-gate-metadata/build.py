#!/usr/bin/env python3
"""从现用九齿分片门控表达构建 owner 元数据复用对照。"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys

from reduction_lowering import restore_owner_reductions


REFERENCE_SHA = "bd991cfda1df2ab008f246560ec4bf594709df8487cbbf23f5d7f5ab6318cf0a"


def rotate_order(source):
    first = "    r=(ringbits>>(3*((owner+1)&7)))&7"
    rotation = ("    rotation=3*((owner+1)&7)\n"
                "    order=((ringbits>>rotation)|(ringbits<<(24-rotation)))&0xffffff\n"
                "    r=order&7")
    step = "        r=(ringbits>>(3*((owner+step)&7)))&7"
    assert source.count(first) == source.count(step) == 1
    return source.replace(first, rotation).replace(step, "        r=(order>>(3*(step-1)))&7")


def share_owner(source, tile):
    owner = "    owner=owners.source[index//8].to(ntl.int32)\n"
    assert source.count(owner) == 1
    prefix, body = source.split(owner)
    scalar = ("    block_owners=owners.source[index//8].to(ntl.int32)\n"
              "    first_owner=ntl.min(block_owners,0).to(ntl.int32)\n"
              "    last_owner=ntl.max(block_owners,0).to(ntl.int32)\n")
    nested = "".join("    " + line if line.strip() else line for line in body.splitlines(keepends=True))
    return (prefix + scalar + "    if first_owner==last_owner:\n"
            + "        owner=first_owner\n" + nested
            + "    else:\n" + "        owner=owners.source[index//8].to(ntl.int32)\n" + nested)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("rotate", "shared", "shared-rotate"), required=True)
    parser.add_argument("--tile", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出目录须尚不存在")
    reference = Path(__file__).resolve().parent.parent / "peer-sharded-norm"
    path = reference / "row_gate_kernel.py"
    source = path.read_text()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_SHA
    if "rotate" in args.mode:
        source = rotate_order(source)
    if args.mode.startswith("shared"):
        source = share_owner(source, args.tile)
    expression = args.output.parent / f"{args.output.name}-expression.py"
    with expression.open("x") as stream:
        stream.write(source)
    spec = importlib.util.spec_from_file_location("peer_gate_metadata_expression", expression)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.insert(0, str(reference))
    sys.path.insert(0, str(reference.parents[2] / "adaln"))
    builder = reference / "build-row-gate.py"
    main_function = runpy.run_path(str(builder))["main"]
    main_function.__globals__.update(arrangement=module.arrangement, application=module.application)
    passes = []
    if args.mode.startswith("shared"):
        import aot_codegen
        original_lower = aot_codegen.lower_kernel
        original_compile = aot_codegen.compile_image

        def lower(*values, **options):
            generated, kernel, bindings = original_lower(*values, **options)
            passes.append(restore_owner_reductions(generated, args.tile))
            lowered_spec = importlib.util.spec_from_file_location("peer_gate_metadata_lowered", generated)
            lowered = importlib.util.module_from_spec(lowered_spec)
            lowered_spec.loader.exec_module(lowered)
            return generated, getattr(lowered, kernel.__name__), bindings

        def compile_with_shared(*values, **options):
            return original_compile(*values, **(options | {"allow_shared": True}))

        aot_codegen.lower_kernel = lower
        aot_codegen.compile_image = compile_with_shared
    sys.argv = [str(builder), str(args.output), "--tile", str(args.tile), "--warps", "4"]
    main_function()
    metadata_path = args.output / "peer-gate-build.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(reference_expression_sha256=REFERENCE_SHA,
                    source_sha256=hashlib.sha256(expression.read_bytes()).hexdigest(),
                    metadata_reuse_mode=args.mode, compiler_passes=passes)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    (args.output / "expression.py").write_text(source)


if __name__ == "__main__":
    main()
