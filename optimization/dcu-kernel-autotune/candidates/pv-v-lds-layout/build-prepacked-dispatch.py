#!/usr/bin/env python3
"""将预排布 V 候选接入原注意力主机 ABI，保留非 packed 及短长度回退。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "combined", "prepack", "output"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    scripts = Path(__file__).resolve().parent
    root = os.environ.get("H3_AUTOTUNE_CODE_ROOT")
    adaln = (Path(root) / "optimization" if root else scripts.parents[2]) / "adaln"
    sys.path.insert(0, str(adaln))
    from aot_codegen import compile_host
    reference = json.loads((args.reference / "dense-attention-aot-build.json").read_text())
    record = json.loads((args.combined / "dense-attention-aot-build.json").read_text())
    pack = json.loads((args.prepack / "prepack-value-build.json").read_text())
    assert record["experiment"]["prepacked_v"]
    assert record["query_block"] == 128 and record["warps"] == 4
    assert digest(args.prepack / "prepack-value.hsaco") == pack["hsaco_sha256"]
    assert digest(args.reference / "dense-attention-packed.hsaco") == reference["packed_strides"]["hsaco_sha256"]
    shutil.copytree(args.combined, args.output)
    fallback = args.output / "dense-attention-packed-reference.hsaco"
    shutil.copy2(args.reference / "dense-attention-packed.hsaco", fallback)
    shutil.copy2(args.prepack / "prepack-value.hsaco", args.output / "prepack-value.hsaco")
    source = args.output / "prepacked-dispatch.cc"
    prefix = "\n".join(f"#define {name} {name}_reference" for name in
                       ("dispatch", "songshan_bench_attention", "songshan_launch_attention"))
    suffix = "\n".join(f"#undef {name}" for name in
                       ("dispatch", "songshan_bench_attention", "songshan_launch_attention"))
    source.write_text(prefix + "\n" + (adaln / "attention_buffer_host.cc").read_text() +
                      "\n" + suffix + "\n" + (scripts / "prepacked_dispatch_extension.cc").read_text())
    host = args.output / "dense-attention-aot.so"
    command = compile_host(host, (source.resolve(),), defines=(("H3_ATTENTION_QUERY_BLOCK", 128),))
    record["prepacked_value_dispatch"] = dict(
        version=1, minimum_rows=4096, prepack=pack, reference_packed=dict(
            file=fallback.name, hsaco_sha256=digest(fallback), shared=reference["packed_strides"]["shared"]),
        host_sha256=digest(host), host_command=command, host_source_sha256=digest(source),
        workspace="按 device/stream 分配，仅复用分配空间，每次重新计算 V；短长度和其他 stride 回退")
    (args.output / "dense-attention-aot-build.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
