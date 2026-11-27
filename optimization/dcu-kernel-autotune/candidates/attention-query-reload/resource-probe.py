#!/usr/bin/env python3
"""只查询资源门槛，不发射 kernel；探针使用独立符号且禁止接入模型。"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy

import torch

from songshan_h3.compiler_compat import configure_hygon_compiler
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget


NAME = "h3_resource_descriptor_probe"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("build", "query"))
    cli.add_argument("reference", type=Path)
    cli.add_argument("directory", type=Path)
    args = cli.parse_args()
    if args.mode == "build":
        source = (args.reference / "dense-attention-packed.amdgcn").read_text()
        descriptor = re.search(r"\t\.amdhsa_kernel .*?\t\.end_amdhsa_kernel", source, re.S)[0]
        metadata = re.search(r"\t\.amdgpu_metadata.*?\t\.end_amdgpu_metadata", source, re.S)[0]
        configure_hygon_compiler()
        backend = HIPBackend(GPUTarget("hip", "gfx936", 64))
        options = backend.parse_options(dict(num_warps=4, num_stages=1, kpack=2))
        args.directory.mkdir()
        builds = []
        for vgpr in (160, 176, 184, 188, 192, 196, 198, 219, 221, 228, 244, 256):
            for sgpr in (54, 66):
                desc = descriptor.replace(".amdhsa_next_free_vgpr 219", f".amdhsa_next_free_vgpr {vgpr}")
                desc = desc.replace(".amdhsa_next_free_sgpr 54", f".amdhsa_next_free_sgpr {sgpr}")
                meta = metadata.replace(".vgpr_count:     219", f".vgpr_count:     {vgpr}")
                meta = meta.replace(".sgpr_count:     58", f".sgpr_count:     {sgpr+4}")
                text = '\t.amdgcn_target "amdgcn-amd-amdhsa--gfx936"\n\t.text\n'
                text += f"\t.globl {NAME}\n\t.p2align 8\n\t.type {NAME},@function\n{NAME}:\n\ts_endpgm\n"
                text += f"\t.size {NAME},.-{NAME}\n\t.section .rodata,#alloc\n\t.p2align 6\n"
                text += (desc + "\n" + meta + "\n").replace("dense_attention_aot_kernel", NAME)
                assert "dense_attention_aot_kernel" not in text
                image = backend.make_hsaco(text, {}, options)
                stem = f"resource-v{vgpr}-s{sgpr}"
                (args.directory / (stem + ".amdgcn")).write_text(text)
                (args.directory / (stem + ".hsaco")).write_bytes(image)
                builds.append(dict(vgpr_descriptor=vgpr, sgpr_descriptor=sgpr,
                    image=stem + ".hsaco", sha256=hashlib.sha256(image).hexdigest()))
        (args.directory / "builds.json").write_text(json.dumps(builds, indent=2) + "\n")
        return
    torch.cuda.init()
    occupancy = runpy.run_path(str(Path(__file__).with_name("screen.py")))["occupancy"]
    rows = []
    for build in json.loads((args.directory / "builds.json").read_text()):
        path = args.directory / build["image"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == build["sha256"]
        for shared in (16384, 24576, 32768):
            rows.append(dict(**build, shared_bytes=shared,
                max_active_blocks_per_CU=occupancy(path, NAME.encode(), shared)))
    result = dict(scope="HIP资源上限API查询，未发射kernel；不是实际动态occupancy或性能成绩",
        device=str(torch.cuda.get_device_properties(0)), block_threads=256, records=rows)
    with (args.directory / "resources.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
