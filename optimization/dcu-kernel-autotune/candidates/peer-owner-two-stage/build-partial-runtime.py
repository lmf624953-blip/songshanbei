#!/usr/bin/env python3
"""编译固定owner两阶段桥；明确区分全卡reduced等待与同stream直接发布。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("nt_build", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--async-publish", action="store_true", help="仅省去全卡reduced等待，保留输入ready及全部发布done")
    cli.add_argument("--reuse-host", type=Path, help="C++及ABI相同时只重编译transport")
    args = cli.parse_args()
    here = Path(__file__).resolve().parent
    args.output.mkdir()
    meta = json.loads((args.nt_build / "peer-gate-build.json").read_text())
    assert meta["partial_output"] and meta["order_model_constants"] == dict(channels=16, chunk=524288, alignment=1024)
    assert sha(args.nt_build / "peer-gate.hsaco") == meta["hsaco_sha256"]
    for path in args.nt_build.iterdir():
        if path.is_file():
            shutil.copy2(path, args.output / path.name)
    command = ["hipcc", "-O3", "-std=c++17", "--offload-arch=gfx936", "-shared", "-fPIC",
               "-DOWNER_WAIT_ALL_REDUCED=" + str(int(not args.async_publish)),
               "-DOWNER_CHANNEL_ROTATION=" + str(meta["channel_rotation"]),
               str(here / "owner_transport.hip"), "-o", str(args.output / "peer-transport.so")]
    subprocess.run(command, check=True)
    sys.path.insert(0, str(Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization/adaln"))
    from aot_codegen import compile_host
    bridge = (here / "owner_bridge.cc").read_text()
    if meta.get("grid_2d", False):
        old = "16*unsigned((owner_capacity+tile-1)/tile),1,1,warps*64"
        assert bridge.count(old) == 1
        bridge = bridge.replace(old, "unsigned((owner_capacity+tile-1)/tile),16,1,warps*64")
    generated = args.output / "owner_bridge_generated.cc"
    generated.write_text(bridge)
    if args.reuse_host:
        prior = json.loads((args.reuse_host / "peer-runtime-build.json").read_text())
        assert prior["source_sha256"]["owner_bridge.cc"] == sha(here / "owner_bridge.cc")
        assert prior.get("grid_2d", False) == meta.get("grid_2d", False)
        shutil.copy2(args.reuse_host / "peer-gate-bridge.so", args.output / "peer-gate-bridge.so")
        host = prior["bridge_command"]
    else:
        host = compile_host(args.output / "peer-gate-bridge.so", [generated])
    tensors = {f"p{i}": i for i in range(8)} | dict(residual=9, gate=10, indices=11, out0=14)
    codes = []
    for arg in meta["arguments"]:
        source = arg["source"]
        if arg["kind"] == "tensor":
            codes.append(tensors[source])
        elif arg["kind"] == "scalar":
            assert source == "rank"
            codes.append(12)
        elif arg["kind"] == "shape":
            codes.append(109 if source == "out0" else 102 if source == "gate" else 103 if source == "indices" else 100)
        elif arg["kind"] == "stride":
            assert source == "out0" and arg["dim"] == 0
            codes.append(109)
        else:
            raise ValueError(arg)
    if args.reuse_host:
        assert codes == prior["abi_codes"]
    record = dict(grid_2d=meta.get("grid_2d", False), generated_bridge_sha256=sha(generated),
                  owner_partition=True, partial_output=True, wait_all_reduced=not args.async_publish,
                  channel_rotation=meta["channel_rotation"], nt=meta, abi_codes=codes,
                  transport_command=command, bridge_command=host,
                  reused_host=str(args.reuse_host) if args.reuse_host else None,
                  sha256={p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
                  source_sha256={p.name: sha(p) for p in (here / "owner_bridge.cc", here / "owner_transport.hip",
                                                         here / "partial_kernel.py", Path(__file__))})
    (args.output / "peer-runtime-build.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
