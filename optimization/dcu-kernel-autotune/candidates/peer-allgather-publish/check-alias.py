#!/usr/bin/env python3
"""独立进程验证八卡桥拒绝输入/残差与输出的部分地址区间重叠。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rccl-sequence-parallel"))
from bridge_micro import InfiniCCLProxy, install_adapter


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("rccl", type=Path)
    cli.add_argument("build", type=Path)
    cli.add_argument("indexed", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--target", choices=("input", "residual"), default="input")
    args = cli.parse_args()
    assert torch.cuda.device_count() == 8
    proxy = InfiniCCLProxy(args.rccl)
    comms = (ct.c_void_p * 8)()
    proxy.ncclCommInitAll(comms, 8, (ct.c_int * 8)(*range(8)))
    peer = ct.CDLL(str((args.build / "peer-gate-bridge.so").resolve()))
    install_adapter(peer, args.build, args.indexed, proxy)
    count = 5376 * 128
    try:
        peer.peer_setup()
        streams = [torch.cuda.Stream(device=r) for r in range(8)]
        inputs = [torch.empty(count + 64, device=r, dtype=torch.bfloat16) for r in range(8)]
        residuals = [torch.empty_like(x) for x in inputs]
        tables = [torch.empty(3 * 5376, device=r, dtype=torch.bfloat16) for r in range(8)]
        indices = [torch.empty(128, device=r, dtype=torch.int64) for r in range(8)]
        targets = inputs if args.target == "input" else residuals

        def call(rank):
            error = peer.h3_peer_bridge_raw(
                rank, inputs[rank].data_ptr(), targets[rank].data_ptr() + 32,
                residuals[rank].data_ptr(), tables[rank].data_ptr(), indices[rank].data_ptr(),
                count, 3, comms[rank], streams[rank].cuda_stream)
            return error.decode() if error else None

        with ThreadPoolExecutor(max_workers=8) as pool:
            errors = list(pool.map(call, range(8)))
        stats = json.loads(peer.h3_peer_bridge_stats())
        assert all(error and "requires separate outputs" in error for error in errors), errors
        assert stats["batches"] == stats["fused"] == stats["fallback"] == 0, stats
        assert stats["calibrations"] == [], stats
        args.output.write_text(json.dumps(dict(build=str(args.build), target=args.target,
                                              overlap_offset_bytes=32, errors=errors, stats=stats,
                                              passed=True), ensure_ascii=False, indent=2) + "\n")
    finally:
        peer.peer_cleanup()
        for rank, comm in enumerate(comms):
            with torch.cuda.device(rank):
                proxy.ncclCommDestroy(comm)


if __name__ == "__main__":
    main()
