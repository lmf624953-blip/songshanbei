#!/usr/bin/env python3
"""在独立 hipprof 进程中发射 AOT V 预排布和注意力。"""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

import torch

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("build", type=Path)
    cli.add_argument("prepack", type=Path)
    args = cli.parse_args()
    code = Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization"
    sys.path.insert(0, str(code))
    functions = runpy.run_path(str(code / "bench-attention.py"))
    module = runpy.run_path(str(Path(__file__).with_name("screen-prepacked-v.py")))
    candidate = module["launcher"](args.build, args.prepack, functions["native_launcher"](args.reference))
    generator = torch.Generator(device="cuda").manual_seed(42)
    for rows in (27570, 22016):
        def packed(heads):
            return torch.randn((1, rows, heads, 128), dtype=torch.bfloat16,
                               device="cuda", generator=generator)
        q, k = (packed(7).transpose(1, 2) for _ in range(2))
        v = packed(21)[:, :, -7:].transpose(1, 2)
        output = torch.empty_like(q)
        samples = functions["timings"](lambda: candidate(q, k, v, 128 ** -0.5, output), 1)
        assert bool(torch.isfinite(output).all())
        print(json.dumps(dict(sequence=rows, samples_ms=samples, finite=True)), flush=True)


if __name__ == "__main__":
    main()
