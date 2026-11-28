#!/usr/bin/env python3
"""逐位筛选一维 VAE RoPE，随后交替计时或独立采集 PMC。"""

import argparse
import ctypes
import json
import os
from pathlib import Path
from runpy import run_path
from time import sleep

import torch

from rope_flat import build_rope_flat

reference_tools = run_path(str(Path(__file__).with_name("bench-elementwise.py")))
torch_rope = reference_tools["torch_rope"]
pair = reference_tools["pair"]
capture_assembly = reference_tools["capture_assembly"]


def make_inputs(generator, tokens, strength, pattern):
    def random(shape):
        return torch.randn(shape, dtype=torch.bfloat16, device="cuda:0", generator=generator)

    q, k = [random((tokens, 32, 64)) * strength for _ in range(2)]
    if pattern == "random":
        angles = random((tokens, 24)).float().repeat(1, 2)
        cosine, sine = angles.cos().bfloat16(), angles.sin().bfloat16()
    else:
        cosine = torch.full((tokens, 48), float(pattern == "identity"),
                            dtype=torch.bfloat16, device="cuda")
        sine = torch.full_like(cosine, float(pattern == "quarter_turn"))
    return q, k, cosine, sine


def launch(kernel, q, k, cosine, sine):
    out_q, out_k = torch.empty_like(q), torch.empty_like(k)
    kernel(q.view(-1), k.view(-1), cosine, sine, out_q.view(-1), out_k.view(-1))
    return out_q, out_k


def load_aot(root, tile, code):
    harness = (Path(os.environ["H3_AUTOTUNE_CODE_ROOT"])
               / "optimization/dcu-kernel-autotune/candidates/wide-norm-indexed/bench.py")
    image = run_path(str(harness))["Fusion"](root / str(tile), symbol=b"apply_rope_flat_kernel")
    code.update(image.meta)

    def enqueue(q, k, cosine, sine, output_q, output_k):
        elements = q.numel()
        image.launch(torch.cuda.current_stream().cuda_stream,
                     rows=(elements + tile - 1) // tile, elements=elements, tokens=cosine.shape[0],
                     q=q, k=k, cosine=cosine, sine=sine, output_q=output_q, output_k=output_k)

    return enqueue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tiles", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--pairs", type=int, default=31)
    parser.add_argument("--pmc", action="store_true")
    parser.add_argument("--aot", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.pairs < 3:
        parser.error("需要新输出目录与至少三组配对")
    if args.pmc and len(args.tiles) != 1:
        parser.error("PMC 仅采集单个指定候选")
    args.output.mkdir()
    report = {}
    with torch.inference_mode():
        for tile in args.tiles:
            generator = torch.Generator(device="cuda:0").manual_seed(411582)
            row = report[str(tile)] = dict(validation=[], code={})
            if args.aot:
                kernel = load_aot(args.aot, tile, row["code"])
            else:
                kernel = build_rope_flat(tile)
                capture_assembly(kernel, args.output / f"rope-flat-{tile}", row["code"])
            cases = [(1797, 1, "random"), (1797, 8, "random"), (1029, 1, "random"),
                     (1796, 4, "random"), (1798, 4, "random"),
                     (17, 1, "identity"), (17, 1, "quarter_turn")]
            inputs = make_inputs(generator, *cases[0])
            if args.pmc:
                expected = torch_rope(*inputs)
                result = launch(kernel, *inputs)
                torch.cuda.synchronize()
                if not all(torch.equal(a.view(torch.int16), b.view(torch.int16))
                           for a, b in zip(expected, result, strict=True)):
                    raise RuntimeError("采集前的 RoPE 逐位校验失败")
                hip = ctypes.CDLL("libamdhip64.so")
                assert hip.hipProfilerStart() == 0
                result = launch(kernel, *inputs)
                torch.cuda.synchronize()
                sleep(2)
                assert hip.hipProfilerStop() == 0
                cases = cases[:1]
            for case_index, (tokens, strength, pattern) in enumerate(cases):
                values = inputs if case_index == 0 else make_inputs(generator, tokens, strength, pattern)
                expected = torch_rope(*values)
                result = launch(kernel, *values)
                mismatch = [int(torch.count_nonzero(a.view(torch.int16) != b.view(torch.int16)))
                            for a, b in zip(expected, result, strict=True)]
                row["validation"].append(dict(tokens=tokens, strength=strength,
                                              pattern=pattern, mismatches=mismatch))
            exact = all(not any(case["mismatches"]) for case in row["validation"])
            if exact and not args.pmc:
                row["timing"] = pair(lambda: torch_rope(*inputs), lambda: launch(kernel, *inputs), args.pairs)
            (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print("rope_flat", tile, row["validation"],
                  {k: v for k, v in row.get("timing", {}).items() if k != "pairs"}, flush=True)


if __name__ == "__main__":
    main()
