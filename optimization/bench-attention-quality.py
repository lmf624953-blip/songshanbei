#!/usr/bin/env python3
"""Compare attention arithmetic with independent PyTorch and FP64 references."""
import argparse
import json
from pathlib import Path
from statistics import median

import ninetoothed as nt
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from songshan_h3.attention_kernel import definition, kernel
from songshan_h3.compiler_compat import configure_hygon_compiler
from adaln.accurate_attention import application


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fp16-pv", action="store_true")
    args = parser.parse_args()
    configure_hygon_compiler()
    arrangement, _, params = definition()
    candidate = nt.make(arrangement, application, params, backend="triton",
                        platform="hygon-bw1000", num_warps=4, num_stages=1,
                        max_num_configs=1, kernel_name="h3_fp32_scale_quality")
    variants = [("original", kernel()), ("fp32_scale", candidate)]
    if args.fp16_pv:
        from adaln.precise_attention import application as precise_application
        variants.append(("fp16_pv", nt.make(arrangement, precise_application, params, backend="triton",
                                            platform="hygon-bw1000", num_warps=4, num_stages=1,
                                            max_num_configs=1, kernel_name="h3_fp16_pv_quality")))
    generator = torch.Generator(device="cuda").manual_seed(42)
    records = []
    for length in (512, 4096):
        q, k, v = [torch.randn((1, 2, length, 128), device="cuda", dtype=torch.bfloat16,
                              generator=generator) for _ in range(3)]
        scale = 128 ** -0.5
        with sdpa_kernel(SDPBackend.MATH):
            reference = F.scaled_dot_product_attention(q, k, v)
        # Small-case FP64 serves as an independent mathematical reference.
        high = None
        if length == 512:
            high = (torch.softmax((q.double() @ k.double().transpose(-1, -2)) * scale, -1)
                    @ v.double()).float()
        for name, launch in variants:
            out = torch.empty_like(q)
            launch(q, k, v, False, scale, out)
            torch.cuda.synchronize()
            samples = []
            for _ in range(9):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                launch(q, k, v, False, scale, out)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            delta = out.float() - reference.float()
            record = {"length": length, "variant": name, "median_ms": median(samples),
                      "torch_reference_backend": "MATH",
                      "finite": bool(torch.isfinite(out).all()),
                      "vs_torch_rmse": float(delta.square().mean().sqrt()),
                      "vs_torch_max_abs": float(delta.abs().max()),
                      "vs_torch_exact_fraction": float((out == reference).float().mean())}
            if high is not None:
                record["vs_fp64_rmse"] = float((out.float() - high).square().mean().sqrt())
            print(record, flush=True)
            records.append(record)
    with args.output.open("x") as stream:
        json.dump(records, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
