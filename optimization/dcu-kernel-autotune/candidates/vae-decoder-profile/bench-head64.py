#!/usr/bin/env python3
"""维持 VAE 九齿注意力的 K64 在线归约，筛选原生 64 维 tile。"""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
from statistics import median

import ninetoothed
import torch
from songshan_h3.attention_kernel import definition, kernel
from songshan_h3.compiler_compat import configure_hygon_compiler


def arrange(query, key, value, is_causal, scale, output, *, query_block):
    def query_layout(tensor):
        result = tensor.tile((1, 1, query_block, 64)).tile(
            (1, query.shape[-3] // key.shape[-3], 1, 1))
        result.dtype = result.dtype.squeeze((0, 2, 3))
        result.dtype.dtype = result.dtype.dtype.squeeze((0, 1))
        return result

    query_arranged = query_layout(query)

    def kv_layout(tensor):
        result = tensor.tile((1, 1, 64, 64)).tile((1, 1, -1, -1)).expand(
            (-1, -1, query_arranged.shape[-2], -1))
        result.dtype = result.dtype.squeeze((0, 1, 3))
        result.dtype.dtype = result.dtype.dtype.squeeze((0, 1))
        return result

    return query_arranged, kv_layout(key), kv_layout(value), is_causal, scale, query_layout(output)


def build(query_block, warps):
    _, expression, args = definition(head_dim=64, dtype=ninetoothed.bfloat16, causal=False)
    return ninetoothed.make(partial(arrange, query_block=query_block), expression, args,
                           backend="triton", platform="hygon-bw1000",
                           num_warps=warps, num_stages=1, max_num_configs=1)


def track_assembly(handle, record, path):
    generated = handle._kernel
    run = generated.run

    def traced(*args, **kwargs):
        result = run(*args, **kwargs)
        if result is not None and not record:
            assembly = result.asm.get("amdgcn", "")
            if assembly:
                path.write_text(assembly)
            record.update(mmac=assembly.count("v_mmac_f32"), shared=result.metadata.shared,
                          warps=result.metadata.num_warps)
            if "hsaco" in result.asm:
                record["hsaco_sha256"] = hashlib.sha256(result.asm["hsaco"]).hexdigest()
        return result

    generated.run = traced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pairs", type=int, default=31)
    args = parser.parse_args()
    if args.pairs < 3 or args.output.exists():
        parser.error("需要新输出目录和至少三组配对")
    args.output.mkdir()
    configure_hygon_compiler()
    candidates = [("base", kernel())]
    for query_block, warps in ((32, 4), (64, 4), (64, 2), (128, 4)):
        candidates.append((f"q{query_block}-w{warps}-d64", build(query_block, warps)))
    report = dict(scope="VAE tile 的独立配对，K64，32 heads，D64，BF16，原在线归约", geometry=[],
                  kernels={name: {} for name, _ in candidates})
    for name, implementation in candidates:
        track_assembly(implementation, report["kernels"][name], args.output / f"{name}.amdgcn")
    scale = 64 ** -0.5

    def save():
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    for tokens in (1797, 1029, 1796, 1798):
        generator = torch.Generator(device="cuda").manual_seed(20261128 + tokens)
        values = [torch.randn((1, tokens, 32, 64), device="cuda", dtype=torch.bfloat16,
                              generator=generator).transpose(1, 2) for _ in range(3)]
        output = [torch.empty_like(values[0]) for _ in range(2)]

        def launch(implementation, target, inputs=values):
            implementation(*inputs, False, scale, target)

        for name, implementation in candidates[1:]:
            row = dict(tokens=tokens, candidate=name, validation=[])
            for multiplier in (1, 3):
                inputs = [values[0] * multiplier, values[1] * multiplier, values[2]]
                launch(candidates[0][1], output[0], inputs)
                launch(implementation, output[1], inputs)
                counts = int(torch.count_nonzero(output[0].view(torch.int16) != output[1].view(torch.int16)))
                row["validation"].append(dict(multiplier=multiplier, mismatches=counts,
                                               finite=bool(torch.isfinite(output[1]).all())))
            if tokens == 1797 and all(item["mismatches"] == 0 and item["finite"] for item in row["validation"]):
                events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

                def measured(op, target):
                    events[0].record()
                    launch(op, target)
                    events[1].record()
                    events[1].synchronize()
                    return events[0].elapsed_time(events[1])

                pairs = []
                for trial in range(args.pairs):
                    order = (0, 1) if trial % 2 == 0 else (1, 0)
                    ops = (candidates[0][1], implementation)
                    samples = {side: measured(ops[side], output[side]) for side in order}
                    pairs.append(dict(reference_ms=samples[0], candidate_ms=samples[1]))
                row.update(pairs=pairs, reference_median_ms=median(p["reference_ms"] for p in pairs),
                           candidate_median_ms=median(p["candidate_ms"] for p in pairs),
                           paired_gain_percent=100 * median(1 - p["candidate_ms"] / p["reference_ms"] for p in pairs))
            report["geometry"].append(row)
            save()
            print({key: value for key, value in row.items() if key != "pairs"}, flush=True)
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
