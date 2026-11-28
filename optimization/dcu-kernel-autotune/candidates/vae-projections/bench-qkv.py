#!/usr/bin/env python3
"""比较 VAE 的三次 Q/K/V 投影与拼权重后的单次投影。"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
from safetensors import safe_open


def load_weights(vae, layers):
    weights = json.loads((vae / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    with ExitStack() as stack:
        shards = {}
        for layer in layers:
            pairs = []
            for name in ("q", "k", "v"):
                parts = []
                for suffix in ("weight", "bias"):
                    key = f"decoder.transformer_blocks.{layer}.attn.to_{name}.{suffix}"
                    path = vae / weights[key]
                    if path not in shards:
                        shards[path] = stack.enter_context(safe_open(path, framework="pt", device="cpu"))
                    parts.append(shards[path].get_tensor(key).to(device="cuda:0", dtype=torch.bfloat16))
                pairs.append(tuple(parts))
            yield layer, pairs


def measure(operation, first, last):
    first.record()
    output = operation()
    last.record()
    last.synchronize()
    return first.elapsed_time(last), output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pairs", type=int, default=31)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 17, 35])
    parser.add_argument("--tokens", type=int, nargs="+", default=[1797, 1029, 1796, 1798])
    args = parser.parse_args()
    if args.output.exists() or args.pairs < 3:
        parser.error("需要新输出路径和至少三组配对")
    torch.set_num_threads(4)
    generator = torch.Generator(device="cuda:0").manual_seed(20261128)
    report = dict(complete=False, scope="真实 BF16 VAE QKV 权重和固定种子输入，独立投影", results=[])
    first, last = [torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    with torch.inference_mode():
        for layer, separate in load_weights(args.model / "vae", args.layers):
            width, input_width = separate[0][0].shape
            assert all(weight.shape == (width, input_width) and bias.shape == (width,)
                       for weight, bias in separate)
            packed_weight, packed_bias = (torch.cat(values, dim=0) for values in zip(*separate, strict=True))
            for tokens in args.tokens:
                features = torch.randn((1, tokens, input_width), device="cuda:0",
                                       dtype=torch.bfloat16, generator=generator)

                def reference():
                    return tuple(F.linear(features, weight, bias) for weight, bias in separate)

                def packed_views():
                    return F.linear(features, packed_weight, packed_bias).split(width, dim=-1)

                def packed_copies():
                    return tuple(part.contiguous() for part in packed_views())

                for mode, candidate in [("views", packed_views), ("copies", packed_copies)]:
                    checks = []
                    for multiplier in (1, 3):
                        saved = features
                        features = saved * multiplier
                        baseline, result = reference(), candidate()
                        checks.append(dict(multiplier=multiplier,
                            mismatches=[int(torch.count_nonzero(a.view(torch.int16) != b.view(torch.int16)))
                                        for a, b in zip(baseline, result, strict=True)],
                            candidate_finite=all(bool(torch.isfinite(part).all()) for part in result),
                            output_strides=[list(part.stride()) for part in result]))
                        features = saved
                    row = dict(layer=layer, tokens=tokens, width=width, input_width=input_width,
                               mode=mode, validation=checks)
                    if tokens == 1797 and all(not any(c["mismatches"]) and c["candidate_finite"] for c in checks):
                        for _ in range(3):
                            reference()
                            candidate()
                        pairs = []
                        for trial in range(args.pairs):
                            operations = (reference, candidate)
                            order = (0, 1) if trial % 2 == 0 else (1, 0)
                            values = {}
                            for index in order:
                                values[index], _ = measure(operations[index], first, last)
                            pairs.append(dict(reference_ms=values[0], packed_ms=values[1]))
                        row.update(pairs=pairs, reference_median_ms=median(p["reference_ms"] for p in pairs),
                                   candidate_median_ms=median(p["packed_ms"] for p in pairs),
                                   paired_gain_percent=100 * median(1 - p["packed_ms"] / p["reference_ms"] for p in pairs))
                    report["results"].append(row)
                    save()
                    print({key: value for key, value in row.items() if key != "pairs"}, flush=True)
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
