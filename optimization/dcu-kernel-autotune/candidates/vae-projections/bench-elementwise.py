#!/usr/bin/env python3
"""验证 RoPE 和 FFN 激活的 BF16 边界，独立测量正收益候选。"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F

from elementwise import build_rope, build_swiglu, silu_div, silu_fast, silu_mul


def torch_rope(q, k, cos, sin):
    out = []
    for x in (q, k):
        rotary, passthrough = x[..., :48], x[..., 48:]
        first, second = rotary.chunk(2, dim=-1)
        swapped = torch.cat([-second, first], dim=-1)
        out.append(torch.cat([rotary * cos[:, None] + swapped * sin[:, None], passthrough], dim=-1))
    return out


def pair(reference, candidate, count):
    for _ in range(3):
        reference()
        candidate()
    torch.cuda.synchronize()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    records = []
    functions = (reference, candidate)
    for trial in range(count):
        order = (0, 1) if trial % 2 == 0 else (1, 0)
        timings = {}
        for side in order:
            events[0].record()
            result = functions[side]()
            events[1].record()
            events[1].synchronize()
            timings[side] = events[0].elapsed_time(events[1])
            del result
        records.append(dict(reference_ms=timings[0], candidate_ms=timings[1]))
    return dict(pairs=records, reference_ms=median(x["reference_ms"] for x in records),
                candidate_ms=median(x["candidate_ms"] for x in records),
                gain_percent=100 * median(1 - x["candidate_ms"] / x["reference_ms"] for x in records))


def capture_assembly(kernel, prefix, record):
    wrapper = kernel._kernel
    original = wrapper.run

    def track(*args, **kwargs):
        compiled = original(*args, **kwargs)
        if compiled is not None and not record:
            asm = compiled.asm.get("amdgcn", "")
            prefix.with_suffix(".amdgcn").write_text(asm)
            record.update(shared=compiled.metadata.shared,
                          hsaco_sha256=hashlib.sha256(compiled.asm["hsaco"]).hexdigest())
        return compiled

    wrapper.run = track


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pairs", type=int, default=31)
    args = parser.parse_args()
    if args.output.exists() or args.pairs < 3:
        parser.error("需要新输出目录与至少三组配对")
    args.output.mkdir()
    generator = torch.Generator(device="cuda:0").manual_seed(411582)
    def random(shape):
        return torch.randn(shape, dtype=torch.bfloat16, device="cuda:0", generator=generator)

    report = dict(complete=False, rope={}, swiglu={})
    def save():
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    with torch.inference_mode():
        for heads in (4, 8, 16):
            kernel = build_rope(heads)
            row = report["rope"][str(heads)] = dict(validation=[], code={})
            capture_assembly(kernel, args.output / f"rope-{heads}", row["code"])
            for tokens, strength in [(1797, 1), (1797, 8), (1029, 1), (1798, 4)]:
                q, k = [random((tokens, 32, 64)) * strength for _ in range(2)]
                angles = random((tokens, 24)).float()
                repeated = torch.cat([angles, angles], dim=-1)
                cos, sin = repeated.cos().bfloat16(), repeated.sin().bfloat16()
                qo, ko = torch.empty_like(q), torch.empty_like(k)
                def base():
                    return torch_rope(q, k, cos, sin)
                def candidate():
                    kernel(q, k, cos, sin, qo, ko)
                    return qo, ko
                expected, result = base(), candidate()
                mismatch = [int(torch.count_nonzero(a.view(torch.int16) != b.view(torch.int16)))
                            for a, b in zip(expected, result, strict=True)]
                row["validation"].append(dict(tokens=tokens, strength=strength, mismatches=mismatch))
                if tokens == 1797 and strength == 1 and not any(mismatch):
                    row["timing"] = pair(base, candidate, args.pairs)
            save()
            print("rope", heads, row["validation"], {k:v for k,v in row.get("timing", {}).items() if k != "pairs"},
                  flush=True)

        patterns = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.bfloat16)
        finite = torch.where(torch.isfinite(patterns), patterns, 0).reshape(8, 8192)
        for function in (silu_div, silu_fast, silu_mul):
            kernel = build_swiglu(function)
            row = report["swiglu"][function.__name__] = dict(validation=[], code={})
            capture_assembly(kernel, args.output / function.__name__, row["code"])
            for tokens, strength in [(8, 0), (1797, 1), (1797, 8), (1029, 1)]:
                packed = random((tokens, 16384))
                up, gate = packed.chunk(2, dim=-1)
                if strength == 0:
                    gate.copy_(finite)
                    up.fill_(1)
                else:
                    gate.mul_(strength)
                out = torch.empty((tokens, 8192), dtype=torch.bfloat16, device="cuda")
                def base():
                    return up * F.silu(gate)
                def candidate():
                    kernel(up, gate, out)
                    return out
                expected, result = base(), candidate()
                mismatch = int(torch.count_nonzero(expected.view(torch.int16) != result.view(torch.int16)))
                row["validation"].append(dict(tokens=tokens, strength=strength, mismatches=mismatch))
                if tokens == 1797 and strength == 1 and mismatch == 0:
                    row["timing"] = pair(base, candidate, args.pairs)
            save()
            print(function.__name__, row["validation"], {k:v for k,v in row.get("timing", {}).items() if k != "pairs"},
                  flush=True)
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
