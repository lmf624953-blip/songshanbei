#!/usr/bin/env python3
"""比较九齿 MRoPE 行分块与原 InfiniCore；仅作独立筛选，不代替真实回归。"""
import argparse
from functools import partial
import json
from pathlib import Path
from statistics import median
import time

import infinicore as ic
import ninetoothed as nt
import torch

from kernel import application, arrange
from songshan_h3.compiler_compat import configure_hygon_compiler


def wrap(tensor):
    return ic.strided_from_blob(tensor.data_ptr(), list(tensor.shape), list(tensor.stride()),
                                dtype=ic.int64 if tensor.dtype == torch.int64 else ic.bfloat16,
                                device=ic.device("cuda", tensor.device.index))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    configure_hygon_compiler()
    ic.set_device(ic.device("cuda", 0))
    generator = torch.Generator(device="cuda").manual_seed(42)
    results = []
    for length in (27570, 22016):
        q, k = (torch.randn((length, 7, 128), device="cuda", dtype=torch.bfloat16, generator=generator)
                for _ in range(2))
        # 原模型 cos/sin 是同一 96 列缓存的两个 48 列视图。
        angles = torch.randn((4096, 48), device="cuda", dtype=torch.float32, generator=generator)
        cache = torch.cat((angles.cos(), angles.sin()), dim=1).to(torch.bfloat16)
        cos, sin = cache[:, :48], cache[:, 48:]
        positions = torch.randint(-1, 4098, (3, length), device="cuda", generator=generator)
        expected = [torch.empty_like(q), torch.empty_like(k)]
        actual = [torch.empty_like(q), torch.empty_like(k)]
        qi, ki = wrap(q.flatten(1)), wrap(k.flatten(1))
        ci, si, pi = map(wrap, (cos, sin, positions))
        qo, ko = (wrap(x.flatten(1)) for x in expected)
        def original():
            ic.mrope(qi, ki, ci, si, pi, 128, 96, 16, 16, 16, False, out=(qo, ko))
        original();torch.cuda.synchronize()
        for rows, warps in ((1, 1), (2, 2), (4, 4), (8, 4), (16, 4)):
            row = dict(length=length, rows=rows, warps=warps)
            try:
                kernel = nt.make(partial(arrange, rows=rows), application,
                    (nt.Tensor(3, dtype=nt.bfloat16), nt.Tensor(3, dtype=nt.bfloat16),
                     nt.Tensor(2, dtype=nt.bfloat16), nt.Tensor(2, dtype=nt.bfloat16),
                     nt.Tensor(2, dtype=nt.int64), nt.Tensor(3, dtype=nt.bfloat16),
                     nt.Tensor(3, dtype=nt.bfloat16)), backend="triton", platform="hygon-bw1000",
                    num_warps=warps, num_stages=1, max_num_configs=1,
                    kernel_name=f"h3_mrope_m{rows}_w{warps}")
                def candidate():
                    kernel(q, k, cos, sin, positions, *actual)
                candidate();torch.cuda.synchronize()
                errors = []
                for a, b in zip(actual, expected):
                    delta = a.float() - b.float()
                    errors.append(dict(exact=torch.equal(a.view(torch.int16), b.view(torch.int16)),
                                       finite=bool(torch.isfinite(a).all()), max_abs=float(delta.abs().max()),
                                       rmse=float(delta.square().mean().sqrt())))
                row['errors'] = errors
                timings = [[], []]
                for _ in range(2):original();candidate()
                torch.cuda.synchronize()
                # InfiniCore 和 Torch 使用不同的原生 stream；独立同步墙钟涵盖二者。
                for i in range(11):
                    for variant in (i % 2, 1 - i % 2):
                        started = time.perf_counter()
                        (original, candidate)[variant]()
                        torch.cuda.synchronize()
                        timings[variant].append(1000 * (time.perf_counter() - started))
                row.update(reference_ms=median(timings[0]), candidate_ms=median(timings[1]),
                           samples_ms=timings,
                           paired_reduction_percent=100 * median(1 - b / a for a, b in zip(*timings)))
            except Exception as error:
                row['error'] = repr(error)
            results.append(row)
            args.output.write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
