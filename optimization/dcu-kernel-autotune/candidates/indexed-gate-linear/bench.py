#!/usr/bin/env python3
"""在原库 stream 配对微测门控，检查动态行数、表长及尾部越界。"""
import argparse
import ctypes as ct
import json
import os
import runpy
from pathlib import Path
from statistics import median

import infinicore as ic
import torch


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('build', type=Path)
    cli.add_argument('--output', type=Path, required=True)
    a = cli.parse_args()
    assert not a.output.exists()
    helper = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'wide-norm-indexed/bench.py'))
    Fusion, load_indexed = helper['Fusion'], helper['load_indexed']
    torch.cuda.set_device(0)
    ic.set_device(ic.device('cuda', 0))
    stream = ic.get_stream()
    if type(stream).__name__ == 'PyCapsule':
        get = ct.pythonapi.PyCapsule_GetPointer
        get.argtypes, get.restype = [ct.py_object, ct.c_char_p], ct.c_void_p
        stream = get(stream, None)
    events_stream = torch.cuda.ExternalStream(stream, device=0)
    host = load_indexed(Path(os.environ['H3_INDEXED_BUILD']))
    indexed = host.songshan_bench_indexed
    indexed.argtypes = [ct.c_bool] + [ct.c_int] * 3 + [ct.c_void_p] * 6
    indexed.restype = ct.c_char_p
    modules = {f.parent.name: Fusion(f.parent) for f in sorted(a.build.glob('*/build.json'))}
    assert modules
    gen = torch.Generator(device='cuda').manual_seed(42)
    records = []
    for rows, table_rows, magnitude in ((1, 3, 1), (33, 6, 8), (1027, 3, 64),
                                         (22016, 6, 1), (27570, 6, 1)):
        residual = torch.randn((rows, 5376), device='cuda', dtype=torch.bfloat16, generator=gen) * magnitude
        activation = torch.randn((rows, 5376), device='cuda', dtype=torch.bfloat16, generator=gen) * magnitude
        table = torch.randn((table_rows, 6 * 5376), device='cuda', dtype=torch.bfloat16, generator=gen) / 8
        gate = table[:, 2 * 5376:3 * 5376]
        indices = torch.randint(table_rows, (rows,), device='cuda', generator=gen)
        ref = torch.empty_like(residual)
        storage = torch.full((rows * 5376 + 512,), -17, dtype=torch.bfloat16, device='cuda')
        output = storage[:rows * 5376]

        def baseline():
            error = indexed(True, 0, rows, table_rows, residual.data_ptr(), activation.data_ptr(),
                            gate.data_ptr(), indices.data_ptr(), ref.data_ptr(), stream)
            if error:
                raise RuntimeError(error.decode())

        torch.cuda.synchronize()
        baseline()
        torch.cuda.synchronize()
        for name, module in modules.items():
            def candidate():
                module.launch(stream, rows=(output.numel() + module.meta['tile'] - 1) // module.meta['tile'],
                              input_rows=rows, table_rows=table_rows, elements=output.numel(),
                              residual=residual, activation=activation, gate=gate, indices=indices, output=output)

            output.fill_(float('nan'))
            torch.cuda.synchronize()
            candidate()
            torch.cuda.synchronize()
            diff = output.float() - ref.flatten().float()
            record = dict(rows=rows, table_rows=table_rows, magnitude=magnitude, variant=name,
                          exact=torch.equal(output.view(torch.int16), ref.flatten().view(torch.int16)),
                          mismatches=int((output.view(torch.int16) != ref.flatten().view(torch.int16)).sum()),
                          finite=bool(torch.isfinite(output).all() and torch.isfinite(ref).all()),
                          tail_ok=bool((storage[output.numel():] == -17).all()),
                          max_abs=float(diff.abs().max()), rmse=float(diff.square().mean().sqrt()))
            if rows >= 22016 and all(record[k] for k in ('exact', 'finite', 'tail_ok')):
                samples = [[], []]
                for trial in range(17):
                    for side in (trial % 2, 1 - trial % 2):
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record(events_stream)
                        (baseline, candidate)[side]()
                        end.record(events_stream)
                        end.synchronize()
                        if trial >= 4:
                            samples[side].append(begin.elapsed_time(end))
                record.update(device_ms=[median(x) for x in samples], samples_ms=samples,
                              paired_reduction_percent=100 * median(1 - b / a for a, b in zip(*samples)))
            records.append(record)
            print(json.dumps(record), flush=True)
            a.output.write_text(json.dumps(dict(scope='独立门控微测，不作为完整模型验收', cases=records), ensure_ascii=False, indent=2) + '\n')
    assert all(r[k] for r in records for k in ('exact', 'finite', 'tail_ok'))


if __name__ == '__main__':
    main()
