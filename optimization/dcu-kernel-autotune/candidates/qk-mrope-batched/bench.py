#!/usr/bin/env python3
"""对照已验收的 Q/K 融合，检查归约顺序、动态步长、末段和输入幅度。"""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import runpy
from statistics import median
import sys

import infinicore as ic
import torch

sys.path.insert(0, str(Path(os.environ['H3_AUTOTUNE_CODE_ROOT']) / 'optimization/adaln'))
from aot_runtime import load_qknorm_mrope


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('reference', type=Path)
    cli.add_argument('builds', type=Path)
    cli.add_argument('output', type=Path)
    a = cli.parse_args()
    assert not a.output.exists()
    for path in a.builds.glob('*/build.json'):
        assert json.loads(path.read_text())['row_tile'] == 1, '多行原型归约域未通过，禁止直接发射'
    Fusion = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'wide-norm-indexed/bench.py'))['Fusion']
    torch.cuda.set_device(0)
    ic.set_device(ic.device('cuda', 0))
    stream = ic.get_stream()
    if type(stream).__name__ == 'PyCapsule':
        get = ct.pythonapi.PyCapsule_GetPointer
        get.argtypes, get.restype = [ct.py_object, ct.c_char_p], ct.c_void_p
        stream = get(stream, None)
    timing_stream = torch.cuda.ExternalStream(stream, device=0)
    host = load_qknorm_mrope(a.reference)
    native = host.songshan_bench_qknorm_mrope
    native.argtypes = [ct.c_int] * 6 + [ct.c_void_p] * 10
    native.restype = ct.c_char_p
    modules = {f.parent.name: Fusion(f.parent) for f in sorted(a.builds.glob('*/build.json'))}
    assert modules
    gen = torch.Generator(device='cuda').manual_seed(42)
    records = []
    for rows, packed, broadcast_positions, magnitude in (
            (1, False, False, 1), (33, True, False, 8), (1027, True, True, 64),
            (22016, True, True, 1), (27570, True, True, 1), (27570, False, False, 1)):
        def rand(*shape):
            return torch.randn(shape, device='cuda', dtype=torch.bfloat16, generator=gen)
        if packed:
            qkv = rand(rows, 21, 128) * magnitude
            query, key = qkv[:, :7], qkv[:, 7:14]
        else:
            query, key = rand(rows, 7, 128) * magnitude, rand(rows, 7, 128) * magnitude
        weights = (rand(128), rand(128))
        angles = torch.randn((4096, 48), device='cuda', generator=gen)
        cache = torch.cat((angles.cos(), angles.sin()), dim=1).to(torch.bfloat16)
        cos, sin = cache[:, :48], cache[:, 48:]
        positions = torch.randint(-1, 4098, (1 if broadcast_positions else 3, rows), device='cuda', generator=gen)
        if broadcast_positions:
            positions = positions.expand(3, -1)
        reference = [torch.empty((rows, 7, 128), device='cuda', dtype=torch.bfloat16) for _ in range(2)]
        storage = [torch.full((rows + 2, 7, 128), -17, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
        output = [x[:rows] for x in storage]
        values = dict(input_rows=rows, query_stride=query.stride(0), key_stride=key.stride(0),
            cache_rows=4096, positions_stride=positions.stride(0), query=query, key=key,
            query_weight=weights[0], key_weight=weights[1], cos=cos, sin=sin, positions=positions,
            query_out=output[0], key_out=output[1])

        def baseline():
            tensors = (query, key, *weights, cos, sin, positions, *reference)
            error = native(0, rows, query.stride(0), key.stride(0), 4096, positions.stride(0),
                           *(x.data_ptr() for x in tensors), stream)
            if error:
                raise RuntimeError(error.decode())

        torch.cuda.synchronize()
        baseline()
        torch.cuda.synchronize()
        for name, module in modules.items():
            def candidate():
                module.launch(stream, rows=((rows + module.meta['row_tile'] - 1) // module.meta['row_tile']) * 7,
                              **values)
            for x in output:
                x.fill_(float('nan'))
            torch.cuda.synchronize()
            candidate()
            torch.cuda.synchronize()
            errors = [dict(exact=torch.equal(x.view(torch.int16), y.view(torch.int16)),
                mismatches=int((x.view(torch.int16) != y.view(torch.int16)).sum()),
                finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
                tail_ok=bool((z[rows:] == -17).all()), max_abs=float((x.float()-y.float()).abs().max()))
                for x, y, z in zip(output, reference, storage, strict=True)]
            item = dict(rows=rows, packed=packed, broadcast_positions=broadcast_positions,
                        magnitude=magnitude, variant=name, errors=errors)
            if rows >= 22016 and all(e[k] for e in errors for k in ('exact', 'finite', 'tail_ok')):
                samples = [[], []]
                for trial in range(17):
                    for side in (trial % 2, 1 - trial % 2):
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record(timing_stream)
                        (baseline, candidate)[side]()
                        end.record(timing_stream)
                        end.synchronize()
                        if trial >= 4:
                            samples[side].append(begin.elapsed_time(end))
                item.update(device_ms=[median(x) for x in samples], samples_ms=samples,
                    paired_gain_percent=100 * median(1-b/a for a,b in zip(*samples, strict=True)))
            records.append(item)
            print(json.dumps(item), flush=True)
            a.output.write_text(json.dumps(dict(scope='独立融合算子，非完整模型验收', cases=records),
                                           ensure_ascii=False, indent=2) + '\n')
    assert all(e[k] for r in records for e in r['errors'] for k in ('finite', 'tail_ok'))


if __name__ == '__main__':
    main()
