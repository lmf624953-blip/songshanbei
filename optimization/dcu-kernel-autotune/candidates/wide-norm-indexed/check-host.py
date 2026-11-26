#!/usr/bin/env python3
"""对照既有直接发射，验证新主机桥的动态行数与调制表 ABI。"""

import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import runpy
import sys

import torch


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('host', type=Path)
    cli.add_argument('reference', type=Path)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    code = Path(os.environ['H3_AUTOTUNE_CODE_ROOT']) / 'optimization'
    sys.path.insert(0, str(code))
    from adaln.aot_runtime import load_widenorm_indexed
    Fusion = runpy.run_path(str(Path(__file__).with_name('bench.py')))['Fusion']
    direct = Fusion(args.reference)
    host = load_widenorm_indexed(args.host)
    launch = host.songshan_bench_widenorm_indexed
    launch.argtypes = [ct.c_int] * 3 + [ct.c_void_p] * 6 + [ct.c_size_t]
    launch.restype = ct.c_char_p
    generator = torch.Generator(device='cuda').manual_seed(42)
    records = []
    for rows, table_rows in ((1, 3), (33, 6), (4096, 3), (27570, 6)):
        def rand(*shape):
            return torch.randn(shape, dtype=torch.bfloat16, device='cuda', generator=generator)
        hidden, weight = rand(rows, 5376), rand(5376)
        table = rand(table_rows, 6 * 5376)
        shift, scale = table[:, :5376], table[:, 5376:2 * 5376]
        indices = torch.randint(table_rows, (rows,), generator=generator, device='cuda')
        ref = torch.full_like(hidden, float('nan'))
        storage = torch.full((rows + 2, 5376), -17, dtype=hidden.dtype, device='cuda')
        output = storage[:rows]
        output.fill_(float('nan'))
        stream = torch.cuda.current_stream().cuda_stream
        direct.launch(stream, rows=rows, table_rows=table_rows, hidden=hidden, weight=weight,
                      scale=scale, shift=shift, indices=indices, output=ref)
        error = launch(0, rows, table_rows, *(t.data_ptr() for t in
                       (hidden, weight, scale, shift, indices, output)), stream)
        if error:
            raise RuntimeError(error.decode())
        torch.cuda.synchronize()
        record = dict(rows=rows, table_rows=table_rows,
                      exact=torch.equal(ref.view(torch.int16), output.view(torch.int16)),
                      finite=bool(torch.isfinite(output).all()),
                      tail_ok=bool((storage[rows:] == -17).all()))
        records.append(record)
        print(json.dumps(record), flush=True)
    args.output.write_text(json.dumps(dict(cases=records), indent=2) + '\n')
    assert all(all(row[k] for k in ('exact', 'finite', 'tail_ok')) for row in records)


if __name__ == '__main__':
    main()
