#!/usr/bin/env python3
"""比较动态长序列，保留逐长度配对样本与普通/放大输入误差。"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import socket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--lengths', nargs='+', type=int, default=[4096,8192,16384,24577,27570,32768,65535])
    parser.add_argument('--iterations', type=int, default=11)
    parser.add_argument('--output', required=True, type=Path)
    args=parser.parse_args()
    if args.output.exists():parser.error('输出路径已存在，拒绝覆盖')
    if args.iterations<3 or any(n<1 for n in args.lengths):parser.error('至少3次配对、长度必须为正')
    benchmark=runpy.run_path(str(Path(__file__).with_name('bench-h3-attention.py')))
    load=benchmark['native_loader']()
    reference,candidate=load(args.reference.resolve()),load(args.candidate.resolve())
    rows=[]
    for length in args.lengths:
        row=benchmark['compare'](reference,candidate,length,args.iterations)
        rows.append(row)
        print(json.dumps(row),flush=True)
    passed=all(row[mode]['finite'] and row[mode]['exact'] for row in rows for mode in ('normal','stress'))
    binaries={label:{name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in ('dense-attention-aot.so','dense-attention-aot.hsaco')}
              for label,path in (('reference',args.reference),('candidate',args.candidate))}
    report=dict(verdict='PASS' if passed else 'FAIL',hostname=socket.gethostname(),reference=str(args.reference.resolve()),candidate=str(args.candidate.resolve()),binaries=binaries,iterations=args.iterations,cases=rows)
    with args.output.open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
