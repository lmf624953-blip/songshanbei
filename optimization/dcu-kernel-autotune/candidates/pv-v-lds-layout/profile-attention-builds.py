#!/usr/bin/env python3
"""在独立hipprof进程依次发射正式基准和已验证候选，保留输出位模式核验。"""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

import torch


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('reference',type=Path);cli.add_argument('candidates',type=Path,nargs='+');cli.add_argument('--length',type=int,default=27570);args=cli.parse_args()
    code=Path(os.environ['H3_AUTOTUNE_CODE_ROOT'])/'optimization';sys.path.insert(0,str(code))
    load=runpy.run_path(str(code/'bench-attention.py'))['native_launcher']
    n=args.length;gen=torch.Generator(device='cuda').manual_seed(42)
    def packed(heads):return torch.randn((1,n,heads,128),device='cuda',dtype=torch.bfloat16,generator=gen)
    q,k=(packed(7).transpose(1,2) for _ in range(2));v=packed(21)[:,:,-7:].transpose(1,2)
    expected=None
    for path in [args.reference,*args.candidates]:
        output=torch.empty((1,n,7,128),device='cuda',dtype=q.dtype).transpose(1,2)
        load(path)(q,k,v,128**-0.5,output);torch.cuda.synchronize()
        if expected is None:expected=output
        else:assert torch.equal(expected.view(torch.int16),output.view(torch.int16))
        print(json.dumps({'build':str(path),'sequence':n,'bitwise_equal':True,'scope':'独立profiler诊断，不作为正式计时'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
