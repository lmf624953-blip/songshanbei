#!/usr/bin/env python3
"""独立hipprof进程核对单/双Q128的实际VGPR、LDS和scratch；非正式计时。"""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

import torch


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('reference',type=Path);cli.add_argument('builds',type=Path);args=cli.parse_args()
    code=Path(os.environ['H3_AUTOTUNE_CODE_ROOT'])/'optimization';sys.path.insert(0,str(code))
    baseline=runpy.run_path(str(code/'bench-attention.py'))['native_launcher'](args.reference)
    factory=runpy.run_path(str(Path(__file__).with_name('screen-dualq-no-spill.py')))['launcher']
    n=27570;gen=torch.Generator(device='cuda').manual_seed(42)
    def packed(heads):return torch.randn((1,n,heads,128),device='cuda',dtype=torch.bfloat16,generator=gen)
    q,k=(packed(7).transpose(1,2) for _ in range(2));v=packed(21)[:,:,-7:].transpose(1,2)
    expected=torch.empty((1,n,7,128),device='cuda',dtype=q.dtype).transpose(1,2)
    baseline(q,k,v,128**-0.5,expected);torch.cuda.synchronize()
    print(json.dumps({'variant':'selected','sequence':n,'scope':'独立PMC，不作性能成绩'}),flush=True)
    for mode in ['single-resource','single-lds32','independent','shared']:
        fn=factory(args.builds/mode,args.reference,baseline)
        output=torch.empty((1,n,7,128),device='cuda',dtype=q.dtype).transpose(1,2)
        fn(q,k,v,128**-0.5,output);torch.cuda.synchronize()
        exact=torch.equal(expected.view(torch.int16),output.view(torch.int16));assert exact
        print(json.dumps({'variant':mode,'sequence':n,'bitwise_equal':exact}),flush=True)


if __name__=='__main__':main()
