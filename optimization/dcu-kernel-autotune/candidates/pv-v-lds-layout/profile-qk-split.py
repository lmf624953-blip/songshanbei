#!/usr/bin/env python3
"""独立hipprof进程内记录融合基准和QK拆分，时间仅用于诊断。"""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("build", type=Path)
    parser.add_argument("--batch", type=int, default=1024)
    args = parser.parse_args()
    code = Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization"
    sys.path.insert(0, str(code))
    baseline = runpy.run_path(str(code / "bench-attention.py"))["native_launcher"](args.reference)
    split = runpy.run_path(str(Path(__file__).with_name("screen-qk-split.py")))["launcher"](args.build, args.reference, args.batch, baseline)
    gen = torch.Generator(device="cuda").manual_seed(42)
    rows = 27570
    def packed(heads):
        return torch.randn((1, rows, heads, 128), device="cuda", dtype=torch.bfloat16, generator=gen)
    q,k = (packed(7).transpose(1,2) for _ in range(2))
    v = packed(21)[:,:,-7:].transpose(1,2)
    outputs = [torch.empty((1,rows,7,128),device="cuda",dtype=q.dtype).transpose(1,2) for _ in range(2)]
    for i,fn in enumerate((baseline,split)):
        fn(q,k,v,128**-0.5,outputs[i])
        torch.cuda.synchronize()
    assert torch.equal(outputs[0].view(torch.int16),outputs[1].view(torch.int16))
    print(json.dumps(dict(sequence=rows,batch=args.batch,bitwise_equal=True,
                          scope="独立profiler诊断，不作正式成绩"),ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
