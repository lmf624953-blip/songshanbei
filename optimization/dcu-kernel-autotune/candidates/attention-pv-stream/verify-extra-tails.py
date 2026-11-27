#!/usr/bin/env python3
"""补验实际packed分派中65/96/127等Q尾行，诊断时间不作性能成绩。"""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference",type=Path)
    cli.add_argument("experiment",type=Path)
    cli.add_argument("output",type=Path)
    args=cli.parse_args()
    code=Path(os.environ["H3_AUTOTUNE_CODE_ROOT"])/"optimization"
    sys.path.insert(0,str(code))
    load=runpy.run_path(str(code/"bench-attention.py"))["native_launcher"]
    screen=runpy.run_path(str(Path(__file__).parents[1]/"pv-v-lds-layout/screen.py"))["screen"]
    baseline=load(args.reference)
    with args.output.open("x") as stream:
        for name in ("stream","staged","staged-compact-v2","k16","k16-compact","k32"):
            candidate=load(args.experiment/name)
            for length in (4161,4192,4223,4224,4225):
                result=screen(baseline,candidate,length,1)
                result.update(variant=name,kind="extra-tail-diagnostic",performance_claim=False)
                stream.write(json.dumps(result,ensure_ascii=False)+"\n");stream.flush()
                print(json.dumps(result,ensure_ascii=False),flush=True)
                if not result["passed"]:
                    raise ValueError(result)


if __name__=="__main__":
    main()
