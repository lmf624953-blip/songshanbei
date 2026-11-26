#!/usr/bin/env python3
"""同 stream 对照已验收的分离门控与宽融合，检查两个输出再交替计时。"""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import runpy
from statistics import median
import sys

import torch
sys.path.insert(0,str(Path(os.environ['H3_AUTOTUNE_CODE_ROOT'])/'optimization/adaln'))
from indexed_runtime import load_indexed


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('build',type=Path)
    cli.add_argument('--reference',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    Fusion=runpy.run_path(str(Path(__file__).resolve().parents[1]/'wide-norm-indexed/bench.py'))['Fusion']
    base=Fusion(args.reference)
    host=load_indexed(Path(os.environ['H3_INDEXED_BUILD']))
    gate=host.songshan_bench_indexed
    gate.argtypes=[ct.c_bool]+[ct.c_int]*3+[ct.c_void_p]*6
    gate.restype=ct.c_char_p
    candidates={f.parent.name:Fusion(f.parent) for f in sorted(args.build.glob('*/build.json'))}
    assert candidates
    generator=torch.Generator(device='cuda').manual_seed(42)
    stream=torch.cuda.current_stream().cuda_stream
    records=[]
    for rows,table_rows,magnitude in ((1,3,1),(33,6,8),(22016,6,1),(27570,6,1)):
        def rand(*shape):
            return torch.randn(shape,dtype=torch.bfloat16,device='cuda',generator=generator)
        hidden,update,weight=rand(rows,5376)*magnitude,rand(rows,5376),rand(5376)
        table=rand(table_rows,6*5376)/8
        shift,scale,modgate=table[:,:5376],table[:,5376:2*5376],table[:,2*5376:3*5376]
        indices=torch.randint(table_rows,(rows,),generator=generator,device='cuda')
        storage=[torch.empty((rows+2,5376),dtype=hidden.dtype,device='cuda') for _ in range(4)]
        residuals=[storage[0][:rows],storage[2][:rows]]
        outputs=[storage[1][:rows],storage[3][:rows]]
        def reference():
            error=gate(True,0,rows,table_rows,hidden.data_ptr(),update.data_ptr(),modgate.data_ptr(),
                       indices.data_ptr(),residuals[0].data_ptr(),stream)
            if error:raise RuntimeError(error.decode())
            base.launch(stream,rows=rows,table_rows=table_rows,hidden=residuals[0],weight=weight,
                        scale=scale,shift=shift,indices=indices,output=outputs[0])
        for name,module in candidates.items():
            def candidate():
                module.launch(stream,rows=rows,table_rows=table_rows,hidden=hidden,update=update,
                              weight=weight,gate=modgate,scale=scale,shift=shift,indices=indices,
                              residual=residuals[1],output=outputs[1])
            for t in storage:
                t.fill_(float('nan'));t[rows:].fill_(-19)
            reference();candidate();torch.cuda.synchronize()
            record=dict(rows=rows,table_rows=table_rows,magnitude=magnitude,variant=name,
                        exact_residual=torch.equal(residuals[0].view(torch.int16),residuals[1].view(torch.int16)),
                        exact_normalized=torch.equal(outputs[0].view(torch.int16),outputs[1].view(torch.int16)),
                        mismatches=int((outputs[0].view(torch.int16)!=outputs[1].view(torch.int16)).sum()),
                        finite=all(bool(torch.isfinite(t).all()) for t in (*residuals,*outputs)),
                        tail_ok=all(bool((t[rows:]==-19).all()) for t in storage),
                        max_abs=float((outputs[0].float()-outputs[1].float()).abs().max()))
            if rows>=22016 and record['finite'] and record['tail_ok']:
                samples=[[],[]]
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                for trial in range(13):
                    for index in (trial%2,1-trial%2):
                        begin.record();(reference,candidate)[index]();end.record();end.synchronize()
                        if trial>=2:samples[index].append(begin.elapsed_time(end))
                record.update(samples_ms=samples,medians_ms=[median(x) for x in samples],
                              paired_gain_percent=100*median(1-b/a for a,b in zip(*samples,strict=True)))
            records.append(record);print(json.dumps(record),flush=True)
    args.output.write_text(json.dumps(dict(cases=records),indent=2)+'\n')
    assert all(r['finite'] and r['tail_ok'] and r['exact_residual'] for r in records)


if __name__=='__main__':
    main()
