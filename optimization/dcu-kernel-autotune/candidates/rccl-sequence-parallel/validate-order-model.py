#!/usr/bin/env python3
"""按位置验证通道/循环/尾块索引假设；对混合边界逐元素核对，保留失败模型。"""
import argparse
import json
from pathlib import Path
import numpy as np
from ring_order_map import owners,RING


def rn(x):
    b=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32)
    return ((b+0x7fff+((b>>16)&1))>>16).astype(np.uint16)


def fp32(x):return (x.astype(np.uint32)<<16).view(np.float32)


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('output',type=Path);cli.add_argument('probes',type=Path,nargs='+')
    cli.add_argument('--channels',type=int,nargs='+',default=[16]);cli.add_argument('--chunks',type=int,nargs='+',default=[524288]);cli.add_argument('--alignments',type=int,nargs='+',default=[128,256,512,1024,2048]);args=cli.parse_args()
    records=[]
    for probe in args.probes:
        info=json.loads((probe/'inference.json').read_text());z=np.load(probe/'samples.npz');raw=z['raw_patterns'];observed=z['output']
        signatures=[]
        for owner in range(8):
            acc=raw[RING[(owner+1)%8]]
            for j in range(2,9):acc=rn(fp32(acc)+fp32(raw[RING[(owner+j)%8]]))
            signatures.append(acc)
        signatures=np.asarray(signatures);cols=np.arange(raw.shape[1])[None,:];positions=z['group_ids'][:,None]*raw.shape[1]+cols
        for channels in args.channels:
            for chunk in args.chunks:
                for alignment in args.alignments:
                    pred=signatures[owners(positions,info['count'],channels,chunk,alignment),cols]
                    bad=np.argwhere(pred!=observed)
                    row=dict(probe=str(probe),count=info['count'],seed=info.get('seed',82371),channels=channels,chunk=chunk,alignment=alignment,compared_elements=int(pred.size),mismatches=int(len(bad)),first_bad_positions=[int(positions[tuple(p)]) for p in bad[:16]],exact=len(bad)==0)
                    records.append(row);print(json.dumps(row),flush=True)
    args.output.write_text(json.dumps(dict(scope='仅所列探针的采样位置逐元素验证，不是完整数组或任意输入证明',ring=RING,records=records),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
