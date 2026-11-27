#!/usr/bin/env python3
"""验证真实模型单次ROI或原始PMC的采集范围；不生成正式性能成绩。"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import runpy

from model_identity import check,digest,read


HERE=Path(__file__).resolve().parent
CODE=HERE.parents[3]


def trace(root):
    replay=root/'replay';record=read(replay/'results.json')
    # replay输出是列表，保留原始数值字段，不拿旧capture输出当参考。
    assert isinstance(record,list) and len(record)==1
    row=record[0]
    assert row['step']==25 and row['variant']=='candidate' and row['diagnostic_profiler']
    assert row['warmup_calls']==1 and len(row['samples_seconds'])==1
    manifest=read(replay/'source/manifest.json');builds=manifest['builds']
    identity=check(*(builds[k]['path'] for k in ['H3_PEER_SHARDED_NORM_BUILD','H3_ADALN_BUILD','H3_PEER_GATE_BUILD','H3_WIDE_NORM_INDEXED_BUILD']))
    assert manifest['runtime']['native_sha256']==identity['native_sha256']
    c=row['kernel_calls'];expected=dict(peer_gate=1600,peer_deferred=1600,peer_batches=200,peer_fused=200,peer_fallback=0,indexed=0,
        sharded_gate=200,sharded_norm=198,sharded_materialized=2,sharded_full_norm=16,sharded_replica_fallbacks=0,
        adaln=16,qknorm=64,qknorm_mrope=800,attention=800,short_attention=32,widenorm_indexed=1600,swiglu=832)
    assert all(c.get(k)==v for k,v in expected.items()),c
    if 'peer_gate_stats' in row:
        assert row['peer_gate_stats']['comparison']=='accepted-peer-vs-sharded-norm'
        assert row['peer_gate_stats']['candidate']['pending_ranks']==0 and not row['peer_gate_stats']['candidate']['failed']
        state_evidence='direct-runtime-stats'
    else:
        # 旧candidate-only格式没有stats字段。固定桥的materialize_pending在清空
        # 全部pending之后才递增materialized；固定native每次forward恰在最后调用它。
        # 两次完整forward正常退出、各一次物化及全部计算计数共同证明清理。
        # 不向原始记录补写未采集的状态值，也不为日志字段重跑模型。
        assert (replay/'complete').read_text().strip()=='complete'
        state_evidence='legacy-format:fixed-bridge-final-materialization-counters-and-normal-exit'
    expected_sha={'video':'80ba4e37aa043fb988ee3e581205cbdbd5bd8d6be09e5090ba01b389ea64e779',
                  'audio':'b8a5b45150b569216dadef7b46bf46a8aedf68af70ee84dde82093eb30587c04'}
    assert len(row['against_capture'])==2
    assert all(r['finite'] and r['output_sha256']==expected_sha[r['name']] for r in row['against_capture'])
    phases=runpy.run_path(str(CODE/'optimization/summarize-profile.py'))['phase_windows'](root/'run.log')
    assert len(phases)==1
    start,end,name,_=phases[0]
    payload=read(root/'hipprof_stream.json');events=payload['traceEvents'] if isinstance(payload,dict) else payload
    analyze=runpy.run_path(str(CODE/'optimization/analyze-hipprof-timeline.py'))['summarize_window']
    timeline,rows=analyze(events,start,end,name,list(range(8)))
    for dev in range(8):
        assert sum(r['calls'] for r in rows if r['device']==dev and r['name']=='h3_peer_gate_kernel')==100
        assert sum(r['calls'] for r in rows if r['device']==dev and 'publish_rows' in r['name'])==100
        assert sum(r['calls'] for r in rows if r['device']==dev and r['name']=='wide_norm_indexed_kernel')==100
    groups=defaultdict(float)
    for r in rows:
        if r['device']!=0:continue
        n=r['name']
        group=('长短注意力合计' if 'dense_attention' in n else 'GEMM' if n.startswith('Cijk_') or 'gemm' in n.lower()
               else '行分片求和与门控' if n=='h3_peer_gate_kernel' else '行分片发布' if 'publish_rows' in n
               else '首层完整及后续局部归一化' if n=='wide_norm_indexed_kernel' else 'SwiGLU' if 'swiglu' in n.lower()
               else 'QK归一化与MRoPE' if 'qk_norm' in n else '其他设备操作')
        groups[group]+=r['total_ms']
    with (root/'kernel-times.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return dict(scope='真实八卡单次Transformer独立trace；排除加载、预热和2秒排空，非正式成绩',
        identity=identity,state_evidence=state_evidence,trace_sha256=digest(root/'hipprof_stream.json'),output_hashes=expected_sha,
        device0_groups_ms=dict(sorted(groups.items(),key=lambda kv:-kv[1])),timeline=timeline,record=row)


def pmc(root,kind):
    text=(root/'pmc.txt').read_text();blocks=re.split(r'(?=kernel-name:)',text);records=[]
    kernel='h3_peer_gate_kernel' if kind=='pmc-gate' else 'wide_norm_indexed_kernel'
    for block in blocks:
        if not block.strip():continue
        lines=block.splitlines();assert lines[0]==f'kernel-name:"{kernel}"',lines[0]
        fields={}
        for line in lines[1:]:
            if not line.strip():continue
            match=re.fullmatch(r'\s*(.*?)\s{2,}(\S+)\s*',line);assert match,line
            fields[match[1]]=match[2]
        records.append(fields)
    assert len(records)==8 and sorted(int(r['gpu id']) for r in records)==list(range(8))
    for r in records:
        dev=int(r['gpu id']);rows=27570*(dev+1)//8-27570*dev//8
        grid=((rows*5376+1023)//1024*256) if kind=='pmc-gate' else rows*256
        assert int(r['grid size'])==grid,(dev,r['grid size'],grid)
        assert r['work group size']=='256' and r['scratch size']=='0'
        assert r['vgpr count']==('60' if kind=='pmc-gate' else '72')
        assert r['shared memory size']==('0' if kind=='pmc-gate' else '512')
    checks=read(root/'result.json');assert checks['final_materialization_exact'] and len(checks['records'])==9
    assert all(all(v['normalized_exact'] and v['hidden_exact'] and v['unowned_untouched'] and v['tails'] for v in r['validated']) for r in checks['records'])
    return dict(scope='相同模型桥镜像的独立八卡Tensor检查PMC；仅采集已预热候选的一个gate/norm，不是完整模型PMC',
        kernel=kernel,pmc_txt_sha256=digest(root/'pmc.txt'),records=records)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['trace','pmc-gate','pmc-norm']);parser.add_argument('output',type=Path)
    args=parser.parse_args();root=args.output.resolve(strict=True)
    assert (root/'exit-code').read_text().strip()=='0'
    result=trace(root) if args.mode=='trace' else pmc(root,args.mode)
    (root/'validation-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ['scope','kernel','device0_groups_ms']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
