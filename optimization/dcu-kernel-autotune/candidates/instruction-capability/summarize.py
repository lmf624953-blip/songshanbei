#!/usr/bin/env python3
"""只读汇编能力、原始标签和模型约束，不重新执行任何GPU任务。"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import socket


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('experiment',type=Path);cli.add_argument('output',type=Path)
    args=cli.parse_args();root=args.experiment
    assert (root/'matrix-map-exit-code').read_text().strip()=='0'
    capability=json.loads((root/'assembler/capability.json').read_text())
    assert len(capability['records'])==20
    assert [x['name']for x in capability['records']if x['exit_code']==0]==[
        'mmac-control','lds-m32x16','lds-m32x16-alt']
    capture=json.loads((root/'matrix-map-summary.json').read_text())
    assert len(capture['records'])==13 and capture['input_unchanged']
    compact=copy.deepcopy(capture)
    for row in compact['records']:
        values=row['values']
        assert len(values)==256 and all(len(x)==8 for x in values)
        assert row['guard_ok'] and row['repeat_exact'] and row['all_tags_in_range'] and row['waves_equal']
        assert all(values[:64]==values[i:i+64]for i in [64,128,192])
        row['complete_values_sha256']=hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()
        row['values']=values[:64]
    compact['capture_note']='已实测四个wave相同；版本记录仅保留第一wave64lane。完整256lane原始输出留在实验目录并记录哈希。'
    intrinsic=json.loads((root/'intrinsic-probes/results.json').read_text())
    assert [x['exit_code']for x in intrinsic['intrinsics']]==[0,-6]
    assert all(x['exit_code']!=0 for x in intrinsic['assembler'])
    analysis=json.loads((root/'matrix-map-analysis-reference.json').read_text())
    assert all(x['all_five_captured_patterns_match_model'] for x in analysis['records'])
    legacy=root.parents[1]/'rounds/h3-long-attention/round_029/report.md'
    result=dict(node=socket.gethostname(),experiment=str(root),selected=False,
        scope='目标能力和独立内存布局诊断，不是新注意力/模型性能成绩',
        records=compact['records'],input_unchanged=True,capture_note=compact['capture_note'],
        assembler=capability,intrinsics=intrinsic,matrix_analysis=analysis,
        matrix_probe=dict(variants=13,repeats=2,kernel_launches=26,threads=256,VGPR_descriptor=32,
            dynamic_LDS_bytes=65536,note='64KiB仅用于填满地址标签，正式注意力仍为16KiB'),
        raw_capture_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ['matrix-map.jsonl','matrix-map-summary.json']},
        reused_round029=dict(path=str(legacy),sha256=hashlib.sha256(legacy.read_bytes()).hexdigest(),
            interpretation='后端未生成MFMA布局的构建失败，不是硬件性能失败'),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in Path(__file__).parent.iterdir()if p.is_file()},
        no_model_candidate=True,no_model_generation_repeated=True,
        rejected_prototype=dict(build='mmac-flags-build.log',gpu_executed=False,
            note='最初直接构建lit/lts探针在CPU汇编阶段拒绝；原型源码移至实验目录，未执行任何unsupported位编码'),
        decision='本轮没有可直接接入的注意力候选；原生矩阵读取需要生产者重布局/额外交换，当前模型不直接替换。保留正式最优。',
        limits='矩阵读取模型只以五种已捕获地址模式验证；直接替换见证基于该模型及固定lane布局，不证明所有重布局或地址模式不可能。')
    with args.output.open('x')as destination:
        json.dump(result,destination,ensure_ascii=False,indent=2);destination.write('\n')


if __name__=='__main__':main()
