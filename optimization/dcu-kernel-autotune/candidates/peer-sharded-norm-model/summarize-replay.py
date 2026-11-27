#!/usr/bin/env python3
"""核对实际peer对照、早中晚逐位哈希、计算调用数和模型分片生命周期。"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('experiment',type=Path);cli.add_argument('accepted_replay',type=Path);cli.add_argument('output',type=Path)
    args=cli.parse_args();root=args.experiment
    assert (root/'replay8-exit-code').read_text().strip()=='0'
    assert (root/'replay8/complete').exists()
    rows=read(root/'replay8/results.json');assert len(rows)==6
    old=read(args.accepted_replay/'results.json')
    manifest=read(root/'replay8/source/manifest.json')
    package=read(root/'native/candidate-build.json')
    assert manifest['runtime']['native_sha256']==package['candidate_native']['sha256']
    assert manifest['builds']['H3_PEER_SHARDED_NORM_BUILD']['path']==str(root/'bridge-v2')
    assert manifest['builds']['H3_PEER_GATE_BUILD']['path'].endswith('validated-build/peer-gate-nt-f13r1n16/peer')
    comparisons=[]
    for step in [1,25,49]:
        ref=next(r for r in rows if r['step']==step and r['variant']=='candidate-no-peer-gate')
        cand=next(r for r in rows if r['step']==step and r['variant']=='candidate')
        previous=next(r for r in old if r['step']==step and r['variant']=='candidate')
        assert ref['capture_sha256']==cand['capture_sha256']==previous['capture_sha256']
        assert not ref['diagnostic_profiler'] and not cand['diagnostic_profiler']
        assert len(ref['samples_seconds'])==len(cand['samples_seconds'])==5
        assert ref['warmup_calls']==cand['warmup_calls']==1
        # 不仅使用torch.equal：原始输出字节SHA也必须与同实例及已验收peer一致。
        hashes=lambda row:{x['name']:x['output_sha256'] for x in row['against_capture']}
        assert hashes(ref)==hashes(cand)==hashes(previous),(step,hashes(ref),hashes(cand),hashes(previous))
        assert all(x['exact'] and x['finite'] and x['mismatches']==0 for x in cand['against_candidate_without_peer_gate'])
        keys=['adaln','indexed','qknorm','qknorm_mrope','attention','short_attention','widenorm_indexed','swiglu','peer_gate','peer_deferred','peer_batches','peer_fused','peer_fallback']
        assert all(ref['kernel_calls'][k]==cand['kernel_calls'][k] for k in keys),(step,ref['kernel_calls'],cand['kernel_calls'])
        assert ref['kernel_calls']['peer_fused']==cand['kernel_calls']['peer_fused']==600
        assert cand['kernel_calls']['sharded_gate']==600 and ref['kernel_calls']['sharded_gate']==0
        assert cand['kernel_calls']['sharded_norm']==594 and cand['kernel_calls']['sharded_materialized']==6
        assert cand['kernel_calls']['sharded_replica_fallbacks']==0 and cand['kernel_calls']['peer_fallback']==0
        state=cand['peer_gate_stats'];assert state['comparison']=='accepted-peer-vs-sharded-norm'
        assert state['candidate']['pending_ranks']==0 and not state['candidate']['failed']
        a,b=statistics.median(ref['samples_seconds']),statistics.median(cand['samples_seconds'])
        comparisons.append(dict(step=step,reference_seconds=a,candidate_seconds=b,reduction_percent=100*(1-b/a),
                                output_hashes=hashes(cand),full_compute_counts_equal=True,materialization_passed=True))
    a=sum(r['reference_seconds'] for r in comparisons);b=sum(r['candidate_seconds'] for r in comparisons)
    result=dict(scope='真实八卡Transformer早中晚回放；参考为当前peer，非完整生成成绩',
        qualified_end_to_end=False,model_bitwise_passed=True,comparisons=comparisons,
        summed_median_reduction_percent=100*(1-b/a),native_sha256=package['candidate_native']['sha256'],
        expected_full_A_projection_percent=(a-b)/3*49/120.69804317139399*100,
        projection_note='仅三个真实step差额外推49次；必须通过完整官方及稳态A配对才可认定端到端收益',
        inputs={'results':str(root/'replay8/results.json'),'accepted_replay':str(args.accepted_replay)},
        source_manifest_sha256=hashlib.sha256((root/'replay8/source/manifest.json').read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
