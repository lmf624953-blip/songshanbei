#!/usr/bin/env python3
"""将已构建组件复制到独立发布目录；资格与默认选择分别处理。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from model_identity import check


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_top(source,target):
    target.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file():shutil.copy2(path,target/path.name)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output',type=Path)
    cli.add_argument('--qualification',type=Path)
    args=cli.parse_args();out=args.output.resolve();assert not out.exists()
    paths={k:Path(v).resolve(strict=True) for k,v in os.environ.items() if k.startswith('H3_') and k.endswith('_BUILD') and v}
    required=['H3_ADALN_BUILD','H3_PEER_GATE_BUILD','H3_PEER_SHARDED_NORM_BUILD','H3_WIDE_NORM_INDEXED_BUILD']
    assert all(k in paths for k in required)
    identity=check(*(paths[k] for k in ['H3_PEER_SHARDED_NORM_BUILD','H3_ADALN_BUILD','H3_PEER_GATE_BUILD','H3_WIDE_NORM_INDEXED_BUILD']))
    qualification=None
    if args.qualification:
        root=args.qualification.resolve(strict=True)
        assert (root/'qualification-exit-code').read_text().strip()=='0'
        qualification=json.loads((root/'benchmark.json').read_text())
        assert qualification['comparison']=='accepted-peer-vs-sharded-norm'
        assert qualification['matched_previously_accepted_media']
        assert qualification['wall']['mean_time_reduction_percent']>=3
        assert min(qualification['wall']['paired_reduction_percent']['samples'])>0
        assert qualification['runtime']['native_sha256']==identity['native_sha256']
    out.mkdir()
    translated={}
    for name,path in paths.items():
        if name=='H3_ADALN_BUILD':
            target=out/'native';copy_top(path,target)
            overlay=Path((path/'native-overlay.txt').read_text().strip())
            shutil.copytree(overlay,target/'overlay',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            (target/'native-overlay.txt').write_text(str(target/'overlay')+'\n')
            meta=json.loads((target/'candidate-build.json').read_text())
            native=next((target/'overlay/infinilm/lib').glob('_infinilm*.so'))
            assert digest(native)==meta['candidate_native']['sha256']
            meta['candidate_native']['path']=str(native)
            meta['host_shim']['path']=str(target/'adaln-host.so')
            if 'applied_source_archive' in meta:
                meta['applied_source_archive']['path']=str(target/'InfiniLM-applied-source.tar.gz')
            if 'sharded_hidden' in meta:
                meta['sharded_hidden']['patch']=str(target/'sharded-hidden.patch')
            (target/'candidate-build.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
        else:
            folder='reference-peer' if name=='H3_PEER_GATE_BUILD' else 'model' if name=='H3_PEER_SHARDED_NORM_BUILD' else name.removeprefix('H3_').removesuffix('_BUILD').lower().replace('_','-')
            target=out/'components'/folder;copy_top(path,target)
        translated[name]=str(target)
    scalar_names=['H3_PY_ATTENTION_FP32_SCALE','H3_ENCODER_INITIAL_CACHE','H3_ENCODER_TP_COUNT',
                  'H3_VAE_DETERMINISTIC_ENCODE','H3_VAE_DEVICES','H3_VAE_BATCH','H3_VAE_CACHE_REPLICAS','H3_ENCODER_PREFILL']
    config={k:os.environ[k] for k in scalar_names if k in os.environ}|translated|{'H3_PEER_GATE_ENABLED':'1'}
    (out/'candidate.env').write_text('# 完整运行组件已复制；稳态指标与冷启动进程时间分别报告。\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in config.items()))
    check(*(Path(translated[k]) for k in ['H3_PEER_SHARDED_NORM_BUILD','H3_ADALN_BUILD','H3_PEER_GATE_BUILD','H3_WIDE_NORM_INDEXED_BUILD']))
    if qualification:
        shutil.copy2(args.qualification/'benchmark.json',out/'qualification.json')
    here=Path(__file__).resolve().parent;code=here.parents[3]
    record=dict(kind='h3-peer-sharded-norm-v1',qualified=qualification is not None,
        comparison='accepted-peer-vs-sharded-norm',identity=identity,components=translated,
        source_components={k:str(v) for k,v in paths.items()},
        source_commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),
        source_sha256={p.name:digest(p) for p in here.iterdir() if p.is_file()},
        expected_media_sha256={'A':'99d975c6e2537274224d5904eb35dc30ba3f350737ebe333b9d43c61882f0440'},
        qualification=str(args.qualification.resolve()) if args.qualification else None,
        note='发布不自动修改selected-best；仍需验证实际加载及完整资格。')
    (out/'release.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(out/'candidate.env')


if __name__=='__main__':main()
