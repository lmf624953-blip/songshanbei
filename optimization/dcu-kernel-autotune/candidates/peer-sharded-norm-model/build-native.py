#!/usr/bin/env python3
"""重建独立native并保留完整源码、确定的对象链接顺序和来源。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('reference', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--xmake', required=True)
    cli.add_argument('--jobs', type=int, default=8)
    args = cli.parse_args()
    args.output.mkdir()
    source = args.output/'source'
    source.mkdir()
    with tarfile.open(args.reference/'InfiniLM-applied-source.tar.gz') as archive:
        for member in archive.getmembers():
            assert (source/member.name).resolve().is_relative_to(source.resolve())
            assert member.isfile() or member.isdir()
        archive.extractall(source)
    here = Path(__file__).resolve().parent
    tree = source/'InfiniLM'
    patch = args.output/'sharded-hidden.patch'
    subprocess.run([sys.executable, str(here/'prepare-native.py'), str(tree), str(patch)], check=True)
    with tarfile.open(args.output/'InfiniLM-applied-source.tar.gz', 'w:gz') as target:
        target.add(tree, arcname='InfiniLM')
    import pybind11
    env = os.environ.copy()
    env.update(SONGSHAN_PYBIND11_INCLUDE=pybind11.get_include(),
               SONGSHAN_PYTHON_INCLUDE=sysconfig.get_paths()['include'], INFINILM_CXX11_ABI='1')
    commands = [[args.xmake, 'f', '-y', '--ccache=n', '-m', 'release', '--cxx11-abi=1',
                 '--cc='+env.get('CC','gcc'), '--cxx='+env.get('CXX','g++')],
                [args.xmake, 'build', '-y', '-j'+str(args.jobs), '_infinilm']]
    start = time.monotonic()
    with (args.output/'build.log').open('w') as log:
        for command in commands:
            subprocess.run(command, cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        libraries = list((tree/'build/linux/x86_64/release').glob('_infinilm*.so'))
        assert len(libraries) == 1
        plan = json.loads((here.parent/'rccl-sequence-parallel/native-link-order.json').read_text())
        commands.append([env.get('CXX','g++'), '-o', str(libraries[0]),
                         *(row['path'] for row in plan['objects']), '-shared', '-m64', '-fPIC',
                         '-L'+str(Path(env['INFINI_ROOT'])/'lib'),
                         '-linfinicore_cpp_api','-linfiniop','-linfinirt','-linfiniccl'])
        subprocess.run(commands[-1], cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    base = Path((args.reference/'native-overlay.txt').read_text().strip())
    overlay = args.output/'overlay'
    shutil.copytree(base, overlay, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    native = overlay/'infinilm/lib'/libraries[0].name
    shutil.copy2(libraries[0], native)
    for name in ['adaln-host.so','source-commit']:
        shutil.copy2(args.reference/name, args.output/name)
    (args.output/'native-overlay.txt').write_text(str(overlay)+'\n')
    record = json.loads((args.reference/'candidate-build.json').read_text())
    record['reference_native'] = record['candidate_native']
    record['candidate_native'] = dict(path=str(native),sha256=digest(native))
    record['host_shim'] = dict(path=str(args.output/'adaln-host.so'),sha256=digest(args.output/'adaln-host.so'))
    record['reference_peer_native_source'] = record['peer_native_source']
    record['peer_native_source'] = dict(path=str(tree/'csrc/models/minimax_h3/minimax_h3_transformer.cpp'),
        sha256=digest(tree/'csrc/models/minimax_h3/minimax_h3_transformer.cpp'))
    record['applied_source_archive'] = dict(path=str(args.output/'InfiniLM-applied-source.tar.gz'),
        sha256=digest(args.output/'InfiniLM-applied-source.tar.gz'))
    record['sharded_hidden'] = dict(qualified=False, patch=str(patch), patch_sha256=digest(patch),
        source=str(tree/'csrc/models/minimax_h3/minimax_h3_transformer.cpp'),
        source_sha256=digest(tree/'csrc/models/minimax_h3/minimax_h3_transformer.cpp'),
        source_archive_sha256=digest(args.output/'InfiniLM-applied-source.tar.gz'),
        builders={p.name:digest(p) for p in [Path(__file__),here/'prepare-native.py']},
        commands=commands, elapsed_seconds=time.monotonic()-start,
        objects=[dict(path=row['path'],sha256=digest(tree/row['path'])) for row in plan['objects']])
    (args.output/'candidate-build.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(native, digest(native))


if __name__ == '__main__':
    main()
