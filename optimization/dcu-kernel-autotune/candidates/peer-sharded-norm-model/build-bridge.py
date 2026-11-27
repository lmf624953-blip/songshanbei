#!/usr/bin/env python3
"""复用已校验行分片NT镜像，构建具有Tensor状态和模型边界的独立主机桥。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('prototype', type=Path)
    cli.add_argument('output', type=Path)
    args = cli.parse_args()
    here = Path(__file__).resolve().parent
    previous = json.loads((args.prototype/'peer-runtime-build.json').read_text())
    assert previous['row_aligned_partition']
    args.output.mkdir()
    for name in ['peer-gate.hsaco', 'peer-gate.amdgcn', 'peer-gate-build.json']:
        shutil.copy2(args.prototype/name, args.output/name)
    assert sha(args.output/'peer-gate.hsaco') == previous['nt']['hsaco_sha256']
    source = here.parent/'peer-sharded-norm/chain_transport.hip'
    transport = ['hipcc','-O3','-std=c++17','--offload-arch=gfx936','-shared','-fPIC',str(source),'-o',str(args.output/'peer-transport.so')]
    subprocess.run(transport, check=True)
    sys.path.insert(0,str(Path(os.environ['H3_AUTOTUNE_CODE_ROOT'])/'optimization/adaln'))
    from aot_codegen import compile_host
    command = compile_host(args.output/'peer-gate-bridge.so',[here/'model_bridge.cc'])
    record = dict(model_bridge=True,qualified=False,nt=previous['nt'],abi_codes=previous['abi_codes'],
        transport_command=transport,bridge_command=command,
        source_sha256={str(p.relative_to(here.parent)):sha(p) for p in [source,here/'model_bridge.cc',Path(__file__)]},
        sha256={p.name:sha(p) for p in args.output.iterdir() if p.is_file()})
    (args.output/'peer-runtime-build.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
