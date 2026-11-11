#!/usr/bin/env python3
"""Rebuild in an empty directory using verified raw inputs; never import old binaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--destination', type=Path, required=True)
p.add_argument('--cache-root', type=Path, required=True,
               help='Read-only raw source/wheel cache, with artifacts/ and downloads/')
p.add_argument('--download-inputs', action='store_true', help='Fetch original inputs from checksum-pinned release assets')
p.add_argument('--phase', choices=['prepare', 'build', 'accept', 'compute', 'all'], default='all',
               help='Split build/accept across allocations when the platform has a short time limit')
a = p.parse_args()
if sys.version_info[:2] != (3, 11):
    p.error('Use platform CPython 3.11, for example python3.11 rebuild.py ...')
root = a.destination.resolve()
cache = a.cache_root.resolve()
env = dict(os.environ)
for name in ('PYTHONPATH', 'VIRTUAL_ENV', 'PYTHONUSERBASE'):
    env.pop(name, None)
env.update(PYTHONNOUSERSITE='1', IFAB_AFS_HOME=str(root),
           H3_INPUT_CACHE=str(cache), SONGSHAN_STORAGE=str(root/'h3-runtime'),
           H3_VENDORED_SOURCE_ROOT=str(HERE/'frameworks'))

if a.download_inputs:
    subprocess.run([sys.executable, str(HERE/'rebuild/download_inputs.py'),
                    '--destination', str(cache)], check=True)

def run(cmd, log):
    with log.open('a', buffering=1) as f:
        print(time.strftime('%FT%T'), 'RUN', *map(str, cmd), file=f, flush=True)
        subprocess.run(list(map(str, cmd)), cwd=root, env=env, stdout=f,
                       stderr=subprocess.STDOUT, check=True)

if a.phase in ('prepare', 'all'):
    if root.exists():
        raise SystemExit('Destination must not exist; use a new empty path')
    root.mkdir(parents=True)
    manifest = json.loads((HERE/'RELEASE-MANIFEST.json').read_text())
    for name, item in manifest.items():
        source = HERE/name
        data = source.read_bytes()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise SystemExit('Source verification failed: '+name)
        if name.startswith('frameworks/'):
            continue  # Keep one public source checkout; build in node TMPDIR.
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    shutil.copyfile(HERE/'RELEASE-MANIFEST.json', root/'RELEASE-MANIFEST.json')
    (root/'artifacts').mkdir(exist_ok=True)
    (root/'env/machine.local.sh').write_text(
        'export DTK_ROOT=/opt/dtk\nexport SONGSHAN_GCC_ROOT=/usr\n'
        'export HYGON_ARCH=gfx936\n')
    for helper in ('prepare.py', 'native.py', 'accept.py'):
        shutil.copyfile(HERE/'rebuild'/helper, root/'artifacts'/('rebuild-'+helper))
    run([sys.executable, root/'artifacts/rebuild-prepare.py'], root/'artifacts/rebuild-prepare.log')
    status = json.loads((root/'artifacts/rebuild-prepare-status.json').read_text())
    if not status.get('complete'):
        raise SystemExit('Prepare stage failed; inspect artifacts/rebuild-prepare.log')

if a.phase in ('build', 'accept', 'compute', 'all'):
    if not Path('/dev/kfd').exists():
        raise SystemExit('Compute phase requires an allocated DCU node')
    py = root/'.venv/bin/python'

if a.phase in ('build', 'compute', 'all'):
    run([py, root/'artifacts/rebuild-native.py'], root/'artifacts/rebuild-compute.log')
    status = json.loads((root/'artifacts/clean-native-status.json').read_text())
    if not status.get('complete'):
        raise SystemExit('Fresh native build failed; inspect artifacts/clean-native.log')
if a.phase in ('accept', 'compute', 'all'):
    native_status = json.loads((root/'artifacts/clean-native-status.json').read_text())
    if not native_status.get('complete'):
        raise SystemExit('Finish --phase build successfully before acceptance')
    for name in ('InfiniCore', 'InfiniLM'):
        hashes = json.loads((root/'artifacts'/(name+'-fresh-install.json')).read_text())
        package = root/'.venv/lib/python3.11/site-packages'/name.lower()
        for relative, expected in hashes.items():
            path = package/relative
            with path.open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != expected:
                    raise SystemExit('Fresh native package changed: '+str(path))
    libraries = json.loads((root/'artifacts/fresh-libraries.json').read_text())
    for relative, expected in libraries.items():
        with (root/'.infini/lib'/relative).open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != expected:
                raise SystemExit('Fresh native library changed: '+relative)
    run([py, root/'artifacts/rebuild-accept.py'], root/'artifacts/rebuild-accept.log')
    print('CLEAN_REBUILD_ACCEPTED', root)
elif a.phase == 'build':
    print('BUILD_COMPLETE; run --phase accept on an allocated DCU node')
else:
    print('PREPARE_COMPLETE; run --phase build then --phase accept, or --phase compute')
