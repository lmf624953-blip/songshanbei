"""Accept fresh builds using operator checks, official dog inference and SSIM."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(os.environ['IFAB_AFS_HOME']).resolve()
assert Path('/dev/kfd').exists(), 'Allocated DCU node required'
(root/'logs').mkdir(exist_ok=True)
raw = subprocess.check_output(['bash', '-c',
    'source "$IFAB_AFS_HOME/scripts/unified_dog_env.sh" >/dev/null && env -0'])
env = {k.decode(): v.decode() for x in raw.split(bytes([0])) if b'=' in x
       for k, v in [x.split(b'=', 1)]}
env['H3_EXPORT_DIR'] = str(root/'artifacts/clean-rebuild-exports')
Path(env['H3_EXPORT_DIR']).mkdir(exist_ok=True)
py = root/'.venv/bin/python'
commands = [
    [py, root/'verify_source.py'],
    [py, root/'prepare_runtime_layout.py', '--first-frame='+str(root/'unified-dog-input/first_frame.png')],
    [py, root/'scripts/build_attention_bridge.py'],
    [py, root/'scripts/check_ninetoothed_lowering.py'],
    [py, root/'scripts/check_attention_codegen.py'],
    [py, root/'scripts/check_attention_bridge.py', '--devices=0,1,2,3,4,5,6,7'],
    [py, root/'scripts/run_unified_dog.py'],
]
record = {'complete': False, 'started': time.time(), 'commands': []}
try:
    for cmd in commands:
        print('ACCEPT_RUN', *map(str, cmd), flush=True)
        subprocess.run(list(map(str, cmd)), cwd=root, env=env, check=True)
        record['commands'].append(list(map(str, cmd)))
    output = Path((root/'artifacts/unified-dog-current-path.txt').read_text().strip())
    dog = json.loads((output/'run.json').read_text())
    assert dog['complete'] and all(dog[k+'_returncode'] == 0
                                  for k in ('infinilm', 'reference', 'official-compare'))
    record.update(complete=True, finished=time.time(), output_directory=str(output),
                  dog_run=dog)
    print('CLEAN_REBUILD_ACCEPTANCE_OK', output, flush=True)
finally:
    (root/'artifacts/clean-rebuild-acceptance.json').write_text(json.dumps(record, indent=2))
