#!/usr/bin/env python3
"""Reproduce the libnl exit crash without GPUs and check the project aliases."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_netlink_runtime import prepare

ROOT = Path(__file__).resolve().parents[1]

WORKER = '''
import ctypes, json, os, sys, torch
handles = []
for library in json.loads(sys.argv[1]):
    handles.append(ctypes.CDLL(library, mode=os.RTLD_NOW | os.RTLD_GLOBAL))
    print('Loaded ' + library, flush=True)
mapped = sorted({line.split()[-1] for line in open('/proc/self/maps')
                 if '/libnl-' in line})
print('Mapped libnl: ' + json.dumps(mapped), flush=True)
if sys.argv[2] == 'single':
    assert len(mapped) == 2, mapped
print('Normal Python completion; testing actual process exit', flush=True)
'''


def main():
    compatibility = prepare()
    bundled = Path(importlib.util.find_spec('torch').origin).parent.parent / 'torch.libs'
    route, = bundled.glob('libnl-route-3-*.so.200.26.0')
    paths = [p for p in os.getenv('LD_LIBRARY_PATH', '').split(':')
             if p and Path(p).resolve() != compatibility.resolve()]
    base = {**os.environ, 'LD_LIBRARY_PATH': ':'.join(paths)}
    fixed = {**base, 'LD_LIBRARY_PATH': ':'.join([str(compatibility), *paths])}
    cases = [
        ('bundled_only', [str(route)], base, False),
        ('system_only', ['libnl-route-3.so.200'], base, False),
        ('duplicate_reproduction', [str(route), 'libnl-route-3.so.200'], base, False),
        ('fixed_bundled_first', [str(route), 'libnl-route-3.so.200', 'libibverbs.so.1'], fixed, True),
        ('fixed_system_first', ['libnl-route-3.so.200', str(route), 'libibverbs.so.1'], fixed, True),
    ]
    report = dict(diagnostic_only=True, gpu_execution_checked=False,
                  inference_passed=False, cases=[])
    for name, libraries, environment, single in cases:
        path = ROOT / 'logs' / ('netlink-' + name + '.log')
        try:
            process = subprocess.run([sys.executable, '-u', '-c', WORKER,
                                      json.dumps(libraries), 'single' if single else 'any'],
                                     env=environment, capture_output=True, text=True, timeout=45)
            path.write_text(process.stdout + process.stderr)
            status = process.returncode
        except subprocess.TimeoutExpired as error:
            path.write_bytes((error.stdout or b'') + (error.stderr or b''))
            status = 'timeout'
        report['cases'].append(dict(name=name, returncode=status, log=str(path)))
        print(f'{name}: exit {status}', flush=True)
    by_name = {case['name']: case['returncode'] for case in report['cases']}
    report['original_crash_reproduced'] = by_name['duplicate_reproduction'] == -11
    report['fixed_cases_passed'] = all(by_name[name] == 0 for name in
                                     ('fixed_bundled_first', 'fixed_system_first'))
    report['passed'] = (report['fixed_cases_passed'] and
                        by_name['bundled_only'] == by_name['system_only'] == 0)
    (ROOT / 'artifacts/netlink-runtime-check.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
