#!/usr/bin/env python3
"""Run isolated attention, VAE and shutdown checks; no full H3 generation."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch
    if torch.cuda.device_count() < 8:
        raise SystemExit('Run this diagnostic from the ordinary terminal exposing all 8 devices')
    baseline = ROOT / 'artifacts/vae-diagnostics/cpu_fp32_math'
    math_result = ROOT / 'artifacts/vae-diagnostics/cuda-0_reference-mixed_math'
    tasks = [
        ('long_attention', ['scripts/check_attention_long_sequence.py']),
        ('vae_math', ['scripts/diagnose_video_vae.py', '--device', 'cuda:0',
                      '--precision', 'reference-mixed', '--attention', 'math',
                      '--reference', str(baseline)]),
        ('vae_ninetoothed', ['scripts/diagnose_video_vae.py', '--device', 'cuda:0',
                             '--precision', 'reference-mixed', '--attention', 'ninetoothed',
                             '--reference', str(math_result)]),
        ('shutdown', ['scripts/diagnose_runtime_shutdown.py', '--gdb-on-failure']),
    ]
    report = dict(diagnostic_only=True, inference_passed=False, tasks={})
    for name, arguments in tasks:
        print(f'[{name}] Starting isolated diagnostic', flush=True)
        path = ROOT / 'logs' / ('h3-diagnostic-' + name + '.log')
        with path.open('w') as log:
            try:
                process = subprocess.run([sys.executable, '-u', *arguments], cwd=ROOT,
                                         env={**os.environ, 'PYTHONFAULTHANDLER': '1'},
                                         stdout=log, stderr=subprocess.STDOUT, timeout=240)
                status = process.returncode
            except subprocess.TimeoutExpired:
                status = 'timeout'
        report['tasks'][name] = dict(returncode=status, log=str(path))
        (ROOT / 'artifacts/h3-runtime-diagnostics.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'[{name}] Exit {status}; {path}', flush=True)
        if status:
            print('\n'.join(path.read_text(errors='replace').splitlines()[-24:]), flush=True)
    return 0 if all(r['returncode'] == 0 for r in report['tasks'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
