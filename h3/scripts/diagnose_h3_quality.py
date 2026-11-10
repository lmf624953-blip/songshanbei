#!/usr/bin/env python3
"""Isolate persistent mosaic output using real-weight component comparisons."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch
    if torch.cuda.device_count() < 8:
        raise SystemExit('Run from the ordinary terminal exposing all 8 devices')
    tasks = [
        ('vae_public_decode', ['scripts/diagnose_video_vae.py', '--device', 'cuda:0',
                               '--precision', 'reference-mixed', '--attention', 'ninetoothed',
                               '--decode-path', 'public', '--reference',
                               'artifacts/vae-diagnostics/cpu_fp32_math_public_128x224'], 240),
        ('transformer_components', ['scripts/diagnose_h3_transformer.py'], 2400),
    ]
    report = dict(diagnostic_only=True, inference_passed=False, tasks=[])
    for name, arguments, timeout in tasks:
        log = ROOT / 'logs' / ('h3-quality-' + name + '.log')
        print(f'[{name}] Starting; progress: {log}', flush=True)
        # Keep the child in the caller's process group, so Ctrl-C reaches all
        # diagnostic workers. No scored generation is run by this suite.
        try:
            with log.open('w') as output:
                process = subprocess.run([sys.executable, '-u', *arguments], cwd=ROOT,
                                         env={**os.environ, 'HF_HUB_OFFLINE': '1'},
                                         stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
                status = process.returncode
        except subprocess.TimeoutExpired:
            status = 'timeout'
        report['tasks'].append(dict(name=name, returncode=status, log=str(log)))
        print(f'[{name}] Exit {status}', flush=True)
        print('\n'.join(log.read_text(errors='replace').splitlines()[-25:]), flush=True)
        (ROOT / 'artifacts/h3-quality-diagnostics.json').write_text(json.dumps(report, indent=2) + '\n')
    report['component_checks_passed'] = all(task['returncode'] == 0 for task in report['tasks'])
    (ROOT / 'artifacts/h3-quality-diagnostics.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Diagnostic suite finished; full inference quality remains unvalidated.', flush=True)
    return 0 if report['component_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
