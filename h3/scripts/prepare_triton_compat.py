#!/usr/bin/env python3
"""Check/apply the recorded LLVM 17 compatibility patch in a project-owned venv."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--check', action='store_true')
    action.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    spec = importlib.util.find_spec('triton')
    if spec is None or spec.origin is None:
        raise SystemExit('Install the pinned vendor Triton wheel first')
    package = Path(spec.origin).resolve().parent
    compiler = package / 'backends/amd/compiler.py'
    source = compiler.read_text()
    present = 'DPA4_TRITON_LLVM17_COMPAT' in source and 'original_src = src' in source
    if not present and args.apply:
        if not compiler.is_relative_to((ROOT / '.venv').resolve()):
            raise SystemExit('Refusing to modify an inherited/base environment. Install Triton into this project .venv first.')
        if shutil.which('patch') is None:
            raise SystemExit('The system patch command is required')
        patch = ROOT / 'patches/triton/0001-llvm17-gep-compat.patch'
        command = ['patch', '--batch', '--forward', '-p1', '-d', str(package), '-i', str(patch)]
        subprocess.run([*command, '--dry-run'], check=True)
        backup = compiler.with_name(compiler.name + '.before-songshan-llvm17')
        if not backup.exists():
            backup.write_text(source)
        subprocess.run(command, check=True)
        source = compiler.read_text()
        present = 'DPA4_TRITON_LLVM17_COMPAT' in source and 'original_src = src' in source
    report = dict(compiler_path=str(compiler), compatibility_patch_present=present,
                  compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),
                  required_environment='DPA4_TRITON_LLVM17_COMPAT=1',
                  device_execution_checked=False)
    (ROOT / 'artifacts').mkdir(exist_ok=True)
    (ROOT / 'artifacts/triton-compat-check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if present else 1


if __name__ == '__main__':
    raise SystemExit(main())
