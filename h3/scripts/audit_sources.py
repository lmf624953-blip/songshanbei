#!/usr/bin/env python3
"""Record available source revisions without changing downloaded checkouts."""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
storage = Path(os.getenv('SONGSHAN_STORAGE',
                        '/work1/share/sugonhpcapp01/zhanghengyu/songshanbei'))


def git(path, *args):
    result = subprocess.run(['git', '-C', str(path), *args], capture_output=True,
                            text=True, timeout=60,
                            env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


sources = {}
for name in ('InfiniCore', 'InfiniLM', 'ninetoothed', 'ntops', 'diffusers', 'xmake'):
    path = storage / 'sources' / name
    entry = dict(present=path.is_dir(), path=str(path))
    if path.is_dir():
        try:
            entry.update(commit=git(path, 'rev-parse', 'HEAD'),
                         branch=git(path, 'branch', '--show-current'),
                         submodules=git(path, 'submodule', 'status', '--recursive').splitlines())
            # Git LFS status filters may create temporary files even for a
            # nominally read-only command. Keep revisions if status cannot run.
            if os.access(path, os.W_OK):
                entry['changes'] = git(path, 'status', '--short').splitlines()
            else:
                entry['changes'] = None
                entry['changes_note'] = 'Not collected: read-only checkout; LFS filters may write'
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            entry['error'] = str(exc)
    sources[name] = entry

packages = {}
for name in ('torch', 'triton', 'numpy', 'diffusers', 'accelerate', 'transformers',
             'huggingface-hub', 'safetensors', 'ninetoothed', 'ntops', 'av',
             'imageio', 'imageio-ffmpeg', 'pybind11', 'hatchling'):
    try:
        packages[name] = md.version(name)
    except md.PackageNotFoundError:
        packages[name] = None

hashes = {}
for filename in ('minimax_h3.py', 'minimax_h3_compare.py'):
    path = storage / 'sources/InfiniLM/examples' / filename
    if path.is_file():
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

report = dict(timestamp=time.strftime('%Y-%m-%d %H:%M:%S %z'), sources=sources,
              installed_packages=packages, reference_script_sha256=hashes,
              model_directory_present=(storage / 'models/MiniMax-H3').is_dir(),
              inference_passed=False)
output = ROOT / 'artifacts/source-audit-current.json'
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(sources={k: v.get('commit', 'missing') for k, v in sources.items()},
                      installed_packages=packages, report=str(output)), indent=2))
