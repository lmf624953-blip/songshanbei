#!/usr/bin/env python3
"""Apply the unchanged recorded insertion without AFS chmod/chown operations."""
import ast
import importlib.util
import json
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parent
spec=importlib.util.find_spec('triton')
if spec is None or spec.origin is None:raise SystemExit('Install the pinned vendor Triton into project .venv first')
compiler=Path(spec.origin).resolve().parent/'backends/amd/compiler.py'
if not compiler.is_relative_to((ROOT/'.venv').resolve()):raise SystemExit('Refusing to patch a public or inherited Triton')
source=compiler.read_text()
if 'DPA4_TRITON_LLVM17_COMPAT' not in source:
    patch=(ROOT/'patches/triton/0001-llvm17-gep-compat.patch').read_text()
    insertion=''.join(line[1:] for line in patch.splitlines(keepends=True) if line.startswith('+') and not line.startswith('+++'))
    anchor='        amdgcn = llvm.translate_to_asm('
    if source.count(anchor)!=1:raise SystemExit('Unexpected source: no file changed')
    updated=source.replace(anchor,insertion+anchor,1);ast.parse(updated)
    backup=compiler.with_name(compiler.name+'.before-songshan-llvm17')
    if not backup.exists():backup.write_text(source)
    compiler.write_text(updated)
print(json.dumps(dict(compiler=str(compiler),sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),compatibility_patch_present=True)))
