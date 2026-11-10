#!/usr/bin/env python3
"""Check released source files against the successful platform run."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'PLATFORM-SOURCE-MANIFEST.json').read_text())
checked=0
for name,entry in manifest.items():
    if name.startswith('evidence/'):continue  # Public evidence paths are normalized.
    path=ROOT/name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
        raise SystemExit('Source mismatch: '+name)
    checked+=1
provenance=json.loads((ROOT/'docs/PLATFORM_PROVENANCE.json').read_text())
for name,expected in provenance['runtime_adaptation_hashes'].items():
    if name.endswith('.so'):continue  # Build the host extension for the target ABI.
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
        raise SystemExit('Runtime source mismatch: '+name)
assert hashlib.sha256((ROOT/'runtime_entry/minimax_h3.py').read_bytes()).hexdigest()=='9bee7f79ad0968cb2af1ade301ef59a4db31d2c06ba9d2ade3435dc2bd98ca46'
assert hashlib.sha256((ROOT/'runtime_entry/minimax_h3_compare.py').read_bytes()).hexdigest()=='e143c28e4b3acb885662e0aa238aa00ca17cbcc82077f465f2ce57d0f1225590'
print('Verified',checked,'exported source/config files and deployed attention/entry hashes')
