#!/usr/bin/env python3
"""Place the exact tested entry files and competition input in the expected layout."""
import argparse
import hashlib
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--first-frame',type=Path,required=True)
a=p.parse_args()
expected='b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54'
if hashlib.sha256(a.first_frame.read_bytes()).hexdigest()!=expected:p.error('Use the unchanged dog PNG from the official report template')
target=ROOT/'unified-dog-input/first_frame.png';target.parent.mkdir(exist_ok=True)
if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=expected:p.error('Existing input differs; no file replaced')
if not target.exists():shutil.copyfile(a.first_frame,target)
for name,expected_sha in [('minimax_h3.py','9bee7f79ad0968cb2af1ade301ef59a4db31d2c06ba9d2ade3435dc2bd98ca46'),('minimax_h3_compare.py','e143c28e4b3acb885662e0aa238aa00ca17cbcc82077f465f2ce57d0f1225590')]:
 source=ROOT/'runtime_entry'/name
 if hashlib.sha256(source.read_bytes()).hexdigest()!=expected_sha:raise SystemExit('Entry hash mismatch: '+name)
 target=ROOT/'h3-runtime/sources/InfiniLM/examples'/name;target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists() and target.read_bytes()!=source.read_bytes():raise SystemExit('Refusing to replace a different entry: '+str(target))
 if not target.exists():shutil.copyfile(source,target)
(ROOT/'artifacts').mkdir(exist_ok=True)
print('TESTED_ENTRY_AND_TEMPLATE_INPUT_READY')
