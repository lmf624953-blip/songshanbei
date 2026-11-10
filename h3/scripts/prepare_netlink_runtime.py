#!/usr/bin/env python3
"""Give system libnl SONAMEs aliases to the exact libraries bundled with Torch.

The vendor Torch wheel renames its libnl dependencies. Loading a second copy
through RCCL/verbs can interpose their registration symbols and crash in fini.
Only project-local symlinks are created; no vendor or system files are changed.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    spec = importlib.util.find_spec('torch')
    if spec is None or spec.origin is None:
        raise RuntimeError('Activate the inference environment first')
    bundled = Path(spec.origin).parent.parent / 'torch.libs'
    destination = ROOT / '.local/lib/torch-netlink'
    pairs = []
    for stem in ('libnl-3', 'libnl-route-3'):
        matches = list(bundled.glob(stem + '-*.so.200.26.0'))
        if len(matches) != 1:
            raise RuntimeError(f'Expected one vendor {stem} ABI 200.26.0 library: {matches}')
        source = matches[0].resolve(strict=True)
        pairs.append((stem, source))
    destination.mkdir(parents=True, exist_ok=True)
    report = dict(purpose='Share Torch bundled libnl with RCCL/verbs',
                  directory=str(destination), libraries=[])
    for stem, source in pairs:
        aliases = []
        for name in (stem + '.so.200', source.name):
            alias = destination / name
            if alias.is_symlink():
                if alias.resolve() != source:
                    raise RuntimeError(f'Unexpected existing symlink: {alias}')
            elif alias.exists():
                raise RuntimeError(f'Refusing to replace an existing file: {alias}')
            else:
                alias.symlink_to(source)
            aliases.append(str(alias))
        report['libraries'].append(dict(source=str(source), aliases=aliases,
                                       sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    (ROOT / 'artifacts').mkdir(exist_ok=True)
    (ROOT / 'artifacts/netlink-runtime.json').write_text(json.dumps(report, indent=2) + '\n')
    return destination


if __name__ == '__main__':
    print(prepare())
