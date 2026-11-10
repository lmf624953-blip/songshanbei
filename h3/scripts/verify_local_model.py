#!/usr/bin/env python3
"""Verify copied FL2VA files offline and bind the manifest to their new location."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(directory, plan, size_only=False):
    verified = []
    if not plan['files']:
        raise ValueError('Empty model hash inventory')
    for index, record in enumerate(plan['files'], 1):
        name = Path(record['path'])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError(f'Unsafe model filename: {name}')
        path = directory / name
        if path.stat().st_size != record['bytes']:
            raise ValueError(f'Model size mismatch: {name}')
        if not size_only:
            kind = record['hash_type']
            if kind == 'sha256':
                digest = hashlib.sha256()
            elif kind == 'git_blob_sha1':
                digest = hashlib.sha1()
                digest.update(f"blob {record['bytes']}\0".encode())
            else:
                raise ValueError(f'Unsupported hash type: {kind}')
            with path.open('rb') as handle:
                for data in iter(lambda: handle.read(8 * 2**20), b''):
                    digest.update(data)
            if digest.hexdigest() != record['expected_hash']:
                raise ValueError(f'Model content hash mismatch: {name}')
        verified.append(dict(path=str(name), reused=True, hash_verified=not size_only))
        print(f'[{index}/{len(plan["files"])}] {"size" if size_only else "hash"} verified: {name}', flush=True)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_path', type=Path, help='Directory containing model_index.json')
    parser.add_argument('--hashes', type=Path, default=ROOT / 'env/model-content-hashes.json')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'artifacts/model-download.json')
    parser.add_argument('--size-only', action='store_true', help='Check sizes only; never write a hash-verified manifest')
    args = parser.parse_args()
    directory = args.model_path.resolve(strict=True)
    plan = json.loads(args.hashes.read_text())
    index = json.loads((directory / 'model_index.json').read_text())
    if index.get('_class_name') != 'MiniMaxH3ModularPipeline':
        raise SystemExit('Expected the verified Diffusers FL2VA layout')
    verified = verify(directory, plan, args.size_only)
    if args.size_only:
        print('Sizes match. Content hashes were NOT checked; no manifest was written.')
        return 0
    manifest = dict(repo=plan['hf_repo'], revision=plan['hf_revision'], source='offline-content-verification',
                    selected_files=[r['path'] for r in plan['files']],
                    bytes=sum(r['bytes'] for r in plan['files']), model_path=str(directory),
                    all_content_hashes_verified=True, verified_files=verified, inference_passed=False)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_name(args.manifest.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(args.manifest)
    print(f'All content hashes verified. Current model manifest: {args.manifest}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
