#!/usr/bin/env python3
"""Prepare the official public FL2VA input, reading its request as data only."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from PIL import Image
import requests

from probe_modelscope import REPO, error_summary, file_url, list_files

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / 'samples/minimax_h3_public'
REQUEST_PATH = 'scripts/readme/reproducible-768p-fl2va-request.sh'


def bounded_get(session, url, limit):
    with session.get(url, stream=True, timeout=(10, 30)) as response:
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_content(chunk_size=65536):
            data.extend(chunk)
            if len(data) > limit:
                raise ValueError('Public sample exceeds the configured size limit')
        return bytes(data)


def parse_request(data):
    text = data.decode('utf-8')
    match = re.search(r"<<'JSON'\r?\n(.*?)\r?\nJSON(?:\r?\n|$)", text, re.DOTALL)
    if not match:
        raise ValueError('Official script has no recognized JSON data block')
    payload = json.loads(match[1])
    if payload.get('task') != 'fl2va' or not isinstance(payload.get('prompt'), str):
        raise ValueError('Public request is not an FL2VA prompt')
    images = [item for item in payload.get('conditions', [])
              if item.get('type') == 'image' and item.get('role') == 'keyframe'
              and item.get('frame_index') == 0]
    if len(images) != 1:
        raise ValueError('Expected exactly one public first-frame image')
    url = images[0]['uri']
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or parsed.hostname != 'cdn.hailuoai.com':
        raise ValueError('Unexpected public image origin; inspect the upstream request')
    return payload, url


def validate_local():
    case = json.loads((SAMPLE / 'case.json').read_text())
    for key in ('first_frame', 'prompt_file', 'upstream_request'):
        # Prefer the hash-verified fixture beside case.json after a project
        # relocation. Keep explicit external fixtures working as a fallback.
        path = SAMPLE / Path(case[key]['path']).name
        if not path.is_file():
            path = Path(case[key]['path'])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != case[key]['sha256']:
            raise ValueError(f'Public sample file is missing or changed: {key}')
        case[key]['path'] = str(path.resolve())
    manifest_path = ROOT / 'artifacts/model-download.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest['revision'] == case['model_revision']:
            case['model_path'] = manifest['model_path']
    with Image.open(case['first_frame']['path']) as image:
        image.verify()
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate existing sample without network requests')
    args = parser.parse_args()
    try:
        if (SAMPLE / 'case.json').exists() or args.check:
            validate_local()
            print(f'Public sample files verified: {SAMPLE / "case.json"}')
            return 0
        manifest = json.loads((ROOT / 'artifacts/model-download.json').read_text())
        if not manifest.get('all_content_hashes_verified'):
            raise ValueError('Expected a completed, hash-verified model download')
        with requests.Session() as session:
            files = list_files(session, 'master')
            if REQUEST_PATH not in files:
                raise ValueError('Official public request script is missing from the ModelScope mirror')
            raw = bounded_get(session, file_url(REQUEST_PATH, 'master'), 2**20)
            expected = files[REQUEST_PATH]
            if len(raw) != expected['bytes'] or expected['sha256'] and hashlib.sha256(raw).hexdigest() != expected['sha256']:
                raise ValueError('Official request failed its metadata size/hash check')
            payload, image_url = parse_request(raw)
            image_data = bounded_get(session, image_url, 20 * 2**20)
        with Image.open(BytesIO(image_data)) as image:
            image_format, dimensions = image.format, image.size
            image.verify()
        extensions = {'PNG': '.png', 'JPEG': '.jpg', 'WEBP': '.webp'}
        if image_format not in extensions:
            raise ValueError('Unsupported public image format')
        SAMPLE.mkdir(parents=True, exist_ok=True)
        request_path = SAMPLE / 'official_request.sh'
        image_path = SAMPLE / ('first_frame' + extensions[image_format])
        prompt_path = SAMPLE / 'prompt.txt'
        prompt_data = payload['prompt'].encode('utf-8')
        for path, data in ((request_path, raw), (image_path, image_data), (prompt_path, prompt_data)):
            if path.exists() and path.read_bytes() != data:
                raise ValueError(f'Existing sample file differs; retained: {path}')
            path.write_bytes(data)
        def entry(path):
            return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        case = dict(task='fl2va', kind='public_smoke_validation',
                    upstream_request=entry(request_path), first_frame=entry(image_path),
                    prompt_file=entry(prompt_path), image_url=image_url,
                    image_size=list(dimensions), original_target=payload.get('target'),
                    original_seed=payload.get('seed'), model_path=manifest['model_path'],
                    model_revision=manifest['revision'],
                    generation=dict(gpus=list(range(8)), num_frames=124, steps=50, seed=42, fps=24),
                    official_competition_sample=False, inference_passed=False)
        (SAMPLE / 'case.json').write_text(json.dumps(case, ensure_ascii=False, indent=2) + '\n')
        validate_local()
        print(f'Public sample prepared: {SAMPLE / "case.json"}')
        print(f'Image: {dimensions[0]}x{dimensions[1]}; prompt characters: {len(payload["prompt"])}')
        print('Generation configuration: 124 frames, 50 steps, seed 42, 24 fps, 8 visible devices.')
        return 0
    except requests.RequestException as exc:
        print(f'Public sample network request failed: {json.dumps(error_summary(exc))}')
    except (OSError, ValueError, KeyError) as exc:
        print(f'Public sample preparation failed: {exc}')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
