#!/usr/bin/env python3
"""Measure bounded HTTP reads from the official ModelScope MiniMax-H3 repo."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlencode, urlsplit

import requests

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = 'https://modelscope.cn'
REPO = 'MiniMax/MiniMax-H3'


def error_summary(exc):
    # Requests exceptions can embed proxy passwords or signed redirect URLs.
    result = {'type': type(exc).__name__}
    response = getattr(exc, 'response', None)
    if response is not None:
        result['http_status'] = response.status_code
    if isinstance(exc, requests.exceptions.ProxyError):
        result['category'] = 'proxy_connection_failed'
    elif isinstance(exc, requests.exceptions.Timeout):
        result['category'] = 'network_timeout'
    elif isinstance(exc, requests.exceptions.ConnectionError):
        result['category'] = 'network_connection_failed'
    elif isinstance(exc, ValueError):
        result['category'] = 'unexpected_server_metadata'
    else:
        result['category'] = 'request_failed'
    return result


def read_json(session, url, params=None, limit=8 * 2**20):
    with session.get(url, params=params, stream=True, timeout=(8, 15)) as response:
        response.raise_for_status()
        body = bytearray()
        for chunk in response.iter_content(chunk_size=65536):
            body.extend(chunk)
            if len(body) > limit:
                raise ValueError('Metadata exceeds the size limit')
        data = json.loads(body)
    return data


def file_url(path, revision):
    return f'{ENDPOINT}/api/v1/models/{REPO}/repo?' + urlencode(
        {'Revision': revision, 'FilePath': path})


def list_files(session, revision):
    data = read_json(session, f'{ENDPOINT}/api/v1/models/{REPO}/repo/files',
                     {'Revision': revision, 'Recursive': 'True'})
    if data.get('Success') is False or data.get('Code', 200) not in (200, '200'):
        raise ValueError('ModelScope file-list API did not return success')
    items = data.get('Data', {}).get('Files')
    if not isinstance(items, list):
        raise ValueError('ModelScope file-list API returned no Files list')
    result = {}
    for item in items:
        if str(item.get('Type', '')).lower() in ('tree', 'dir', 'directory'):
            continue
        path = item.get('Path')
        if not isinstance(path, str) or not path or path.startswith('/') or '..' in path.split('/'):
            continue
        size = item.get('Size')
        if size is None:
            continue
        result[path] = dict(bytes=int(size), sha256=item.get('Sha256') or item.get('SHA256'))
    if not result:
        raise ValueError('No file metadata returned')
    return result


def find_layout(session, files, revision):
    for prefix in ('', 'diffusers/', 'FL2VA/'):
        index_path = prefix + 'model_index.json'
        config_path = prefix + 'transformer/config.json'
        if index_path not in files or config_path not in files:
            continue
        index = read_json(session, file_url(index_path, revision), limit=2**20)
        config = read_json(session, file_url(config_path, revision), limit=2**20)
        if (index.get('_class_name') == 'MiniMaxH3ModularPipeline'
                and config.get('_class_name') == 'MiniMaxH3Transformer3DModel'):
            return prefix
    return None


def choose_sample(files, prefix, plan, minimum_size):
    sizes = plan.get('file_sizes', {})
    if prefix is not None:
        for name, size in sizes.items():
            # The current HF plan has a repository-root Diffusers layout.
            candidate = prefix + name
            if (name.endswith('.safetensors') and size >= minimum_size
                    and candidate in files and files[candidate]['bytes'] == size):
                return candidate, True
        candidates = [name for name, meta in files.items()
                      if name.startswith(prefix + 'text_encoder/')
                      and name.endswith('.safetensors') and meta['bytes'] >= minimum_size]
        if candidates:
            return sorted(candidates)[0], False
    candidates = [name for name, meta in files.items()
                  if name.endswith('.safetensors') and meta['bytes'] >= minimum_size
                  and ('fl2va' in name.lower() or prefix is not None and name.startswith(prefix))]
    if not candidates:
        raise ValueError('No suitable large FL2VA weight file is visible in the official repo')
    return sorted(candidates)[0], False


def measure_range(session, url, start, byte_limit, seconds):
    started = time.monotonic()
    result = dict(range_start=start, requested_bytes=byte_limit, received_bytes=0)
    last_print = started
    try:
        with session.get(url, stream=True, timeout=(8, min(5, seconds)), headers={
                'Range': f'bytes={start}-{start + byte_limit - 1}',
                'Accept-Encoding': 'identity'}) as response:
            result.update(http_status=response.status_code,
                          header_seconds=round(time.monotonic() - started, 3),
                          final_host=urlsplit(response.url).hostname,
                          range_supported=response.status_code == 206)
            response.raise_for_status()
            if response.status_code not in (200, 206):
                raise ValueError('Download did not return HTTP 200 or 206')
            if response.status_code == 206:
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)',
                                     response.headers.get('Content-Range', ''))
                if not match or int(match[1]) != start:
                    raise ValueError('Server returned an unexpected byte range')
            elif start:
                raise ValueError('Server ignored the nonzero range request')
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' in content_type or 'application/json' in content_type:
                raise ValueError('Download URL returned a document rather than weight bytes')
            for chunk in response.iter_content(chunk_size=min(65536, byte_limit)):
                if not chunk:
                    continue
                if 'first_data_seconds' not in result:
                    result['first_data_seconds'] = round(time.monotonic() - started, 3)
                result['received_bytes'] += min(len(chunk), byte_limit - result['received_bytes'])
                now = time.monotonic()
                if now - last_print >= 5:
                    print(f"  received {result['received_bytes'] / 2**20:.2f} MiB, "
                          f"average {result['received_bytes'] / max(now - started, 1e-9) / 1e6:.2f} MB/s",
                          flush=True)
                    last_print = now
                if result['received_bytes'] >= byte_limit or now - started >= seconds:
                    break
    except (requests.RequestException, ValueError) as exc:
        result['error'] = error_summary(exc)
    elapsed = max(time.monotonic() - started, 1e-9)
    result.update(elapsed_seconds=round(elapsed, 6),
                  average_MBps=round(result['received_bytes'] / elapsed / 1e6, 3),
                  sample_complete=result['received_bytes'] == byte_limit,
                  passed=result['received_bytes'] > 0 and 'error' not in result)
    return result


def run(args, report):
    session = requests.Session()
    session.max_redirects = 5
    session.headers.update({'User-Agent': 'songshanbei-minimax-h3-source-probe/1.0'})
    plan_path = ROOT / 'artifacts/model-download-plan.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    report['hf_reference'] = {key: plan.get(key) for key in ('repo', 'revision', 'bytes')}
    print(f'Official ModelScope repo: {REPO}; revision: {args.revision}', flush=True)
    print(f'At most {args.trials * args.mib} MiB of weight data; no model files will be written.', flush=True)
    print('Proxy selection follows the current terminal environment.', flush=True)
    report['phase'] = 'file_metadata'
    files = list_files(session, args.revision)
    report['visible_files'] = len(files)
    report['phase'] = 'layout_check'
    prefix = find_layout(session, files, args.revision)
    report['diffusers_layout'] = prefix if prefix is not None else None
    report['h3_diffusers_layout_found'] = prefix is not None
    if prefix is None:
        print('No matching H3 Diffusers layout confirmed; speed testing an official FL2VA weight only.', flush=True)
    else:
        print(f'H3 Diffusers layout: {prefix or "repository root"}', flush=True)
    byte_limit = args.mib * 2**20
    path, size_match = choose_sample(files, prefix, plan, byte_limit * args.trials)
    report.update(sample_path=path, sample_size_bytes=files[path]['bytes'],
                  sample_sha256=files[path]['sha256'],
                  sample_path_and_size_match_hf_plan=size_match,
                  full_weight_identity_verified=False)
    print(f"Sample file: {path} ({files[path]['bytes'] / 2**30:.2f} GiB); "
          f'path/size match HF plan: {size_match}', flush=True)
    report['phase'] = 'speed_test'
    report['samples'] = []
    for index in range(args.trials):
        print(f'Trial {index + 1}/{args.trials}: reading at most {args.mib} MiB, '
              f'for about {args.seconds} seconds.', flush=True)
        result = measure_range(session, file_url(path, args.revision),
                               index * byte_limit, byte_limit, args.seconds)
        report['samples'].append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not result['passed']:
            break
    valid = [r for r in report['samples'] if r['passed']]
    report['passed'] = bool(valid)
    if valid:
        speed = sum(r['received_bytes'] for r in valid) / max(
            sum(r['elapsed_seconds'] for r in valid), 1e-9)
        report['average_MBps'] = round(speed / 1e6, 3)
        if plan.get('bytes') and speed:
            report['estimated_full_download_hours_if_sustained'] = round(plan['bytes'] / speed / 3600, 2)
        print(f'ModelScope measured average: {speed / 1e6:.2f} MB/s. '
              'This is a small-sample result, not guaranteed sustained throughput.', flush=True)
    report['phase'] = 'complete'
    session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mib', type=int, default=16, help='Maximum bytes per trial in MiB (1-64)')
    parser.add_argument('--trials', type=int, default=2, choices=(1, 2, 3))
    parser.add_argument('--seconds', type=int, default=30, help='Approximate transfer duration cap per trial (5-60)')
    parser.add_argument('--revision', default='master')
    args = parser.parse_args()
    if not 1 <= args.mib <= 64 or not 5 <= args.seconds <= 60:
        parser.error('--mib must be 1-64 and --seconds must be 5-60')
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                  hostname=os.uname().nodename, repo=REPO, revision=args.revision,
                  max_weight_sample_bytes=args.mib * 2**20 * args.trials,
                  phase='start', passed=False)
    try:
        run(args, report)
    except (requests.RequestException, ValueError, OSError) as exc:
        report['error'] = error_summary(exc)
        print(f"Probe failed during {report['phase']}: {json.dumps(report['error'])}", flush=True)
    except KeyboardInterrupt:
        report['cancelled'] = True
        print('Probe cancelled.', flush=True)
    output = ROOT / 'artifacts/modelscope-speed.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'Report: {output}', flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
