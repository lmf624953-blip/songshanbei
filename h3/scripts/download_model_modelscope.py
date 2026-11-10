#!/usr/bin/env python3
"""Download the pinned HF FL2VA file set from official ModelScope, checking hashes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time

from huggingface_hub import HfApi
import httpx
import requests

from probe_modelscope import ENDPOINT, REPO, error_summary, file_url, find_layout, list_files

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def new_hasher(record):
    if record['hash_type'] == 'sha256':
        return hashlib.sha256()
    result = hashlib.sha1()
    result.update(f"blob {record['bytes']}\0".encode())
    return result


def hash_prefix(path, record):
    result = new_hasher(record)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            result.update(chunk)
    return result


def make_plan(hf_plan, session, revision, local):
    if hf_plan.get('repo') != 'MiniMaxAI/MiniMax-H3' or not re.fullmatch(
            r'[0-9a-f]{40}', hf_plan.get('revision', '')):
        raise ValueError('A pinned official HF download plan is required')
    selected = hf_plan['selected_files']
    if len(selected) != len(set(selected)):
        raise ValueError('HF plan contains duplicate filenames')
    for name in selected:
        if not name or name.startswith('/') or '..' in name.split('/'):
            raise ValueError('HF plan contains an unsafe relative path')
    print('Reading hashes from the fixed HF revision (metadata only).', flush=True)
    info = HfApi().model_info(hf_plan['repo'], revision=hf_plan['revision'],
                              files_metadata=True, timeout=30)
    if info.sha != hf_plan['revision']:
        raise ValueError('HF returned an unexpected revision')
    expected = {item.rfilename: item for item in info.siblings}
    files = list_files(session, revision)
    prefix = find_layout(session, files, revision)
    if prefix is None:
        raise ValueError('Official ModelScope repo has no compatible H3 Diffusers layout')
    records = []
    for name in selected:
        if name not in expected:
            raise ValueError(f'Missing required file in HF metadata: {name}')
        hf = expected[name]
        ms = files.get(prefix + name)
        size = hf_plan['file_sizes'][name]
        if hf.size != size:
            raise ValueError(f'File size differs from the pinned HF plan: {name}')
        if hf.lfs is not None:
            digest = hf.lfs.sha256
            kind = 'sha256'
            if not re.fullmatch(r'[0-9a-f]{64}', digest):
                raise ValueError(f'HF has no usable SHA256 for {name}')
        else:
            digest = hf.blob_id
            kind = 'git_blob_sha1'
            if not digest or not re.fullmatch(r'[0-9a-f]{40}', digest):
                raise ValueError(f'HF has no usable Git blob hash for {name}')
        record = dict(path=name, modelscope_path=prefix + name, bytes=size,
                      hash_type=kind, expected_hash=digest)
        source_matches = bool(ms and ms['bytes'] == size and (
            kind != 'sha256' or not ms['sha256'] or ms['sha256'].lower() == digest))
        record['source_metadata_matches'] = source_matches
        if not source_matches:
            # A verified local HF file can be retained even if the other hub's
            # README/config has changed. Never download a mismatched replacement.
            path = local / name
            if (not path.is_file() or path.stat().st_size != size
                    or hash_prefix(path, record).hexdigest() != digest):
                raise ValueError(f'ModelScope metadata differs from pinned HF content: {name}')
        records.append(record)
    return dict(hf_repo=hf_plan['repo'], hf_revision=hf_plan['revision'],
                modelscope_repo=REPO, modelscope_revision=revision,
                bytes=sum(r['bytes'] for r in records), files=records,
                note='ModelScope contents must match pinned HF hashes; a mutable branch is not trusted by name alone')


class Progress:
    def __init__(self, plan):
        self.total = plan['bytes']
        self.count = len(plan['files'])
        self.done = {}
        self.active = {}
        self.network_bytes = 0
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def update(self, name, amount, received=0):
        with self.lock:
            self.active[name] = amount
            self.network_bytes += received

    def complete(self, record):
        with self.lock:
            self.done[record['path']] = record['bytes']
            self.active.pop(record['path'], None)

    def monitor(self):
        previous_time = time.monotonic()
        previous_bytes = 0
        while not self.stop.wait(15):
            now = time.monotonic()
            with self.lock:
                snapshot = dict(verified_files=len(self.done), total_files=self.count,
                                verified_bytes=sum(self.done.values()),
                                active_bytes=sum(self.active.values()),
                                total_bytes=self.total, network_bytes=self.network_bytes,
                                active_files=list(self.active))
            speed = (snapshot['network_bytes'] - previous_bytes) / max(now - previous_time, 1e-9)
            snapshot.update(timestamp=time.strftime('%Y-%m-%d %H:%M:%S %z'),
                            recent_MBps=round(speed / 1e6, 2))
            save_json(ROOT / 'artifacts/modelscope-download-progress.json', snapshot)
            print(f"[download] verified {snapshot['verified_files']}/{self.count}, "
                  f"verified {snapshot['verified_bytes'] / 2**30:.2f} GiB, "
                  f"active partial {snapshot['active_bytes'] / 2**30:.2f} GiB, "
                  f"recent {speed / 1e6:.2f} MB/s", flush=True)
            previous_time, previous_bytes = now, snapshot['network_bytes']


def download_one(record, local, revision, progress):
    name, size = record['path'], record['bytes']
    dest = local / name
    if dest.exists():
        if dest.stat().st_size != size or hash_prefix(dest, record).hexdigest() != record['expected_hash']:
            raise ValueError(f'Existing final file differs from the pinned model; retained unchanged: {name}')
        progress.complete(record)
        return dict(path=name, reused=True, hash_verified=True)
    if not record.get('source_metadata_matches', True):
        raise ValueError(f'No matching ModelScope source for missing local file: {name}')
    part = local / '.cache/modelscope/parts' / (name + '.' + record['expected_hash'] + '.part')
    part.parent.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.max_redirects = 5
        for attempt in range(1, 5):
            if progress.stop.is_set():
                raise InterruptedError('Download stopped')
            offset = part.stat().st_size if part.exists() else 0
            if offset > size:
                raise ValueError(f'Temporary file exceeds expected size; retained: {name}')
            digest = hash_prefix(part, record) if offset else new_hasher(record)
            progress.update(name, offset)
            try:
                if size == 0 and not part.exists():
                    part.touch()
                if offset < size:
                    headers = {'Range': f'bytes={offset}-{size - 1}', 'Accept-Encoding': 'identity'}
                    with session.get(file_url(record['modelscope_path'], revision), headers=headers,
                                     stream=True, timeout=(10, 20)) as response:
                        response.raise_for_status()
                        if response.status_code not in (200, 206):
                            raise ValueError(f'Unexpected HTTP status for {name}')
                        if response.status_code == 206:
                            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',
                                                 response.headers.get('Content-Range', ''))
                            if (not match or int(match[1]) != offset or int(match[3]) != size
                                    or int(match[2]) >= size):
                                raise ValueError(f'Unexpected Content-Range for {name}')
                        elif offset:
                            # A server ignoring Range must restart rather than append a full file.
                            offset = 0
                            digest = new_hasher(record)
                            progress.update(name, 0)
                        content_type = response.headers.get('Content-Type', '').lower()
                        if name.endswith('.safetensors') and (
                                'text/html' in content_type or 'application/json' in content_type):
                            raise ValueError(f'Weight URL returned a document: {name}')
                        with part.open('ab' if offset else 'wb') as stream:
                            for chunk in response.iter_content(chunk_size=2**20):
                                if progress.stop.is_set():
                                    raise InterruptedError('Download stopped')
                                if not chunk:
                                    continue
                                if offset + len(chunk) > size:
                                    raise ValueError(f'Received more bytes than expected: {name}')
                                stream.write(chunk)
                                digest.update(chunk)
                                offset += len(chunk)
                                progress.update(name, offset, len(chunk))
                if offset != size:
                    raise requests.ConnectionError('Transfer ended before the expected file size')
                if digest.hexdigest() != record['expected_hash']:
                    raise ValueError(f'Content hash mismatch; temporary file retained: {name}')
                dest.parent.mkdir(parents=True, exist_ok=True)
                part.replace(dest)
                progress.complete(record)
                print(f"[verified] {name} ({size / 2**30:.2f} GiB)", flush=True)
                return dict(path=name, reused=False, hash_verified=True)
            except requests.RequestException as exc:
                code = getattr(getattr(exc, 'response', None), 'status_code', None)
                if attempt == 4 or code and 400 <= code < 500 and code not in (408, 429):
                    raise
                print(f'[retry {attempt}/4] {name}: {json.dumps(error_summary(exc))}', flush=True)
                if progress.stop.wait(min(2**attempt, 10)):
                    raise InterruptedError('Download stopped')


def assert_no_hf_download(local):
    for path in (local / '.cache/huggingface/download').rglob('*.lock'):
        with path.open('rb') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('An HF downloader still holds file locks; stop it before switching sources') from None
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2, choices=range(1, 9))
    parser.add_argument('--revision', default='master', help='Official ModelScope revision; content hashes remain pinned to HF')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    storage = Path(os.getenv('SONGSHAN_STORAGE', '/work1/share/sugonhpcapp01/zhanghengyu/songshanbei'))
    local = storage / 'models/MiniMax-H3'
    progress = None
    try:
        hf_plan = json.loads((ROOT / 'artifacts/model-download-plan.json').read_text())
        assert_no_hf_download(local)
        with requests.Session() as session:
            plan = make_plan(hf_plan, session, args.revision, local)
        save_json(ROOT / 'artifacts/modelscope-download-plan.json', plan)
        print(f"Source: {REPO}@{args.revision}; expected content: {plan['hf_repo']}@{plan['hf_revision']}", flush=True)
        print(f"Selected {len(plan['files'])} FL2VA files, {plan['bytes'] / 2**30:.2f} GiB. Metadata checks passed.", flush=True)
        if args.plan_only:
            return 0
        local.mkdir(parents=True, exist_ok=True)
        lock_path = local / '.cache/modelscope/download.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('Another ModelScope downloader is already running') from None
            remaining = sum(r['bytes'] for r in plan['files'] if not (local / r['path']).is_file())
            if shutil.disk_usage(local).free < remaining + 10 * 2**30:
                raise ValueError('Insufficient space for remaining model files plus 10 GiB reserve')
            progress = Progress(plan)
            monitor = threading.Thread(target=progress.monitor, daemon=True)
            monitor.start()
            completed = []
            pool = ThreadPoolExecutor(max_workers=args.workers)
            try:
                futures = [pool.submit(download_one, record, local, args.revision, progress)
                           for record in plan['files']]
                for future in as_completed(futures):
                    completed.append(future.result())
            finally:
                progress.stop.set()
                pool.shutdown(wait=True, cancel_futures=True)
                monitor.join(timeout=2)
            manifest = dict(repo=plan['hf_repo'], revision=plan['hf_revision'],
                            source='modelscope', source_repo=REPO, source_revision=args.revision,
                            selected_files=hf_plan['selected_files'], bytes=plan['bytes'],
                            model_path=str(local), all_content_hashes_verified=True,
                            verified_files=completed, inference_passed=False)
            save_json(ROOT / 'artifacts/model-download.json', manifest)
            print(f'All files verified against the fixed HF revision. Download complete. --model-path={local}', flush=True)
            return 0
    except KeyboardInterrupt:
        print('Download interrupted; partial files retained for resume.', flush=True)
    except (requests.RequestException, httpx.HTTPError, OSError) as exc:
        print(f'Download failed: {json.dumps(error_summary(exc))}. Partial files retained.', flush=True)
    except (ValueError, KeyError) as exc:
        print(f'Download stopped: {exc}', flush=True)
    finally:
        if progress is not None:
            progress.stop.set()
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
