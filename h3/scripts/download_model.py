#!/usr/bin/env python3
"""Download only unmodified FL2VA Diffusers components at one pinned HF commit.

Select layout using actual configuration, not the example's folder name.
Run after prepare_dependencies.sh in a terminal with network and storage access.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
storage = Path(os.environ.get('SONGSHAN_STORAGE',
    '/work1/share/sugonhpcapp01/zhanghengyu/songshanbei'))
repo = 'MiniMaxAI/MiniMax-H3'
cache = storage / '.cache/huggingface'
local = storage / 'models/MiniMax-H3'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--transport', choices=('auto', 'http'), default='auto')
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--revision', default=os.getenv(
    'SONGSHAN_MODEL_REVISION', '42ed227ee7df40d41602854ae760620d6eb651fe'))
parser.add_argument('--status', action='store_true', help='Show local files without network access')
args = parser.parse_args()
if args.workers < 1 or args.workers > 16:
    parser.error('--workers must be between 1 and 16')


def local_progress(file_sizes=None):
    completed = {}
    if file_sizes is None:
        for base, dirs, names in os.walk(local):
            dirs[:] = [name for name in dirs if name != '.cache']
            for name in names:
                path = Path(base) / name
                try:
                    completed[str(path.relative_to(local))] = path.stat().st_size
                except OSError:
                    pass
    else:
        for name, expected in file_sizes.items():
            path = local / name
            try:
                if path.stat().st_size == expected:
                    completed[name] = expected
            except OSError:
                pass
    temporary = []
    for path in (local / '.cache/huggingface/download').rglob('*.incomplete'):
        try:
            stat = path.stat()
            temporary.append(dict(path=str(path.relative_to(local)), bytes=stat.st_size,
                                  age_seconds=round(time.time() - stat.st_mtime, 1)))
        except OSError:
            pass
    report = dict(timestamp=time.strftime('%Y-%m-%d %H:%M:%S %z'),
                  completed_files=len(completed), completed_bytes=sum(completed.values()),
                  total_files=len(file_sizes) if file_sizes else None,
                  total_bytes=sum(file_sizes.values()) if file_sizes else None,
                  temporary_files=temporary,
                  note='Temporary files may include leftovers from earlier attempts; byte counts are not network throughput')
    (ROOT / 'artifacts/model-download-progress.json').write_text(json.dumps(report, indent=2) + '\n')
    total = f"{report['total_bytes'] / 2**30:.2f} GiB" if file_sizes else 'unknown'
    print(f"[download] complete files {len(completed)}/{len(file_sizes) if file_sizes else '?'}, "
          f"complete {report['completed_bytes'] / 2**30:.2f}/{total}; "
          f"temporary files {len(temporary)}, total size "
          f"{sum(item['bytes'] for item in temporary) / 2**30:.2f} GiB", flush=True)
    return report


if args.status:
    plan_path = ROOT / 'artifacts/model-download-plan.json'
    sizes = json.loads(plan_path.read_text()).get('file_sizes') if plan_path.is_file() else None
    print(json.dumps(local_progress(sizes), indent=2))
    raise SystemExit(0)

# Hub reads these settings at import time. An already running download is not
# affected by editing this script or changing variables in another terminal.
if args.transport == 'http':
    os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
os.environ.setdefault('HF_XET_CACHE', str(storage / '.cache/huggingface/xet'))
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

api = HfApi()
info = api.model_info(repo, revision=args.revision, files_metadata=True)
revision = info.sha
files = {item.rfilename: item for item in info.siblings}

def read_json(name):
    path = hf_hub_download(repo, name, revision=revision, cache_dir=cache)
    return json.loads(Path(path).read_text())

prefix = None
for candidate in ('diffusers/', '', 'FL2VA/'):
    index_name = candidate + 'model_index.json'
    config_name = candidate + 'transformer/config.json'
    if index_name not in files or config_name not in files:
        continue
    index = read_json(index_name)
    config = read_json(config_name)
    if (index.get('_class_name') == 'MiniMaxH3ModularPipeline'
            and config.get('_class_name') == 'MiniMaxH3Transformer3DModel'):
        prefix = candidate
        break
if prefix is None:
    raise RuntimeError('No compatible Diffusers FL2VA layout found. '
                       'Do not substitute original-format weights; inspect the repository.')

components = ('text_encoder', 'transformer', 'tokenizer', 'processor',
              'vae', 'audio_vae', 'scheduler', 'audio_scheduler')
selected = []
for component in components:
    matches = [name for name in files if name.startswith(prefix + component + '/')]
    if not matches:
        raise RuntimeError(f'Missing component: {prefix}{component}')
    selected.extend(matches)
for name in (prefix + 'model_index.json', prefix + 'modular_model_index.json',
             'LICENSE', 'README.md'):
    if name in files:
        selected.append(name)
storage.mkdir(parents=True, exist_ok=True)
needed = sum(files[name].size or 0 for name in selected)
existing = sum((local / name).stat().st_size for name in selected
               if (local / name).is_file())
free = shutil.disk_usage(storage).free
remaining = max(0, needed - existing)
print(f'Repo: {repo}@{revision}', flush=True)
print(f'Transport: {args.transport}; workers: {args.workers}', flush=True)
print(f'Layout: {prefix or "repository root"}; {len(selected)} files; '
      f'{needed / 2**30:.1f} GiB; free {free / 2**30:.1f} GiB', flush=True)
if free < remaining + 10 * 2**30:
    raise RuntimeError('Insufficient free storage for the remaining files plus 10 GiB reserve')
file_sizes = {name: files[name].size or 0 for name in selected}
plan = dict(repo=repo, revision=revision, selected_files=selected, file_sizes=file_sizes,
            bytes=needed, model_path=str(local / prefix if prefix else local))
(ROOT / 'artifacts/model-download-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
stop_progress = threading.Event()


def monitor_progress():
    while not stop_progress.wait(30):
        try:
            local_progress(file_sizes)
        except OSError as exc:
            print(f'[download] Cannot record local progress: {exc}', flush=True)


local_progress(file_sizes)
progress_thread = threading.Thread(target=monitor_progress, daemon=True)
progress_thread.start()
try:
    snapshot_download(repo, revision=revision, allow_patterns=selected,
                      local_dir=local, cache_dir=cache, max_workers=args.workers)
finally:
    stop_progress.set()
    progress_thread.join(timeout=2)
    local_progress(file_sizes)
bad_sizes = [name for name, size in file_sizes.items()
             if not (local / name).is_file() or (local / name).stat().st_size != size]
if bad_sizes:
    raise RuntimeError(f'Download returned with missing or wrong-size files: {bad_sizes}')

# The upstream resolver requires a directory named diffusers, except when it is
# supplied an arbitrary name. Expose a symlink to the exact downloaded layout.
# No weight or model configuration is rewritten.
actual = local / prefix if prefix else local
model_path = actual
if prefix == 'FL2VA/':
    alias = local / 'diffusers'
    if alias.exists() and alias.resolve() != actual.resolve():
        raise RuntimeError(f'Conflicting existing path: {alias}')
    if not alias.exists():
        alias.symlink_to('FL2VA', target_is_directory=True)
    model_path = alias
manifest = dict(repo=repo, revision=revision, selected_files=selected,
                bytes=needed, model_path=str(model_path))
(ROOT / 'artifacts/model-download.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(f'Download complete. --model-path={model_path}')
