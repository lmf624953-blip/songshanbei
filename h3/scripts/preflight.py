#!/usr/bin/env python3
"""Record actual dependencies and execute small BF16 checks on visible DCUs.

This checks infrastructure only; it is not an H3 inference or competition test.
"""
import argparse
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/preflight.json')
args = parser.parse_args()
official = json.loads((ROOT / 'env/official-packages.json').read_text())
comparison = {}
for name, expected in official.items():
    if not name or not name.strip('-'):
        continue
    try:
        # Respect venv precedence when the base environment has the same package.
        actual = md.version(name)
    except md.PackageNotFoundError:
        actual = None
    comparison[name] = dict(expected=expected, actual=actual,
                            status='match' if actual == expected else
                            'missing' if actual is None else 'different')
report = dict(timestamp=time.strftime('%Y-%m-%d %H:%M:%S %z'),
              hostname=platform.node(), python=sys.version, executable=sys.executable,
              platform=platform.platform(), slurm_job_id=os.getenv('SLURM_JOB_ID'),
              dtk_root=os.getenv('DTK_ROOT'),
              commands={k: shutil.which(k) for k in ['hipcc', 'xmake', 'ffmpeg', 'g++']},
              filesystem=dict(zip(['total', 'used', 'free'], shutil.disk_usage(ROOT))),
              official_comparison=comparison, devices=[], inference_passed=False)
report['tools'] = {}
for name, command in [('xmake', ['xmake', '--version']),
                      ('cxx', [os.getenv('CXX', 'g++'), '--version'])]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        report['tools'][name] = dict(returncode=result.returncode,
                                     output=(result.stdout + result.stderr).strip()[:2000])
    except (OSError, subprocess.TimeoutExpired) as exc:
        report['tools'][name] = dict(error=str(exc))
report['storage'] = dict(path=os.getenv('SONGSHAN_STORAGE'),
                         writable=os.access(os.getenv('SONGSHAN_STORAGE', ROOT), os.W_OK))
host_cards = []
for card in sorted(Path('/sys/class/drm').glob('card[0-9]*')):
    try:
        if (card / 'device/vendor').read_text().strip() == '0x1d94':
            host_cards.append(dict(card=card.name,
                                   pci_device=str((card / 'device').resolve()),
                                   device_id=(card / 'device/device').read_text().strip()))
    except OSError:
        pass
report['device_visibility'] = dict(
    host_hygon_cards=host_cards,
    dev_kfd_present=Path('/dev/kfd').exists(),
    dev_dri_present=Path('/dev/dri').exists(),
    note='Host PCI devices do not establish access from this execution sandbox or SLURM allocation')
required_packages = ('InfiniCore', 'InfiniLM', 'diffusers', 'accelerate', 'transformers', 'huggingface-hub',
                     'safetensors', 'ninetoothed', 'ntops', 'av', 'imageio-ffmpeg',
                     'pybind11', 'hatchling')
missing_packages = []
for package in required_packages:
    try:
        md.version(package)
    except md.PackageNotFoundError:
        missing_packages.append(package)
report['inference_prerequisites'] = dict(
    missing_packages=missing_packages,
    model_download_manifest_present=(ROOT / 'artifacts/model-download.json').is_file(),
    note='Package presence and basic compute checks do not validate full H3 inference')
failures = []
try:
    import torch
    report['torch'] = dict(version=torch.__version__,
                           vendor_version=getattr(torch.version, '__hcu_version__', None),
                           hip=torch.version.hip)
    if report['torch']['vendor_version'] != official['torch']:
        failures.append('Torch vendor version differs from official inventory')
    count = torch.cuda.device_count()
    if not count:
        failures.append(f'No DCU accessible to PyTorch in this execution session; '
                        f'{len(host_cards)} Hygon PCI devices observed on host')
    torch.manual_seed(42)
    # Use identical BF16 operands for CPU reference and each device.
    a = torch.randn(256, 256).to(torch.bfloat16)
    b = torch.randn(256, 256).to(torch.bfloat16)
    ref = a.float() @ b.float()
    for i in range(count):
        prop = torch.cuda.get_device_properties(i)
        entry = dict(index=i, name=prop.name, arch=prop.gcnArchName,
                     total_memory=prop.total_memory)
        try:
            aa, bb = a.to(f'cuda:{i}'), b.to(f'cuda:{i}')
            result = (aa @ bb).float().cpu()
            torch.cuda.synchronize(i)
            relative_l2 = ((result-ref).norm()/ref.norm()).item()
            entry.update(relative_l2=relative_l2,
                         bf16_matmul_pass=bool(torch.isfinite(result).all()) and relative_l2 < 0.01)
            if not entry['bf16_matmul_pass']:
                failures.append(f'DCU {i} BF16 check failed')
            del aa, bb, result
        except Exception as exc:
            entry['error'] = str(exc)
            failures.append(f'DCU {i}: {exc}')
        report['devices'].append(entry)
except Exception as exc:
    failures.append(repr(exc))
report['infrastructure_failures'] = failures
report['infrastructure_pass'] = not failures
out = args.output
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k:v for k,v in report.items() if k != 'official_comparison'}, ensure_ascii=False, indent=2))
print(f'Full dependency comparison: {out}')
raise SystemExit(bool(failures))
