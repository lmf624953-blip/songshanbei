#!/usr/bin/env python3
"""Separate native library verification from device-dependent Python imports."""
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import sysconfig
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = dict(timestamp=time.strftime('%Y-%m-%d %H:%M:%S %z'),
                  framework_build_passed=False, dynamic_libraries_passed=False,
                  native_imports_passed=None, inference_passed=False,
                  competition_operator_audit_passed=False, ninetoothed_aot_enabled=False,
                  cpp_abi=int(torch._C._GLIBCXX_USE_CXX11_ABI),
                  visible_devices=torch.cuda.device_count())
    try:
        libraries = [ROOT / '.infini/lib' / name for name in (
            'libinfinirt.so', 'libinfiniop.so', 'libinfiniccl.so', 'libinfinicore_cpp_api.so')]
        suffix = sysconfig.get_config_var('EXT_SUFFIX')
        report['packages'] = {}
        for name in ('InfiniCore', 'InfiniLM'):
            package = name.lower()
            source = ROOT / 'sources' / (name + '-build')
            module = source / 'python' / package / 'lib' / ('_' + package + suffix)
            libraries.append(module)
            report[package + '_build_dir'] = str(source)
            report['packages'][name] = importlib.metadata.version(name)
        for library in libraries:
            if not library.is_file():
                raise FileNotFoundError(library)
            ctypes.CDLL(str(library), mode=os.RTLD_NOW | os.RTLD_GLOBAL)
        report['libraries'] = [str(path) for path in libraries]
        report['dynamic_libraries_passed'] = True
        report['framework_build_passed'] = True
        if report['visible_devices']:
            report['native_imports_passed'] = False
            import infinicore
            import infinilm
            from infinilm.multimodal.minimax_h3 import MiniMaxH3ConditionEncoder, MiniMaxH3TransformerRunner
            report['native_imports_passed'] = True
        else:
            report['native_imports_pending_reason'] = (
                'No accessible accelerator in this session; Core import initializes the Hygon runtime. '
                'Run this check from the ordinary terminal with devices exposed.')
        print(json.dumps(report, indent=2))
        return 0
    except Exception:
        report['error'] = traceback.format_exc()
        print(report['error'])
        return 1
    finally:
        (ROOT / 'artifacts/framework-build-current.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
