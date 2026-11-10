#!/usr/bin/env python3
"""Build the host-only InfiniCore to NineToothed dispatcher extension."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sysconfig

import pybind11
import torch

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'songshan_h3/attention_bridge.cc'
output = source.parent / ('_attention_bridge' + sysconfig.get_config_var('EXT_SUFFIX'))
torch_root = Path(torch.__file__).resolve().parent
core_root = Path(os.environ.get('INFINI_ROOT', ROOT / '.infini'))
library = core_root / 'lib/libinfinicore_cpp_api.so'
if not library.is_file():
    raise SystemExit('Build and install InfiniCore first: missing ' + str(library))
command = [os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-shared', '-fPIC',
           '-DENABLE_ATEN', f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}',
           '-I' + pybind11.get_include(), '-I' + sysconfig.get_paths()['include'],
           '-I' + str(torch_root / 'include'),
           '-I' + str(torch_root / 'include/torch/csrc/api/include'),
           '-I' + str(core_root / 'include'), str(source),
           '-L' + str(core_root / 'lib'), '-linfinicore_cpp_api',
           '-L' + str(torch_root / 'lib'), '-ltorch_python', '-ltorch', '-ltorch_cpu', '-lc10',
           '-Wl,-rpath,' + str(core_root / 'lib'),
           '-Wl,-rpath,' + str(torch_root / 'lib'), '-o', str(output)]
subprocess.run(command, check=True)
spec = importlib.util.spec_from_file_location('_attention_bridge', output)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.register_hygon()
report = dict(command=command, output=str(output), host_extension_built=True,
              host_import_and_registration_passed=True,
              device_execution_passed=False, inference_passed=False)
(ROOT / 'artifacts/attention-bridge-build.json').write_text(json.dumps(report, indent=2) + '\n')
print('Host extension built; device correctness has not been checked: ' + str(output))
