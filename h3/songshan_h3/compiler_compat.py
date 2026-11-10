"""Select the installed Hygon backend's opt-in LLVM 17 compatibility pass."""
import inspect
import hashlib
import os
from pathlib import Path


def configure_hygon_compiler():
    from triton.backends.amd.compiler import HIPBackend

    flag = 'DPA4_TRITON_LLVM17_COMPAT'
    if flag not in inspect.getsource(HIPBackend.make_amdgcn):
        raise RuntimeError(
            'This DTK 26.04 setup requires the Triton LLVM 17 GEP compatibility '
            'pass; the installed backend does not expose ' + flag)
    os.environ.setdefault(flag, '1')
    if os.environ[flag] != '1':
        raise RuntimeError(flag + '=1 is required for this Hygon/DTK 26.04 setup')
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault('DPA4_TRITON_IR_LOG', str(root / 'artifacts/triton-llvm17-ir'))
    backend = Path(inspect.getsourcefile(HIPBackend))
    return dict(flag=flag, enabled=True, ir_log=os.environ['DPA4_TRITON_IR_LOG'],
                backend_file=str(backend),
                backend_sha256=hashlib.sha256(backend.read_bytes()).hexdigest())
