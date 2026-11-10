#!/usr/bin/env python3
"""Compile NineToothed attention to gfx936 binaries without launching a GPU."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DIRECTORY = ROOT / 'artifacts/attention-codegen'


def worker(head_dim, causal):
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    import ninetoothed
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from ninetoothed.compiler.driver import DEFAULT_COMPILER, CompileRequest
    from ninetoothed.compiler.runtime import import_python_module
    from ninetoothed.backends.materializers.triton import _compile_block
    from songshan_h3.attention_kernel import definition
    from songshan_h3.compiler_compat import configure_hygon_compiler

    compatibility = configure_hygon_compiler()
    arrangement, application, tensors = definition(head_dim, ninetoothed.bfloat16, causal)
    compilation = DEFAULT_COMPILER.compile(CompileRequest(
        arrangement=arrangement, application=application, tensors=tensors,
        backend='triton', platform='hygon-bw1000', num_warps=4, num_stages=1,
        max_num_configs=1))
    stem = DIRECTORY / f'head{head_dim}-causal{int(causal)}'
    source_path = stem.with_suffix('.triton.py')
    # Persist the compiler output unchanged, then compile its generated entry.
    source_path.write_text(compilation.artifact.primary_source)
    module = import_python_module(source_path)
    function = getattr(module, compilation.artifact.kernel_name + '_kernel')
    constants = {'is_causal': causal, 'BLOCK': _compile_block(compilation)}
    # Runtime JIT infers fp32 for Python float and i32 for these model sizes.
    signature = {name: ('*bf16' if name in ('query', 'key', 'value', 'output')
                        else 'fp32' if name == 'scale' else 'i32')
                 for name in function.arg_names if name not in constants}
    compiled = triton.compile(ASTSource(function, signature, constexprs=constants),
                              target=GPUTarget('hip', 'gfx936', 64),
                              options={'num_warps': 4, 'num_stages': 1})
    binary = compiled.asm['hsaco']
    stem.with_suffix('.hsaco').write_bytes(binary)
    stem.with_suffix('.llir').write_text(compiled.asm['llir'])
    result = dict(head_dim=head_dim, causal=causal, dtype='bfloat16',
                  codegen_passed=True, binary_bytes=len(binary),
                  binary_sha256=hashlib.sha256(binary).hexdigest(),
                  shared_memory_bytes=compiled.metadata.shared,
                  compatibility=compatibility, device_execution_passed=False)
    stem.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'head_dim={head_dim}, causal={causal}: gfx936 codegen PASS, {len(binary)} bytes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', nargs=2, metavar=('HEAD_DIM', 'CAUSAL'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(int(args.worker[0]), bool(int(args.worker[1])))
        return 0
    report = dict(codegen_passed=False, device_execution_passed=False,
                  inference_passed=False, target='hip:gfx936', cases=[],
                  versions={name: importlib.metadata.version(name) for name in ('triton', 'ninetoothed')})
    try:
        for head_dim in (64, 72, 128):
            for causal in (False, True):
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), '--worker', str(head_dim), str(int(causal))],
                    capture_output=True, text=True)
                log = ROOT / f'logs/attention-codegen-head{head_dim}-causal{int(causal)}.log'
                log.write_text(result.stdout + result.stderr)
                print(result.stdout, end='', flush=True)
                if result.returncode:
                    raise RuntimeError(f'Codegen exited {result.returncode}; see {log}')
                path = DIRECTORY / f'head{head_dim}-causal{int(causal)}.json'
                report['cases'].append(json.loads(path.read_text()))
        report['codegen_passed'] = True
        print(f'All {len(report["cases"])} device binaries compiled; no GPU execution performed.')
        return 0
    except Exception as error:
        report['error'] = str(error)
        print(report['error'], flush=True)
        return 1
    finally:
        (ROOT / 'artifacts/attention-codegen.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
