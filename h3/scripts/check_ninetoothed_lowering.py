#!/usr/bin/env python3
"""Check attention lowering and JIT wrapper creation without a GPU launch."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    report = dict(lowering_passed=False, jit_wrapper_passed=False,
                  host_argument_contract_passed=False,
                  device_execution_passed=False, inference_passed=False)
    try:
        import torch
        from ninetoothed.compiler.driver import lower, DEFAULT_COMPILER, CompileRequest
        from ninetoothed.compiler.runtime import _public_values
        from songshan_h3.attention_kernel import definition, kernel
        arrangement, application, tensors = definition()
        compilation = DEFAULT_COMPILER.compile(CompileRequest(
            arrangement=arrangement, application=application, tensors=tensors,
            platform='hygon-bw1000', backend='triton',
            num_warps=4, num_stages=1, max_num_configs=1))
        artifact = compilation.artifact
        report['lowering_passed'] = True
        expected_arguments = ('query', 'key', 'value', 'is_causal', 'scale', 'output')
        if compilation.launch_abi.public_args != expected_arguments:
            raise ValueError('Unexpected public attention kernel arguments')
        report['public_arguments'] = list(expected_arguments)
        # Exercise the same contract checker that rejected the meta mask. CPU
        # is allowed only in this host check; it never launches the GPU kernel.
        for head_dim in (72, 128):
            q, k, v, out = (torch.empty((1, 2, 4, head_dim), dtype=torch.bfloat16)
                            for _ in range(4))
            for causal in (False, True):
                _public_values(compilation.launch_abi,
                               (q, k, v, causal, head_dim**-0.5, out), {},
                               specs=compilation.kernel.tensors, device_types=('cpu',))
        report['host_argument_contract_passed'] = True
        report['generated_source_sha256'] = hashlib.sha256(artifact.primary_source.encode()).hexdigest()
        report['definition_sha256'] = hashlib.sha256(
            (ROOT / 'songshan_h3/attention_kernel.py').read_bytes()).hexdigest()
        report['versions'] = {name: importlib.metadata.version(name) for name in ('ninetoothed', 'ntops')}
        report['head_dimensions_lowered'] = []
        for head_dim in (72, 128):
            lower(*definition(head_dim), platform='hygon-bw1000', backend='triton',
                  num_warps=4, num_stages=1, max_num_configs=1)
            report['head_dimensions_lowered'].append(head_dim)
        kernel()
        report['jit_wrapper_passed'] = True
        print('NineToothed lowering, argument contract and JIT wrapper: PASS; no GPU execution performed')
        return 0
    except Exception:
        report['error'] = traceback.format_exc()
        print(report['error'])
        return 1
    finally:
        (ROOT / 'artifacts/attention-lowering.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
