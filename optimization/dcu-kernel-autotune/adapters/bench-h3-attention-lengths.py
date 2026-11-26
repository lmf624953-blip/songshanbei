#!/usr/bin/env python3
"""在生产的三镜像分派路径上比较不同动态长度的完整注意力。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--lengths', type=int, nargs='+',
                        default=[4096, 8192, 16384, 22016, 27570, 32768, 49152, 65535])
    parser.add_argument('--iterations', type=int, default=9)
    args = parser.parse_args()
    framework = Path(__file__).resolve().parents[1]
    code = Path(os.environ['H3_AUTOTUNE_CODE_ROOT']) / 'optimization'
    sys.path.insert(0, str(code))
    loader = runpy.run_path(str(code / 'bench-attention.py'))['native_launcher']
    screen = runpy.run_path(str(framework / 'candidates/pv-v-lds-layout/screen.py'))['screen']
    reference, candidate = loader(args.reference.resolve()), loader(args.candidate.resolve())
    result = dict(scope='动态长度、七个 head、dim128、BF16、实际 packed stride；生产镜像及主机分派',
                  reference_build_sha256=hashlib.sha256((args.reference / 'dense-attention-aot-build.json').read_bytes()).hexdigest(),
                  candidate_build_sha256=hashlib.sha256((args.candidate / 'dense-attention-aot-build.json').read_bytes()).hexdigest(),
                  cases=[])
    for length in args.lengths:
        case = screen(reference, candidate, length, args.iterations)
        result['cases'].append(case)
        print(json.dumps(case, ensure_ascii=False), flush=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    assert all(case['passed'] for case in result['cases'])


if __name__ == '__main__':
    main()
