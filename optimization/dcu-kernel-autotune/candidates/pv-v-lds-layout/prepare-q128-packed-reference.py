#!/usr/bin/env python3
"""从正式三镜像构建中提取仅供 packed stride 微测的 Q128 基准。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('production', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.production.resolve(strict=True)
    record = json.loads((root / 'dense-attention-aot-build.json').read_text())
    assert (record['query_block'], record['key_block'], record['warps']) == (128, 32, 4)
    assert record['packed_strides']['strides'] == [896, 896, 2688, 896]
    assert 'direct_attention_query' in record['codegen_passes']
    for name, digest in [('dense-attention-aot.hsaco', record['hsaco_sha256']),
                         ('dense-attention-packed.hsaco', record['packed_strides']['hsaco_sha256']),
                         ('dense-attention-fallback.hsaco', record['fallback']['hsaco_sha256'])]:
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    out = args.output.resolve()
    out.mkdir()
    for name in ('dense-attention-aot.so', 'dense-attention-fallback.hsaco'):
        shutil.copy2(root / name, out / name)
    shutil.copy2(root / 'dense-attention-packed.hsaco', out / 'dense-attention-aot.hsaco')
    shutil.copy2(root / 'dense-attention-packed-pv-swizzle.ttgir', out / 'pv-swizzle.ttgir')
    shutil.copy2(root / 'dense-attention-packed.amdgcn', out / 'dense-attention-aot.amdgcn')
    packed = record.pop('packed_strides')
    record.update(hsaco_sha256=packed['hsaco_sha256'], shared=packed['shared'],
                  source=str(out / 'pv-swizzle.ttgir'))
    asm = (out / 'dense-attention-aot.amdgcn').read_text()
    record['experiment'] = dict(mode='both', compact_args='common-length-fixed-strides',
                                v_layout=[4, 2, 8, [0, 1]],
                                requires_strides=packed['strides'],
                                scope='仅供七 head 的 packed stride 微测，正式推理使用原三镜像构建',
                                isa=dict(Counter(re.findall(
                                    r'^\s*(ds_\w+|v_mmac_\w+|global_\w+|s_barrier|s_waitcnt)\b', asm, re.M))))
    (out / 'dense-attention-aot-build.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
