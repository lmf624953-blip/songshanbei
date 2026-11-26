#!/usr/bin/env python3
"""对独立 packed PV 镜像做无 profiler 校验，跳过 spills 与相同代码。"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('reference', type=Path)
    cli.add_argument('builds', type=Path)
    cli.add_argument('output', type=Path)
    a = cli.parse_args()
    assert not a.output.exists()
    code = Path(os.environ['H3_AUTOTUNE_CODE_ROOT']) / 'optimization'
    sys.path.insert(0, str(code))
    load = runpy.run_path(str(code / 'bench-attention.py'))['native_launcher']
    screen = runpy.run_path(str(Path(__file__).with_name('screen.py')))['screen']
    framework = Path(__file__).resolve().parents[2]
    elf = runpy.run_path(str(framework / 'adapters/compare-h3-device.py'))['elf_loads']
    reference_sections = elf(a.reference / 'dense-attention-packed.hsaco')['sections']
    baseline = load(a.reference)
    records, winners = [], []
    for f in sorted(a.builds.glob('*/dense-attention-aot-build.json')):
        record = json.loads(f.read_text())
        experiment = record['experiment']
        item = dict(variant=f.parent.name, build=str(f.parent), experiment=experiment)
        same = elf(f.parent / 'dense-attention-packed.hsaco')['sections']['.text'] == reference_sections['.text']
        if experiment['resources']['private_segment_fixed_size']:
            item.update(skipped=True, reason='寄存器溢出，保留资源证据，不占用微测时间')
        elif same:
            item.update(skipped=True, reason='可执行 .text 与对照相同')
        else:
            candidate = load(f.parent)
            cases = [screen(baseline, candidate, n, 11) for n in (27570, 22016)]
            item.update(skipped=False, cases=cases)
            if all(c['passed'] and c['paired_gain_percent'] >= 0.5 for c in cases):
                winners.append(f.parent.name)
        records.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        a.output.write_text(json.dumps(dict(scope='单算子，非端到端验收', records=records,
            winners=winners), ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
