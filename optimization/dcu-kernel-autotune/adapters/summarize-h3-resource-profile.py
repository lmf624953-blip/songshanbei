#!/usr/bin/env python3
"""汇总已有 hipprof 分段时间线、设备事件和活跃 wave 原始计数。"""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import runpy
from statistics import mean


def group(row):
    name = row['name']
    if 'dense_attention' in name:
        return '九齿注意力（长短合计）'
    if row['category'] == 'RCCLOPS':
        return 'RCCL'
    if name.startswith('Cijk_') or 'gemm' in name.lower():
        return 'GEMM'
    if 'rmsnormKernel' in name:
        return '宽 RMSNorm'
    if 'indexed_gate' in name:
        return '索引门控'
    if 'indexed_adaln' in name:
        return '索引 AdaLN'
    if 'SwiGLU' in name:
        return 'SwiGLU'
    if 'qk_norm' in name:
        return 'Q/K RMSNorm 与 MRoPE'
    return '其他设备操作'


def summarize(root, code, compute_units):
    log, trace = root / 'replay.log', root / 'hipprof_stream.json'
    module = runpy.run_path(str(code / 'optimization/analyze-hipprof-timeline.py'))
    phase_windows = runpy.run_path(str(code / 'optimization/summarize-profile.py'))['phase_windows']
    payload = json.loads(trace.read_text())
    events = payload['traceEvents'] if isinstance(payload, dict) else payload
    windows, summary, all_rows = [], {}, []
    for start, end, name, _ in phase_windows(log):
        value, rows = module['summarize_window'](events, start, end, name, list(range(8)))
        summary[name] = value
        all_rows.extend(rows)
        groups = defaultdict(float)
        for row in rows:
            if row['device'] == 0:
                groups[group(row)] += row['total_ms']
        windows.append(dict(phase=name, wall_ms=value['wall_ms'], device0_activity_ms=dict(groups),
                            all_cards_idle_ms=value['all_devices_idle']['idle_ms'],
                            all_cards_internal_idle_ms=value['all_devices_idle']['internal_idle_ms']))
    (root / 'timeline-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (root / 'kernel-times.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    wave_rows = list(csv.DictReader((root / 'waves.csv').open()))
    wave_groups = defaultdict(list)
    for row in wave_rows:
        if row['KernelName'] == 'dense_attention_aot_kernel':
            wave_groups[row['grd']].append(float(row['SQ_ACCUM_PREV_HIRES']) /
                                            float(row['GRBM_GUI_ACTIVE']) / compute_units)
    wall = mean(row['wall_ms'] for row in windows)
    names = {name for row in windows for name in row['device0_activity_ms']}
    groups = [dict(name=name, mean_ms=mean(row['device0_activity_ms'].get(name, 0) for row in windows))
              for name in names]
    for item in groups:
        item['pct_of_window'] = item['mean_ms'] / wall * 100
    groups.sort(key=lambda row: -row['mean_ms'])
    return dict(scope='独立 profiler 诊断，设备 0 累计事件耗时除以同步窗口；正式收益使用无 profiler 数据',
                compute_units=compute_units, mean_window_ms=wall, device0_groups=groups,
                waves_per_cu_by_grid={grid: dict(samples=values, mean=mean(values))
                                      for grid, values in wave_groups.items()}, windows=windows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--compute-units', type=int, default=80)
    args = parser.parse_args()
    value = summarize(args.profile, args.code, args.compute_units)
    (args.profile / '去噪算子占比.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
