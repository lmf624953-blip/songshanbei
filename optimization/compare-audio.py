#!/usr/bin/env python3
"""补充比较音轨的长度、幅度和逐样本误差；不替代官方视频比较。"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import imageio_ffmpeg


def decode(path):
    process = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path), "-map", "0:a:0",
                              "-f", "f32le", "-acodec", "pcm_f32le", "-ar", "32000",
                              "-ac", "2", "pipe:1"], check=True, capture_output=True)
    return np.frombuffer(process.stdout, dtype="<f4").astype(np.float64).reshape(-1, 2)


def compare(reference, candidate):
    left, right = decode(reference), decode(candidate)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise RuntimeError("解码音轨含非有限值")
    count = min(len(left), len(right))
    if not count:
        raise RuntimeError("音轨为空")
    x, y = left[:count], right[:count]
    error = x - y
    signal, noise = float(np.mean(x * x)), float(np.mean(error * error))
    centered_x, centered_y = x - x.mean(axis=0), y - y.mean(axis=0)
    denominator = np.sqrt(np.sum(centered_x ** 2) * np.sum(centered_y ** 2))
    return dict(sample_rate=32000, channels=2, reference_samples=len(left), candidate_samples=len(right),
                sample_length_match=len(left) == len(right), compared_samples=count,
                reference_rms=float(np.sqrt(signal)), candidate_rms=float(np.sqrt(np.mean(y * y))),
                max_absolute_error=float(np.max(np.abs(error))), mean_absolute_error=float(np.mean(np.abs(error))),
                rmse=float(np.sqrt(noise)), snr_db=float(10 * np.log10(signal / noise)) if signal and noise else None,
                correlation=float(np.sum(centered_x * centered_y) / denominator) if denominator else None,
                reference_peak=float(np.max(np.abs(x))), candidate_peak=float(np.max(np.abs(y))),
                note="AAC 解码后逐样本比较，受编码及生成波动影响；没有人为对齐、重采样搜索或相位调整。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
