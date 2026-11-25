#!/usr/bin/env python3
"""从两侧正式常驻测量生成中文对照报告，不混用预热或诊断计时。"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path


def load(path):
    return json.loads((path / "measurement.json").read_text())


def metric(stat):
    return f"{stat['mean']:.3f} ± {stat['stdev']:.3f}"


def sampled_total_peak(run, report):
    measured = set(report["measured_indices"])
    windows = [(row["started_epoch_ns"], row["finished_epoch_ns"])
               for row in report["requests"] if row["index"] in measured]
    devices = set(report["sampled_measured_device_peak_mib"])
    readings = []
    for line in Path(f"{run}.vram.jsonl").read_text().splitlines():
        sample = json.loads(line)
        if set(sample["used_mib"]) != devices:
            continue
        if any(start <= sample["epoch_ns"] <= end for start, end in windows):
            readings.append(sum(sample["used_mib"].values()))
    if not readings:
        raise RuntimeError(f"No complete VRAM samples for {run}")
    return max(readings)


def implementation(run):
    manifest = json.loads((run / "source/manifest.json").read_text())
    builds = manifest["builds"]
    settings = manifest["environment"]
    attention = builds["H3_ATTENTION_AOT_BUILD"]["artifacts"]["dense-attention-aot-build.json"]
    features = [
        f"长注意力 Q/K 分块为 {attention.get('query_block', 64)}×{attention.get('key_block', 64)}，"
        f"{attention['warps']} warps，kpack={attention.get('kpack', 1)}。",
        "索引调制、门控残差、Q/K RMSNorm 和长注意力由九齿编译生成，C++ 负责主机发射。",
    ]
    if attention["fp32_scale"]:
        features.append("长注意力在 QK 累积之后使用 FP32 缩放。")
    if "H3_SHORT_ATTENTION_AOT_BUILD" in builds:
        features.append("短序列条件精炼注意力单独使用 64×64 的九齿 AOT 实现，覆盖形状兼容的输入。")
    for name, component in (("H3_REDUCECAST_BUILD", "Transformer"), ("H3_ENCODER_REDUCECAST_BUILD", "条件编码器")):
        if name in builds:
            dtype = builds[name]["artifacts"]["reducecast-build.json"]["dtype"]
            features.append(f"{component}的选定行并行层使用 {dtype} 通信中间格式。")
    if settings.get("H3_ENCODER_PREFILL") == "1":
        features.append("条件编码器的单样本完整预填充直接使用九齿因果注意力。")
    if capacity := settings.get("H3_ENCODER_INITIAL_CACHE"):
        features.append(f"条件编码器初始 KV 工作区为 {capacity} 个 token，超过容量时按输入长度扩容。")
    if tp_count := settings.get("H3_ENCODER_TP_COUNT"):
        features.append(f"条件编码器使用 {tp_count} 卡张量并行，主 Transformer 继续使用八卡。")
    if devices := settings.get("H3_VAE_DEVICES"):
        features.append(f"原视频 VAE 时域窗口并行安排到设备 {devices}，沿用原网络和拼接。")
    if settings.get("H3_VAE_CACHE_REPLICAS") == "1":
        features.append("重复请求复用通过版本校验的 VAE decoder 权重副本。")
    if settings.get("H3_VAE_DETERMINISTIC_ENCODE") == "1":
        features.append("单帧条件编码使用确定性 DNN，作用域限于关键帧 VAE 编码。")
    return "".join(features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline, candidate = load(args.baseline), load(args.candidate)
    b, c = baseline["stage_seconds"], candidate["stage_seconds"]
    wall_reduction = 100 * (1 - candidate["wall_seconds"]["mean"] / baseline["wall_seconds"]["mean"])
    pipeline_reduction = 100 * (1 - c["generation.pipeline_total"]["mean"] / b["generation.pipeline_total"]["mean"])
    speedup = baseline["wall_seconds"]["mean"] / candidate["wall_seconds"]["mean"]
    dates = [
        datetime.fromtimestamp(run["requests"][0]["started_epoch_ns"] / 1e9).astimezone().isoformat(timespec="seconds")
        for run in (baseline, candidate)
    ]
    baseline_count = len(baseline["requests"])
    matched = list(zip(baseline["requests"], candidate["requests"][:baseline_count], strict=True))
    extra = candidate["requests"][baseline_count:]
    completion = json.loads((args.candidate / "complete.json").read_text())
    accounting = (
        "候选在第六条完整请求之后的附加 C 用例中遇到会话中断。表内六条请求均已完整完成，"
        "三次正式 A 来自同一常驻实例。原进程退出记录为 130，未落盘的加载分项和完整进程墙钟标记为缺失。"
        "C 在新进程补跑并单独记录，附加 C 的计时不进入此表。"
        if completion.get("interrupted") else
        "模型首次加载、进程墙钟、原始分项及每次请求结果均单独保存。"
    )
    lines = ["# MiniMax-H3 决赛优化实测报告", "",
             f"同节点、同八卡、同权重和生成配置，排除每侧第一次预热，默认 A 的正式请求墙钟平均减少 **{wall_reduction:.2f}%**，吞吐比为 **{speedup:.3f}×**；生成管线平均减少 **{pipeline_reduction:.2f}%**。", "",
             "默认输入为官方小狗首帧和 `A dog falls asleep`，画布 768×928、50 steps、124 帧、seed 42。50-step 原调度器每次执行 49 次 Transformer 调用。每个请求都重新执行条件编码、VAE 编码、完整去噪和音视频解码，没有复用生成结果。", "",
             "两侧各用独立常驻实例依次运行 `A,A,A,A,B,A`，第 0 次仅预热，第 1、2、3 次用于下表。第 4 次切换到方形帆船 B，第 5 次返回 A。表中时间为均值 ± 样本标准差，单位为秒。", "",
             f"原版和候选分别于 `{dates[0]}`、`{dates[1]}` 开始运行。候选后续迭代复用已有同条件固定原版测量。", "",
             "| 指标 | 固定原版 | 候选 |", "|---|---:|---:|",
             f"| 请求墙钟，包含媒体封装 | {metric(baseline['wall_seconds'])} | {metric(candidate['wall_seconds'])} |"]
    labels = {"generation.pipeline_total": "生成管线", "compute.h3_transformer": "Transformer", "compute.condition_encoder": "条件编码器",
              "input.video_vae_encode": "视频 VAE 编码", "output.video_vae_decode": "视频 VAE 解码",
              "output.audio_vae_decode": "音频 VAE 解码", "output.media_encode": "媒体封装"}
    lines.extend(f"| {label} | {metric(b[key])} | {metric(c[key])} |" for key, label in labels.items())
    lines += ["", "请求墙钟从输入准备开始，到同步生成和媒体编码结束，包含输入处理、条件编码、完整生成与输出封装，排除模型首次加载及成片检查。正式计时关闭 profiler 和去噪张量采集。" + accounting, "",
              "## 规格与质量", "",
              "下列质量来自未修改的官方视频比较函数及独立 PyTorch 参考；音频另附无相位搜索的逐样本误差诊断。官方规则没有公开可直接换算为总分的数值公式，因此此报告不给出推算比赛分数。", "",
              "| 请求 | 原版平均 SSIM | 候选平均 SSIM | 原版最低帧 | 候选最低帧 |", "|---|---:|---:|---:|---:|"]
    for left, right in matched:
        if (left["case"] != right["case"] or left["first_frame_sha256"] != right["first_frame_sha256"] or
                left.get("last_frame_sha256") != right.get("last_frame_sha256")):
            raise RuntimeError("两侧请求输入不一致")
        lq, rq = left["quality"], right["quality"]
        lines.append(f"| {left['index']} / {left['case']} | {lq['ssim_match_percent']:.3f}% | {rq['ssim_match_percent']:.3f}% | {lq['ssim_min_percent']:.3f}% | {rq['ssim_min_percent']:.3f}% |")
    for right in extra:
        rq = right["quality"]
        lines.append(f"| {right['index']} / {right['case']} | — | {rq['ssim_match_percent']:.3f}% | — | {rq['ssim_min_percent']:.3f}% |")
    if extra:
        lines += ["", "候选额外运行首末帧用例，使用其独立 PyTorch 参考验证规格和质量。"]
    lines += ["", f"默认 A 三次正式输出平均 SSIM：原版 **{metric(baseline['ssim_percent'])}%**，候选 **{metric(candidate['ssim_percent'])}%**。全部请求的媒体规格与 49 次 Transformer 调用核验见原始记录。", "",
              "原版重复生成的 SSIM 有波动，候选的重复性以本表及实际成片哈希为据。算子消融使用同一真实输入的完整 Transformer 输出配对证据。", "",
              "| 请求 | 原版音频相关系数 | 候选音频相关系数 | 原版 SNR / dB | 候选 SNR / dB |", "|---|---:|---:|---:|---:|"]
    def audio_value(record, key):
        value = (record.get("audio_quality") or {}).get(key)
        return "未记录" if value is None else f"{value:.4f}"
    for left, right in matched:
        lines.append(f"| {left['index']} / {left['case']} | {audio_value(left, 'correlation')} | {audio_value(right, 'correlation')} | {audio_value(left, 'snr_db')} | {audio_value(right, 'snr_db')} |")
    for right in extra:
        lines.append(f"| {right['index']} / {right['case']} | — | {audio_value(right, 'correlation')} | — | {audio_value(right, 'snr_db')} |")
    lines += ["", "音轨从独立参考和候选成片解码为 32 kHz 双声道，按原位置逐样本比较，不进行相位搜索或人为时间对齐；AAC 编码和生成本身也影响这些数字。该诊断用于观察音视频质量取舍，不等同于主观听感或比赛的官方音频评分。", "",
              "## 资源与实现来源", "",
              "| GPU | 原版正式请求显存采样最高读数 / MiB | 候选正式请求显存采样最高读数 / MiB |", "|---|---:|---:|"]
    for device in sorted(baseline["sampled_measured_device_peak_mib"]):
        lines.append(f"| {device} | {baseline['sampled_measured_device_peak_mib'][device]} | {candidate['sampled_measured_device_peak_mib'][device]} |")
    total_b = sampled_total_peak(args.baseline, baseline)
    total_c = sampled_total_peak(args.candidate, candidate)
    lines.append(f"| 八卡同一次采样合计的最高读数 | {total_b:,} | {total_c:,} |")
    lines += ["", "显存由 `hy-smi` 每五秒读取整卡已用量，按请求实际时间窗口筛选，覆盖 InfiniCore 原生分配及 PyTorch 分配。该读数是采样最高值，可能漏掉短暂峰值；进程完整采样、首次加载峰值和主机最大 RSS 保留在测量文件中。", "",
              implementation(args.candidate), "",
              "候选版本、二进制哈希、实际加载路径、源码快照和运行命令均随结果保存。", "",
              "## 原始证据", ""]
    for name, run in (("固定原版", args.baseline.resolve()), ("候选", args.candidate.resolve())):
        relative = Path(os.path.relpath(run, args.output.resolve().parent))
        lines.append(f"- {name}：[测量汇总]({relative}/measurement.json)、[规格及质量]({relative}/sequence-validation.json)、[源码清单]({relative}/source/manifest.json)、[运行命令]({relative}.command.txt)、[完整日志]({relative}.log)、[整卡显存采样]({relative}.vram.jsonl)。")
    with args.output.open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
