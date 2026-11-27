#!/usr/bin/env python3
"""核验分片门控的官方入口、完整请求配对、质量、来源和整卡显存。"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median, stdev

ORDER = ["A"] * 8 + ["B", "C", "A"]
PEERS = ["off", "on", "off", "on", "on", "off", "off", "on", "on", "on", "on"]
PAIRS = [(2, 3), (5, 4), (6, 7)]
FIRST_FRAME = "b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54"
STABLE_PACKED = "a00cc85245e936216531503c3c1dfb57293d00d907dd517b4b1b6c8932f8a3cd"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stats(values):
    return dict(samples=values, mean=mean(values), median=median(values), stdev=stdev(values))


def timing(records, getter):
    base = [getter(records[a]) for a, _ in PAIRS]
    candidate = [getter(records[b]) for _, b in PAIRS]
    return dict(reference=stats(base), candidate=stats(candidate),
                paired_reduction_percent=stats([100 * (1 - b / a) for a, b in zip(base, candidate)]),
                mean_time_reduction_percent=100 * (1 - mean(candidate) / mean(base)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("qualification", type=Path)
    cli.add_argument("accepted", type=Path)
    args = cli.parse_args()
    root, accepted = args.qualification.resolve(strict=True), args.accepted.resolve(strict=True)
    official, sequence = root / "official", root / "steady-paired"
    for path in (official / "exit-code", official / "validation-exit-code", Path(str(sequence) + ".exit-code")):
        require(path.read_text().strip() == "0", f"运行失败：{path}")
    public = read(official / "validation.json")
    require(read(official / "entry-defaults.json")["defaults"] ==
            {"--steps": 50, "--num-frames": 124, "--seed": 42}, "未使用官方默认规格")
    require(public["local_checks_passed"] and public["transformer_calls"] == 49 and
            digest(official / "video.mp4") == digest(accepted / "official/video.mp4"), "普通入口质量回归失败")
    completed = read(sequence / "complete.json")
    require(completed["sequence"] == ORDER and completed["peer_gate_variants"] == PEERS
            and completed["variants"] == ["candidate"] * 11, "配对或多用例请求不完整")
    manifest = read(sequence / "source/manifest.json")
    env = manifest["environment"]
    for key, value in dict(H3_PY_ATTENTION_FP32_SCALE="1", H3_VAE_DEVICES="0,1",
                           H3_VAE_DETERMINISTIC_ENCODE="1", H3_ENCODER_INITIAL_CACHE="512",
                           H3_ENCODER_TP_COUNT="4", HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", OMP_NUM_THREADS="4").items():
        require(env.get(key) == value, f"环境不符：{key}")
    require(not env.get("H3_BENCH_JIT_CACHE"), "不得引入编译缓存优化")
    for name, build in manifest["builds"].items():
        for filename, artifact in build["artifacts"].items():
            path = Path(artifact["path"]) if filename == "loaded_native" else Path(build["path"]) / filename
            if isinstance(artifact, dict) and isinstance(artifact.get("sha256"), str):
                require(digest(path) == artifact["sha256"], f"运行时产物被修改：{name}/{filename}")
            elif filename.endswith(".json"):
                require(read(path) == artifact, f"运行时元数据被修改：{name}/{filename}")
    native = read(Path(manifest["builds"]["H3_ADALN_BUILD"]["path"]) / "candidate-build.json")
    require(manifest["runtime"]["native_sha256"] == native["candidate_native"]["sha256"], "实际导入native不符")
    attention = manifest["builds"]["H3_ATTENTION_AOT_BUILD"]["artifacts"]
    require(attention["dense-attention-packed.hsaco"]["sha256"] == STABLE_PACKED, "未沿用已验收注意力")
    peer = manifest["builds"]["H3_PEER_GATE_BUILD"]["artifacts"]["peer-runtime-build.json"]
    require(peer["nt"]["tile"] == 1024 and peer["nt"]["warps"] == 4, "九齿候选配置不符")
    quality = read(sequence / "sequence-validation.json")
    require(len(quality) == 11, "缺少独立参考质量比较")
    accepted_order = read(accepted / "steady-paired/complete.json")["sequence"]
    accepted_hashes = {case: digest(accepted / f"steady-paired/{accepted_order.index(case):02d}-{case}/video.mp4")
                       for case in ("A", "B", "C")}
    accepted_counts = {case: read(accepted / f"steady-paired/{accepted_order.index(case):02d}-{case}/record.json")["candidate_calls"]
                       for case in ("A", "B", "C")}
    samples = [json.loads(line) for line in Path(str(sequence) + ".vram.jsonl").read_text().splitlines()]
    records = []
    for i, case in enumerate(ORDER):
        directory = sequence / f"{i:02d}-{case}"
        row = read(directory / "record.json")
        options, counts = row["command_options"], row["candidate_calls"]
        require(row["index"] == i and row["case"] == case and row["peer_gate_variant"] == PEERS[i], "请求分派不符")
        require(row["transformer_calls"] == 49 and options["steps"] == 50 and options["num_frames"] == 124
                and options["seed"] == 42 and options["gpus"] == "0,1,2,3,4,5,6,7" and not options["legacy"], "工作量不符")
        require(not row["diagnostic_capture"] and not row["diagnostic_finite_checks"]
                and row["media"]["video_specs_passed"] and counts["attention"] > 0, "诊断或规格不合格")
        require(all(counts.get(name) == value for name, value in accepted_counts[case].items()
                    if name != "indexed"), "保留的注意力、归约、SwiGLU等计算调用数与已验收版本不符")
        if case == "A":
            spec = row["media"]["video"]
            require(row["first_frame_sha256"] == FIRST_FRAME and options["text"] == "A dog falls asleep"
                    and options["last_frame"] is None and options["height"] is None and options["width"] is None
                    and (spec["width"], spec["height"], spec["frames"], spec["fps"]) == (768, 928, 124, 24), "官方A被改动")
        output_hash = digest(directory / "video.mp4")
        require(output_hash == accepted_hashes[case], f"{i}-{case}成片与已验收版本不一致")
        q = quality[i]
        require(q["index"] == i and q["case"] == case and q["comparison"]["ssim_match_percent"] is not None,
                "独立参考比较不符")
        if PEERS[i] == "on":
            require(counts["peer_gate"] == counts["peer_deferred"] == 39200
                    and counts["peer_batches"] == counts["peer_fused"] == 4900
                    and counts["peer_fallback"] == 0 and counts["indexed"] == 0,
                    "候选未完整覆盖通信及门控，或触发回退")
        else:
            require(all(counts.get(key, 0) == 0 for key in
                        ("peer_gate", "peer_deferred", "peer_batches", "peer_fused", "peer_fallback"))
                    and counts["indexed"] == 39200, "关闭候选的对照不符")
        window = [s for s in samples if row["started_epoch_ns"] <= s["epoch_ns"] <= row["finished_epoch_ns"]]
        require(bool(window), "缺少请求显存采样")
        row.update(output_sha256=output_hash, warmup=i < 2,
                   ssim_percent=q["comparison"]["ssim_match_percent"],
                   sampled_device_peak_mib={device: max(s["used_mib"][device] for s in window)
                                            for device in window[0]["used_mib"]}, vram_samples=len(window))
        records.append(row)
    fields = ("generation.pipeline_total", "compute.h3_transformer", "compute.condition_encoder",
              "input.video_vae_encode", "output.video_vae_decode", "output.audio_vae_decode", "output.media_encode")
    result = dict(protocol="minimax-h3-public-a-peer-gate-paired-v1", comparison="selected-stable-paired-vs-peer-gate",
                  metric="wall_seconds", units="seconds", timed_pairs=PAIRS, completed_sequence=ORDER,
                  wall=timing(records, lambda r: r["wall_seconds"]),
                  stages={field: timing(records, lambda r: r["stage_seconds"][field]) for field in fields},
                  runtime=manifest["runtime"], environment=env, native=native["candidate_native"],
                  peer_build=peer, official_validation=public, matched_previously_accepted_media=True,
                  records=records, model_load_seconds=completed["model_load_seconds"],
                  resource_note="hy-smi每五秒整卡显存采样；同实例关闭候选的请求也可能保留候选工作区，不代表独立进程的显存差。")
    (root / "benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("protocol", "comparison", "wall")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
