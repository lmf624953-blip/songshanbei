#!/usr/bin/env python3
"""汇总同实例8次官方A的三组配对；不代替独立的B/C完整资格验收。"""
import argparse
import json
from pathlib import Path
import runpy
from peer_identity import check, digest, read


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("sequence", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    helper = runpy.run_path(str(Path(__file__).with_name("summarize-peer-gate.py")))
    require, timing = helper["require"], helper["timing"]
    root = args.sequence.resolve(strict=True)
    completed = read(root / "complete.json")
    order = helper["PEERS"][:8]
    full_order = completed["sequence"]
    require(full_order in (["A"] * 8, helper["ORDER"])
            and completed["peer_gate_variants"] == helper["PEERS"][:len(full_order)]
            and completed["variants"] == ["candidate"] * len(full_order), "8次官方A配对顺序不符")
    require(Path(str(root) + ".exit-code").read_text().strip() == "0", "完整请求进程失败")
    manifest = read(root / "source/manifest.json")
    builds, env = manifest["builds"], manifest["environment"]
    identity = check(builds["H3_PEER_GATE_BUILD"]["path"], builds["H3_ADALN_BUILD"]["path"])
    require(identity["native_sha256"] == manifest["runtime"]["native_sha256"]
            and identity["nt_sha256"] == builds["H3_PEER_GATE_BUILD"]["artifacts"]["peer-gate.hsaco"]["sha256"],
            "实际运行镜像或原生库与固定版本不同")
    require(builds["H3_ATTENTION_AOT_BUILD"]["artifacts"]["dense-attention-packed.hsaco"]["sha256"] ==
            helper["STABLE_PACKED"], "注意力不是固定前序版本")
    expected_env = dict(H3_PY_ATTENTION_FP32_SCALE="1", H3_VAE_DEVICES="0,1", H3_VAE_DETERMINISTIC_ENCODE="1",
                        H3_ENCODER_INITIAL_CACHE="512", H3_ENCODER_TP_COUNT="4",
                        HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", OMP_NUM_THREADS="4")
    require(all(env.get(k) == v for k, v in expected_env.items()) and not env.get("H3_BENCH_JIT_CACHE"), "复测环境不符")
    preserved = dict(adaln=392, qknorm=1568, qknorm_mrope=19600, attention=19600,
                     short_attention=784, widenorm_indexed=39200, swiglu=20384)
    samples = [json.loads(line) for line in Path(str(root) + ".vram.jsonl").read_text().splitlines()]
    records = []
    for i, peer in enumerate(order):
        directory = root / f"{i:02d}-A"
        row = read(directory / "record.json")
        options, counts, spec = row["command_options"], row["candidate_calls"], row["media"]["video"]
        require(row["index"] == i and row["case"] == "A" and row["peer_gate_variant"] == peer
                and row["transformer_calls"] == 49 and not row["diagnostic_capture"]
                and not row["diagnostic_finite_checks"] and row["media"]["video_specs_passed"], "请求工作量或计时模式不符")
        expected_options = dict(task="fl2va", steps=50, num_frames=124, seed=42, text="A dog falls asleep",
                                gpus="0,1,2,3,4,5,6,7", last_frame=None, height=None, width=None, legacy=False)
        require(all(options[k] == v for k, v in expected_options.items())
                and row["first_frame_sha256"] == helper["FIRST_FRAME"]
                and (spec["width"], spec["height"], spec["frames"], spec["fps"]) == (768, 928, 124, 24), "官方输入或成片规格不符")
        require(all(counts.get(k) == v for k, v in preserved.items()), "完整模型保留计算计数不符")
        expected_peer = dict(peer_gate=39200, peer_deferred=39200, peer_batches=4900, peer_fused=4900,
                             peer_fallback=0, indexed=0) if peer == "on" else dict(
                             peer_gate=0, peer_deferred=0, peer_batches=0, peer_fused=0, peer_fallback=0, indexed=39200)
        require(all(counts.get(k) == v for k, v in expected_peer.items()), "分片门控调用计数不符")
        sha = digest(directory / "video.mp4")
        require(sha == "99d975c6e2537274224d5904eb35dc30ba3f350737ebe333b9d43c61882f0440", "成片与已验收官方A不一致")
        window = [s for s in samples if row["started_epoch_ns"] <= s["epoch_ns"] <= row["finished_epoch_ns"]]
        require(bool(window), "缺少整卡显存采样")
        row.update(output_sha256=sha, warmup=i < 2,
                   sampled_device_peak_mib={d: max(s["used_mib"][d] for s in window) for d in window[0]["used_mib"]})
        records.append(row)
    fields = ("generation.pipeline_total", "compute.h3_transformer", "compute.condition_encoder",
              "input.video_vae_encode", "output.video_vae_decode", "output.audio_vae_decode", "output.media_encode")
    report = dict(protocol="minimax-h3-public-a-peer-gate-paired-v1", comparison="selected-stable-paired-vs-peer-gate",
                  metric="wall_seconds", units="seconds", timed_pairs=helper["PAIRS"],
                  completed_sequence=completed["sequence"], full_qualification=False,
                  summarized_request_indices=list(range(8)),
                  scope="8次官方A性能复测；排除双方各一次预热，B/C资格另见完整11请求验收。",
                  wall=timing(records, lambda r: r["wall_seconds"]),
                  stages={key: timing(records, lambda r: r["stage_seconds"][key]) for key in fields},
                  runtime=manifest["runtime"], environment=env, identity=identity, records=records,
                  model_load_seconds=completed["model_load_seconds"],
                  resource_note="每5秒整卡显存采样；同实例对照可能保留候选工作区，不代表独立进程显存差。")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["wall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
