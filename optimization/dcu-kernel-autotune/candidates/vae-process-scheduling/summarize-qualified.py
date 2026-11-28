#!/usr/bin/env python3
"""核对VAE进程调度的11次完整请求及同实例计时，普通入口另作资格。"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy


ORDER = ["A"] * 8 + ["B", "C", "A"]
VARIANTS = ["off", "on", "off", "on", "on", "off", "off", "on", "on", "on", "on"]
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qualification", type=Path)
    parser.add_argument("accepted", type=Path)
    args = parser.parse_args()
    root, accepted = args.qualification.resolve(strict=True), args.accepted.resolve(strict=True)
    sequence = root / "steady-paired"
    require(Path(str(sequence) + ".exit-code").read_text().strip() == "0", "完整请求进程未正常退出")
    ipc = read(root / "ipc-check.json")
    require(ipc["complete"] and len(ipc["checks"]) == 6 and ipc["worker_exit_codes"] == [0, 0], "IPC资格不完整")
    completed = read(sequence / "complete.json")
    require(completed["sequence"] == ORDER and completed["vae_process_variants"] == VARIANTS
            and completed["variants"] == ["candidate"] * 11 and completed["peer_gate_variants"] is None,
            "完整请求顺序或固定Transformer要求不符")
    manifest = read(sequence / "source/manifest.json")
    for name, expected in dict(H3_PY_ATTENTION_FP32_SCALE="1", H3_VAE_DEVICES="0,1",
                               H3_VAE_PROCESS_WORKERS="7", H3_VAE_DETERMINISTIC_ENCODE="1",
                               H3_ENCODER_INITIAL_CACHE="512", H3_ENCODER_TP_COUNT="4",
                               HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", OMP_NUM_THREADS="4").items():
        require(manifest["environment"].get(name) == expected, f"环境不符：{name}")
    require(not manifest["environment"].get("H3_BENCH_JIT_CACHE"), "不得引入编译缓存优化")
    project = HERE.parents[3]
    for name, expected in manifest["sources"].items():
        require(digest(sequence / "source" / name) == expected, f"源码快照变化：{name}")
        require(digest(project / name) == expected, f"实际运行期间源码变化：{name}")
    builds = manifest["builds"]
    check_identity = runpy.run_path(str(HERE.parent / "peer-sharded-norm-model/model_identity.py"))["check"]
    identity = check_identity(builds["H3_PEER_SHARDED_NORM_BUILD"]["path"], builds["H3_ADALN_BUILD"]["path"],
                              builds["H3_PEER_GATE_BUILD"]["path"], builds["H3_WIDE_NORM_INDEXED_BUILD"]["path"])
    require(manifest["runtime"]["native_sha256"] == identity["native_sha256"], "实际导入native不符")
    for name, build in builds.items():
        for filename, artifact in build["artifacts"].items():
            path = Path(artifact["path"]) if filename == "loaded_native" else Path(build["path"]) / filename
            if isinstance(artifact, dict) and isinstance(artifact.get("sha256"), str):
                require(digest(path) == artifact["sha256"], f"产物改变：{name}/{filename}")
    quality = read(sequence / "sequence-validation.json")
    require(len(quality) == 11, "缺少独立参考质量比较")
    expected = {case: accepted / f"steady-paired/{index:02d}-{case}"
                for case, index in (("A", 1), ("B", 8), ("C", 9))}
    expected_hashes = {case: digest(path / "video.mp4") for case, path in expected.items()}
    expected_counts = {case: read(path / "record.json")["candidate_calls"] for case, path in expected.items()}
    records, requests = [], 0
    for index, case in enumerate(ORDER):
        directory = sequence / f"{index:02d}-{case}"
        row = read(directory / "record.json")
        options, counts = row["command_options"], row["candidate_calls"]
        require(row["case"] == case and row["index"] == index and row["vae_process_variant"] == VARIANTS[index],
                "请求分派不符")
        require(row["transformer_calls"] == 49 and options["steps"] == 50 and options["num_frames"] == 124
                and options["seed"] == 42 and options["gpus"] == "0,1,2,3,4,5,6,7" and not options["legacy"],
                "工作量不符")
        require(not row["diagnostic_capture"] and not row["diagnostic_finite_checks"]
                and row["media"]["video_specs_passed"], "诊断或规格不符")
        require(all(value == expected_counts[case].get(name) for name, value in counts.items()
                    if name != "sharded_replica_checks") and counts.keys() == expected_counts[case].keys(),
                "Transformer完整计算或算子分派与已验收版本不符")
        state = row["peer_gate_stats"]
        require(not state["candidate"]["pending_ranks"] and not state["candidate"]["failed"], "分片计算未完成")
        require(digest(directory / "video.mp4") == expected_hashes[case], "成片与对应已验收输出不一致")
        if case == "A":
            v = row["media"]["video"]
            require((v["width"], v["height"], v["frames"]) == (768, 928, 124), "公开A规格不符")
            require(row["first_frame_sha256"] == "b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54"
                    and options["text"] == "A dog falls asleep", "公开A输入不符")
        requests += int(VARIANTS[index] == "on")
        process = row["vae_process_stats"]
        require(process["requests"] == requests and process["enabled"] == (VARIANTS[index] == "on"), "VAE计算次数不符")
        if process["enabled"]:
            require(len(process["last_samples"]) == 7 and
                    sorted(i for worker in process["last_samples"] for i in worker["clips"]) == list(range(7)),
                    "原7个时间窗未完整计算")
        row.update(output_sha256=expected_hashes[case], warmup=index < 2,
                   ssim_percent=quality[index]["comparison"]["ssim_match_percent"])
        records.append(row)
    require("[vae_process] worker exits [0, 0, 0, 0, 0, 0, 0]" in Path(str(sequence) + ".log").read_text(),
            "VAE子进程未全部正常退出")
    timing = runpy.run_path(str(HERE.parent / "peer-sharded-norm-model/summarize-qualified.py"))["timing"]
    fields = ("generation.pipeline_total", "compute.h3_transformer", "compute.condition_encoder",
              "input.video_vae_encode", "output.video_vae_decode", "output.audio_vae_decode", "output.media_encode")
    result = dict(protocol="minimax-h3-public-a-vae-process-paired-v1", comparison="accepted-sharded-vs-vae-process",
                  metric="wall_seconds", units="seconds", timed_pairs=[[2, 3], [5, 4], [6, 7]],
                  resident_qualification_passed=True, full_release_qualified=False,
                  pending=["普通公开入口新进程", "独立profile/PMC及资源稳定性复核", "独立重建或源码身份复核"],
                  wall=timing(records, lambda row: row["wall_seconds"]),
                  stages={field: timing(records, lambda row: row["stage_seconds"][field]) for field in fields},
                  completed_sequence=ORDER, records=records, environment=manifest["environment"],
                  runtime=manifest["runtime"], identity=identity, all_11_media_byte_equal=True,
                  source_sha256=manifest["sources"], model_load_seconds=completed["model_load_seconds"],
                  resource_note="子进程显存见逐请求样本，整卡采样另存steady-paired.vram.jsonl；父进程allocator与RSS不涵盖子进程")
    (root / "benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["wall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
