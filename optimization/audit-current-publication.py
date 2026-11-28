#!/usr/bin/env python3
"""只读核对已选发布、VAE候选与本地原始证据；不执行推理或替代赛事判定。"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import socket
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "审计输出必须是新文件")
    code = Path(__file__).resolve().parents[1]
    release = args.release.resolve(strict=True)
    experiment = args.experiment.resolve(strict=True)
    prior = read(code / "optimization/当前最优版本核查.json")
    require(digest(args.rules) == prior["rules"]["sha256"], "规则文件与既有核查不一致")
    release_info = read(release / "release.json")
    selection = read(release / "selection.json")
    require(release_info["selected"] and release_info["qualified"], "当前发布未完成选版资格")
    require(digest(release / "selection.json") == release_info["selection_sha256"], "选版记录变化")
    require(digest(release / "candidate.env") == selection["candidate_config_sha256"], "发布配置变化")
    components = release_info["components"]
    candidates = code / "optimization/dcu-kernel-autotune/candidates"
    check = runpy.run_path(str(candidates / "peer-sharded-norm-model/model_identity.py"))["check"]
    identity = check(*(components[name] for name in (
        "H3_PEER_SHARDED_NORM_BUILD", "H3_ADALN_BUILD", "H3_PEER_GATE_BUILD", "H3_WIDE_NORM_INDEXED_BUILD")))
    require(identity == release_info["identity"], "发布实际native/桥/设备镜像变化")
    baseline = "45b2c3094d8d189e030f8984d7c07dead4db690a"
    entries = {}
    for name in ("minimax_h3.py", "minimax_h3_compare.py"):
        relative = "h3/frameworks/InfiniLM/examples/" + name
        original = subprocess.check_output(["git", "show", baseline + ":" + relative], cwd=code)
        require((code / relative).read_bytes() == original, "官方入口或比较脚本被修改：" + name)
        deployed = code / "runtime/h3-runtime/sources/InfiniLM/examples" / name
        require(deployed.read_bytes() == original, "部署入口或比较脚本不符：" + name)
        entries[relative] = digest(code / relative)
    official = read(experiment / "official-summary.json")
    require(official["official_default_passed"] and official["media_byte_equal_to_accepted"], "普通入口资格不完整")
    require(digest(experiment / "official/video.mp4") == official["media_sha256"], "普通入口成片变化")
    resident = read(experiment / "qualification/benchmark.json")
    require(resident["resident_qualification_passed"] and resident["all_11_media_byte_equal"], "常驻资格不完整")
    require(len(resident["records"]) == 11, "完整请求数不符")
    require(not resident["environment"].get("H3_BENCH_JIT_CACHE"), "不得启用额外编译缓存优化")
    for relative, expected in resident["source_sha256"].items():
        require(digest(code / relative) == expected, "运行源码变化：" + relative)
    media = []
    for row in resident["records"]:
        path = experiment / "qualification/steady-paired" / f'{row["index"]:02d}-{row["case"]}' / "video.mp4"
        require(digest(path) == row["output_sha256"], "既有成片变化：" + str(path))
        require(row["transformer_calls"] == 49, "Transformer调用数变化")
        options = row["command_options"]
        require((options["steps"], options["num_frames"], options["seed"]) == (50, 124, 42), "生成工作量变化")
        media.append(dict(index=row["index"], case=row["case"], sha256=row["output_sha256"],
                          transformer_calls=49, ssim_percent=row["ssim_percent"]))
    verification = read(experiment / "verification/result.json")
    require(all(verification[key] for key in ("complete", "numerical_checks_passed", "steady_allocated_exactly_stable", "source_unchanged")),
            "扩展数值或资源资格不完整")
    require(len(verification["records"]) == 52 and len(verification["stability"]) == 32
            and verification["worker_exit_codes"] == [0] * 7, "扩展检查或worker生命周期不符")
    for path, expected in verification["source_sha256"].items():
        require(digest(path) == expected, "独立检查源码变化：" + path)
    trace = experiment / "trace-all"
    capture = read(trace / "capture.json")
    log = (trace / "run.log").read_text(errors="replace")
    require(capture["complete"] and capture["source_unchanged"] and capture["worker_exit_codes"] == [0] * 7,
            "诊断模型运行本身失败")
    result = dict(
        reviewed_at=datetime.now(timezone.utc).isoformat(), node=socket.gethostname(),
        source_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip(),
        scope="只读重核已有完整输出、源码与设备身份；本次未重复推理，未重散列全部模型权重",
        rules_sha256=digest(args.rules), official_entry_and_compare_sha256=entries,
        selected_release=release.name, selected_qualified=True, selected_identity=identity,
        selected_resident_wall=selection["full_A_wall"],
        vae_candidate=dict(resident_passed=True, ordinary_default_passed=True,
            unchanged_functional_sources=len(resident["source_sha256"]), media=media,
            wall=resident["wall"], stages=resident["stages"], extra_numeric_records=52, stable_requests=32,
            selected=False, full_release_qualified=False,
            pending=["有效的独立时间线与PMC", "发布身份与统一复测入口整合"],
            cold_process_wall=official["process_time"]),
        profiler=dict(process_exit_code=int((trace / "exit-code").read_text()),
            model_capture_complete=True, json_export_error="err: invalid_argument" in log,
            diagnostic_qualified=False, raw_database_sha256=digest(trace / "hipprof.db"),
            capture_sha256=digest(trace / "capture.json"), log_sha256=digest(trace / "run.log")),
        rule_review=dict(new_scheduling_violation_found=False,
            findings=["官方入口、固定工作量与逐请求完整输出核对通过",
                      "七进程重新计算当前输入，原_decode_clip及拼接顺序不变；新增调度没有新增设备数学kernel",
                      "既有Diffusers/Torch神经网络VAE明确属于模型计算，不能按媒体辅助工作申报",
                      "九齿编译后端定向变换及既有第三方基础能力需按实际依赖申报许可范围"],
            conclusion="未发现本轮新增宿主调度违反给定条款；不能据本地检查保证全部既有第三方能力已获出题方明确许可或代表赛事合规裁定"))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(selected=result["selected_release"], checked_functional_sources=len(resident["source_sha256"]),
                          checked_media=len(media) + 1, numerical_records=52, stability_requests=32,
                          trace_json_export_error=result["profiler"]["json_export_error"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
