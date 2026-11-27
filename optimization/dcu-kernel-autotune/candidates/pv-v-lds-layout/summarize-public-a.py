#!/usr/bin/env python3
"""Validate and summarize the reproducible public A attention benchmark."""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys


HERE = Path(__file__).resolve().parent
PAIRING = runpy.run_path(str(HERE / "summarize-softmax-dpp-steady.py"))
VARIANTS = list(PAIRING["VARIANTS"][:8])
PACKED = "dense-attention-packed.hsaco"
BUILD_META = "dense-attention-aot-build.json"
TEXT_HASHES = {
    "base": "c63d3793cfee35a6eeab6c2abf4945796f05edcb7fdeb1158aad462a692a8157",
    "fast": "077490ced4882c937fbc2c7d4982589c8706da1fa726e66a2d759e91555e7700",
}
FIRST_FRAME = "b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54"
REFERENCE_VIDEO = "99d975c6e2537274224d5904eb35dc30ba3f350737ebe333b9d43c61882f0440"
RUN_ENV = {
    "H3_PY_ATTENTION_FP32_SCALE": "1",
    "H3_VAE_DEVICES": "0,1",
    "H3_VAE_DETERMINISTIC_ENCODE": "1",
    "H3_ENCODER_INITIAL_CACHE": "512",
    "H3_ENCODER_TP_COUNT": "4",
    "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    "OMP_NUM_THREADS": "4",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_environment(environment):
    for name, expected in RUN_ENV.items():
        require(environment.get(name) == expected, f"配对测量需要 {name}={expected}")
    require(not environment.get("H3_BENCH_JIT_CACHE"), "配对测量使用未配置 H3_BENCH_JIT_CACHE 的构建")


def images(base, fast):
    framework = HERE.parents[1]
    elf_loads = runpy.run_path(str(framework / "adapters/compare-h3-device.py"))["elf_loads"]
    result = {}
    for variant, path in (("base", base), ("fast", fast)):
        metadata = PAIRING["read"](path / BUILD_META)
        packed = metadata["packed_strides"]
        actual = elf_loads(path / PACKED)
        require(actual["sections"][".text"]["sha256"] == TEXT_HASHES[variant],
                f"{variant} 的 packed 指令与公布成绩所用版本不同：{path}")
        require(packed["strides"] == [896, 896, 2688, 896]
                and packed["shared"] == (8192 if variant == "base" else 16384)
                and packed["hsaco_sha256"] == actual["sha256"],
                f"{variant} 的 packed 元数据与产物不符：{path}")
        result[variant] = {"path": str(path.resolve()), "packed_sha256": actual["sha256"],
                           "packed_text_sha256": actual["sections"][".text"]["sha256"],
                           "packed_lds_bytes": packed["shared"]}
    return result


def report(sequence):
    read, digest, timing = (PAIRING[key] for key in ("read", "digest", "timing"))
    complete = read(sequence / "complete.json")
    require(complete["sequence"] == ["A"] * len(VARIANTS)
            and complete["attention_variants"] == VARIANTS, "常驻请求的顺序或镜像配对不符")
    manifest = read(sequence / "source/manifest.json")
    overrides = read(sequence / "attention-overrides.json")
    base = manifest["builds"]["H3_ATTENTION_AOT_BUILD"]
    identity = images(Path(base["path"]), Path(overrides["path"]))
    require(identity["base"]["packed_sha256"] == base["artifacts"][PACKED]["sha256"]
            and identity["fast"]["packed_sha256"] == overrides["metadata"]["packed_strides"]["hsaco_sha256"]
            and overrides["sequence"] == VARIANTS, "当前产物与运行时快照不符")
    validate_environment(manifest["environment"])
    expected = {"task": "fl2va", "steps": 50, "num_frames": 124, "seed": 42,
                "gpus": "0,1,2,3,4,5,6,7", "text": "A dog falls asleep",
                "legacy": False, "last_frame": None, "height": None, "width": None}
    rows = []
    for index, variant in enumerate(VARIANTS):
        directory = sequence / f"{index:02d}-A"
        row = read(directory / "record.json")
        spec = row["media"]["video"]
        require(row["index"] == index and row["case"] == "A"
                and row["variant"] == "candidate" and row["attention_variant"] == variant,
                f"请求 {index} 的镜像分派不符")
        require(all(row["command_options"][key] == value for key, value in expected.items())
                and row["first_frame_sha256"] == FIRST_FRAME and row["transformer_calls"] == 49
                and row["media"]["video_specs_passed"]
                and (spec["width"], spec["height"], spec["frames"], spec["fps"]) == (768, 928, 124, 24),
                f"请求 {index} 的官方输入或成片规格不符")
        require(not row["diagnostic_capture"] and not row["diagnostic_finite_checks"]
                and row["candidate_calls"]["attention"] > 0, f"请求 {index} 的计时模式不符")
        rows.append(dict(index=index, attention_variant=variant, warmup=index < 2,
                         wall_seconds=row["wall_seconds"], stage_seconds=row["stage_seconds"],
                         transformer_calls=row["transformer_calls"], candidate_calls=row["candidate_calls"],
                         output_sha256=digest(directory / "video.mp4")))
    require(all(row["candidate_calls"] == rows[0]["candidate_calls"] for row in rows),
            "各次请求的算子调用数不一致")
    require(len({row["output_sha256"] for row in rows}) == 1, "同进程配对的成片哈希不一致")
    stages = {stage: timing(rows, lambda row: row["stage_seconds"][stage])
              for stage in rows[2]["stage_seconds"]}
    return dict(protocol="minimax-h3-public-a-resident-paired-v1",
                metric="wall_seconds", units="seconds", timed_pairs=PAIRING["PAIRS"],
                runtime=manifest["runtime"], environment=manifest["environment"], images=identity,
                input=dict(expected, width=768, height=928, transformer_calls=49,
                           first_frame_sha256=FIRST_FRAME),
                model_load_seconds=complete["model_load_seconds"], requests=rows,
                output_matches_reference_sha256=rows[0]["output_sha256"] == REFERENCE_VIDEO,
                wall=timing(rows, lambda row: row["wall_seconds"]), stages=stages)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    subparsers = cli.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("base", type=Path)
    check.add_argument("fast", type=Path)
    check.add_argument("--first-frame", type=Path)
    summarize = subparsers.add_parser("report")
    summarize.add_argument("sequence", type=Path)
    summarize.add_argument("output", type=Path)
    args = cli.parse_args()
    try:
        if args.command == "check":
            images(args.base, args.fast)
            if args.first_frame:
                require(PAIRING["digest"](args.first_frame) == FIRST_FRAME, "首帧与官方公开 A 不符")
                validate_environment(os.environ)
            return
        result = report(args.sequence)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        wall = result["wall"]
        print(f"wall_seconds {wall['reference']['mean']:.3f} -> {wall['candidate']['mean']:.3f} s "
              f"({wall['mean_time_reduction_percent']:.3f}% less)")
        print(args.output)
    except (OSError, ValueError, KeyError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
