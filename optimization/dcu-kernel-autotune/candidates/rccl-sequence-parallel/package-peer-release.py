#!/usr/bin/env python3
"""将已重建的等价产物打包到稳定目录；真实装载检查后再选为正式配置。"""
import argparse
import json
from pathlib import Path
import shlex
import shutil
from peer_identity import check, digest, read


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("release", type=Path)
    cli.add_argument("qualification", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    source, qualification = args.release.resolve(strict=True), args.qualification.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    assert (qualification / "qualification-exit-code").read_text().strip() == "0"
    benchmark = read(qualification / "benchmark.json")
    assert len(benchmark["records"]) == 11 and benchmark["matched_previously_accepted_media"]
    assert benchmark["wall"]["mean_time_reduction_percent"] > 3
    identity = check(source / "peer", source / "native")
    assert identity["native_sha256"] == benchmark["native"]["sha256"]
    shutil.copytree(source, output, ignore=shutil.ignore_patterns("source", "__pycache__", "*.pyc"))
    native = output / "native"
    overlay = native / "overlay"
    (native / "native-overlay.txt").write_text(str(overlay) + "\n")
    meta = read(native / "candidate-build.json")
    meta["candidate_native"]["path"] = str(next((overlay / "infinilm/lib").glob("_infinilm*.so")))
    for key, filename in (("host_shim", "adaln-host.so"), ("peer_native_patch", "infinilm-peer-gate.patch"),
                          ("peer_compile_log", "native-build.log"), ("applied_source_archive", "InfiniLM-applied-source.tar.gz")):
        meta[key]["path"] = str(native / filename)
        assert digest(native / filename) == meta[key]["sha256"]
    (native / "candidate-build.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    (output / "candidate.env").write_text('# 完整资格验收版本的独立发布配置。\n' +
        'source ' + shlex.quote(str(output / "reference.env")) + '\n' +
        'export H3_ADALN_BUILD=' + shlex.quote(str(native)) + '\n' +
        'export H3_PEER_GATE_BUILD=' + shlex.quote(str(output / "peer")) + '\n' +
        'export H3_PEER_GATE_ENABLED=1\n')
    identity = check(output / "peer", native)
    record = read(output / "release.json")
    record.update(candidate_config=str(output / "candidate.env"), qualified=False,
                  qualification=str(qualification), native_build=meta, identity=identity,
                  wall_seconds=benchmark["wall"]["candidate"]["mean"],
                  reference_wall_seconds=benchmark["wall"]["reference"]["mean"],
                  wall_reduction_percent=benchmark["wall"]["mean_time_reduction_percent"],
                  qualification_note="原测量与完整11请求通过；本目录为已核对加载内容的重建产物，等待实际装载回放。")
    (output / "release.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(output / "candidate.env")


if __name__ == "__main__":
    main()
