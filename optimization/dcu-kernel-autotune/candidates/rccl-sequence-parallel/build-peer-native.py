#!/usr/bin/env python3
"""从已验收源码包重建独立InfiniLM overlay；不覆盖正式库或安装目录。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(reference, output, tree, commands, env, elapsed):
    """打包已完成的独立构建；分开调用可修复打包失败，不重复编译。"""
    libraries = list((tree / "build/linux/x86_64/release").glob("_infinilm*.so"))
    if len(libraries) != 1:
        raise ValueError(f"native产物不唯一：{libraries}")
    base_overlay = Path((reference / "native-overlay.txt").read_text().strip())
    overlay = output / "overlay"
    shutil.copytree(base_overlay, overlay, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    native = overlay / "infinilm/lib" / libraries[0].name
    shutil.copy2(libraries[0], native)
    for name in ("adaln-host.so", "source-commit"):
        shutil.copy2(reference / name, output / name)
    (output / "native-overlay.txt").write_text(str(overlay) + "\n")
    record = json.loads((reference / "candidate-build.json").read_text())
    record["reference_native"] = record["candidate_native"]
    record["candidate_native"] = dict(path=str(native), sha256=digest(native))
    record["host_shim"] = dict(path=str(output / "adaln-host.so"), sha256=digest(output / "adaln-host.so"))
    for key, path in (("peer_native_source", tree / "csrc/models/minimax_h3/minimax_h3_transformer.cpp"),
                      ("peer_native_patch", output / "infinilm-peer-gate.patch"),
                      ("peer_compile_log", output / "native-build.log"),
                      ("reference_source_archive", reference / "InfiniLM-applied-source.tar.gz"),
                      ("applied_source_archive", output / "InfiniLM-applied-source.tar.gz")):
        record[key] = dict(path=str(path), sha256=digest(path))
    record["rebuild"] = dict(commands=commands, elapsed_seconds=elapsed,
                             xmake_program_dir=env.get("XMAKE_PROGRAM_DIR"),
                             source_sha256={p.name: digest(p) for p in
                                            (Path(__file__), Path(__file__).with_name("prepare-native-hook.py"),
                                             Path(__file__).with_name("link-peer-native.py"),
                                             Path(__file__).with_name("native-link-order.json"))})
    (output / "candidate-build.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(native=str(native), sha256=digest(native)), ensure_ascii=False))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path, help="含源码包和native-overlay.txt的已验收构建")
    cli.add_argument("output", type=Path, help="尚不存在的构建目录")
    cli.add_argument("--xmake", default=shutil.which("xmake"), help="可复用已有私有xmake")
    cli.add_argument("--jobs", type=int, default=8)
    args = cli.parse_args()
    if not args.xmake or args.jobs < 1:
        cli.error("需要可执行的xmake与正数jobs；源码附带xmake可离线构建")
    args.reference = args.reference.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir()
    source_parent = args.output / "source"
    source_parent.mkdir()
    archive = args.reference / "InfiniLM-applied-source.tar.gz"
    with tarfile.open(archive) as source:
        # 只接受单一源码目录及普通文件；拒绝目录外写入与链接。
        for member in source.getmembers():
            destination = (source_parent / member.name).resolve()
            if not destination.is_relative_to(source_parent) or not (member.isfile() or member.isdir()):
                raise ValueError(f"不支持的源码包成员：{member.name}")
        source.extractall(source_parent)
    tree = source_parent / "InfiniLM"
    here = Path(__file__).resolve().parent
    patch = args.output / "infinilm-peer-gate.patch"
    subprocess.run([sys.executable, str(here / "prepare-native-hook.py"), str(tree), str(patch)], check=True)
    with tarfile.open(args.output / "InfiniLM-applied-source.tar.gz", "w:gz") as target:
        target.add(tree, arcname="InfiniLM")
    env = os.environ.copy()
    import pybind11
    import sysconfig
    env.update(SONGSHAN_PYBIND11_INCLUDE=pybind11.get_include(),
               SONGSHAN_PYTHON_INCLUDE=sysconfig.get_paths()["include"], INFINILM_CXX11_ABI="1")
    commands = [[args.xmake, "f", "-y", "--ccache=n", "-m", "release", "--cxx11-abi=1",
                 "--cc=" + env.get("CC", "gcc"), "--cxx=" + env.get("CXX", "g++")],
                [args.xmake, "build", "-y", "-j" + str(args.jobs), "_infinilm"],
                [sys.executable, str(here / "link-peer-native.py"), str(tree)]]
    started = time.monotonic()
    with (args.output / "native-build.log").open("w") as log:
        for command in commands:
            subprocess.run(command, cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    package(args.reference, args.output, tree, commands, env, time.monotonic() - started)


if __name__ == "__main__":
    main()
