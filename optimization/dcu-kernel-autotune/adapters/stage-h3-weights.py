#!/usr/bin/env python3
"""把模型完整复制到节点本地盘，逐文件核对 SHA-256；不改动源权重。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def model_files(root):
    # Hugging Face 下载缓存不是模型输入，可能由权重目录所有者创建且不可读。
    paths = []
    for directory, children, names in os.walk(root):
        children[:] = [name for name in children if not name.startswith(".")]
        paths.extend(Path(directory) / name for name in names if not name.startswith("."))
    return sorted(p for p in paths if p.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    destination = args.destination.absolute()
    if not source.is_dir() or destination.exists() or not 1 <= args.workers <= 8:
        parser.error("源必须是目录，目标必须不存在，workers 必须为 1–8")
    files = model_files(source)
    total = sum(p.stat().st_size for p in files)
    if not files or shutil.disk_usage(destination.parent).free < total + 10 * 1024**3:
        parser.error("源目录为空或目标盘剩余空间不足（预留 10 GiB）")
    destination.mkdir()
    started = time.time()
    report = {"source": str(source), "destination": str(destination), "bytes": total,
              "files": [], "verified": False, "started": started}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    def copy_and_verify(path):
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = path.stat()
        original = hashlib.sha256()
        with path.open("rb") as reader, target.open("xb") as writer:
            while chunk := reader.read(16 * 1024**2):
                original.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"复制期间源文件发生变化：{relative}")
        copied = hashlib.sha256()
        with target.open("rb") as reader:
            while chunk := reader.read(16 * 1024**2):
                copied.update(chunk)
        if original.digest() != copied.digest() or target.stat().st_size != before.st_size:
            raise RuntimeError(f"目标校验失败：{relative}")
        shutil.copystat(path, target)
        result = {"path": str(relative), "bytes": before.st_size, "sha256": original.hexdigest()}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            report["files"] = list(pool.map(copy_and_verify, files))
        # 不把文件复制期间新增或删除的内容静默漏掉。
        if [str(p.relative_to(source)) for p in model_files(source)] != [r["path"] for r in report["files"]]:
            raise RuntimeError("复制期间源目录的文件列表发生变化")
        report["verified"] = True
    finally:
        report["elapsed_s"] = time.time() - started
        args.manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
