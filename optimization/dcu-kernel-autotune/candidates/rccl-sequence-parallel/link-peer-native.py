#!/usr/bin/env python3
"""使用冻结对象顺序链接实测native；逐对象哈希不符时拒绝声明重建一致。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("tree", type=Path)
    args = cli.parse_args()
    tree = args.tree.resolve(strict=True)
    plan_path = Path(__file__).with_name("native-link-order.json")
    plan = json.loads(plan_path.read_text())
    for row in plan["objects"]:
        digest = hashlib.sha256((tree / row["path"]).read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"编译对象与实测版本不同：{row['path']}")
    candidates = list((tree / "build/linux/x86_64/release").glob("_infinilm*.so"))
    if len(candidates) != 1:
        raise ValueError("native产物不唯一")
    command = [os.environ.get("CXX", "g++"), "-o", str(candidates[0]),
               *(row["path"] for row in plan["objects"]), "-shared", "-m64", "-fPIC",
               "-L" + str(Path(os.environ["INFINI_ROOT"]) / "lib"),
               "-linfinicore_cpp_api", "-linfiniop", "-linfinirt", "-linfiniccl"]
    subprocess.run(command, cwd=tree, check=True)
    digest = hashlib.sha256(candidates[0].read_bytes()).hexdigest()
    if digest != plan["native_sha256"]:
        raise ValueError(f"冻结链接后native仍不同：{digest}")
    print(json.dumps(dict(command=command, sha256=digest,
                         link_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest())))


if __name__ == "__main__":
    main()
