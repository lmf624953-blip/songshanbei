#!/usr/bin/env python3
"""核对重建的主机库和所有设备可加载区段，保留完整文件哈希。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def allocated_sections(path):
    table = subprocess.check_output(["readelf", "--wide", "--section-headers", str(path)], text=True)
    raw = path.read_bytes()
    result = {}
    for line in table.splitlines():
        match = re.match(
            r"^\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)"
            r"\s+[0-9a-f]+\s+(\S+)\s+", line,
        )
        if not match:
            continue
        name, kind, address, offset, size, flags = match.groups()
        if "A" not in flags:
            continue
        start, length = int(offset, 16), int(size, 16)
        digest = None if kind == "NOBITS" else hashlib.sha256(raw[start:start + length]).hexdigest()
        result[name] = dict(address=address, size=length, type=kind, flags=flags, sha256=digest)
    if not {".text", ".note", ".rodata"} <= result.keys():
        raise RuntimeError(f"Expected GPU executable sections are missing: {path}")
    return result


def compare(left, right):
    a, b = (hashlib.sha256(path.read_bytes()).hexdigest() for path in (left, right))
    row = dict(original=str(left), rebuilt=str(right), original_sha256=a, rebuilt_sha256=b, file_equal=a == b)
    if left.suffix == ".hsaco":
        original_sections, rebuilt_sections = allocated_sections(left), allocated_sections(right)
        row["loaded_sections_equal"] = original_sections == rebuilt_sections
        row["changed_loaded_sections"] = [
            name for name in sorted(original_sections.keys() | rebuilt_sections.keys())
            if original_sections.get(name) != rebuilt_sections.get(name)
        ]
        row["loaded_sections"] = original_sections
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_json", type=Path)
    parser.add_argument("rebuilt", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--short-attention", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config_json.read_text())["environment"]
    native = (args.native or Path(config["H3_ADALN_BUILD"])) / "overlay/infinilm/lib"
    native_artifact, = native.glob("_infinilm*.so")
    rows = [compare(native_artifact, args.rebuilt / "infinilm/overlay/infinilm/lib" / native_artifact.name)]
    builds = {
        "adaln": Path(config["H3_ADALN_AOT_BUILD"]),
        "indexed": Path(config["H3_INDEXED_BUILD"]),
        "qknorm": Path(config["H3_QKNORM_BUILD"]),
        "attention": Path(config["H3_ATTENTION_AOT_BUILD"]),
    }
    if args.short_attention:
        builds["short-attention"] = args.short_attention
    for target, build in builds.items():
        files = sorted(path for path in build.iterdir() if path.suffix in {".hsaco", ".so"})
        if not files:
            raise RuntimeError(f"No binary artifacts in {build}")
        rows.extend(compare(path, args.rebuilt / target / path.name) for path in files)
    passed = all(row.get("loaded_sections_equal", row["file_equal"]) for row in rows)
    proof = dict(source_commit=(args.rebuilt / "source-commit").read_text().strip(),
                 runtime_artifacts_match=passed, artifacts=rows)
    with args.output.open("x") as stream:
        json.dump(proof, stream, indent=2)
        stream.write("\n")
    if not passed:
        raise RuntimeError(f"Rebuilt runtime artifacts differ, see {args.output}")
    print(f"{len(rows)} verified artifacts")


if __name__ == "__main__":
    main()
