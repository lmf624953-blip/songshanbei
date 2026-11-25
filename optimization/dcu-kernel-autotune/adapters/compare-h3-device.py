#!/usr/bin/env python3
"""核对两份 H3 AOT 主机库、ELF 加载区段和映射，允许调试路径不同。"""

import argparse
import hashlib
import json
from pathlib import Path
import struct


def digest(data):
    return hashlib.sha256(data).hexdigest()


def elf_loads(path):
    data = path.read_bytes()
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError(f"只支持 ELF64 little endian：{path}")
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data)
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, header[6] + i * header[11])
                for i in range(header[12])]
    strings = sections[header[13]]
    names = data[strings[4]:strings[4] + strings[5]]
    allocated = {}
    for name, kind, flags, address, offset, size, *_ in sections:
        if flags & 2:
            label = names[name:names.index(b"\0", name)].decode()
            allocated[label] = {"type": kind, "flags": flags, "address": address,
                                "size": size, "sha256": None if kind == 8 else digest(data[offset:offset + size])}
    segments = [list(struct.unpack_from("<IIQQQQQQ", data, header[5] + i * header[9]))
                for i in range(header[10])]
    return {"sha256": digest(data), "sections": allocated, "segments": segments,
            "elf_target": {"machine": header[2], "flags": header[7], "entry": header[4]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("rebuilt", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = [elf_loads(p / "dense-attention-aot.hsaco") for p in (args.original, args.rebuilt)]
    host_digests = [digest((p / "dense-attention-aot.so").read_bytes()) for p in (args.original, args.rebuilt)]
    same_sections = records[0]["sections"] == records[1]["sections"]
    same_mapping = all(records[0][k] == records[1][k] for k in ("segments", "elf_target"))
    passed = same_sections and same_mapping and host_digests[0] == host_digests[1]
    result = {"verdict": "PASS" if passed else "FAIL", "original": str(args.original.resolve()),
              "rebuilt": str(args.rebuilt.resolve()), "loaded_sections_equal": same_sections,
              "mapping_equal": same_mapping, "host_sha256": host_digests,
              "original_device": records[0], "rebuilt_device": records[1]}
    with args.output.open("x") as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("verdict", "loaded_sections_equal", "mapping_equal")}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
