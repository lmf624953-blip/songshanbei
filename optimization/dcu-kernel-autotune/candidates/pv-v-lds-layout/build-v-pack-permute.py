#!/usr/bin/env python3
"""用八条字节置换构造原 V 的 LDS 数据，不改变 BF16 位模式。"""

import argparse
from pathlib import Path
import runpy


def compact_pack(lines, delayed):
    first = "\tv_lshlrev_b32_e32 v122, 16, v177"
    last = "\tds_write2_b64 v172, v[122:123], v[176:177] offset1:8"
    start = lines.index(first) if not delayed else lines.index(first.replace("v177", "v208"))
    last_source = last.replace("v[176:177]", "v[207:208]") if delayed else last
    end = lines.index(last_source, start) + 1
    region = [line for line in lines[start:end] if not line.lstrip().startswith(".loc")]
    assert len(region) == 19
    sources = list(range(175, 182)) + [104]
    if delayed:
        sources = list(range(206, 214))
    a, b, c, d, e, f, g, h = (f"v{x}" for x in sources)
    pack = lambda dest, high, low, selector: f"\tv_perm_b32 {dest}, {high}, {low}, s{selector}"
    lines[start:end] = [
        pack("v122", c, a, 43),
        pack("v123", g, e, 43),
        pack("v158", c, a, 44),
        pack("v159", g, e, 44),
        "\tv_add_u32_e32 v160, 8192, v134",
        "\tds_write2_b64 v160, v[122:123], v[158:159] offset1:8",
        pack("v122", d, b, 43),
        pack("v123", h, f, 43),
        pack("v176", d, b, 44),
        pack("v177", h, f, 44),
        "\tv_add_u32_e32 v172, 8192, v148",
        last,
    ]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--wait-at", choices=("early", "p-fence", "pack"))
    args = cli.parse_args()
    scripts = Path(__file__).parent
    build = runpy.run_path(str(scripts / "build-disjoint-k-lds.py"))
    lines = build["verified_assembly"]((args.reference / (build["STEM"] + ".amdgcn")).read_text())
    if args.wait_at:
        pipe = runpy.run_path(str(scripts / "build-v-wait-pipeline.py"))
        pipe["register_pipeline"](lines, args.wait_at, 206)
    compact_pack(lines, bool(args.wait_at))
    build["write_variant"](args.reference, args.output, "\n".join(lines) + "\n", 16384, "v_pack_permute",
                           dict(wait_at=args.wait_at, full_block_permutations=8))


if __name__ == "__main__":
    main()
