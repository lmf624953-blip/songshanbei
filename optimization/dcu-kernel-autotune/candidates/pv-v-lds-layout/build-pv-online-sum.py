#!/usr/bin/env python3
"""将完整块的部分在线求和移到 PV 的最后一批 MMAC 附近。"""

import argparse
from pathlib import Path
import runpy


MMAC = "\tv_mmac_f32_16x16x16_bf16 "
WAIT = "\ts_waitcnt lgkmcnt(0)"
ASM_START = "\t;;#ASMSTART"
ASM_END = "\t;;#ASMEND"


def transform(lines, mode, references):
    pack = lines.index("\tv_perm_b32 v186, v186, v206, s43")
    ready = lines.index(".Lh3_next_k_ready:", pack)
    mmacs = [i for i in range(pack, ready) if lines[i].startswith(MMAC)]
    assert len(mmacs) == 32
    wait = mmacs[16] - 1
    assert lines[wait] == WAIT and mmacs[16:] == list(range(wait + 1, wait + 17))
    add_begin = next(i for i in range(ready + 1, len(lines)) if not lines[i].lstrip().startswith(".loc"))
    assert all(line.startswith("\tv_add_f32_e32 ") for line in lines[add_begin:add_begin + 8])
    lead = 4 if mode == "adds4" else 8
    adds = lines[add_begin:add_begin + lead]
    if mode.startswith("adds"):
        assert references(adds).isdisjoint(references(lines[wait:ready]))
        del lines[add_begin:add_begin + lead]
        lines[wait:wait] = adds
        return dict(adds=lead, dpp=0)

    first = add_begin + lead
    assert lines[first:first + 2] == [ASM_START, "\ts_nop 1"]
    assert lines[first + 22] == ASM_END
    dpp = lines[first + 2:first + 22]
    updates = lines[first + 23:first + 27]
    assert all(line.startswith("\tv_add_f32_dpp ") for line in dpp)
    assert all(line.startswith("\tv_fma_f32 ") for line in updates)
    crossed = lines[wait:ready]
    assert references(adds + dpp + updates).isdisjoint(references(crossed))
    del lines[add_begin:first + 27]
    if mode == "dpp-front":
        lines[wait:wait] = [*adds, ASM_START, "\ts_nop 1", *dpp, ASM_END, *updates]
    else:
        # The four original PV chains and five DPP phases retain their
        # respective order. The LSU wait still gates the last PV group.
        body = lines[wait + 1:wait + 17]
        scheduled = [*adds, ASM_START, "\ts_nop 1", *dpp[:4], WAIT]
        for phase in range(4):
            scheduled.extend(body[4 * phase:4 * phase + 4])
            scheduled.extend(dpp[4 * phase + 4:4 * phase + 8])
        lines[wait:wait + 17] = [*scheduled, ASM_END, *updates]
    return dict(adds=lead, dpp=len(dpp))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("mode", choices=("adds4", "adds8", "dpp-front", "dpp-split"))
    args = cli.parse_args()
    scripts = Path(__file__).parent
    helper = runpy.run_path(str(scripts / "build-disjoint-k-lds.py"))
    references = runpy.run_path(str(scripts / "build-v-wait-pipeline.py"))["references"]
    lines = helper["verified_assembly"]((args.reference / (helper["STEM"] + ".amdgcn")).read_text())
    details = transform(lines, args.mode, references)
    helper["write_variant"](args.reference, args.output, "\n".join(lines) + "\n", 16384, "pv_online_sum",
                            dict(mode=args.mode, **details))


if __name__ == "__main__":
    main()
