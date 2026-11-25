#!/usr/bin/env python3
"""Parse hipprof trace (kernel.csv) and full PMC (.txt) into readable stats.

Usage:
  analyze_pmc.py trace <kernel.csv> [topN]
      Print the top-N kernels by % of trace GPU time (hotspot identification).

  analyze_pmc.py pmc <full.txt> <kernel-substring> [<kernel-substring>...]
      For each target substring, aggregate that kernel's PMC dispatches and
      print per-variant resource stats (work-group, VGPR, scratch, kernel time
      median, L1 stall, L2 hit).

Example:
  python3 analyze_pmc.py trace rounds/reaxff/round_001/trace/hotspot.kernel.csv
  python3 analyze_pmc.py pmc rounds/reaxff/round_001/pmc/full.txt \
      TagPairReaxComputeTorsionPreprocessed TagPairReaxComputeAngularPreprocessed
"""
import csv
import re
import statistics
import sys


def shorten(name, maxlen=100):
    """Make a (de)mangled hipprof kernel name readable, keeping the Tag* tag."""
    m = re.search(r"LAMMPS_NS::(Tag[A-Za-z_]+<[^>]*>)", name)
    if not m:
        m = re.search(r"LAMMPS_NS::([A-Za-z_]+<[^>]*>|[A-Za-z_]+)", name)
    tag = m.group(1) if m else name
    mb = re.search(r">,\s*(\d+)u,\s*(\d+)u>\(\)$", name)
    suffix = " <{0}u,{1}u>".format(*mb.groups()) if mb else ""
    out = tag + suffix
    return out if len(out) <= maxlen else out[: maxlen - 1] + "…"


def parse_trace(csvpath, top_n):
    rows = []
    with open(csvpath, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            print("no kernel rows in", csvpath)
            return
        names = {name.strip().lower(): index for index, name in enumerate(header)}
        name_i = names.get("name", 0)
        calls_i = names.get("calls", 1)
        total_i = names.get("totaldurationns", names.get("total_ns", 2))
        pct_i = names.get("percentage", 5)
        for row in reader:
            if len(row) <= max(name_i, calls_i, total_i, pct_i) or not row[name_i].strip():
                continue
            if row[name_i].strip().lower() == "total":
                continue
            try:
                pct = float(row[pct_i])
            except ValueError:
                continue
            rows.append((pct, row[name_i], row[calls_i], row[total_i]))
    rows.sort(key=lambda r: -r[0])
    if not rows:
        print("no kernel rows in", csvpath)
        return
    print("{:>11}  {:>7}  {:>13}  kernel".format("%time", "calls", "total_ns"))
    for pct, name, calls, total in rows[:top_n]:
        print("{:10.3f}%  {:>7}  {:>13}  {}".format(pct, calls, total, shorten(name)))
    print("… top {} of {} kernels".format(min(top_n, len(rows)), len(rows)))


def parse_pmc(txtpath):
    if str(txtpath).endswith(".csv"):
        with open(txtpath, newline="") as stream:
            return [
                {
                    "name": row["KernelName"],
                    "ktime": (int(row["EndNs"]) - int(row["BeginNs"])) * 1e-9,
                    "vgpr": int(row["arch_vgpr"]),
                    "scratch": int(row["scr"]),
                    "wg": int(row["wgr"]),
                }
                for row in csv.DictReader(stream) if row.get("KernelName")
            ]
    pats = {
        "ktime": re.compile(r"kernel time\s+([\d.]+)\(s\)"),
        "vgpr": re.compile(r"vgpr count\s+(\d+)"),
        "scratch": re.compile(r"scratch size\s+(\d+)"),
        "wg": re.compile(r"work group size\s+(\d+)"),
        "l1stall": re.compile(r"L1 cache unit is stalled\s+([\d.]+)"),
        "l2hit": re.compile(r"L2 cache hit rate\s+([\d.]+)"),
    }
    blocks = []
    cur = None
    with open(txtpath, errors="replace") as f:
        for line in f:
            m = re.search(r'kernel-name:"([^"]+)"', line)
            if m:
                if cur:
                    blocks.append(cur)
                cur = {"name": m.group(1)}
            elif cur is not None:
                for k, p in pats.items():
                    mm = p.search(line)
                    if mm and k not in cur:
                        cur[k] = float(mm.group(1))
    if cur:
        blocks.append(cur)
    return blocks


def _median(blocks, key):
    vals = [b[key] for b in blocks if key in b]
    return statistics.median(vals) if vals else float("nan")


def summarize_pmc(txtpath, substrings):
    blocks = parse_pmc(txtpath)
    if not blocks:
        print("no PMC blocks parsed from", txtpath)
        return
    for sub in substrings:
        sel = [b for b in blocks if sub in b["name"]]
        if not sel:
            print("\n## {}: no dispatches found".format(sub))
            continue
        variants = {}
        for b in sel:
            variants.setdefault(b["name"], []).append(b)
        print("\n## {}: {} dispatches".format(sub, len(sel)))
        for vname, vblks in sorted(variants.items(), key=lambda kv: -len(kv[1])):
            wg = _median(vblks, "wg")
            vgpr = _median(vblks, "vgpr")
            sc = _median(vblks, "scratch")
            kt = _median(vblks, "ktime")
            l1 = _median(vblks, "l1stall")
            l2 = _median(vblks, "l2hit")
            print(
                "  wg={:>4}  vgpr={:>4}  scratch={:>5}B  ktime={:8.1f}ms  "
                "L1stall={:6.2f}%  L2hit={:6.2f}%   {}".format(
                    int(wg), int(vgpr), int(sc), kt * 1000.0, l1, l2, shorten(vname)
                )
            )


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    mode = sys.argv[1]
    if mode == "trace":
        parse_trace(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 12)
    elif mode == "pmc":
        summarize_pmc(sys.argv[2], sys.argv[3:])
    else:
        print(__doc__, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
