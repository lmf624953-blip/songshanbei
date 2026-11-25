#!/usr/bin/env python3
"""Compare E2E and per-kernel PMC metrics between a round and baseline.

Usage:
  metrics.py <round_dir> <baseline_dir> <kernel-substring>...

Reads <dir>/eval/run.log for E2E metrics and <dir>/pmc/full.txt for per-kernel
PMC stats. E2E extraction is software-specific: the profile's METRIC_PARSER env
var points at a Python module exposing parse_run_log(log_path) -> {label: value};
E2E_METRIC names the primary metric (marked with "*" and summarized in the
headline). PMC parsing is reused from analyze_pmc.parse_pmc.

For each target kernel substring the dominant dispatch variant (largest dispatch
count) is selected, and its stats are reported as medians -- consistent with how
`analyze_pmc.py pmc` presents variants.

Writes the comparison table to <round_dir>/eval/metrics.txt and prints it.
"""
import importlib.util
import os
import statistics
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from analyze_pmc import parse_pmc  # noqa: E402

INT_KEYS = ("vgpr", "scratch", "wg")
STAT_LABELS = {
    "ktime": "ktime (ms)",
    "vgpr": "vgpr",
    "scratch": "scratch (B)",
    "wg": "work-group",
    "l1stall": "L1stall (%)",
    "l2hit": "L2hit (%)",
}


def load_parser(path):
    """Load a Python module from path and return its parse_run_log function."""
    spec = importlib.util.spec_from_file_location("metric_parser", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "parse_run_log"):
        raise SystemExit("METRIC_PARSER %r has no parse_run_log()" % path)
    return mod.parse_run_log


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _dominant(blocks, sub):
    """Pick the dominant dispatch variant for a kernel substring and median its stats."""
    sel = [b for b in blocks if sub in b["name"]]
    if not sel:
        return None
    variants = {}
    for b in sel:
        variants.setdefault(b["name"], []).append(b)
    dominant = max(variants.values(), key=len)
    stats = {}
    for key in ("ktime", "vgpr", "scratch", "wg", "l1stall", "l2hit"):
        stats[key] = _median(b.get(key) for b in dominant)
    stats["dispatches"] = len(dominant)
    return stats


def kernel_stats(round_dir, substrings):
    """Per-substring dominant-variant medians. Prefers pmc/full.txt; falls back
    to the focused pmc/<kernel>.txt produced by `profile ROUND` when the full
    PMC file is absent (e.g. older baseline runs)."""
    result = {}
    full = os.path.join(round_dir, "pmc", "full.txt")
    if not os.path.exists(full):
        full = os.path.join(round_dir, "pmc", "full.csv")
    if os.path.exists(full):
        blocks = parse_pmc(full)
        for sub in substrings:
            result[sub] = _dominant(blocks, sub)
    else:
        for sub in substrings:
            focused = os.path.join(round_dir, "pmc", sub + ".txt")
            if os.path.exists(focused):
                result[sub] = _dominant(parse_pmc(focused), sub)
            else:
                result[sub] = None
    return result


def _fmt(v, digits=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        return "{:.{}f}".format(v, digits)
    return str(v)


def _pct_delta(base, cand):
    if base is None or cand is None or base == 0:
        return None
    return (cand - base) / base * 100.0


def build_table(base_e2e, cand_e2e, e2e_metric):
    """E2E rows for every label the parser returned, marking the primary one."""
    labels = []
    for d in (base_e2e, cand_e2e):
        for k in d:
            if k not in labels:
                labels.append(k)
    lines = ["metric         baseline   candidate   delta"]
    for label in labels:
        b, c = base_e2e.get(label), cand_e2e.get(label)
        speed = _pct_delta(b, c)
        mark = " *" if label == e2e_metric else ""
        lines.append("{:<14} {:>9}   {:>9}   {:>8}{}".format(
            label,
            _fmt(b),
            _fmt(c),
            "{:+.1f}%".format(speed) if speed is not None else "-",
            mark,
        ))
    return "\n".join(lines)


def build_kernel_table(base_stat, cand_stat):
    if not cand_stat:
        return "(no PMC data)"
    lines = []
    lines.append("stat         baseline   candidate   delta")
    for key in ("ktime", "vgpr", "scratch", "wg", "l1stall", "l2hit"):
        label = STAT_LABELS[key]
        b = base_stat.get(key) if base_stat else None
        c = cand_stat.get(key)
        if key == "ktime":
            b = b * 1000.0 if b is not None else None
            c = c * 1000.0 if c is not None else None
            delta = _pct_delta(b, c)
            delta_s = "{:+.1f}%".format(delta) if delta is not None else "-"
        elif key in INT_KEYS:
            b_i = int(b) if b is not None else None
            c_i = int(c) if c is not None else None
            b, c = b_i, c_i
            delta = (c - b) if (b is not None and c is not None) else None
            delta_s = "{:+d}".format(int(delta)) if delta is not None else "-"
        else:
            delta = (c - b) if (b is not None and c is not None) else None
            delta_s = "{:+.2f}".format(delta) if delta is not None else "-"
        lines.append("{:<12} {:>9}   {:>9}   {:>8}".format(label, _fmt(b), _fmt(c), delta_s))
    if cand_stat.get("dispatches"):
        lines.append("dispatches      {}   (dominant variant)".format(cand_stat["dispatches"]))
    return "\n".join(lines)


def main():
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    parser_path = os.environ.get("METRIC_PARSER")
    e2e_metric = os.environ.get("E2E_METRIC", "ns/day")
    measurement_scope = os.environ.get("AUTOTUNE_MEASUREMENT_SCOPE", "E2E")
    if not parser_path or not os.path.exists(parser_path):
        print("error: METRIC_PARSER not set or not found: %r" % parser_path, file=sys.stderr)
        sys.exit(2)
    parse_run_log = load_parser(parser_path)

    round_dir, base_dir = sys.argv[1], sys.argv[2]
    substrings = sys.argv[3:]

    cand_e2e = parse_run_log(os.path.join(round_dir, "eval", "run.log"))
    base_e2e = parse_run_log(os.path.join(base_dir, "eval", "run.log"))
    cand_kern = kernel_stats(round_dir, substrings)
    base_kern = kernel_stats(base_dir, substrings)

    lines = ["# metrics: {} vs {}".format(os.path.basename(round_dir), os.path.basename(base_dir))]
    lines.append("")
    lines.append(f"## {measurement_scope}")
    b = base_e2e.get(e2e_metric)
    c = cand_e2e.get(e2e_metric)
    if b is None or c is None:
        lines.append("primary metric '{}' not found in run log".format(e2e_metric))
    else:
        d = _pct_delta(b, c)
        lines.append("primary metric ({}): baseline {} -> candidate {} ({:+.1f}%)".format(
            e2e_metric, _fmt(b), _fmt(c), d if d is not None else 0.0))
    lines.append(build_table(base_e2e, cand_e2e, e2e_metric))
    for sub in substrings:
        lines.append("")
        lines.append("## Kernel: {}".format(sub))
        lines.append(build_kernel_table(base_kern.get(sub), cand_kern.get(sub)))
    text = "\n".join(lines) + "\n"

    metrics_path = os.path.join(round_dir, "eval", "metrics.txt")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        f.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
