#!/usr/bin/env python3
"""Summarize the official and resident qualification of the DPP arithmetic build."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median, stdev


ORDER = tuple("AAAAAAAABCA")
VARIANTS = ("base", "fast", "base", "fast", "fast", "base", "base", "fast", "fast", "fast", "fast")
PAIRS = ((2, 3), (5, 4), (6, 7))
TRANSFORMER = "compute.h3_transformer"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(samples):
    return dict(samples=samples, mean=mean(samples), median=median(samples), stdev=stdev(samples))


def timing(records, values):
    base = [values(records[a]) for a, _ in PAIRS]
    fast = [values(records[b]) for _, b in PAIRS]
    reductions = [100 * (1 - b / a) for a, b in zip(base, fast, strict=True)]
    return dict(reference=stats(base), candidate=stats(fast), paired_reduction_percent=stats(reductions),
                mean_time_reduction_percent=100 * (1 - mean(fast) / mean(base)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("validation", type=Path)
    cli.add_argument("accepted_steady", type=Path)
    cli.add_argument("accepted_official", type=Path)
    args = cli.parse_args()
    official = args.validation / "official"
    steady = args.validation / "steady-paired"
    assert (args.validation / "qualification-exit-code").read_text().strip() == "0"
    assert (official / "exit-code").read_text().strip() == "0"
    assert (official / "validation-exit-code").read_text().strip() == "0"
    public = read(official / "validation.json")
    accepted = read(args.accepted_official / "validation.json")
    defaults = read(official / "entry-defaults.json")["defaults"]
    assert defaults == {"--steps": 50, "--num-frames": 124, "--seed": 42}
    assert public["local_checks_passed"] and public["transformer_calls"] == 49
    assert public["media"]["video_specs_passed"]
    assert public["comparison"]["ssim_match_percent"] == accepted["comparison"]["ssim_match_percent"]
    official_sha256 = digest(official / "video.mp4")
    assert official_sha256 == digest(args.accepted_official / "video.mp4")

    assert tuple(read(steady / "complete.json")["sequence"]) == ORDER
    assert tuple(read(steady / "attention-overrides.json")["sequence"]) == VARIANTS
    quality = read(steady / "sequence-validation.json")
    assert len(quality) == len(ORDER)
    accepted_requests = {case: args.accepted_steady / f"{index:02d}-{case}"
                         for case, index in (("A", 1), ("B", 4), ("C", 5))}
    hashes = {case: digest(path / "video.mp4") for case, path in accepted_requests.items()}
    accepted_records = {case: read(path / "record.json") for case, path in accepted_requests.items()}
    rows = []
    for index, (case, variant) in enumerate(zip(ORDER, VARIANTS, strict=True)):
        request = steady / f"{index:02d}-{case}"
        record = read(request / "record.json")
        assert record["case"] == case and record["attention_variant"] == variant
        assert record["transformer_calls"] == 49 and record["media"]["video_specs_passed"]
        assert quality[index]["index"] == index and quality[index]["case"] == case
        options = record["command_options"]
        assert (options["steps"], options["num_frames"], options["seed"]) == (50, 124, 42)
        actual = digest(request / "video.mp4")
        assert actual == hashes[case], f"Unexpected media bytes for request {index} ({case}, {variant})"
        expected = accepted_records[case]
        assert record["media"]["video"] == expected["media"]["video"]
        assert record["candidate_calls"] == expected["candidate_calls"]
        attention_calls = record["candidate_calls"]["attention"]
        rows.append(dict(index=index, case=case, attention_variant=variant,
                         wall_seconds=record["wall_seconds"], transformer_seconds=record["stage_seconds"][TRANSFORMER],
                         transformer_calls=record["transformer_calls"], attention_calls=attention_calls,
                         sha256=actual, peak_torch_allocated_bytes=record["torch_peak_allocated_bytes"],
                         independent_quality=quality[index]))
    wall = timing(rows, lambda row: row["wall_seconds"])
    transformer = timing(rows, lambda row: row["transformer_seconds"])
    eligible = (wall["paired_reduction_percent"]["median"] >= 3
                and wall["mean_time_reduction_percent"] >= 3
                and all(gain > 0 for gain in wall["paired_reduction_percent"]["samples"]))
    result = dict(official=dict(validation=public, sha256=official_sha256, entry_defaults=defaults),
                  matched_previously_accepted_media=True, records=rows, timed_pairs=PAIRS,
                  wall=wall, transformer=transformer, eligible=eligible)
    (args.validation / "qualified.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(wall=wall, transformer=transformer, eligible=eligible), indent=2))


if __name__ == "__main__":
    main()
