#!/usr/bin/env python3
"""Check resident FL2VA requests against the same independent reference clips."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"


def compare_function():
    path = RUNTIME / "h3-runtime/sources/InfiniLM/examples/minimax_h3_compare.py"
    spec = importlib.util.spec_from_file_location("h3_resident_comparator", path)
    comparator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = comparator
    spec.loader.exec_module(comparator)
    return comparator.compare_video_streams


def config(record):
    keys = ("task", "text", "steps", "num_frames", "seed", "height", "width")
    return {key: record["command_options"][key] for key in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--dog-reference", type=Path, required=True)
    parser.add_argument("--square-reference", type=Path)
    parser.add_argument("--last-frame-reference", type=Path)
    args = parser.parse_args()
    candidate = args.candidate.resolve(strict=True)
    order = json.loads((candidate / "complete.json").read_text())["sequence"]
    references = {"A": args.dog_reference.resolve(strict=True)}
    square = None
    if "B" in order:
        if args.square_reference is None:
            parser.error("--square-reference is required for sequence B")
        square = json.loads((args.square_reference / "record.json").read_text())
        if square["case"] != "B" or not square["command_options"]["legacy"]:
            raise RuntimeError("Expected an independently generated legacy B reference")
        references["B"] = (args.square_reference / "video.mp4").resolve(strict=True)
    last_frame_case = None
    if "C" in order:
        if args.last_frame_reference is None:
            parser.error("--last-frame-reference is required for sequence C")
        last_frame_case = json.loads((args.last_frame_reference / "record.json").read_text())
        if last_frame_case["case"] != "C" or not last_frame_case["command_options"]["legacy"]:
            raise RuntimeError("Expected an independently generated legacy C reference")
        references["C"] = (args.last_frame_reference / "video.mp4").resolve(strict=True)
    compare = compare_function()
    audio_spec = importlib.util.spec_from_file_location("h3_audio_comparator", Path(__file__).with_name("compare-audio.py"))
    audio_module = importlib.util.module_from_spec(audio_spec)
    audio_spec.loader.exec_module(audio_module)
    results = []
    for index, case in enumerate(order):
        directory = candidate / f"{index:02d}-{case}"
        recorded = json.loads((directory / "record.json").read_text())
        options = config(recorded)
        if (recorded["case"] != case or recorded["index"] != index or
            any(options[k] != v for k, v in {"steps": 50, "num_frames": 124, "seed": 42}.items())):
            raise RuntimeError(f"Unexpected resident request configuration: {directory}")
        if recorded.get("transformer_calls") != 49:
            raise RuntimeError(f"Unexpected transformer invocation count: {directory}")
        if case in {"A", "B"} and recorded["command_options"]["last_frame"] is not None:
            raise RuntimeError(f"An unexpected last frame was supplied: {directory}")
        if case == "A" and (options["text"] != "A dog falls asleep" or
                            recorded["first_frame_sha256"] != "b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54"):
            raise RuntimeError(f"Public sample input differs: {directory}")
        if case == "B" and (options != config(square) or recorded["first_frame_sha256"] != square["first_frame_sha256"]):
            raise RuntimeError(f"First frame or generation settings differ: {directory}")
        if case == "C" and (options != config(last_frame_case) or
                            recorded["first_frame_sha256"] != last_frame_case["first_frame_sha256"] or
                            not recorded.get("last_frame_sha256") or
                            recorded["last_frame_sha256"] != last_frame_case.get("last_frame_sha256")):
            raise RuntimeError(f"First/last frame or generation settings differ: {directory}")
        video, reference = directory / "video.mp4", references[case]
        if video.samefile(reference):
            raise RuntimeError(f"The reference must be independent: {directory}")
        result = compare(reference, video)
        if not (recorded["media"]["video_specs_passed"] and result["decoded_length_match"]
                and result["metadata_size_match"] and result["shape_mismatches"] == 0
                and result["ssim_match_percent"] is not None):
            raise RuntimeError(f"Video validation failed: {directory}")
        results.append({"case": case, "index": index, "variant": recorded.get("variant", "original"),
                        "wall_seconds": recorded["wall_seconds"],
                        "reference": str(reference), "comparison": result,
                        "audio_comparison": audio_module.compare(reference, video)})
        print(case, index, f"SSIM {result['ssim_match_percent']:.3f}%", flush=True)
    (candidate / "sequence-validation.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
