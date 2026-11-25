#!/usr/bin/env python3
"""核验兼容入口的公开样例成片，调用原官方视频比较函数。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

from songshan_h3.output_validation import validate_generated_media


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("first_frame", type=Path)
    args = parser.parse_args()
    first_hash = hashlib.sha256(args.first_frame.read_bytes()).hexdigest()
    if first_hash != "b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54":
        raise RuntimeError("The public sample first frame differs")
    log = (args.run / "run.log").read_text()
    if not re.search(r"compute\.h3_transformer\s+[\d.]+ s \(49 calls,", log):
        raise RuntimeError("Expected all 49 Transformer calls from the 50-step schedule")
    runtime = Path(__file__).resolve().parents[1] / "runtime"
    comparator_path = runtime / "h3-runtime/sources/InfiniLM/examples/minimax_h3_compare.py"
    spec = importlib.util.spec_from_file_location("h3_official_comparator", comparator_path)
    comparator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = comparator
    spec.loader.exec_module(comparator)
    video = args.run / "video.mp4"
    media = validate_generated_media(video, {"width": 768, "height": 928, "num_frames": 124, "fps": 24})
    comparison = comparator.compare_video_streams(args.reference, video)
    if not (media["video_specs_passed"] and comparison["decoded_length_match"]
            and comparison["metadata_size_match"] and comparison["shape_mismatches"] == 0
            and comparison["ssim_match_percent"] is not None):
        raise RuntimeError("Public sample media validation failed")
    result = {"local_checks_passed": True, "first_frame_sha256": first_hash, "transformer_calls": 49,
              "reference": str(args.reference.resolve()), "media": media, "comparison": comparison}
    (args.run / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"公开样例规格通过，平均 SSIM {comparison['ssim_match_percent']:.3f}%")


if __name__ == "__main__":
    main()
