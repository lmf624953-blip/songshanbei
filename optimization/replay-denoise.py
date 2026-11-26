#!/usr/bin/env python3
"""Replay captured real transformer inputs to isolate host and kernel changes."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
from statistics import mean, median
import sys
import time

import torch
from torch.utils._pytree import tree_map
from infinilm.multimodal.minimax_h3 import MiniMaxH3DiffusersAdapter, MiniMaxH3TransformerRunner

sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_snapshot import snapshot


def outputs(result):
    return result if isinstance(result, tuple) else (result.sample, result.audio_sample)


def compare(actual, expected):
    records = []
    for name, x, y in zip(("video", "audio"), outputs(actual), expected, strict=True):
        x = x.detach().cpu()
        delta = (x.float() - y.float()).abs()
        records.append({"name": name, "shape": list(x.shape), "exact": bool(torch.equal(x, y)),
                        "mismatches": int((x != y).sum()), "max_abs_error": float(delta.max()),
                        "mean_abs_error": float(delta.mean()),
                        "rmse": float(delta.double().square().mean().sqrt()),
                        "finite": bool(torch.isfinite(x).all()),
                        "output_sha256": hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--captures", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", default="1,49")
    parser.add_argument("--iterations", default=3, type=int)
    parser.add_argument("--qknorm-ablation", action="store_true")
    parser.add_argument("--qknorm-mrope-ablation", action="store_true")
    parser.add_argument("--wide-ablation", action="store_true", help="Compare wide RMSNorm with original on the same real input")
    parser.add_argument("--reference-results", type=Path, help="Independent PyTorch outputs for these exact captures")
    parser.add_argument("--precision-ablation", action="store_true", help="Compare original, BF16/FP32 communication and optional precision kernels")
    parser.add_argument("--wire-ablation", action="store_true", help="Compare BF16 communication against the registered communication variant")
    parser.add_argument("--short-attention-ablation", action="store_true", help="Compare JIT and AOT for short Transformer attention")
    parser.add_argument("--pv-build", type=Path, help="Optional FP16 PV attention candidate for the precision ablation")
    parser.add_argument("--fast-attention-build", type=Path, help="Optional BF16 PV candidate for the precision ablation")
    parser.add_argument("--profile-once", action="store_true", help="Only one warmed candidate call under hipprof; diagnostic timing")
    parser.add_argument("--profile-drain-seconds", type=float, default=0,
                        help="Wait after the measured window before stopping hipprof")
    parser.add_argument("--elementwise-reference-builds", type=Path, nargs=2, metavar=("WIDE_NORM", "SWIGLU"), help="Compare reference, compact norm, row SwiGLU and their combination on identical inputs")
    parser.add_argument("--swiglu-ablation", action="store_true", help="Compare packed SwiGLU on identical real inputs")
    parser.add_argument("--widenorm-indexed-ablation", action="store_true", help="Compare wide RMSNorm/AdaLN fusion on identical real inputs")
    parser.add_argument("--candidate-only", action="store_true", help="Only exercise the selected build, including its first call")
    parser.add_argument("--indexed-ablation", type=Path, help="Compare the enabled indexed build against this original indexed build")
    args = parser.parse_args()
    if args.iterations < 3:
        parser.error("Use at least three timed trials")
    if args.profile_drain_seconds < 0 or (args.profile_drain_seconds and not args.profile_once):
        parser.error("A nonnegative profile drain delay requires --profile-once")
    control = sys.modules["songshan_candidate_control"]
    pv_host = pv_callback = None
    if args.pv_build:
        if not args.precision_ablation:
            parser.error("--pv-build requires --precision-ablation")
        from adaln.aot_runtime import load_attention
        pv_host = load_attention(args.pv_build.resolve(strict=True))
        pv_callback = ctypes.cast(pv_host.songshan_launch_attention, ctypes.c_void_p)
    fast_host = fast_callback = None
    if args.fast_attention_build:
        if not (args.precision_ablation or args.wire_ablation):
            parser.error("--fast-attention-build requires a precision or wire ablation")
        from adaln.aot_runtime import load_attention
        fast_host = load_attention(args.fast_attention_build.resolve(strict=True))
        fast_callback = ctypes.cast(fast_host.songshan_launch_attention, ctypes.c_void_p)
    old_indexed_callback = None
    if args.indexed_ablation:
        from adaln.indexed_runtime import load_indexed
        old_indexed_host = load_indexed(args.indexed_ablation.resolve(strict=True))
        old_indexed_callback = ctypes.cast(old_indexed_host.songshan_launch_indexed, ctypes.c_void_p)
    old_elementwise = None
    if args.elementwise_reference_builds:
        if args.swiglu_ablation or args.widenorm_indexed_ablation or args.profile_once or args.candidate_only:
            parser.error("elementwise references require a separate four-variant replay")
        from adaln.aot_runtime import load_widenorm_indexed, load_swiglu
        wide,swiglu = (path.resolve(strict=True) for path in args.elementwise_reference_builds)
        old_elementwise = (load_widenorm_indexed(wide), load_swiglu(swiglu))
        old_wide_callback = ctypes.cast(old_elementwise[0].songshan_launch_widenorm_indexed, ctypes.c_void_p)
        old_swiglu_callback = ctypes.cast(old_elementwise[1].songshan_launch_swiglu, ctypes.c_void_p)
    out = args.out.resolve()
    out.mkdir()
    snapshot(out)
    if args.elementwise_reference_builds:
        proof = {}
        for name,path,stem in zip(("wide_norm","swiglu"),args.elementwise_reference_builds,("wide-norm-indexed","swiglu")):
            proof[name] = dict(path=str(path.resolve()),metadata=json.loads((path / (stem+"-build.json")).read_text()),
                              host_sha256=hashlib.sha256((path / (stem+"-host.so")).read_bytes()).hexdigest())
        (out / "elementwise-reference-builds.json").write_text(json.dumps(proof,indent=2)+"\n")
    overrides = {}
    for name, path in (("fp16_pv", args.pv_build), ("bf16_pv_fast", args.fast_attention_build)):
        if path:
            overrides[name] = dict(path=str(path.resolve()),
                                   metadata=json.loads((path / "dense-attention-aot-build.json").read_text()),
                                   host_sha256=hashlib.sha256((path / "dense-attention-aot.so").read_bytes()).hexdigest())
    (out / "attention-overrides.json").write_text(json.dumps(overrides, indent=2) + "\n")
    model = args.model.resolve()
    if (model / "fl2va/transformer").is_dir():
        model = model / "fl2va"
    runner = MiniMaxH3TransformerRunner(model / "transformer", device="nvidia",
                                       tp_device_ids=list(range(8)))
    actor = MiniMaxH3DiffusersAdapter(runner, dtype=torch.bfloat16).eval()
    records = []

    def synchronize():
        for device in range(8):
            torch.cuda.synchronize(device)

    try:
        with torch.inference_mode():
            for step in map(int, args.steps.split(",")):
                path = args.captures / f"step-{step:02d}.pt"
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if args.reference_results:
                    references = json.loads((args.reference_results / "results.json").read_text())
                    matched = next(row for row in references if row["step"] == step)
                    if matched["input_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                        raise RuntimeError("Independent reference used different live inputs")
                positional, kwargs = tree_map(
                    lambda x, d: x.to(d) if isinstance(x, torch.Tensor) else x,
                    saved["inputs"], saved["devices"],
                )
                paired = None
                variants = ("original", "candidate-no-qknorm", "candidate") if args.qknorm_ablation else ("original", "candidate")
                if args.swiglu_ablation:
                    variants = ("candidate-no-swiglu", "candidate")
                if args.widenorm_indexed_ablation:
                    variants = ("candidate-no-widenorm-indexed", "candidate")
                if args.indexed_ablation:
                    variants = ("original", "candidate-old-indexed", "candidate")
                if args.wide_ablation:
                    variants = ("candidate-no-widenorm", "candidate")
                if args.precision_ablation:
                    variants = ("original", "candidate-bf16-wire", "candidate-fp32-wire")
                    if fast_callback:
                        variants += ("candidate-bf16-wire-fast-attention",)
                    if pv_callback:
                        variants += ("candidate-fp32-wire-fp16-pv",)
                    variants += ("candidate-fp32-wire-wide-norm",)
                if args.wire_ablation:
                    variants = ("candidate-tf-no-tf-reducecast", "candidate-tf")
                    if fast_callback:
                        variants += ("candidate-bf16-wire-fast-attention",)
                if args.short_attention_ablation:
                    variants = ("candidate-no-short-attention", "candidate")
                if args.qknorm_mrope_ablation:
                    variants = ("candidate-no-qknorm-mrope", "candidate")
                if args.elementwise_reference_builds:
                    variants = ("candidate-reference-elementwise", "candidate-compact-norm", "candidate-row-swiglu", "candidate")
                if args.profile_once or args.candidate_only:
                    variants = ("candidate",)
                for variant in variants:
                    selected_variant = variant
                    if variant in {"candidate-no-qknorm", "candidate-old-indexed", "candidate-no-widenorm", "candidate-no-short-attention", "candidate-no-qknorm-mrope", "candidate-no-widenorm-indexed", "candidate-no-swiglu"}:
                        selected_variant = "candidate"
                    if args.precision_ablation and variant != "original":
                        selected_variant = "candidate-tf-no-reducecast" if variant.startswith("candidate-bf16-wire") else "candidate-tf"
                    if args.wire_ablation and variant == "candidate-bf16-wire-fast-attention":
                        selected_variant = "candidate-tf-no-tf-reducecast"
                    if args.elementwise_reference_builds:
                        selected_variant = "candidate"
                    control.select_variant(selected_variant)
                    if args.elementwise_reference_builds:
                        if variant in {"candidate-reference-elementwise", "candidate-row-swiglu"}:
                            control.set_widenorm_indexed_callback(old_wide_callback)
                        if variant in {"candidate-reference-elementwise", "candidate-compact-norm"}:
                            control.set_swiglu_callback(old_swiglu_callback)
                    if args.precision_ablation:
                        control.set_widenorm_enabled(variant == "candidate-fp32-wire-wide-norm")
                        if variant == "candidate-fp32-wire-fp16-pv":
                            control.set_attention_callback(pv_callback)
                    if variant == "candidate-bf16-wire-fast-attention":
                        control.set_attention_callback(fast_callback)
                    if args.swiglu_ablation:
                        control.set_swiglu_enabled(variant != "candidate-no-swiglu")
                    if args.widenorm_indexed_ablation:
                        control.set_widenorm_indexed_enabled(variant != "candidate-no-widenorm-indexed")
                    if variant == "candidate-no-widenorm":
                        control.set_widenorm_enabled(False)
                    if variant == "candidate-no-short-attention":
                        control.set_short_attention_enabled(False)
                    if variant == "candidate-no-qknorm":
                        control.set_qknorm_enabled(False)
                    if variant == "candidate-no-qknorm-mrope":
                        control.set_qknorm_mrope_enabled(False)
                    if variant == "candidate-old-indexed":
                        control.set_indexed_callback(old_indexed_callback)
                    before = control.read_counters()
                    started = time.perf_counter()
                    result = actor(*positional, **kwargs)
                    synchronize()
                    warmup_seconds = time.perf_counter() - started
                    samples = []
                    if args.profile_once:
                        hip = ctypes.CDLL("libamdhip64.so")
                        if hip.hipProfilerStart() != 0:
                            raise RuntimeError("hipProfilerStart failed")
                        print("[hipprof.begin]", json.dumps({"name": f"compute.h3_transformer.{step}", "epoch_ns": time.time_ns()}), flush=True)
                    for _ in range(1 if args.profile_once else args.iterations):
                        started = time.perf_counter()
                        result = actor(*positional, **kwargs)
                        synchronize()
                        samples.append(time.perf_counter() - started)
                    if args.profile_once:
                        print("[hipprof.end]", json.dumps({"name": f"compute.h3_transformer.{step}", "epoch_ns": time.time_ns()}), flush=True)
                        if args.profile_drain_seconds:
                            time.sleep(args.profile_drain_seconds)
                        if hip.hipProfilerStop() != 0:
                            raise RuntimeError("hipProfilerStop failed")
                    record = {"step": step, "variant": variant, "warmup_calls": 1,
                              "samples_seconds": samples, "mean_seconds": mean(samples),
                              "median_seconds": median(samples),
                              "warmup_seconds": warmup_seconds,
                              "kernel_calls": {key: count - before[key] for key, count in control.read_counters().items()},
                              "capture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "against_capture": compare(result, saved["outputs"])}
                    record["diagnostic_profiler"] = args.profile_once
                    if args.precision_ablation or args.wire_ablation:
                        record["rejected_nonfinite"] = not all(row["finite"] for row in record["against_capture"])
                    if args.reference_results:
                        expected = torch.load(args.reference_results / f"step-{step:02d}.pt",
                                              map_location="cpu", weights_only=True)
                        record["against_independent_pytorch"] = compare(result, expected)
                    if variant in {"candidate-no-qknorm", "candidate-old-indexed", "candidate-no-widenorm", "candidate-no-short-attention", "candidate-no-qknorm-mrope", "candidate-no-widenorm-indexed", "candidate-no-swiglu"}:
                        paired = tuple(x.detach().cpu().clone() for x in outputs(result))
                    elif variant == "candidate" and paired is not None and not args.elementwise_reference_builds:
                        key = "against_candidate_with_original_indexed" if args.indexed_ablation else "against_candidate_without_qknorm"
                        if args.wide_ablation:
                            key = "against_candidate_without_widenorm"
                        if args.short_attention_ablation:
                            key = "against_candidate_with_jit_short_attention"
                        if args.qknorm_mrope_ablation:
                            key = "against_candidate_without_qknorm_mrope"
                        if args.swiglu_ablation:
                            key = "against_candidate_without_swiglu"
                        if args.widenorm_indexed_ablation:
                            key = "against_candidate_without_widenorm_indexed"
                        record[key] = compare(result, paired)
                    if args.elementwise_reference_builds:
                        if variant == "candidate-reference-elementwise":
                            paired = tuple(x.detach().cpu().clone() for x in outputs(result))
                        else:
                            record["against_elementwise_reference"] = compare(result, paired)
                    print(json.dumps(record), flush=True)
                    records.append(record)
                    (out / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    finally:
        runner.close()
    (out / "complete").write_text("complete\n")


if __name__ == "__main__":
    main()
