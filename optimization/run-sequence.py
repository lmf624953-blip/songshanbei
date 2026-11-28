#!/usr/bin/env python3
"""Run independent FL2VA requests against one model instance for steady-state comparison."""
import argparse
import ctypes
from functools import wraps
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

import torch
from torch.utils._pytree import tree_map
from PIL import Image, ImageChops, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_snapshot import snapshot

PROJECT = Path(__file__).resolve().parents[1]
RUNTIME = PROJECT / "runtime"


def load_entry():
    path = RUNTIME / "h3-runtime/sources/InfiniLM/examples/minimax_h3.py"
    spec = importlib.util.spec_from_file_location("h3_sequence_entry", path)
    entry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = entry
    spec.loader.exec_module(entry)
    return entry


def save_media(entry, result, destination):
    entry.encode_video(
        result["videos"][0],
        fps=entry.FPS,
        output_path=str(destination),
        audio=result["audio"][0],
        audio_sample_rate=result["sampling_rate"],
    )


def make_square_case(path):
    image = Image.new("RGB", (640, 640), "#9ed5f1")
    draw = ImageDraw.Draw(image)
    draw.ellipse((484, 65, 559, 140), fill="#ffd86b")
    draw.rectangle((0, 395, 640, 640), fill="#216a9f")
    draw.polygon([(150, 414), (475, 414), (422, 468), (209, 468)], fill="#8d452e")
    draw.line((325, 155, 325, 414), fill="#423633", width=8)
    draw.polygon([(315, 165), (315, 396), (170, 396)], fill="#f9f6df")
    draw.polygon([(336, 207), (336, 396), (445, 396)], fill="#f6a758")
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sequence", default="A,B,A,A,A")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--variants", help="Comma-separated original/candidate/candidate-tf modes, one per request")
    parser.add_argument("--fast-attention-build", type=Path, help="Optional attention-only override for paired complete requests")
    parser.add_argument("--attention-variants", help="Comma-separated base/fast selections, one per request")
    parser.add_argument("--peer-gate-variants", help="Comma-separated off/on selections for the peer gate candidate")
    parser.add_argument("--vae-process-variants", help="Comma-separated off/on selections for independent VAE processes")
    parser.add_argument("--capture-denoise", action="store_true", help="Save live inputs from the first request; diagnostic timing")
    parser.add_argument("--capture-boundaries", action="store_true", help="Save the first denoising input/output from every request; diagnostic timing")
    parser.add_argument("--check-finite", action="store_true", help="Check all transformer outputs before the scheduler; diagnostic timing")
    args = parser.parse_args()
    if args.capture_denoise and args.capture_boundaries:
        parser.error("Use one capture mode per run")
    order = args.sequence.split(",")
    if not order or any(case not in {"A", "B", "C"} for case in order):
        parser.error("--sequence must contain only comma-separated A, B and C")
    control = sys.modules.get("songshan_candidate_control")
    variants = args.variants.split(",") if args.variants else ["candidate" if control else "original"] * len(order)
    if len(variants) != len(order) or any(v not in {"original", "candidate", "candidate-tf", "candidate-no-reducecast", "candidate-tf-no-reducecast", "candidate-tf-no-encodercast", "candidate-tf-no-tf-reducecast"} for v in variants):
        parser.error("--variants must specify an original/candidate/candidate-tf mode for every request")
    if not control and any(v != "original" for v in variants):
        parser.error("Candidate variants require the native candidate launcher")
    vae_order = args.vae_process_variants.split(",") if args.vae_process_variants else None
    if vae_order is not None:
        if (len(vae_order) != len(order) or any(v not in {"off", "on"} for v in vae_order)
                or not control or not hasattr(control, "set_vae_process_enabled")
                or any(v != "candidate" for v in variants) or args.peer_gate_variants
                or args.fast_attention_build):
            parser.error("VAE process pairing requires off/on for each candidate request and fixed Transformer kernels")
    peer_order = args.peer_gate_variants.split(",") if args.peer_gate_variants else None
    if peer_order is not None:
        if (len(peer_order) != len(order) or any(v not in {"off", "on"} for v in peer_order)
                or not control or not hasattr(control, "set_peer_gate_enabled")
                or any(v != "candidate" for v in variants) or args.fast_attention_build):
            parser.error("Peer gate pairing requires off/on for each candidate request and one fixed attention build")

    attention_order = args.attention_variants.split(",") if args.attention_variants else ["base"] * len(order)
    if len(attention_order) != len(order) or any(v not in {"base", "fast"} for v in attention_order):
        parser.error("--attention-variants must specify base/fast for every request")
    if bool(args.fast_attention_build) != bool(args.attention_variants):
        parser.error("--fast-attention-build and --attention-variants must be used together")
    if args.fast_attention_build and (not control or not hasattr(control, "set_attention_callback")
                                     or any(v == "original" for v in variants)):
        parser.error("Attention pairing requires candidate variants with a registered attention kernel")
    fast_host = fast_callback = None
    if args.fast_attention_build:
        from adaln.aot_runtime import load_attention
        args.fast_attention_build = args.fast_attention_build.resolve(strict=True)
        fast_host = load_attention(args.fast_attention_build)
        fast_callback = ctypes.cast(fast_host.songshan_launch_attention, ctypes.c_void_p)

    entry = load_entry()
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size
    from songshan_h3.output_validation import validate_generated_media

    out = args.out.resolve()
    out.mkdir()
    snapshot(out)
    if args.fast_attention_build:
        path = args.fast_attention_build
        override = dict(path=str(path),
                        metadata=json.loads((path / "dense-attention-aot-build.json").read_text()),
                        sha256={name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                                for name in ("dense-attention-aot.hsaco", "dense-attention-aot.so")},
                        sequence=attention_order)
        (out / "attention-overrides.json").write_text(json.dumps(override, indent=2) + "\n")
    if args.capture_denoise:
        from adaln.capture_denoise import install_capture
        install_capture(out / "denoise-captures")
    capture_reset = None
    if args.capture_boundaries:
        from adaln.capture_denoise import install_capture
        capture_reset = install_capture(out / "boundary-captures", steps=(1,))
    first_frame = RUNTIME / "unified-dog-input/first_frame.png"
    second_frame = out / "square-first-frame.png"
    if {"B", "C"} & set(order):
        make_square_case(second_frame)
    last_frame = out / "square-last-frame.png"
    if "C" in order:
        with Image.open(second_frame) as image:
            ImageChops.offset(image, 64, 0).save(last_frame)
    cases = {
        "A": (first_frame, "A dog falls asleep", None),
        "B": (second_frame, "A small sailboat moves across the water under a bright sun.", None),
        "C": (second_frame, "A small sailboat moves to the right across the water under a bright sun.", str(last_frame)),
    }

    request = argparse.Namespace(
        task="fl2va", model_path=str(args.model.resolve()), last_frame=None,
        gpus="0,1,2,3,4,5,6,7", steps=50, num_frames=124, seed=42,
        height=None, width=None, legacy=args.legacy, legacy_text_gpus=None,
        legacy_transformer_gpus=None, legacy_memory_fraction=0.4,
    )
    devices, text_devices, transformer_devices = entry.validate_args(request)
    model_timings = entry.TimingReport(devices)
    paths = entry.resolve_model_paths(request.model_path, request.task)
    pipe, text_encoder, transformer = entry.load_pipeline(
        paths, request.task, devices, model_timings,
        legacy=args.legacy, legacy_text_devices=text_devices,
        legacy_transformer_devices=transformer_devices,
        legacy_memory_fraction=request.legacy_memory_fraction,
    )
    encoder_capture_counts = {}
    if capture_reset:
        original_encoder_forward = text_encoder.model.forward

        @wraps(original_encoder_forward)
        def capture_encoder(*positional, **kwargs):
            result = original_encoder_forward(*positional, **kwargs)
            directory = capture_reset.current_directory()
            count = encoder_capture_counts.get(directory, 0) + 1
            encoder_capture_counts[directory] = count
            inputs = (positional, kwargs)
            saved = dict(step=0,
                         inputs=tree_map(lambda x: x.detach().cpu() if isinstance(x, torch.Tensor) else x, inputs),
                         devices=tree_map(lambda x: str(x.device) if isinstance(x, torch.Tensor) else x, inputs),
                         outputs=(result.hidden_states[50].detach().cpu(),))
            destination = directory / f"encoder-{count:02d}.pt"
            torch.save(saved, destination)
            print("[diagnostic.encoder_capture]", destination, flush=True)
            return result

        text_encoder.model.forward = capture_encoder
    finite_calls = 0
    if args.check_finite:
        original_forward = transformer.forward

        @wraps(original_forward)
        def checked_forward(*positional, **kwargs):
            nonlocal finite_calls
            result = original_forward(*positional, **kwargs)
            tensors = result if isinstance(result, tuple) else (result.sample, result.audio_sample)
            if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
                raise RuntimeError("Transformer returned non-finite video/audio before scheduling")
            finite_calls += 1
            return result

        transformer.forward = checked_forward

    try:
        for index, case in enumerate(order):
            variant = variants[index]
            if control:
                control.select_variant(variant)
                if attention_order[index] == "fast":
                    control.set_attention_callback(fast_callback)
                if peer_order is not None:
                    control.set_peer_gate_enabled(peer_order[index] == "on")
                if vae_order is not None:
                    control.set_vae_process_enabled(vae_order[index] == "on")
            vae_before = control.vae_process_stats()["requests"] if vae_order is not None else None
            counters_before = control.read_counters() if control else {}
            finite_before = finite_calls
            print(f"[request] index={index} case={case} variant={variant}", flush=True)
            request.first_frame, request.text, request.last_frame = cases[case]
            request.first_frame = str(request.first_frame)
            case_dir = out / f"{index:02d}-{case}"
            case_dir.mkdir()
            if capture_reset:
                capture_reset(out / "boundary-captures" / f"{index:02d}-{case}")
            timings = entry.TimingReport(devices)
            for device in devices:
                torch.cuda.reset_peak_memory_stats(device)
            started_epoch_ns = time.time_ns()
            started = time.perf_counter()
            _, generation, prepared = entry.prepare_generation_inputs(request, timings)
            width, height = prepared["first_image"].size
            height, width = resolve_canvas_size(
                width, height, pipe.canvas_multiple,
                pipe.config.canvas_short_edge, pipe.config.canvas_max_pixels,
            )
            profiler = entry.install_component_profilers(
                timings, text_encoder=text_encoder, transformer=transformer,
                vae=pipe.vae, audio_vae=pipe.audio_vae,
            )
            try:
                with torch.inference_mode(), timings.measure("generation.pipeline_total", synchronize=True):
                    result = pipe(**generation)
            finally:
                profiler.close()
            destination = case_dir / "video.mp4"
            with timings.measure("output.media_encode"):
                save_media(entry, result, destination)
            timings.synchronize()
            elapsed = time.perf_counter() - started
            finished_epoch_ns = time.time_ns()
            del result
            media = validate_generated_media(destination, dict(
                width=width, height=height, num_frames=request.num_frames, fps=entry.FPS,
            ))
            record = {
                "case": case, "index": index, "variant": variant, "command_options": vars(request),
                "first_frame_sha256": hashlib.sha256(Path(request.first_frame).read_bytes()).hexdigest(),
                "last_frame_sha256": hashlib.sha256(Path(request.last_frame).read_bytes()).hexdigest() if request.last_frame else None,
                "wall_seconds": elapsed, "stage_seconds": {key: sum(values) for key, values in timings.values.items()},
                "started_epoch_ns": started_epoch_ns, "finished_epoch_ns": finished_epoch_ns,
                "torch_peak_allocated_bytes": {device: torch.cuda.max_memory_allocated(device) for device in devices},
                "process_peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "transformer_calls": len(timings.values.get("compute.h3_transformer", ())),
                "diagnostic_capture": (args.capture_denoise and index == 0) or args.capture_boundaries,
                "diagnostic_finite_checks": args.check_finite,
                "encoder_capture_calls": encoder_capture_counts.get(capture_reset.current_directory(), 0) if capture_reset else 0,
                "finite_transformer_calls": finite_calls - finite_before,
                "media": media,
            }
            if args.fast_attention_build:
                record["attention_variant"] = attention_order[index]
            if peer_order is not None:
                record["peer_gate_variant"] = peer_order[index]
                record["peer_gate_stats"] = control.peer_gate_stats()
            if vae_order is not None:
                record["vae_process_variant"] = vae_order[index]
                record["vae_process_stats"] = control.vae_process_stats()
                record["peer_gate_stats"] = control.peer_gate_stats()
                expected_requests = int(vae_order[index] == "on")
                if record["vae_process_stats"]["requests"] - vae_before != expected_requests:
                    raise RuntimeError("完整请求没有执行指定VAE进程分支")
            record["candidate_calls"] = {name: count - counters_before[name]
                                         for name, count in (control.read_counters() if control else {}).items()}
            if vae_order is not None:
                control.validate_peer_request("on", record["candidate_calls"], record["peer_gate_stats"])
            if variant == "original" and any(record["candidate_calls"].values()):
                raise RuntimeError("An original request unexpectedly executed candidate kernels")
            if peer_order is not None:
                counts = record["candidate_calls"]
                if hasattr(control, "validate_peer_request"):
                    control.validate_peer_request(peer_order[index], counts, record["peer_gate_stats"])
                elif peer_order[index] == "off" and any(counts.get(name, 0) for name in
                        ("peer_gate", "peer_deferred", "peer_batches", "peer_fused", "peer_fallback")):
                    raise RuntimeError("A disabled peer gate request executed the candidate")
                if peer_order[index] == "on" and (counts.get("peer_gate") != 39200
                        or counts.get("peer_deferred") != 39200 or counts.get("peer_batches") != 4900
                        or counts.get("peer_fused", 0) + counts.get("peer_fallback", 0) != 4900):
                    raise RuntimeError("Peer gate counters do not cover the complete 49-call Transformer")
            if variant in {"candidate-no-reducecast", "candidate-tf-no-reducecast"} and any(record["candidate_calls"].get(name, 0) for name in ("reducecast", "encodercast")):
                raise RuntimeError("The disabled communication precision candidate executed")
            if variant.endswith("no-encodercast") and record["candidate_calls"].get("encodercast", 0):
                raise RuntimeError("The disabled encoder communication candidate executed")
            if variant.endswith("no-tf-reducecast") and record["candidate_calls"].get("reducecast", 0):
                raise RuntimeError("The disabled transformer communication candidate executed")
            if args.check_finite and record["finite_transformer_calls"] != 49:
                raise RuntimeError("The finite-value diagnostic did not check all transformer calls")
            if record["transformer_calls"] != 49:
                raise RuntimeError("The fixed 50-step scheduler did not execute 49 transformer calls")
            (case_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n")
            print(case_dir, elapsed, media["video_specs_passed"], flush=True)
            if not media["video_specs_passed"]:
                raise RuntimeError(f"Media specification failed in {case_dir}")
    finally:
        if not args.legacy:
            transformer.close()
            text_encoder.close()
    (out / "complete.json").write_text(json.dumps({
        "sequence": order, "variants": variants, "attention_variants": attention_order,
        "peer_gate_variants": peer_order,
        "vae_process_variants": vae_order,
        "model_load_seconds": {key: sum(values) for key, values in model_timings.values.items()},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
