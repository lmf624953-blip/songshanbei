#!/usr/bin/env python3
"""验证真实Popen VAE的尺寸、窗口、线程对照与持续请求资源；非整模型成绩。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import socket
import sys
from time import perf_counter
from types import FunctionType

import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import video_latent_num_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steady-repeats", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() or torch.cuda.device_count() != 8 or args.steady_repeats < 8:
        parser.error("需要八卡、新输出目录和至少八次稳态重复")
    assert os.environ.get("H3_PY_ATTENTION_FP32_SCALE") == "1"
    assert not os.environ.get("H3_BENCH_JIT_CACHE") and not os.environ.get("H3_VAE_CACHE_REPLICAS")
    args.output.mkdir()
    code = Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]).resolve(strict=True)
    candidate = code / "optimization/dcu-kernel-autotune/candidates/vae-process-scheduling"
    sys.path.insert(0, str(candidate))
    sys.path.insert(0, str(code / "optimization"))
    from process_runtime import Workers, core
    from video_decode_parallel import decode_parallel

    paths = [candidate / name for name in ("bench.py", "process_runtime.py", "worker-entry.py")]
    paths += [code / "optimization/video_decode_parallel.py", code / "optimization/adaln/accurate_attention.py",
              Path(__file__).resolve()]
    frozen_source = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    torch.set_num_threads(4)
    core.initialize_attention()
    vae = AutoencoderKLMiniMaxH3.from_pretrained(str(args.model / "vae"), dtype=torch.bfloat16,
                                                local_files_only=True).eval().requires_grad_(False).to("cuda:0")
    devices = tuple(torch.device("cuda", rank) for rank in range(7))
    ratio, chunk, drop = vae.spatial_compression_ratio, vae.tokens_chunk_size, vae.config.token_drop
    temporal = video_latent_num_frames(124, vae.config.clip_length, chunk)
    report = dict(scope="独立VAE原始输出位模式、同七卡线程/进程配对与持续资源检查，不是完整模型计时",
                  complete=False, records=[], thread_pairs=[], stability=[],
                  python_attention_mode="1", compiler_cache_optimization=False,
                  source_sha256=frozen_source,
                  runtime=dict(hostname=socket.gethostname(), python=sys.version,
                               torch_version=torch.__version__, hip_version=torch.version.hip,
                               gpu_name=torch.cuda.get_device_name(0)),
                  settings=dict(tokens_chunk_size=chunk, token_drop=drop, official_A_latent_frames=temporal))
    pool = None

    def save():
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def latent(height, width, frames, seed):
        return torch.randn((1, vae.config.latent_channels, frames, height // ratio, width // ratio),
                           device="cuda:0", dtype=vae.post_quant_conv.weight.dtype,
                           generator=torch.Generator(device="cuda:0").manual_seed(seed))

    def windows(z):
        count = z.shape[2] + drop
        return (count + (-count) % chunk) // chunk - int(drop > 0)

    def measure(name, mode, z, oracle=None, *, strict=True):
        for device in devices:
            torch.cuda.synchronize(device)
        before = pool.request if pool else 0
        started = perf_counter()
        if mode == "process":
            output = process_decode(vae, z, devices, serial=AutoencoderKLMiniMaxH3._decode)
        else:
            output = decode_parallel(vae, z, devices if mode == "thread7" else devices[:2])
        torch.cuda.synchronize(0)
        seconds = perf_counter() - started
        bits = output.view(torch.int16)
        finite = bool(torch.isfinite(output).all())
        exact = oracle is None or (list(output.shape) == list(oracle[0]) and torch.equal(bits, oracle[1]))
        record = dict(name=name, mode=mode, seconds=seconds, latent_shape=list(z.shape),
                      shape=list(output.shape), dtype=str(output.dtype), windows=windows(z),
                      finite=finite, exact=None if oracle is None else exact)
        if oracle is not None and not exact:
            if list(output.shape) == list(oracle[0]):
                record["mismatched_int16_words"] = int(torch.count_nonzero(bits != oracle[1]))
        if mode == "process":
            expected = int(windows(z) >= 2)
            assert pool.request - before == expected
            record["pool_requests"] = pool.request
            record["worker_samples"] = pool.last_samples if expected else []
            if expected:
                clips = sorted(i for sample in pool.last_samples for i in sample["clips"])
                assert clips == list(range(windows(z)))
        report["records"].append(record)
        save()
        print(json.dumps({k: v for k, v in record.items() if k != "worker_samples"}), flush=True)
        if strict:
            assert finite and exact, f"{name} {mode} 完整输出位模式失败"
        return ((list(output.shape), bits.clone()) if oracle is None else None), record

    try:
        with torch.inference_mode():
            base = latent(928, 768, temporal, 43)
            reference, _ = measure("A-warmup", "thread2", base)
            assert reference[0] == [1, 3, 124, 928, 768]
            pool = Workers(args.model, 7, args.output / "workers")
            process_decode = FunctionType(decode_parallel.__code__,
                dict(decode_parallel.__globals__, schedule_clips=pool.schedule),
                decode_parallel.__name__, decode_parallel.__defaults__, decode_parallel.__closure__)
            process_decode.__kwdefaults__ = dict(decode_parallel.__kwdefaults__ or {})
            _, _ = measure("A-process-warmup", "process", base, reference)
            _, thread_warm = measure("A-thread7-warmup", "thread7", base, reference, strict=False)
            thread_eligible = thread_warm["exact"] and thread_warm["finite"]
            for trial in range(3 if thread_eligible else 0):
                sample = {}
                for mode in (("thread7", "process") if trial % 2 == 0 else ("process", "thread7")):
                    _, record = measure(f"A-thread-control-{trial}", mode, base, reference, strict=mode != "thread7")
                    if not record["exact"] or not record["finite"]:
                        thread_eligible = False
                        break
                    sample[mode] = record["seconds"]
                if not thread_eligible:
                    break
                report["thread_pairs"].append(sample)
                save()
            report["thread_comparison"] = dict(eligible=True,
                thread7_median=median(row["thread7"] for row in report["thread_pairs"]),
                process7_median=median(row["process"] for row in report["thread_pairs"]),
                paired_gain_percent=100 * median(1 - row["process"] / row["thread7"]
                                                for row in report["thread_pairs"])) if thread_eligible else dict(
                    eligible=False, reason="七卡线程控制未通过位模式/有限值门槛，不汇总其性能")
            # 清理线程对照遗留的parent空闲缓存。child的工作区与权重不变。
            for device in devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            cases = [("square", 768, 768, temporal, 47),
                     ("one-window", 256, 256, chunk * (1 + int(drop > 0)) - drop, 53),
                     ("two-windows", 768, 768, chunk * (2 + int(drop > 0)) - drop, 59),
                     ("eight-windows", 928, 768, chunk * (8 + int(drop > 0)) - drop, 61),
                     ("changed-A-values", 928, 768, temporal, 71)]
            for name, height, width, frames, seed in cases:
                z = latent(height, width, frames, seed)
                expected, _ = measure(name, "thread2", z)
                _, _ = measure(name, "process", z, expected)
                del z, expected
            _, _ = measure("return-A", "process", base, reference)
            for trial in range(args.steady_repeats):
                _, record = measure(f"steady-A-{trial}", "process", base, reference)
                report["stability"].append(dict(trial=trial,
                    allocated=[sample["memory"]["allocated_bytes"] for sample in record["worker_samples"]],
                    reserved=[sample["memory"]["reserved_bytes"] for sample in record["worker_samples"]]))
                save()
            first = report["stability"][0]["allocated"]
            report["steady_allocated_exactly_stable"] = all(row["allocated"] == first for row in report["stability"])
            assert report["steady_allocated_exactly_stable"], "同输入连续请求的GC后当前分配量仍增长"
            report["numerical_checks_passed"] = True
    finally:
        if pool is not None:
            pool.close()
            report["worker_exit_codes"] = [p.exitcode for p in pool.processes]
        current_source = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        report["source_unchanged"] = current_source == frozen_source
        report["complete"] = (report.get("numerical_checks_passed", False)
                              and report.get("worker_exit_codes") == [0] * 7 and report["source_unchanged"])
        save()
    assert report["complete"]


if __name__ == "__main__":
    main()
