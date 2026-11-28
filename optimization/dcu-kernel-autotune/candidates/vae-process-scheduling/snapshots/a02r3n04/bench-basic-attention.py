#!/usr/bin/env python3
"""对照两卡线程与多进程 VAE 时间窗分派，保留原解码和拼接。"""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
import traceback
from unittest.mock import patch

import torch
import torch.multiprocessing as mp
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import video_latent_num_frames
from songshan_h3.diffusers_attention import install_video_attention


def worker(model, rank, connection, log):
    with open(log, "a", buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        try:
            torch.set_num_threads(4)
            torch.cuda.set_device(rank)
            install_video_attention()
            vae = AutoencoderKLMiniMaxH3.from_pretrained(
                str(Path(model) / "vae"), dtype=torch.bfloat16,
                local_files_only=True).eval().requires_grad_(False).to(f"cuda:{rank}")
            connection.send(("ready", rank, str(vae.post_quant_conv.weight.dtype)))
            with torch.inference_mode():
                while True:
                    command = connection.recv()
                    if command[0] == "stop":
                        break
                    kind, request, jobs = command
                    assert kind == "run"
                    outputs = []
                    started = perf_counter()
                    for index, window in jobs:
                        decoded = vae._decode_clip(window.to(f"cuda:{rank}"))
                        outputs.append((index, decoded.to("cpu")))
                    torch.cuda.synchronize(rank)
                    connection.send(("done", request, outputs, perf_counter() - started,
                                     torch.cuda.max_memory_allocated(rank)))
        except BaseException:
            message = traceback.format_exc()
            print(message, flush=True)
            try:
                connection.send(("error", rank, message))
            except (BrokenPipeError, EOFError):
                pass
            raise
        finally:
            connection.close()


class Workers:
    def __init__(self, model, count, directory):
        self.connections, self.processes = [], []
        self.request = 0
        self.last_samples = []
        context = mp.get_context("spawn")
        try:
            for rank in range(count):
                parent, child = context.Pipe()
                process = context.Process(target=worker,
                    args=(str(model), rank, child, str(directory / f"worker-{rank}.log")))
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            (directory / "worker-pids.json").write_text(json.dumps([p.pid for p in self.processes]) + "\n")
            for rank in range(count):
                result = self.receive(rank)
                assert result == ("ready", rank, "torch.bfloat16"), result
                print("worker ready", rank, flush=True)
        except BaseException:
            self.close()
            raise

    def receive(self, rank):
        connection = self.connections[rank]
        if not connection.poll(240):
            raise TimeoutError(f"worker {rank} pid {self.processes[rank].pid} 未返回，exit={self.processes[rank].exitcode}")
        result = connection.recv()
        if result[0] == "error":
            raise RuntimeError(str(result))
        return result

    def schedule(self, vae, z, num_chunks, devices, clip_batch):
        assert clip_batch == 1
        count = min(len(devices), num_chunks)
        self.request += 1
        request = self.request
        jobs = [[] for _ in range(count)]
        # 当前请求的输入经 CPU IPC 传递。所有搬运包含在调用计时内。
        torch.cuda.synchronize(z.device)
        host = z.to("cpu")
        for index in range(num_chunks):
            start = index * vae.tokens_chunk_size
            stop = start + vae.tokens_chunk_size + vae.token_overlap
            jobs[index % count].append((index, host[:, :, start:stop].contiguous()))
        for rank in range(count):
            self.connections[rank].send(("run", request, jobs[rank]))
        clips = [None] * num_chunks
        samples = []
        for rank in range(count):
            kind, returned, outputs, seconds, peak = self.receive(rank)
            assert kind == "done" and returned == request
            samples.append(dict(rank=rank, clips=[i for i, _ in outputs], seconds=seconds,
                                torch_peak_allocated_bytes=peak))
            for index, output in outputs:
                assert clips[index] is None
                clips[index] = output.to(z.device)
        assert all(clip is not None for clip in clips)
        self.last_samples = samples
        return clips

    def close(self):
        for process, connection in zip(self.processes, self.connections):
            if process.is_alive():
                try:
                    connection.send(("stop",))
                except (BrokenPipeError, EOFError):
                    pass
        for process in self.processes:
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join(10)
        for connection in self.connections:
            connection.close()


@contextmanager
def selected_schedule(decode, pool, processes):
    if processes:
        with patch.object(decode, "schedule_clips", pool.schedule):
            yield
    else:
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 7])
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or torch.cuda.device_count() != 8:
        parser.error("需要八张可见卡与新输出目录")
    if args.pairs < 1 or any(n < 2 or n > 8 for n in args.workers):
        parser.error("workers须在2到8之间，至少一次配对")
    args.output.mkdir()
    sys.path.insert(0, str(Path(os.environ["H3_AUTOTUNE_CODE_ROOT"]) / "optimization"))
    import video_decode_parallel as decode

    install_video_attention()
    torch.set_num_threads(4)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        str(args.model / "vae"), dtype=torch.bfloat16,
        local_files_only=True).eval().requires_grad_(False).to("cuda:0")
    frames = video_latent_num_frames(124, vae.config.clip_length, vae.tokens_chunk_size)
    ratio = vae.spatial_compression_ratio
    latent = torch.randn((1, vae.config.latent_channels, frames, 928 // ratio, 768 // ratio),
                         dtype=vae.post_quant_conv.weight.dtype, device="cuda:0",
                         generator=torch.Generator(device="cuda:0").manual_seed(43))
    report = dict(scope="独立VAE固定latent：两卡线程对照多进程，包含每请求搬运、解码、汇聚和原拼接",
                  complete=False, records=[], comparisons={}, latent_shape=list(latent.shape),
                  model=str(args.model), compiler_cache_optimization=False)
    pool = None
    reference = None

    def save():
        (args.output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def measure(count, processes, warmup):
        nonlocal reference
        devices = tuple(torch.device("cuda", rank) for rank in range(count))
        for device in devices:
            torch.cuda.synchronize(device)
        started = perf_counter()
        with selected_schedule(decode, pool, processes):
            output = decode.decode_parallel(vae, latent, devices)
            torch.cuda.synchronize(0)
        elapsed = perf_counter() - started
        bits = output.view(torch.int16)
        if reference is None:
            reference = bits.clone()
        exact, finite = torch.equal(reference, bits), bool(torch.isfinite(output).all())
        assert list(output.shape) == [1, 3, 124, 928, 768] and finite
        record = dict(workers=count, processes=processes, warmup=warmup, seconds=elapsed,
                      exact=exact, finite=finite, shape=list(output.shape))
        if processes:
            record["worker_samples"] = pool.last_samples
        if not exact:
            record["mismatches"] = int(torch.count_nonzero(bits != reference))
        report["records"].append(record)
        save()
        print(json.dumps(record), flush=True)
        return elapsed, exact

    try:
        with torch.inference_mode():
            measure(2, False, True)
            pool = Workers(args.model, max(args.workers), args.output)
            for count in dict.fromkeys(args.workers):
                _, exact = measure(count, True, True)
                if not exact:
                    report["comparisons"][str(count)] = dict(eligible=False, reason="完整解码未逐位一致")
                    continue
                pairs = []
                for trial in range(args.pairs):
                    sample = {}
                    for candidate in ((False, True) if trial % 2 == 0 else (True, False)):
                        seconds, exact = measure(count if candidate else 2, candidate, False)
                        assert exact, "重复解码数值不一致"
                        sample["candidate" if candidate else "reference"] = seconds
                    pairs.append(sample)
                report["comparisons"][str(count)] = dict(eligible=True, pairs=pairs,
                    reference_median=median(p["reference"] for p in pairs),
                    candidate_median=median(p["candidate"] for p in pairs),
                    paired_gain_percent=100 * median(1 - p["candidate"] / p["reference"] for p in pairs))
                save()
        report["complete"] = True
    finally:
        if pool is not None:
            pool.close()
            report["worker_exit_codes"] = [process.exitcode for process in pool.processes]
        save()


if __name__ == "__main__":
    main()
