#!/usr/bin/env python3
"""用原始空间 tile 和两份不同 latent 验证单个时间窗的 HIP 图回放。"""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import video_latent_num_frames
from songshan_h3.diffusers_attention import install_video_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() or torch.cuda.device_count() != 1:
        parser.error("需要新输出路径和一张可见卡")

    install_video_attention()
    torch.set_num_threads(4)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(str(args.model / "vae"), dtype=torch.bfloat16,
                                                local_files_only=True).eval().requires_grad_(False).to("cuda:0")
    ratio = vae.spatial_compression_ratio
    latent_frames = video_latent_num_frames(124, vae.config.clip_length, vae.tokens_chunk_size)
    generator = torch.Generator(device="cuda:0").manual_seed(43)
    shape = (1, vae.config.latent_channels, latent_frames, 928 // ratio, 768 // ratio)
    window_frames = vae.tokens_chunk_size + vae.token_overlap
    inputs = [torch.randn(shape, dtype=vae.post_quant_conv.weight.dtype, device="cuda",
                          generator=generator)[:, :, :window_frames] for _ in range(2)]
    static = torch.empty_strided(inputs[0].shape, inputs[0].stride(),
                                 dtype=inputs[0].dtype, device=inputs[0].device)
    static.copy_(inputs[0])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    report = dict(shape=list(inputs[0].shape), strides=list(inputs[0].stride()),
                  tiling=vae.use_tiling, complete=False, validation=[], pairs=[])

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    with torch.inference_mode(), torch.cuda.stream(stream):
        references = [vae._decode_clip(value) for value in inputs]
        stream.synchronize()
        save()
        graph = torch.cuda.CUDAGraph()
        start = perf_counter()
        with torch.cuda.graph(graph, stream=stream):
            output = vae._decode_clip(static)
        stream.synchronize()
        report["capture_seconds"] = perf_counter() - start

        def replay(value):
            static.copy_(value)
            graph.replay()
            return output

        for value, reference in zip(inputs, references, strict=True):
            observed = replay(value)
            stream.synchronize()
            mismatch = int(torch.count_nonzero(reference.view(torch.int16) != observed.view(torch.int16)))
            report["validation"].append(dict(mismatches=mismatch, finite=bool(torch.isfinite(observed).all())))
        report["inputs_produce_different_outputs"] = not torch.equal(*references)
        save()
        if not report["inputs_produce_different_outputs"] or any(
                row["mismatches"] or not row["finite"] for row in report["validation"]):
            raise RuntimeError("图回放逐位校验失败")
        for trial in range(7):
            times = {}
            value = inputs[trial % 2]
            order = ("eager", "graph") if trial % 2 == 0 else ("graph", "eager")
            for side in order:
                start = perf_counter()
                result = vae._decode_clip(value) if side == "eager" else replay(value)
                stream.synchronize()
                times[side] = perf_counter() - start
                del result
            report["pairs"].append(times)
            print(times, flush=True)
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["gain_percent"] = 100 * median(1 - x["graph"] / x["eager"] for x in report["pairs"])
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
