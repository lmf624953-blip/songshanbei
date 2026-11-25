#!/usr/bin/env python3
"""用独立 PyTorch Transformer 重放真实去噪输入，提供相同输入的精度参照。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils._pytree import tree_map
from diffusers import MiniMaxH3Transformer3DModel

from source_snapshot import snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", default="1,25,49")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir()
    snapshot(out)
    path = Path(__file__).resolve().parents[1] / "runtime/h3-runtime/sources/InfiniLM/examples/minimax_h3.py"
    spec = importlib.util.spec_from_file_location("h3_reference_replay_entry", path)
    entry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = entry
    spec.loader.exec_module(entry)
    root = args.model.resolve()
    if (root / "fl2va/transformer").is_dir():
        root = root / "fl2va"
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        str(root), subfolder="transformer", dtype=torch.bfloat16, device_map="balanced",
        max_memory=entry.make_max_memory(list(range(8)), .4),
        low_cpu_mem_usage=True, local_files_only=True)
    entry.install_legacy_output_device_hooks(model)
    model.requires_grad_(False)
    model.eval()
    records = []
    with torch.inference_mode():
        for step in map(int, args.steps.split(",")):
            capture_path = args.captures / f"step-{step:02d}.pt"
            saved = torch.load(capture_path, map_location="cpu", weights_only=True)
            positional, kwargs = tree_map(
                lambda x, device: x.to(device) if isinstance(x, torch.Tensor) else x,
                saved["inputs"], saved["devices"])
            started = time.perf_counter()
            result = model(*positional, **kwargs)
            tensors = result if isinstance(result, tuple) else (result.sample, result.audio_sample)
            outputs = tuple(value.detach().cpu() for value in tensors)
            for device in range(8):
                torch.cuda.synchronize(device)
            if not all(bool(torch.isfinite(value).all()) for value in outputs):
                raise RuntimeError("Independent PyTorch reference produced non-finite output")
            torch.save(outputs, out / f"step-{step:02d}.pt")
            row = dict(step=step, diagnostic_only=True, seconds=time.perf_counter() - started,
                       input_sha256=hashlib.sha256(capture_path.read_bytes()).hexdigest(),
                       output_shapes=[list(value.shape) for value in outputs])
            records.append(row)
            (out / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    (out / "complete").write_text("complete\n")


if __name__ == "__main__":
    main()
