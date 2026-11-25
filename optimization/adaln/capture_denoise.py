"""Capture diagnostic live denoising inputs for independent kernel/host replay."""
from functools import wraps
from pathlib import Path

import torch
from torch.utils._pytree import tree_map
from infinilm.multimodal.minimax_h3 import MiniMaxH3DiffusersAdapter


def install_capture(destination, steps=(1, 25, 49)):
    destination = Path(destination)
    destination.mkdir()
    original = MiniMaxH3DiffusersAdapter.forward
    calls = 0

    @wraps(original)
    def capture(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        output = original(self, *args, **kwargs)
        if calls in steps:
            inputs = (args, kwargs)
            devices = tree_map(lambda x: str(x.device) if isinstance(x, torch.Tensor) else x, inputs)
            values = tree_map(lambda x: x.detach().cpu() if isinstance(x, torch.Tensor) else x, inputs)
            tensors = output if isinstance(output, tuple) else (output.sample, output.audio_sample)
            record = {"step": calls, "inputs": values, "devices": devices,
                      "outputs": tuple(t.detach().cpu() for t in tensors)}
            path = destination / f"step-{calls:02d}.pt"
            torch.save(record, path)
            print("[diagnostic.capture]", path, flush=True)
        return output

    MiniMaxH3DiffusersAdapter.forward = capture

    def reset(new_destination):
        nonlocal calls, destination
        calls = 0
        destination = Path(new_destination)
        destination.mkdir()

    reset.current_directory = lambda: destination
    return reset
