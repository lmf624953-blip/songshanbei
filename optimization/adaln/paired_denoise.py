"""Compare native denoising branches on the same live scheduler inputs."""
from functools import wraps
import json
import sys

import torch
from infinilm.multimodal.minimax_h3 import MiniMaxH3DiffusersAdapter


def tensors(output):
    return output if isinstance(output, tuple) else (output.sample, output.audio_sample)


def compare(first, second):
    results = []
    for name, x, y in zip(("video", "audio"), tensors(first), tensors(second), strict=True):
        equal = torch.equal(x, y)
        result = {"name": name, "shape": list(x.shape), "exact": equal}
        if not equal:
            delta = (x.float() - y.float()).abs()
            result.update(max_abs=float(delta.max()), mean_abs=float(delta.mean()),
                          differences=int((x != y).sum()))
        results.append(result)
    return results


def install_paired_denoise(register, indexed_callback):
    original = MiniMaxH3DiffusersAdapter.forward
    calls = 0

    @wraps(original)
    def paired(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        candidate = original(self, *args, **kwargs)
        if calls not in (1, 8, 25, 49):
            return candidate
        control = sys.modules.get("songshan_candidate_control")
        if control:
            control.select_variant("original")
        else:
            register(None)
        try:
            baseline = original(self, *args, **kwargs)
            repeated = original(self, *args, **kwargs)
        finally:
            if control:
                control.select_variant("candidate")
            else:
                register(indexed_callback)
        results = {"step": calls, "candidate_vs_original": compare(candidate, baseline),
                   "original_vs_repeat": compare(baseline, repeated)}
        print("[adaln.paired]", json.dumps(results), flush=True)
        return candidate

    MiniMaxH3DiffusersAdapter.forward = paired
