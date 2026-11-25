#!/usr/bin/env python3
"""Use hipprof to capture the encoder and two representative denoising calls."""
from collections import Counter
from contextlib import contextmanager
import ctypes
from functools import wraps
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

entry = Path(__file__).resolve().parents[1] / (
    "runtime/h3-runtime/sources/InfiniLM/examples/minimax_h3.py"
)
spec = importlib.util.spec_from_file_location("h3_profiled_entry", entry)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
measure = module.TimingReport.measure
wrap = module.ComponentProfiler.wrap
calls = Counter()
hip = ctypes.CDLL("libamdhip64.so")
roctx = ctypes.CDLL("libroctx64.so")
roctx.roctxRangePushA.argtypes = [ctypes.c_char_p]
stop_after = int(os.environ.get("H3_PROFILE_STOP_AFTER_STEP", "0"))


class ProfileComplete(Exception):
    pass


def capture_call(name, call):
    return name == "compute.condition_encoder" or (
        name == "compute.h3_transformer" and call in (1, 8)
    )


def profile_wrap(profiler, owner, method_name, timing_name):
    wrap(profiler, owner, method_name, timing_name)
    timed = getattr(owner, method_name)

    @wraps(timed)
    def report_shapes(*args, **kwargs):
        call = calls[timing_name] + 1
        if capture_call(timing_name, call):
            inputs = {f"arg_{i}": value for i, value in enumerate(args)}
            inputs.update(kwargs)
            shapes = {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in inputs.items() if module.torch.is_tensor(value)
            }
            print("[hipprof.inputs]", json.dumps({"name": f"{timing_name}.{call}", "tensors": shapes}))
        result = timed(*args, **kwargs)
        if timing_name == "compute.h3_transformer" and stop_after == call:
            raise ProfileComplete
        return result

    setattr(owner, method_name, report_shapes)


@contextmanager
def profile_measure(report, name, *, synchronize=False):
    calls[name] += 1
    call = calls[name]
    if not capture_call(name, call):
        with measure(report, name, synchronize=synchronize):
            yield
        return

    label = f"{name}.{call}"
    if hip.hipProfilerStart() != 0:
        raise RuntimeError(f"Unable to start hipprof for {label}")
    roctx.roctxRangePushA(label.encode())
    try:
        print("[hipprof.begin]", json.dumps({"name": label, "epoch_ns": time.time_ns()}))
        with measure(report, name, synchronize=synchronize):
            yield
    finally:
        print("[hipprof.end]", json.dumps({"name": label, "epoch_ns": time.time_ns()}))
        roctx.roctxRangePop()
        if hip.hipProfilerStop() != 0:
            raise RuntimeError(f"Unable to stop hipprof for {label}")


module.TimingReport.measure = profile_measure
module.ComponentProfiler.wrap = profile_wrap
sys.argv[0] = str(entry)
if stop_after:
    load_pipeline = module.load_pipeline
    native_owners = []

    @wraps(load_pipeline)
    def remember_models(*args, **kwargs):
        pipe, encoder, transformer = load_pipeline(*args, **kwargs)
        if not kwargs["legacy"]:
            native_owners.extend((transformer, encoder))
        return pipe, encoder, transformer

    module.load_pipeline = remember_models
    try:
        module.main()
    except ProfileComplete:
        for owner in native_owners:
            owner.close()
        print(f"[hipprof] Diagnostic capture completed after call {stop_after}", flush=True)
    else:
        raise RuntimeError("The requested diagnostic stop point was not reached")
else:
    module.main()
