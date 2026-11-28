#!/usr/bin/env python3
"""Register the experimental AdaLN host shim and run the original entry."""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
from types import ModuleType

from infinilm.lib import _infinilm

here = Path(__file__).resolve().parent
build = Path(os.environ["H3_ADALN_BUILD"]).resolve(strict=True)
native = Path(_infinilm.__file__).resolve()
if not native.is_relative_to(Path((build / "native-overlay.txt").read_text().strip()).resolve()):
    raise RuntimeError(f"Candidate InfiniLM library is not loaded: {native}")
spec = importlib.util.spec_from_file_location("songshan_adaln_kernel", here / "adaln_kernel.py")
kernel = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = kernel
spec.loader.exec_module(kernel)
kernel.kernel()

aot_build = os.environ.get("H3_ADALN_AOT_BUILD")
if aot_build:
    aot_build = Path(aot_build).resolve(strict=True)
    host = ctypes.CDLL(str(aot_build / "adaln-aot-host.so"))
    prepare = host.songshan_prepare_adaln_aot
    prepare.argtypes = [ctypes.c_char_p]
    prepare.restype = ctypes.c_char_p
    error = prepare(os.fsencode(aot_build / "adaln-aot.hsaco"))
    if error:
        raise RuntimeError(f"Preparing the AdaLN device binary failed: {error.decode()}")
    aot_calls = host.songshan_adaln_aot_calls
    aot_calls.argtypes = []
    aot_calls.restype = ctypes.c_uint64
    callback = host.songshan_launch_adaln_aot
    print("[adaln] device binary", aot_build / "adaln-aot.hsaco", flush=True)
else:
    host = ctypes.CDLL(str(build / "adaln-host.so"))
    callback = host.songshan_launch_adaln
infinilm = ctypes.CDLL(str(native))
register = infinilm.songshan_set_adaln_launch
register.argtypes = [ctypes.c_void_p]
register.restype = None
calls = infinilm.songshan_fused_adaln_calls
calls.argtypes = []
calls.restype = ctypes.c_uint64
register(ctypes.cast(callback, ctypes.c_void_p))
indexed_build = os.environ.get("H3_INDEXED_BUILD")
audit_path = os.environ.get("H3_INDEXED_AUDIT")
if indexed_build:
    sys.path.insert(0, str(here))
    from indexed_runtime import load_indexed
    indexed_build = Path(indexed_build).resolve(strict=True)
    indexed_host = load_indexed(indexed_build)
    indexed_register = infinilm.songshan_set_indexed_launch
    indexed_register.argtypes, indexed_register.restype = [ctypes.c_void_p], None
    indexed_calls = infinilm.songshan_indexed_calls
    indexed_calls.argtypes, indexed_calls.restype = [], ctypes.c_uint64
    indexed_callback = ctypes.cast(indexed_host.songshan_launch_indexed, ctypes.c_void_p)
    if audit_path:
        audit = ctypes.CDLL(audit_path)
        audit_init = audit.songshan_init_indexed_audit
        audit_init.argtypes, audit_init.restype = [ctypes.c_void_p], None
        audit_init(indexed_callback)
        indexed_callback = ctypes.cast(audit.songshan_audit_indexed, ctypes.c_void_p)
        audit_checked = audit.songshan_indexed_audit_checked
        audit_checked.argtypes, audit_checked.restype = [], ctypes.c_uint64
        audit_incorrect = audit.songshan_indexed_audit_incorrect
        audit_incorrect.argtypes, audit_incorrect.restype = [], ctypes.c_uint64
        print("[adaln] indexed audit", audit_path, flush=True)
    indexed_register(indexed_callback)
    if os.environ.get("H3_INDEXED_PAIRED") == "1":
        from paired_denoise import install_paired_denoise
        install_paired_denoise(indexed_register, indexed_callback)
        print("[adaln] paired denoising diagnostic", flush=True)
    print("[adaln] indexed device binaries", indexed_build, flush=True)
optional_kernels = (
    ("qknorm", os.environ.get("H3_QKNORM_BUILD")),
    ("qknorm_mrope", os.environ.get("H3_QKNORM_MROPE_BUILD")),
    ("attention", os.environ.get("H3_ATTENTION_AOT_BUILD")),
    ("short_attention", os.environ.get("H3_SHORT_ATTENTION_AOT_BUILD")),
    ("reducecast", os.environ.get("H3_REDUCECAST_BUILD")),
    ("encodercast", os.environ.get("H3_ENCODER_REDUCECAST_BUILD")),
    ("widenorm", os.environ.get("H3_WIDE_NORM_BUILD")),
    ("widenorm_indexed", os.environ.get("H3_WIDE_NORM_INDEXED_BUILD")),
    ("swiglu", os.environ.get("H3_SWIGLU_BUILD")),
)
registrations = []


def device_callback(name, lib):
    symbol = {"encodercast": "reducecast", "short_attention": "attention"}.get(name, name)
    return ctypes.cast(getattr(lib, f"songshan_launch_{symbol}"), ctypes.c_void_p)


if any(path for _, path in optional_kernels):
    sys.path.insert(0, str(here))
    from aot_runtime import load_attention, load_qknorm, load_qknorm_mrope, load_reducecast, load_widenorm, load_widenorm_indexed, load_swiglu
    loaders = {"qknorm": load_qknorm, "qknorm_mrope": load_qknorm_mrope, "attention": load_attention, "short_attention": load_attention,
               "reducecast": load_reducecast, "encodercast": load_reducecast,
               "widenorm": load_widenorm, "widenorm_indexed": load_widenorm_indexed, "swiglu": load_swiglu}
    for name, path in optional_kernels:
        if not path:
            continue
        host_build = Path(path).resolve(strict=True)
        lib = loaders[name](host_build)
        if name == "encodercast" and json.loads((host_build / "reducecast-build.json").read_text())["dtype"] != "float32":
            raise RuntimeError("The conditioning encoder communication candidate requires FP32 casts")
        if name == "reducecast":
            mode = infinilm.songshan_set_reducecast_mode
            mode.argtypes, mode.restype = [ctypes.c_int], ctypes.c_int
            wire_dtype = json.loads((host_build / "reducecast-build.json").read_text())["dtype"]
            if mode({"float32": 0, "float16": 1}[wire_dtype]) != 0:
                raise RuntimeError(f"Unsupported reduction wire dtype: {wire_dtype}")
        setter = getattr(infinilm, f"songshan_set_{name}_launch")
        setter.argtypes, setter.restype = [ctypes.c_void_p], None
        counter = getattr(infinilm, f"songshan_{name}_calls")
        counter.argtypes, counter.restype = [], ctypes.c_uint64
        setter(device_callback(name, lib))
        registrations.append((name, lib, setter, counter))
        print(f"[{name}] device binary", host_build, flush=True)
print("[adaln] loaded", native, hashlib.sha256(native.read_bytes()).hexdigest(), flush=True)
condition_prefill = None
if os.environ.get("H3_ENCODER_PREFILL") == "1":
    setter = infinilm.songshan_set_condition_prefill
    setter.argtypes, setter.restype = [ctypes.c_int], None
    counter = infinilm.songshan_condition_prefill_calls
    counter.argtypes, counter.restype = [], ctypes.c_uint64
    setter(1)
    condition_prefill = (setter, counter)
    print("[condition_prefill] NineToothed causal attention", flush=True)
cache_capacity = os.environ.get("H3_ENCODER_INITIAL_CACHE")
encoder_tp_count = os.environ.get("H3_ENCODER_TP_COUNT")
if cache_capacity or encoder_tp_count:
    from condition_cache import install as install_condition_cache
    install_condition_cache(
        int(cache_capacity) if cache_capacity else None,
        int(encoder_tp_count) if encoder_tp_count else None,
    )
    if cache_capacity:
        print("[condition_cache] initial capacity", cache_capacity, flush=True)
python_attention_bridge = None
python_attention_mode = os.environ.get("H3_PY_ATTENTION_FP32_SCALE")
if python_attention_mode in {"1", "encoder"}:
    import songshan_h3.attention_bridge as python_attention_bridge
    from accurate_attention import kernel as python_attention_candidate
    python_attention_original = python_attention_bridge.kernel
    python_attention_candidate()

    def encoder_attention(query, key, value, causal, scale, output):
        implementation = python_attention_candidate if causal else python_attention_original
        return implementation()(query, key, value, causal, scale, output)

    def encoder_attention_kernel():
        return encoder_attention

    python_attention_selected = encoder_attention_kernel if python_attention_mode == "encoder" else python_attention_candidate
    python_attention_bridge.kernel = python_attention_selected
    print("[attention] FP32 scaling enabled for Python bridge paths; mode", python_attention_mode, flush=True)
vae_type = None
if parallel_devices := os.environ.get("H3_VAE_DEVICES"):
    sys.path.insert(0, str(here.parent))
    from video_decode_parallel import install_parallel_decode
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
    vae_type = AutoencoderKLMiniMaxH3
    serial_vae_decode = vae_type._decode
    device_ids = tuple(map(int, parallel_devices.split(",")))
    clip_batch = int(os.environ.get("H3_VAE_BATCH", "1"))
    install_parallel_decode(device_ids, clip_batch)
    candidate_vae_decode = vae_type._decode
    print("[vae] temporal decode devices", device_ids, "clip batch", clip_batch, flush=True)

keyframe_encoder = None
if os.environ.get("H3_VAE_DETERMINISTIC_ENCODE") == "1":
    from diffusers.modular_pipelines.minimax_h3 import encoders as keyframe_encoder
    from keyframe_determinism import wrap_keyframe_encode
    keyframe_encode_original = keyframe_encoder.encode_vae_condition
    keyframe_encode_candidate = wrap_keyframe_encode(keyframe_encode_original)
    keyframe_encoder.encode_vae_condition = keyframe_encode_candidate
    print("[vae] deterministic DNN for single-frame condition encoding", flush=True)


def select_variant(variant):
    if variant not in ("original", "candidate", "candidate-tf", "candidate-no-reducecast", "candidate-tf-no-reducecast", "candidate-tf-no-encodercast", "candidate-tf-no-tf-reducecast"):
        raise ValueError(f"Unknown request variant: {variant}")
    enabled = variant != "original"
    register(ctypes.cast(callback, ctypes.c_void_p) if enabled else None)
    if indexed_build:
        indexed_register(indexed_callback if enabled else None)
    if condition_prefill:
        condition_prefill[0](int(enabled))
    for name, lib, setter, counter in registrations:
        kernel_enabled = enabled and not (name in {"reducecast", "encodercast"} and variant.endswith("no-reducecast"))
        kernel_enabled = kernel_enabled and not (name == "encodercast" and variant.endswith("no-encodercast"))
        kernel_enabled = kernel_enabled and not (name == "reducecast" and variant.endswith("no-tf-reducecast"))
        setter(device_callback(name, lib) if kernel_enabled else None)
    if vae_type is not None:
        vae_type._decode = candidate_vae_decode if enabled else serial_vae_decode
    if keyframe_encoder is not None:
        keyframe_encoder.encode_vae_condition = keyframe_encode_candidate if enabled else keyframe_encode_original
    if python_attention_bridge is not None:
        python_attention_bridge.kernel = python_attention_selected if enabled and not variant.startswith("candidate-tf") else python_attention_original


def read_counters():
    result = {"adaln": calls()}
    if indexed_build:
        result["indexed"] = indexed_calls()
    if condition_prefill:
        result["condition_prefill"] = condition_prefill[1]()
    for name, lib, setter, counter in registrations:
        result[name] = counter()
    return result


def set_kernel_enabled(kernel, enabled):
    for name, lib, setter, _ in registrations:
        if name == kernel:
            setter(device_callback(name, lib) if enabled else None)
            return
    raise RuntimeError(f"No registered kernel for {kernel}")


control = ModuleType("songshan_candidate_control")
control.select_variant, control.read_counters = select_variant, read_counters
control.set_qknorm_enabled = lambda enabled: set_kernel_enabled("qknorm", enabled)
control.set_qknorm_mrope_enabled = lambda enabled: set_kernel_enabled("qknorm_mrope", enabled)
control.set_widenorm_enabled = lambda enabled: set_kernel_enabled("widenorm", enabled)
control.set_swiglu_enabled = lambda enabled: set_kernel_enabled("swiglu", enabled)
control.set_widenorm_indexed_enabled = lambda enabled: set_kernel_enabled("widenorm_indexed", enabled)
control.set_short_attention_enabled = lambda enabled: set_kernel_enabled("short_attention", enabled)
for registered_name, _, registered_setter, _ in registrations:
    if registered_name in {"attention", "widenorm_indexed", "swiglu"}:
        setattr(control, f"set_{registered_name}_callback", registered_setter)
if indexed_build:
    control.set_indexed_callback = indexed_register
sys.modules[control.__name__] = control
peer_gate_plugin = None
if os.environ.get("H3_PEER_GATE_BUILD"):
    if not indexed_build:
        raise RuntimeError("Peer gate requires the selected indexed kernel build")
    peer_source = here.parent / "dcu-kernel-autotune/candidates/rccl-sequence-parallel"
    sys.path.insert(0, str(peer_source))
    from peer_gate_runtime import install as install_peer_gate
    peer_gate_plugin = install_peer_gate(infinilm, indexed_host, control, os.environ["H3_PEER_GATE_BUILD"])
sharded_norm_plugin = None
if os.environ.get("H3_PEER_SHARDED_NORM_BUILD"):
    if peer_gate_plugin is None:
        raise RuntimeError("Sharded norm requires the accepted peer reference")
    norm_host = next((lib for name, lib, _, _ in registrations if name == "widenorm_indexed"), None)
    if norm_host is None:
        raise RuntimeError("Sharded norm requires the matched indexed norm image")
    sys.path.insert(0, str(here.parent / "dcu-kernel-autotune/candidates/peer-sharded-norm-model"))
    from model_runtime import install as install_sharded_norm
    sharded_norm_plugin = install_sharded_norm(infinilm, indexed_host, norm_host, control,
        peer_gate_plugin, os.environ["H3_PEER_SHARDED_NORM_BUILD"])
vae_process_plugin = None
if process_workers := os.environ.get("H3_VAE_PROCESS_WORKERS"):
    if vae_type is None:
        raise RuntimeError("VAE process scheduling requires the accepted temporal decoder")
    sys.path.insert(0, str(here.parent / "dcu-kernel-autotune/candidates/vae-process-scheduling"))
    from process_runtime import install as install_vae_process
    vae_process_plugin = install_vae_process(control, int(process_workers),
        os.environ["H3_VAE_PROCESS_LOG_DIR"])
    print("[vae_process] configured workers", process_workers, flush=True)
entry = here.parents[1] / "runtime/h3-runtime/sources/InfiniLM/examples/minimax_h3.py"
if os.environ.get("H3_PROFILE_CANDIDATE") == "1":
    entry = here.parent / "profile-entry.py"
elif sys.argv[1:2] == ["--sequence-run"]:
    entry = here.parent / "run-sequence.py"
    sys.argv.pop(1)
elif sys.argv[1:2] == ["--replay-run"]:
    entry = here.parent / "replay-denoise.py"
    sys.argv.pop(1)
sys.argv[0] = str(entry)
try:
    runpy.run_path(str(entry), run_name="__main__")
finally:
    if sharded_norm_plugin:
        print("[sharded_norm] stats", json.dumps(sharded_norm_plugin["stats"]()), flush=True)
    if peer_gate_plugin:
        peer_gate_plugin["enabled"](False)
        print("[peer_gate] stats", json.dumps(peer_gate_plugin["stats"]()), flush=True)
    optional_counts = {}
    if condition_prefill:
        prefill_set, prefill_count = condition_prefill
        optional_counts["condition_prefill"] = prefill_count()
        prefill_set(0)
        print(f"[condition_prefill] calls: {optional_counts['condition_prefill']}", flush=True)
    for name, lib, setter, counter in registrations:
        optional_counts[name] = counter()
        setter(None)
        print(f"[{name}] calls: {optional_counts[name]}", flush=True)
    count = calls()
    register(None)
    print(f"[adaln] fused calls: {count}", flush=True)
    if aot_build:
        print(f"[adaln] AOT calls: {aot_calls()} of {count}", flush=True)
    if indexed_build:
        indexed_count = indexed_calls()
        indexed_register(None)
        print(f"[adaln] indexed calls: {indexed_count}", flush=True)
        if audit_path:
            checked, incorrect = audit_checked(), audit_incorrect()
            print(f"[adaln] indexed audit compared {checked} tensors, {incorrect} mismatches", flush=True)
if count == 0:
    raise RuntimeError("No MiniMax-H3 AdaLN calls reached the candidate")
if indexed_build and indexed_count == 0 and not (peer_gate_plugin and peer_gate_plugin["calls"]()):
    raise RuntimeError("No MiniMax-H3 calls reached the indexed kernels")
if indexed_build and audit_path and (checked == 0 or incorrect):
    raise RuntimeError("Indexed AdaLN audit did not pass")
if aot_build and not indexed_build and aot_calls() == 0:
    raise RuntimeError("No MiniMax-H3 AdaLN calls reached the specialized kernel")
# 形状不满足融合条件时，模型仍可完整走原来的归一化和 RoPE。
# 回放和正式验收按请求核对融合调用数。
if any(count == 0 for name, count in optional_counts.items() if name != "qknorm_mrope"):
    raise RuntimeError(f"An enabled Q/K specialization did not execute: {optional_counts}")
