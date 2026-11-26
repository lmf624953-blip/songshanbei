"""Preserve the Python, host shim and patch sources used by a measured run."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys


def snapshot(destination):
    project = Path(__file__).resolve().parent.parent
    target = destination / "source"
    target.mkdir()
    files = []
    for root in (project / "optimization", project / "optimization/adaln"):
        files.extend(p for p in root.iterdir() if p.is_file() and p.suffix in {".py", ".sh", ".cc", ".patch"})
    files.append(project / "runtime/h3-runtime/sources/InfiniLM/examples/minimax_h3.py")
    records = {}
    for source in sorted(files):
        relative = source.relative_to(project)
        copied = target / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copied)
        records[str(relative)] = hashlib.sha256(copied.read_bytes()).hexdigest()
    artifacts = {}
    for name in ("H3_ADALN_BUILD", "H3_ADALN_AOT_BUILD", "H3_INDEXED_BUILD",
                 "H3_SWIGLU_BUILD", "H3_QKNORM_BUILD", "H3_QKNORM_MROPE_BUILD", "H3_WIDE_NORM_INDEXED_BUILD", "H3_ATTENTION_AOT_BUILD", "H3_SHORT_ATTENTION_AOT_BUILD",
                 "H3_REDUCECAST_BUILD", "H3_ENCODER_REDUCECAST_BUILD", "H3_WIDE_NORM_BUILD"):
        if not os.environ.get(name):
            continue
        build = Path(os.environ[name]).resolve(strict=True)
        metadata = {}
        for path in build.iterdir():
            if path.suffix in {".so", ".hsaco"} or path.name in {"candidate-build.json", "native-overlay.txt", "host-optimization.txt", "InfiniLM-applied-source.tar.gz"}:
                metadata[path.name] = {"bytes": path.stat().st_size,
                                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            if path.name.endswith("-build.json") and path.name != "candidate-build.json":
                metadata[path.name] = json.loads(path.read_text())
        if (build / "native-overlay.txt").exists():
            overlay = Path((build / "native-overlay.txt").read_text().strip())
            for path in (overlay / "infinilm/lib").glob("_infinilm*.so"):
                metadata["loaded_native"] = {"path": str(path),
                                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        artifacts[name] = {"path": str(build), "artifacts": metadata}
    names = ("H3_PY_ATTENTION_FP32_SCALE", "H3_VAE_DEVICES", "H3_VAE_BATCH", "H3_VAE_CACHE_REPLICAS",
             "H3_VAE_DETERMINISTIC_ENCODE",
             "H3_ENCODER_PREFILL", "H3_ENCODER_INITIAL_CACHE", "H3_ENCODER_TP_COUNT", "H3_INDEXED_AUDIT",
             "H3_INDEXED_PAIRED", "H3_BENCH_JIT_CACHE", "H3_NATIVE_CXX_OPTIMIZE",
             "ROCBLAS_TENSILE_LIBPATH", "TRITON_CACHE_DIR", "NINETOOTHED_CACHE_DIR")
    names += ("NCCL_ALGO", "NCCL_PROTO", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS",
              "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
              "KERNEL_TEST", "TENSILE_DB")
    names += ("ROCBLAS_DEFAULT_ATOMICS_MODE",)
    environment = {name: os.environ[name] for name in names if name in os.environ}
    runtime = {"python": sys.version, "executable": sys.executable, "hostname": socket.gethostname(),
               "cpu_affinity": sorted(os.sched_getaffinity(0))}
    from infinilm.lib import _infinilm
    import torch
    native = Path(_infinilm.__file__).resolve()
    runtime.update(native_path=str(native), native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                   torch_version=torch.__version__, hip_version=torch.version.hip)
    (target / "manifest.json").write_text(json.dumps({"sources": records, "builds": artifacts,
                                                     "environment": environment, "runtime": runtime}, indent=2) + "\n")
