"""Load verified single-kernel HIP modules and their shared-memory metadata."""
import ctypes
import hashlib
import json
import os


def load_kernel(build, stem, host_file, prepare_symbol):
    record = json.loads((build / f"{stem}-build.json").read_text())
    image = build / f"{stem}.hsaco"
    with image.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != record["hsaco_sha256"]:
        raise RuntimeError(f"Unexpected device binary digest: {image}")
    host = ctypes.CDLL(str(build / host_file))
    prepare = getattr(host, prepare_symbol)
    prepare.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    prepare.restype = ctypes.c_char_p
    error = prepare(os.fsencode(image), record["warps"], record["shared"])
    if error:
        raise RuntimeError(f"Preparing {image.name} failed: {error.decode()}")
    return host


def load_qknorm(build):
    return load_kernel(build, "qk-norm", "qk-norm-host.so", "songshan_prepare_qknorm")


def load_widenorm(build):
    return load_kernel(build, "wide-norm", "wide-norm-host.so", "songshan_prepare_widenorm")


def load_attention(build):
    host = load_kernel(build, "dense-attention-aot", "dense-attention-aot.so", "songshan_prepare_attention")
    record = json.loads((build / "dense-attention-aot-build.json").read_text())
    if hasattr(host, "songshan_attention_query_block"):
        host.songshan_attention_query_block.argtypes = []
        host.songshan_attention_query_block.restype = ctypes.c_int
        if host.songshan_attention_query_block() != record.get("query_block", 64):
            raise RuntimeError("Attention host and device query blocks disagree")
    elif record.get("query_block", 64) != 64:
        raise RuntimeError("An older attention host cannot launch this query block")
    return host


def load_reducecast(build):
    record = json.loads((build / "reducecast-build.json").read_text())
    images = [build / f"reducecast-{name}.hsaco" for name in ("up", "down")]
    for name, image in zip(("up", "down"), images):
        if hashlib.sha256(image.read_bytes()).hexdigest() != record[name]["hsaco_sha256"]:
            raise RuntimeError(f"Unexpected reduce cast binary digest: {image}")
    host = ctypes.CDLL(str(build / "reducecast-host.so"))
    prepare = host.songshan_prepare_reducecast
    prepare.argtypes, prepare.restype = [ctypes.c_char_p, ctypes.c_char_p], ctypes.c_char_p
    error = prepare(*(os.fsencode(image) for image in images))
    if error:
        raise RuntimeError(f"Preparing reduction casts failed: {error.decode()}")
    return host
