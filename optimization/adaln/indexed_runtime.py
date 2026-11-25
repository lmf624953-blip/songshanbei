"""Load the two indexed device kernels using their build manifest."""
import ctypes
import hashlib
import json
import os


def load_indexed(build):
    record = json.loads((build / "indexed-build.json").read_text())
    paths = [build / f"{name}.hsaco" for name in ("indexed_adaln", "indexed_gate")]
    warps = {record[path.stem]["warps"] for path in paths}
    if len(warps) != 1:
        raise RuntimeError("The indexed kernels disagree on workgroup size")
    for path in paths:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record[path.stem]["hsaco_sha256"]:
            raise RuntimeError(f"Unexpected indexed device binary digest: {path}")
    host = ctypes.CDLL(str(build / "indexed-host.so"))
    if hasattr(host, "songshan_indexed_tile"):
        tiles = {record[path.stem].get("tile", 256) for path in paths}
        if tiles != {host.songshan_indexed_tile()}:
            raise RuntimeError("Indexed host and device tile sizes disagree")
    prepare = host.songshan_prepare_indexed
    prepare.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    prepare.restype = ctypes.c_char_p
    error = prepare(*(os.fsencode(p) for p in paths), warps.pop())
    if error:
        raise RuntimeError(f"Preparing the indexed device binaries failed: {error.decode()}")
    return host
