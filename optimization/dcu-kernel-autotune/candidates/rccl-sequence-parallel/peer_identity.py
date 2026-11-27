"""公开复测固定已验证的设备代码和主机库，允许HSACO调试路径不同。"""
import hashlib
import json
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(peer_build, native_build):
    peer_build, native_build = Path(peer_build), Path(native_build)
    frozen = read(HERE / "peer-image-identity.json")
    meta = read(peer_build / "peer-runtime-build.json")
    for name, expected in meta["sha256"].items():
        if digest(peer_build / name) != expected:
            raise ValueError(f"分片构建产物与元数据不符：{name}")
    for name, expected in frozen["host_sha256"].items():
        if digest(peer_build / name) != expected:
            raise ValueError(f"主机库与冻结版本不符：{name}")
    elf = runpy.run_path(str(HERE.parents[1] / "adapters/compare-h3-device.py"))["elf_loads"]
    image = elf(peer_build / "peer-gate.hsaco")
    if any(image[key] != value for key, value in frozen["nt"].items()):
        raise ValueError("九齿镜像的加载内容或布局与冻结版本不符")
    overlay = Path((native_build / "native-overlay.txt").read_text().strip())
    natives = list((overlay / "infinilm/lib").glob("_infinilm*.so"))
    if len(natives) != 1 or digest(natives[0]) != frozen["native_sha256"]:
        raise ValueError("InfiniLM原生库与冻结版本不符")
    if meta["nt"]["tile"] != 1024 or meta["nt"]["warps"] != 4:
        raise ValueError("九齿tile/warps与冻结版本不符")
    return dict(native_sha256=digest(natives[0]), nt_sha256=image["sha256"],
                nt_loaded_identity=frozen["nt"], host_sha256=frozen["host_sha256"])
