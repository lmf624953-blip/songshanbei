"""独立核对模型、当前peer参考及归一化镜像；允许NT的非加载调试信息变化。"""
import hashlib
import json
from pathlib import Path
import runpy

HERE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(model_build,native_build,peer_build,norm_build):
    model_build,native_build,peer_build,norm_build=map(Path,(model_build,native_build,peer_build,norm_build))
    frozen=read(HERE/'model-image-identity.json')
    meta=read(model_build/'peer-runtime-build.json')
    assert meta['model_bridge'] and meta['abi_codes']==frozen['abi_codes']
    assert meta['nt']['tile']==1024 and meta['nt']['warps']==4
    for name,expected in meta['sha256'].items():assert digest(model_build/name)==expected,name
    for name,expected in frozen['host_sha256'].items():assert digest(model_build/name)==expected,name
    elf=runpy.run_path(str(HERE.parents[1]/'adapters/compare-h3-device.py'))['elf_loads']
    nt=elf(model_build/'peer-gate.hsaco')
    assert all(nt[k]==value for k,value in frozen['nt'].items())
    norm=elf(norm_build/'wide-norm-indexed.hsaco')
    assert all(norm[k]==value for k,value in frozen['norm'].items())
    assert digest(norm_build/'wide-norm-indexed-host.so')==frozen['norm_host_sha256']
    peer_frozen=read(HERE.parent/'rccl-sequence-parallel/peer-image-identity.json')
    peer_meta=read(peer_build/'peer-runtime-build.json')
    for name,expected in peer_meta['sha256'].items():assert digest(peer_build/name)==expected,name
    for name,expected in peer_frozen['host_sha256'].items():assert digest(peer_build/name)==expected,name
    peer=elf(peer_build/'peer-gate.hsaco');assert all(peer[k]==value for k,value in peer_frozen['nt'].items())
    overlay=Path((native_build/'native-overlay.txt').read_text().strip());natives=list((overlay/'infinilm/lib').glob('_infinilm*.so'))
    assert len(natives)==1 and digest(natives[0])==frozen['native_sha256']
    return dict(native_sha256=frozen['native_sha256'],model_nt_sha256=nt['sha256'],reference_nt_sha256=peer['sha256'],
                host_sha256=frozen['host_sha256'],norm_host_sha256=frozen['norm_host_sha256'])
