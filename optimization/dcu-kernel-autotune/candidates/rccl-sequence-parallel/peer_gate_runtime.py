"""注册已校验的九齿分片门控桥，提供驻留模型对照开关与计数。"""
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path


def install(native,indexed_host,control,build):
    build=Path(build).resolve(strict=True);meta=json.loads((build/'peer-runtime-build.json').read_text())
    for name,digest in meta['sha256'].items():
        assert hashlib.sha256((build/name).read_bytes()).hexdigest()==digest,f'Peer artifact hash mismatch: {name}'
    bridge=ct.CDLL(str(build/'peer-gate-bridge.so'))
    prepare=bridge.h3_peer_bridge_prepare
    prepare.argtypes=[ct.c_char_p,ct.c_int,ct.c_int,ct.POINTER(ct.c_int),ct.c_int,ct.c_char_p,ct.c_void_p];prepare.restype=ct.c_char_p
    codes=(ct.c_int*len(meta['abi_codes']))(*meta['abi_codes'])
    error=prepare(os.fsencode(build/'peer-gate.hsaco'),meta['nt']['tile'],meta['nt']['warps'],codes,len(codes),os.fsencode(build/'peer-transport.so'),ct.cast(indexed_host.songshan_bench_indexed,ct.c_void_p))
    if error:raise RuntimeError(error.decode())
    setter=native.songshan_set_peer_gate_launch;setter.argtypes=[ct.c_void_p];setter.restype=None
    callback=ct.cast(bridge.songshan_launch_peer_gate,ct.c_void_p)
    calls=native.songshan_peer_gate_calls;calls.argtypes=[];calls.restype=ct.c_uint64
    deferred=native.songshan_peer_deferred_rows;deferred.argtypes=[];deferred.restype=ct.c_uint64
    bridge.h3_peer_bridge_stats.argtypes=[];bridge.h3_peer_bridge_stats.restype=ct.c_char_p
    def stats():return json.loads(bridge.h3_peer_bridge_stats())
    def enabled(value):setter(callback if value else None)
    original_select=control.select_variant;original_counts=control.read_counters
    def select(variant):original_select(variant);enabled(variant.startswith('candidate'))
    def counters():
        result=original_counts();state=stats()
        result.update(peer_gate=calls(),peer_deferred=deferred(),peer_batches=state['batches'],peer_fused=state['fused'],peer_fallback=state['fallback'])
        return result
    control.select_variant=select;control.read_counters=counters;control.set_peer_gate_enabled=enabled;control.peer_gate_stats=stats
    enabled(os.environ.get('H3_PEER_GATE_ENABLED','1')=='1')
    print('[peer_gate] NineToothed runtime',build,'tile',meta['nt']['tile'],flush=True)
    return dict(bridge=bridge,stats=stats,calls=calls,enabled=enabled)
