"""注册模型专用分片hidden回调；关闭候选时精确使用当前已验收peer回调。"""
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path


def install(native, indexed_host, norm_host, control, reference_plugin, build):
    build=Path(build).resolve(strict=True)
    meta=json.loads((build/'peer-runtime-build.json').read_text())
    assert meta['model_bridge']
    for name,expected in meta['sha256'].items():
        assert hashlib.sha256((build/name).read_bytes()).hexdigest()==expected,(build,name)
    bridge=ct.CDLL(str(build/'peer-gate-bridge.so'))
    prepare=bridge.h3_peer_bridge_prepare
    prepare.argtypes=[ct.c_char_p,ct.c_int,ct.c_int,ct.POINTER(ct.c_int),ct.c_int,ct.c_char_p,ct.c_void_p]
    prepare.restype=ct.c_char_p
    codes=(ct.c_int*len(meta['abi_codes']))(*meta['abi_codes'])
    def check(message):
        if message:raise RuntimeError(message.decode())
    check(prepare(os.fsencode(build/'peer-gate.hsaco'),meta['nt']['tile'],meta['nt']['warps'],codes,len(codes),
        os.fsencode(build/'peer-transport.so'),ct.cast(indexed_host.songshan_bench_indexed,ct.c_void_p)))
    reference=reference_plugin['bridge']
    bridge.h3_model_prepare.argtypes=[ct.c_void_p]*2;bridge.h3_model_prepare.restype=ct.c_char_p
    check(bridge.h3_model_prepare(ct.cast(norm_host.songshan_bench_widenorm_indexed,ct.c_void_p),
                                  ct.cast(reference.songshan_launch_peer_gate,ct.c_void_p)))
    bridge.h3_model_set_enabled.argtypes=[ct.c_int];bridge.h3_model_set_enabled.restype=ct.c_char_p
    bridge.h3_peer_bridge_stats.argtypes=[];bridge.h3_peer_bridge_stats.restype=ct.c_char_p
    gate_setter=native.songshan_set_peer_gate_launch;gate_setter.argtypes=[ct.c_void_p];gate_setter.restype=None
    norm_setter=native.songshan_set_widenorm_indexed_launch;norm_setter.argtypes=[ct.c_void_p];norm_setter.restype=None
    hidden_setter=native.songshan_set_peer_hidden_hooks;hidden_setter.argtypes=[ct.c_void_p]*2;hidden_setter.restype=None
    cast=lambda function:ct.cast(function,ct.c_void_p)
    selected=False
    def stats():return json.loads(bridge.h3_peer_bridge_stats())
    def enabled(value, reference_on=True):
        nonlocal selected
        check(bridge.h3_model_set_enabled(int(value)))
        selected=bool(value)
        gate_setter(cast(bridge.h3_model_gate) if selected else cast(reference.songshan_launch_peer_gate) if reference_on else None)
        norm_setter(cast(bridge.h3_model_norm) if selected else cast(norm_host.songshan_launch_widenorm_indexed) if reference_on else None)
        hidden_setter(cast(bridge.h3_model_materialize) if selected else None,cast(bridge.h3_model_abort) if selected else None)
    old_select=control.select_variant;old_counts=control.read_counters
    def select(variant):
        old_select(variant)
        enabled(variant.startswith('candidate'), reference_on=variant!='original')
    def counters():
        result=old_counts();state=stats()
        result['peer_batches']+=state['batches'];result['peer_fused']+=state['fused'];result['peer_fallback']+=state['fallback']
        result.update(sharded_gate=state['fused'],sharded_norm=state['norm_batches'],
                      sharded_materialized=state['materialized'],sharded_replica_checks=state['replica_checks'],
                      sharded_replica_fallbacks=state['replica_fallbacks'],sharded_full_norm=state['full_norm_calls'])
        return result
    def combined_stats():return dict(comparison='accepted-peer-vs-sharded-norm',candidate=stats(),reference=reference_plugin['stats']())
    def validate_request(order,counts,state,transformer_calls=49):
        active=order=='on';batches=100*transformer_calls
        expected=dict(peer_gate=8*batches,peer_deferred=8*batches,peer_batches=batches,peer_fused=batches,
                      peer_fallback=0,indexed=0,sharded_gate=batches if active else 0,
                      sharded_norm=99*transformer_calls if active else 0,
                      sharded_materialized=transformer_calls if active else 0,
                      sharded_full_norm=8*transformer_calls if active else 0,sharded_replica_fallbacks=0)
        if any(counts.get(key)!=value for key,value in expected.items()):
            raise RuntimeError(f'分片norm/当前peer完整请求计数不符：expected={expected}, actual={counts}')
        if state['comparison']!='accepted-peer-vs-sharded-norm' or state['candidate']['pending_ranks'] or state['candidate']['failed']:
            raise RuntimeError('完整请求返回时分片hidden未清理，或对照类型不符')
    old_wide_enabled=control.set_widenorm_indexed_enabled
    old_wide_callback=control.set_widenorm_indexed_callback
    def wide_enabled(value):
        if selected:raise RuntimeError('分片norm启用时不能替换其配套回调，请单独建立消融配置')
        return old_wide_enabled(value)
    def wide_callback(value):
        if selected:raise RuntimeError('分片norm启用时不能替换其配套回调，请单独建立消融配置')
        return old_wide_callback(value)
    control.select_variant=select;control.read_counters=counters
    control.set_peer_gate_enabled=enabled;control.peer_gate_stats=combined_stats
    control.set_widenorm_indexed_enabled=wide_enabled;control.set_widenorm_indexed_callback=wide_callback
    control.sharded_norm_stats=stats
    control.peer_comparison='accepted-peer-vs-sharded-norm'
    control.validate_peer_request=validate_request
    enabled(os.environ.get('H3_PEER_GATE_ENABLED','1')=='1')
    print('[sharded_norm] current peer control; candidate',build,flush=True)
    return dict(bridge=bridge,stats=stats,enabled=enabled,reference=reference_plugin)
