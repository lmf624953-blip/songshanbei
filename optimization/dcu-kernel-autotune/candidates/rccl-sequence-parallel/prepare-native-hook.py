#!/usr/bin/env python3
"""在已验收native源码的私有副本中添加可关闭的H3分片门控回调，保留正式默认。"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

BASE_SHA256='e1471b8389f06ddb84715b2a72d40b7ccfcad62dacf9f3ce9976cc64df77c081'
SOURCE='csrc/models/minimax_h3/minimax_h3_transformer.cpp'


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('private_source',type=Path);cli.add_argument('patch',type=Path);args=cli.parse_args()
    p=args.private_source/SOURCE;original=p.read_text();assert hashlib.sha256(p.read_bytes()).hexdigest()==BASE_SHA256
    def replace_once(text,old,new):
        assert text.count(old)==1,old[:100]
        return text.replace(old,new,1)
    text=replace_once(original,'#include <limits>','#include <limits>\n#include <type_traits>')
    marker='namespace {\nstd::atomic<SwiGLULaunch> swiglu_launch{nullptr};'
    declarations='''using PeerGateLaunch = void (*)(const infinicore::Tensor &, const infinicore::Tensor &,
    const infinicore::Tensor &, const infinicore::Tensor &, const infinicore::Tensor &,
    int, int, infinicclComm_t, std::uintptr_t);
namespace {
std::atomic<PeerGateLaunch> peer_gate_launch{nullptr};
std::atomic<std::uint64_t> peer_gate_calls{0}, peer_deferred_rows{0};
thread_local bool peer_defer_row = false;
struct PeerRowScope {
    bool previous;
    explicit PeerRowScope(bool enabled) : previous(peer_defer_row) { peer_defer_row = enabled; }
    ~PeerRowScope() { peer_defer_row = previous; }
};
std::atomic<SwiGLULaunch> swiglu_launch{nullptr};'''
    text=replace_once(text,marker,declarations)
    marker='namespace {\n\nstd::optional<infinicore::Tensor> packed_swiglu('
    exports='''extern "C" __attribute__((visibility("default")))
void songshan_set_peer_gate_launch(PeerGateLaunch launch) { peer_gate_launch.store(launch, std::memory_order_release); }
extern "C" __attribute__((visibility("default")))
std::uint64_t songshan_peer_gate_calls() { return peer_gate_calls.load(std::memory_order_relaxed); }
extern "C" __attribute__((visibility("default")))
std::uint64_t songshan_peer_deferred_rows() { return peer_deferred_rows.load(std::memory_order_relaxed); }

namespace {

std::optional<infinicore::Tensor> packed_swiglu('''
    text=replace_once(text,marker,exports)
    old='''    auto mutable_input = input;
    return linear->forward(mutable_input);'''
    new='''    auto mutable_input = input;
    if constexpr (std::is_same_v<Linear, infinilm::layers::linear::RowParallelLinear>) {
        if (peer_defer_row) {
            if (linear->has_bias()) throw std::runtime_error("H3 peer gate requires bias-free row projection");
            peer_deferred_rows.fetch_add(1, std::memory_order_relaxed);
            return static_cast<const infinilm::nn::BaseLinear &>(*linear).forward(mutable_input);
        }
    }
    return linear->forward(mutable_input);'''
    text=replace_once(text,old,new)
    old='''    auto token_modulation = (launch ? modulation : gather_rows(modulation, adaln_indices))->unsqueeze(0);'''
    new='''    auto peer = peer_gate_launch.load(std::memory_order_acquire);
    const auto &peer_rank = infinilm::global_state::get_tensor_model_parallel_rank_info();
    if (!launch || peer_rank.tp_size != 8 || !peer_rank.comm || hidden_states->shape()[1] < 16384) peer = nullptr;
    auto token_modulation = (launch ? modulation : gather_rows(modulation, adaln_indices))->unsqueeze(0);'''
    text=replace_once(text,old,new)
    old='''        return launch ? apply_indexed(launch, true, input, branch, gate, adaln_indices)
                      : infinicore::op::add(input, infinicore::op::mul(gate, branch));'''
    new='''        if (peer) {
            auto out = infinicore::Tensor::empty(input->shape(), input->dtype(), input->device());
            peer(out, input->contiguous(), branch->contiguous(), gate, adaln_indices,
                 peer_rank.tp_rank, peer_rank.tp_size, peer_rank.comm,
                 reinterpret_cast<std::uintptr_t>(infinicore::context::getStream()));
            peer_gate_calls.fetch_add(1, std::memory_order_relaxed);
            return out;
        }
        return launch ? apply_indexed(launch, true, input, branch, gate, adaln_indices)
                      : infinicore::op::add(input, infinicore::op::mul(gate, branch));'''
    text=replace_once(text,old,new)
    text=replace_once(text,'    auto attention_output = attn_->forward(normalized, rope, position_ids, rotary_cos_sin_cache);',
        '    auto attention_output = [&] { PeerRowScope scope(peer != nullptr); return attn_->forward(normalized, rope, position_ids, rotary_cos_sin_cache); }();')
    text=replace_once(text,'    auto mlp_output = ff_->forward(normalized);',
        '    auto mlp_output = [&] { PeerRowScope scope(peer != nullptr); return ff_->forward(normalized); }();')
    patch=''.join(difflib.unified_diff(original.splitlines(keepends=True),text.splitlines(keepends=True),fromfile='a/'+SOURCE,tofile='b/'+SOURCE))
    with args.patch.open('x') as f:f.write(patch)
    p.write_text(text)
    record=dict(source=str(p),base_sha256=BASE_SHA256,candidate_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),patch_sha256=hashlib.sha256(args.patch.read_bytes()).hexdigest(),scope='仅私有源码；默认回调为空，完整原路径保留。尚未编译、未实现桥接或真实模型验证。')
    args.patch.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':main()
