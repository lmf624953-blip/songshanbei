#!/usr/bin/env python3
"""在已验收peer源码的独立副本补充hidden物化和失败边界，不改变默认路径。"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

SOURCE = 'csrc/models/minimax_h3/minimax_h3_transformer.cpp'
BASE_SHA = 'b3341d79f35b850b2924e383a22100d0250f348cd0859b11cc2dd33c553bea1e'


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('tree', type=Path)
    cli.add_argument('patch', type=Path)
    args = cli.parse_args()
    path = args.tree/SOURCE
    original = path.read_text()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == BASE_SHA
    text = original

    def replace(old, new):
        nonlocal text
        assert text.count(old) == 1, old
        text = text.replace(old, new, 1)

    replace('#include <limits>', '#include <limits>\n#include <exception>')
    replace('namespace {\nstd::atomic<PeerGateLaunch> peer_gate_launch{nullptr};', '''using PeerMaterializeLaunch = void (*)(const infinicore::Tensor &, int, int, infinicclComm_t, std::uintptr_t);
using PeerAbortLaunch = void (*)();
namespace {
std::atomic<PeerMaterializeLaunch> peer_materialize_launch{nullptr};
std::atomic<PeerAbortLaunch> peer_abort_launch{nullptr};
struct PeerForwardGuard {
    int initial = std::uncaught_exceptions();
    ~PeerForwardGuard() noexcept {
        if (std::uncaught_exceptions() > initial) {
            auto abort = peer_abort_launch.load(std::memory_order_acquire);
            if (abort) { try { abort(); } catch (...) {} }
        }
    }
};
std::atomic<PeerGateLaunch> peer_gate_launch{nullptr};''')
    replace('namespace {\n\nstd::optional<infinicore::Tensor> packed_swiglu(', '''extern "C" __attribute__((visibility("default")))
void songshan_set_peer_hidden_hooks(PeerMaterializeLaunch materialize, PeerAbortLaunch abort) {
    peer_materialize_launch.store(materialize, std::memory_order_release);
    peer_abort_launch.store(abort, std::memory_order_release);
}

namespace {
void materialize_peer_hidden(const infinicore::Tensor &input) {
    auto materialize = peer_materialize_launch.load(std::memory_order_acquire);
    if (!materialize) return;
    const auto &rank = infinilm::global_state::get_tensor_model_parallel_rank_info();
    if (rank.tp_size == 8 && rank.comm) {
        materialize(input, rank.tp_rank, rank.tp_size, rank.comm,
                    reinterpret_cast<std::uintptr_t>(infinicore::context::getStream()));
    }
}

std::optional<infinicore::Tensor> packed_swiglu(''')
    replace('        return adaln(norm->forward(input), scale, shift);',
            '        materialize_peer_hidden(input);\n        return adaln(norm->forward(input), scale, shift);')
    replace('InfinilmModel::Output MiniMaxH3Transformer::forward(const Input &input) const {',
            'InfinilmModel::Output MiniMaxH3Transformer::forward(const Input &input) const {\n    PeerForwardGuard peer_forward_guard;')
    replace('    auto normalized = norm_out_->forward(hidden_states, temb, timestep_indices);',
            '    materialize_peer_hidden(hidden_states);\n    auto normalized = norm_out_->forward(hidden_states, temb, timestep_indices);')
    patch = ''.join(difflib.unified_diff(original.splitlines(True), text.splitlines(True), fromfile='a/'+SOURCE, tofile='b/'+SOURCE))
    args.patch.write_text(patch)
    path.write_text(text)
    args.patch.with_suffix('.json').write_text(json.dumps(dict(base_sha256=BASE_SHA,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), patch_sha256=hashlib.sha256(args.patch.read_bytes()).hexdigest(),
        scope='仅独立源码；未注册时保持已验收peer路径；不代表模型资格'), ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
