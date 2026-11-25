// Compare sampled model tensors with the original InfiniCore operation sequence.
#include <array>
#include <atomic>
#include <cstdint>
#include <ATen/ops/equal.h>
#include <infinicore/adaptor/aten_adaptor.hpp>
#include <infinicore/context/context.hpp>
#include <infinicore/ops/add.hpp>
#include <infinicore/ops/embedding.hpp>
#include <infinicore/ops/mul.hpp>

using infinicore::Tensor;
using Launch = void (*)(bool, const Tensor &, const Tensor &, const Tensor &,
                        const Tensor &, const Tensor &, std::uintptr_t);

namespace {
Launch fused = nullptr;
std::array<std::atomic<std::uint64_t>, 8> calls{};
std::atomic<std::uint64_t> checked{0};
std::atomic<std::uint64_t> incorrect{0};

Tensor lookup(const Tensor &part, const Tensor &indices) {
    return infinicore::op::embedding(indices, part->squeeze(0)->contiguous())->unsqueeze(0);
}
} // namespace

extern "C" void songshan_init_indexed_audit(Launch launch) {
    fused = launch;
}

extern "C" void songshan_audit_indexed(bool gate, const Tensor &out, const Tensor &first,
                                       const Tensor &second, const Tensor &table_part,
                                       const Tensor &indices, std::uintptr_t stream) {
    fused(gate, out, first, second, table_part, indices, stream);
    auto ordinal = calls.at(out->device().getIndex()).fetch_add(1, std::memory_order_relaxed);
    // Sample both groups of the first block every 12 denoising steps.
    if (ordinal % 2400 >= 4) {
        return;
    }
    auto mod = lookup(table_part, indices);
    Tensor expected;
    if (gate) {
        expected = infinicore::op::add(first, infinicore::op::mul(mod, second));
    } else {
        auto scale = lookup(second, indices);
        auto scaled = infinicore::op::mul(first, scale);
        expected = infinicore::op::add(infinicore::op::add(first, scaled), mod);
    }
    infinicore::context::syncStream();
    using infinicore::adaptor::to_aten_tensor;
    if (!at::equal(to_aten_tensor(expected), to_aten_tensor(out))) {
        incorrect.fetch_add(1, std::memory_order_relaxed);
    }
    checked.fetch_add(1, std::memory_order_relaxed);
}

extern "C" std::uint64_t songshan_indexed_audit_checked() {
    return checked.load(std::memory_order_relaxed);
}

extern "C" std::uint64_t songshan_indexed_audit_incorrect() {
    return incorrect.load(std::memory_order_relaxed);
}
