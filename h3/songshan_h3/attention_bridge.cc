// Host dispatch and tensor views only. Numerical work lives in attention_kernel.py.
#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/python_variable.h>
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/ops/mha.hpp"
#include "infinicore/ops/mha_varlen.hpp"

namespace py = pybind11;
using infinicore::Tensor;
using infinicore::graph::GraphTensor;

namespace {
struct DensePlan {
    GraphTensor out, q, k, v;
    float scale;
    bool causal;
};
struct VarlenPlan {
    GraphTensor out, q, k, v, cu_q, cu_k;
    float scale;
};

void validate_plan(const std::optional<Tensor> &alibi) {
    if (alibi) throw std::runtime_error("NineToothed H3 attention does not support ALiBi");
    if (infinicore::context::isGraphRecording())
        throw std::runtime_error("NineToothed Python dispatch requires device graph capture disabled");
}

py::object view(const Tensor &tensor) {
    return py::reinterpret_steal<py::object>(
        THPVariable_Wrap(infinicore::adaptor::to_aten_tensor(tensor)));
}

void *dense_plan(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v,
                 std::optional<Tensor> alibi, float scale, bool causal) {
    validate_plan(alibi);
    return new DensePlan{GraphTensor(out), GraphTensor(q), GraphTensor(k), GraphTensor(v), scale, causal};
}

void dense_run(void *meta) {
    auto *p = static_cast<DensePlan *>(meta);
    py::gil_scoped_acquire gil;
    try {
        py::module_::import("songshan_h3.attention_bridge").attr("dense")(
            view(p->out), view(p->q), view(p->k), view(p->v), p->scale, p->causal,
            reinterpret_cast<uintptr_t>(infinicore::context::getStream()));
    } catch (py::error_already_set &error) {
        throw std::runtime_error(error.what());
    }
}

void dense_cleanup(void **meta) {
    delete static_cast<DensePlan *>(*meta);
    *meta = nullptr;
}

void *varlen_plan(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v,
                  const Tensor &cu_q, const Tensor &cu_k, std::optional<Tensor> blocks,
                  int max_q, int max_k, std::optional<Tensor> alibi, float scale) {
    validate_plan(alibi);
    if (blocks) throw std::runtime_error("NineToothed H3 varlen attention does not support paged KV");
    return new VarlenPlan{GraphTensor(out), GraphTensor(q), GraphTensor(k), GraphTensor(v),
                          GraphTensor(cu_q), GraphTensor(cu_k), scale};
}

void varlen_run(void *meta) {
    auto *p = static_cast<VarlenPlan *>(meta);
    py::gil_scoped_acquire gil;
    try {
        py::module_::import("songshan_h3.attention_bridge").attr("varlen")(
            view(p->out), view(p->q), view(p->k), view(p->v), view(p->cu_q), view(p->cu_k),
            p->scale, reinterpret_cast<uintptr_t>(infinicore::context::getStream()));
    } catch (py::error_already_set &error) {
        throw std::runtime_error(error.what());
    }
}

void varlen_cleanup(void **meta) {
    delete static_cast<VarlenPlan *>(*meta);
    *meta = nullptr;
}
} // namespace

PYBIND11_MODULE(_attention_bridge, m) {
    m.def("register_hygon", [] {
        using namespace infinicore::op;
        const auto device = infinicore::Device::Type::HYGON;
        MultiheadAttention::plan_dispatcher().registerDevice(device, &dense_plan);
        MultiheadAttention::run_dispatcher().registerDevice(device, &dense_run);
        MultiheadAttention::cleanup_dispatcher().registerDevice(device, &dense_cleanup);
        MultiheadAttentionVarlen::plan_dispatcher().registerDevice(device, &varlen_plan);
        MultiheadAttentionVarlen::run_dispatcher().registerDevice(device, &varlen_run);
        MultiheadAttentionVarlen::cleanup_dispatcher().registerDevice(device, &varlen_cleanup);
    });
}
