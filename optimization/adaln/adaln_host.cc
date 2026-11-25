// The host shim exposes strided ATen views and launches on the InfiniCore stream.
#include <cstdint>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/python_variable.h>
#include <infinicore/adaptor/aten_adaptor.hpp>

namespace py = pybind11;
using infinicore::Tensor;

namespace {
py::object view(const Tensor &tensor) {
    return py::reinterpret_steal<py::object>(
        THPVariable_Wrap(infinicore::adaptor::to_aten_tensor(tensor)));
}
} // namespace

extern "C" void songshan_launch_adaln(const Tensor &out,
                                     const Tensor &normalized,
                                     const Tensor &scale,
                                     const Tensor &shift,
                                     std::uintptr_t stream) {
    py::gil_scoped_acquire gil;
    try {
        py::module_::import("songshan_adaln_kernel").attr("launch")(
            view(out), view(normalized), view(scale), view(shift), stream);
    } catch (py::error_already_set &error) {
        throw std::runtime_error(error.what());
    }
}
