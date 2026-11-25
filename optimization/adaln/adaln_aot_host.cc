// Launch NineToothed's specialized device binary on the InfiniCore stream.
#include <atomic>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <hip/hip_runtime_api.h>
#include <infinicore/tensor.hpp>

using infinicore::Tensor;

extern "C" void songshan_launch_adaln(const Tensor &, const Tensor &, const Tensor &,
                                     const Tensor &, std::uintptr_t);

namespace {
constexpr int width = 5376;
constexpr int modulation_stride = 6 * width;
constexpr int tile = 256;
constexpr int threads = 256;
std::vector<hipModule_t> modules;
std::vector<hipFunction_t> kernels;
std::atomic<std::uint64_t> aot_calls{0};
thread_local std::string error_message;

void check(hipError_t error) {
    if (error != hipSuccess)
        throw std::runtime_error(hipGetErrorString(error));
}

bool compatible(const Tensor &tensor, int row_stride) {
    return tensor->strides()[1] == row_stride && tensor->strides()[2] == 1
           && reinterpret_cast<std::uintptr_t>(tensor->data()) % 16 == 0;
}

void launch_native(int device, int rows, const void *normalized, const void *scale,
                   const void *shift, const void *output, hipStream_t stream) {
    void *scratch = nullptr;
    void *params[] = {&normalized, &scale, &shift, &output, &rows, &rows, &rows, &rows,
                      &scratch, &scratch};
    check(hipModuleLaunchKernel(kernels.at(device), rows * (width / tile), 1, 1,
                               threads, 1, 1, 0, stream, params, nullptr));
}
} // namespace

extern "C" const char *songshan_prepare_adaln_aot(const char *image) {
    try {
        int count = 0;
        int prior_device = 0;
        check(hipGetDeviceCount(&count));
        check(hipGetDevice(&prior_device));
        modules.resize(count);
        kernels.resize(count);
        for (int i = 0; i < count; ++i) {
            check(hipSetDevice(i));
            check(hipModuleLoad(&modules[i], image));
            check(hipModuleGetFunction(&kernels[i], modules[i], "adaln_aot_kernel"));
        }
        check(hipSetDevice(prior_device));
        return nullptr;
    } catch (const std::exception &error) {
        error_message = error.what();
        return error_message.c_str();
    }
}

extern "C" const char *songshan_bench_adaln_aot(int device, int rows, const void *normalized,
                                              const void *scale, const void *shift,
                                              const void *output, std::uintptr_t stream) {
    try {
        launch_native(device, rows, normalized, scale, shift, output,
                      reinterpret_cast<hipStream_t>(stream));
        return nullptr;
    } catch (const std::exception &error) {
        error_message = error.what();
        return error_message.c_str();
    }
}

extern "C" void songshan_launch_adaln_aot(const Tensor &out, const Tensor &normalized,
                                         const Tensor &scale, const Tensor &shift,
                                         std::uintptr_t stream) {
    const auto &shape = normalized->shape();
    if (shape[2] != width || shape[1] > std::numeric_limits<int>::max() / modulation_stride
        || !compatible(out, width) || !compatible(normalized, width)
        || !compatible(scale, modulation_stride) || !compatible(shift, modulation_stride)) {
        songshan_launch_adaln(out, normalized, scale, shift, stream);
        return;
    }
    launch_native(normalized->device().getIndex(), shape[1], normalized->data(),
                  scale->data(), shift->data(), out->data(), reinterpret_cast<hipStream_t>(stream));
    aot_calls.fetch_add(1, std::memory_order_relaxed);
}

extern "C" std::uint64_t songshan_adaln_aot_calls() {
    return aot_calls.load(std::memory_order_relaxed);
}
