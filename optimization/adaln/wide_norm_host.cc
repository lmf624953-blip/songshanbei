// Check and launch the NineToothed 5376-wide RMSNorm specialization.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <hip/hip_runtime_api.h>
#include <infinicore/tensor.hpp>

using infinicore::Tensor;

namespace {
constexpr int width = 5376;
std::vector<hipModule_t> modules;
std::vector<hipFunction_t> functions;
int threads = 0;
int shared = 0;
thread_local std::string error_message;

void check(hipError_t status) {
    if (status != hipSuccess)
        throw std::runtime_error(hipGetErrorString(status));
}

void dispatch(int device, int rows, const void *input, const void *weight,
              const void *output, hipStream_t stream) {
    void *scratch = nullptr;
    void *args[] = {&input, &weight, &output, &rows, &rows, &scratch, &scratch};
    check(hipModuleLaunchKernel(functions.at(device), rows, 1, 1,
                               threads, 1, 1, shared, stream, args, nullptr));
}
} // namespace

extern "C" const char *songshan_prepare_widenorm(const char *path, int warps, int shared_memory) {
    try {
        int count = 0;
        int prior = 0;
        check(hipGetDeviceCount(&count));
        check(hipGetDevice(&prior));
        modules.resize(count);
        functions.resize(count);
        threads = warps * 64;
        shared = shared_memory;
        for (int i = 0; i < count; ++i) {
            check(hipSetDevice(i));
            check(hipModuleLoad(&modules[i], path));
            check(hipModuleGetFunction(&functions[i], modules[i], "wide_norm_kernel"));
        }
        check(hipSetDevice(prior));
        return nullptr;
    } catch (const std::exception &error) {
        error_message = error.what();
        return error_message.c_str();
    }
}

extern "C" const char *songshan_bench_widenorm(int device, int rows, const void *input,
                                            const void *weight, const void *output, std::uintptr_t stream) {
    try {
        dispatch(device, rows, input, weight, output, reinterpret_cast<hipStream_t>(stream));
        return nullptr;
    } catch (const std::exception &error) {
        error_message = error.what();
        return error_message.c_str();
    }
}

extern "C" void songshan_launch_widenorm(const Tensor &out, const Tensor &input,
                                       const Tensor &weight, std::uintptr_t stream) {
    dispatch(input->device().getIndex(), input->numel() / width, input->data(),
             weight->data(), out->data(), reinterpret_cast<hipStream_t>(stream));
}
