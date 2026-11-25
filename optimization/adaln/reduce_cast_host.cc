// Host dispatch only: the floating casts are compiled from NineToothed.
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <hip/hip_runtime_api.h>
#include <infinicore/tensor.hpp>

using infinicore::Tensor;
namespace {
struct Kernels {
    std::array<hipModule_t, 2> modules{};
    std::array<hipFunction_t, 2> functions{};
};
std::vector<Kernels> kernels;
thread_local std::string error_message;
void check(hipError_t result) {
    if (result != hipSuccess) throw std::runtime_error(hipGetErrorString(result));
}
void dispatch(int device, int down, int count, const void *input, const void *output, hipStream_t stream) {
    void *scratch = nullptr;
    void *args[] = {&input, &output, &count, &count, &scratch, &scratch};
    check(hipModuleLaunchKernel(kernels.at(device).functions.at(down), (count + 1023) / 1024,
                               1, 1, 256, 1, 1, 0, stream, args, nullptr));
}
}

extern "C" const char *songshan_prepare_reducecast(const char *up, const char *down) {
    try {
        int count = 0, prior = 0;
        check(hipGetDeviceCount(&count));
        check(hipGetDevice(&prior));
        kernels.resize(count);
        const char *paths[] = {up, down};
        const char *names[] = {"reducecast_up_kernel", "reducecast_down_kernel"};
        for (int device = 0; device < count; ++device) {
            check(hipSetDevice(device));
            for (int op = 0; op < 2; ++op) {
                check(hipModuleLoad(&kernels[device].modules[op], paths[op]));
                check(hipModuleGetFunction(&kernels[device].functions[op], kernels[device].modules[op], names[op]));
            }
        }
        check(hipSetDevice(prior));
        return nullptr;
    } catch (const std::exception &error) {
        error_message = error.what();
        return error_message.c_str();
    }
}

extern "C" const char *songshan_bench_reducecast(int device, int down, int count,
                                                 const void *input, void *output, std::uintptr_t stream) {
    try {
        dispatch(device, down, count, input, output, reinterpret_cast<hipStream_t>(stream));
        return nullptr;
    } catch (const std::exception &error) {
        error_message = error.what();
        return error_message.c_str();
    }
}

extern "C" void songshan_launch_reducecast(const Tensor &out, const Tensor &input, std::uintptr_t stream) {
    dispatch(input->device().getIndex(), input->dtype() == infinicore::DataType::BF16 ? 0 : 1,
             input->numel(), input->data(), out->data(), reinterpret_cast<hipStream_t>(stream));
}
