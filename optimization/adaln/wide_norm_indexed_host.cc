// 九齿宽 RMSNorm 与索引 AdaLN 融合的主机发射桥。
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <hip/hip_runtime_api.h>
#include <infinicore/tensor.hpp>

using infinicore::Tensor;
namespace {
std::vector<hipModule_t> modules;
std::vector<hipFunction_t> functions;
int threads = 0, shared = 0;
thread_local std::string error_message;
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
void dispatch(int device, int rows, int table_rows, const void *input, const void *weight,
              const void *scale, const void *shift, const void *indices, const void *output,
              hipStream_t stream) {
    void *scratch = nullptr;
    void *args[] = {&input, &weight, &scale, &shift, &indices, &output,
                    &rows, &table_rows, &scratch, &scratch};
    check(hipModuleLaunchKernel(functions.at(device), rows, 1, 1,
                               threads, 1, 1, shared, stream, args, nullptr));
}
}

extern "C" const char *songshan_prepare_widenorm_indexed(const char *path, int warps, int shared_memory) {
    try {
        if (warps != 4 && warps != 8 && warps != 16)
            throw std::runtime_error("Unsupported wide norm indexed launch geometry");
        int count = 0, prior = 0;
        check(hipGetDeviceCount(&count)); check(hipGetDevice(&prior));
        modules.resize(count); functions.resize(count);
        threads = warps * 64; shared = shared_memory;
        for (int device = 0; device < count; ++device) {
            check(hipSetDevice(device)); check(hipModuleLoad(&modules[device], path));
            check(hipModuleGetFunction(&functions[device], modules[device], "wide_norm_indexed_kernel"));
        }
        check(hipSetDevice(prior));
        return nullptr;
    } catch (const std::exception &error) {
        error_message = error.what(); return error_message.c_str();
    }
}

extern "C" const char *songshan_bench_widenorm_indexed(
    int device, int rows, int table_rows, const void *input, const void *weight,
    const void *scale, const void *shift, const void *indices, const void *output,
    std::uintptr_t stream) {
    try {
        dispatch(device, rows, table_rows, input, weight, scale, shift, indices, output,
                 reinterpret_cast<hipStream_t>(stream));
        return nullptr;
    } catch (const std::exception &error) {
        error_message = error.what(); return error_message.c_str();
    }
}

extern "C" void songshan_launch_widenorm_indexed(
    const Tensor &output, const Tensor &input, const Tensor &weight,
    const Tensor &scale, const Tensor &shift, const Tensor &indices, std::uintptr_t stream) {
    dispatch(input->device().getIndex(), input->shape()[1], scale->shape()[1],
             input->data(), weight->data(), scale->data(), shift->data(), indices->data(),
             output->data(), reinterpret_cast<hipStream_t>(stream));
}
