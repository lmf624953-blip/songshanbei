// 九齿融合的主机发射桥；不在 C++ 中执行设备数学计算。
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
void dispatch(int device, int rows, int query_stride, int key_stride, int cache_rows,
              int positions_stride, const void *query, const void *key,
              const void *query_weight, const void *key_weight, const void *cos,
              const void *sin, const void *positions, const void *query_out,
              const void *key_out, hipStream_t stream) {
    void *scratch = nullptr;
    void *args[] = {&query, &key, &query_weight, &key_weight, &cos, &sin, &positions,
                    &query_out, &key_out, &rows, &query_stride, &key_stride,
                    &cache_rows, &positions_stride, &scratch, &scratch};
    check(hipModuleLaunchKernel(functions.at(device), static_cast<unsigned>(rows) * 7u, 1, 1,
                               threads, 1, 1, shared, stream, args, nullptr));
}
} // namespace

extern "C" const char *songshan_prepare_qknorm_mrope(const char *path, int warps, int shared_memory) {
    try {
        int count = 0, prior = 0;
        check(hipGetDeviceCount(&count)); check(hipGetDevice(&prior));
        modules.resize(count); functions.resize(count);
        if (warps != 2) throw std::runtime_error("QK norm MRoPE requires rows=1, warps=2");
        threads = warps * 64; shared = shared_memory;
        for (int i = 0; i < count; ++i) {
            check(hipSetDevice(i)); check(hipModuleLoad(&modules[i], path));
            check(hipModuleGetFunction(&functions[i], modules[i], "qk_norm_mrope_kernel"));
        }
        check(hipSetDevice(prior)); return nullptr;
    } catch (const std::exception &error) {
        error_message = error.what(); return error_message.c_str();
    }
}

extern "C" const char *songshan_bench_qknorm_mrope(
    int device, int rows, int query_stride, int key_stride, int cache_rows, int positions_stride,
    const void *query, const void *key, const void *query_weight, const void *key_weight,
    const void *cos, const void *sin, const void *positions, const void *query_out,
    const void *key_out, std::uintptr_t stream) {
    try {
        dispatch(device, rows, query_stride, key_stride, cache_rows, positions_stride,
                 query, key, query_weight, key_weight, cos, sin, positions, query_out, key_out,
                 reinterpret_cast<hipStream_t>(stream));
        return nullptr;
    } catch (const std::exception &error) {
        error_message = error.what(); return error_message.c_str();
    }
}

extern "C" void songshan_launch_qknorm_mrope(
    const Tensor &query_out, const Tensor &key_out, const Tensor &query, const Tensor &key,
    const Tensor &query_weight, const Tensor &key_weight, const Tensor &cos,
    const Tensor &sin, const Tensor &positions, std::uintptr_t stream) {
    dispatch(query->device().getIndex(), query->shape()[0], query->strides()[0], key->strides()[0],
             cos->shape()[0], 0, query->data(), key->data(),
             query_weight->data(), key_weight->data(), cos->data(), sin->data(), positions->data(),
             query_out->data(), key_out->data(), reinterpret_cast<hipStream_t>(stream));
}
