// Host-only dispatch for the fixed head layout of the NineToothed attention.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <hip/hip_runtime_api.h>
#include <infinicore/tensor.hpp>

using infinicore::Tensor;

namespace {
constexpr int heads = 7;
#ifndef H3_ATTENTION_QUERY_BLOCK
#define H3_ATTENTION_QUERY_BLOCK 64
#endif
constexpr int query_block = H3_ATTENTION_QUERY_BLOCK;
std::vector<hipModule_t> modules;
std::vector<hipFunction_t> functions;
std::vector<hipModule_t> fallback_modules;
std::vector<hipFunction_t> fallback_functions;
#ifndef H3_ATTENTION_BUFFER_LIMIT
#define H3_ATTENTION_BUFFER_LIMIT 2147483648LL
#endif
int fallback_shared = 0;

bool buffer_eligible(int rows, int q_stride, int k_stride, int v_stride, int out_stride) {
    // 32位有符号相对寻址；逐指针检查最后一元素的末字节，包含head/width。
    if (rows <= 0) return false;
    for (int stride : {q_stride, k_stride, v_stride, out_stride}) {
        if (stride <= 0) return false;
        const std::int64_t elements = (static_cast<std::int64_t>(rows) - 1) * stride + heads * 128;
        if (elements * 2 > H3_ATTENTION_BUFFER_LIMIT) return false;
    }
    return true;
}
int threads = 0;
int shared = 0;
thread_local std::string error_message;

void check(hipError_t status) {
    if (status != hipSuccess)
        throw std::runtime_error(hipGetErrorString(status));
}

void dispatch(int device, int rows, float scale, const void *q, const void *k,
              const void *v, const void *out, int q_stride, int k_stride,
              int v_stride, int out_stride, hipStream_t stream) {
    void *scratch = nullptr;
    void *args[] = {&q, &k, &v, &scale, &out, &rows, &q_stride, &rows, &k_stride,
                    &rows, &v_stride, &rows, &out_stride, &scratch, &scratch};
    const bool use_buffer = buffer_eligible(rows, q_stride, k_stride, v_stride, out_stride);
    const auto function = use_buffer ? functions.at(device) : fallback_functions.at(device);
    const int selected_shared = use_buffer ? shared : fallback_shared;
    check(hipModuleLaunchKernel(function, heads * ((rows + query_block - 1) / query_block),
                               1, 1, threads, 1, 1, selected_shared, stream, args, nullptr));
}
} // namespace

extern "C" int songshan_attention_query_block() { return query_block; }
extern "C" int songshan_attention_buffer_eligible(int rows, int q_stride, int k_stride,
                                                  int v_stride, int out_stride) {
    return buffer_eligible(rows, q_stride, k_stride, v_stride, out_stride) ? 1 : 0;
}

extern "C" const char *songshan_prepare_buffer_attention(const char *path, const char *fallback,
    int warps, int shared_memory, int fallback_shared_memory) {
    try {
        int count = 0;
        int prior = 0;
        check(hipGetDeviceCount(&count));
        check(hipGetDevice(&prior));
        modules.resize(count);
        functions.resize(count);
        fallback_modules.resize(count);
        fallback_functions.resize(count);
        threads = warps * 64;
        shared = shared_memory;
        fallback_shared = fallback_shared_memory;
        for (int i = 0; i < count; ++i) {
            check(hipSetDevice(i));
            check(hipModuleLoad(&modules[i], path));
            check(hipModuleGetFunction(&functions[i], modules[i], "dense_attention_aot_kernel"));
            check(hipModuleLoad(&fallback_modules[i], fallback));
            check(hipModuleGetFunction(&fallback_functions[i], fallback_modules[i], "dense_attention_aot_kernel"));
        }
        check(hipSetDevice(prior));
        return nullptr;
    } catch (const std::exception &error) {
        error_message = error.what();
        return error_message.c_str();
    }
}

extern "C" const char *songshan_bench_attention(int device, int rows, float scale,
                                               const void *q, const void *k, const void *v,
                                               const void *out, int q_stride, int k_stride,
                                               int v_stride, int out_stride, std::uintptr_t stream) {
    try {
        dispatch(device, rows, scale, q, k, v, out, q_stride, k_stride, v_stride, out_stride,
                 reinterpret_cast<hipStream_t>(stream));
        return nullptr;
    } catch (const std::exception &error) {
        error_message = error.what();
        return error_message.c_str();
    }
}

extern "C" void songshan_launch_attention(const Tensor &out, const Tensor &q,
                                          const Tensor &k, const Tensor &v,
                                          float scale, std::uintptr_t stream) {
    dispatch(q->device().getIndex(), q->shape()[1], scale, q->data(), k->data(), v->data(),
             out->data(), q->strides()[1], k->strides()[1], v->strides()[1], out->strides()[1],
             reinterpret_cast<hipStream_t>(stream));
}
