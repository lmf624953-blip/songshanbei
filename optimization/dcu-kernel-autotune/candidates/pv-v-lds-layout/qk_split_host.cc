// 从九齿同源镜像发射QK及softmax/PV阶段；只负责模块与发射。
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {
hipModule_t modules[3];
hipFunction_t kernels[3];
thread_local std::string message;
void check(hipError_t code) {
    if (code != hipSuccess) throw std::runtime_error(hipGetErrorString(code));
}
}

extern "C" const char* prepare_split(const char* qk, const char* pv, const char* prepack) {
    try {
        const char* paths[] = {qk, pv, prepack};
        const char* names[] = {"h3_split_qk_kernel", "h3_split_softmax_pv_kernel", "h3_prepack_value_kernel"};
        for (int i = 0; i < 3; ++i) {
            check(hipModuleLoad(&modules[i], paths[i]));
            check(hipModuleGetFunction(&kernels[i], modules[i], names[i]));
        }
        return nullptr;
    } catch (const std::exception& error) { message = error.what(); return message.c_str(); }
}

extern "C" const char* bench_split(int rows, float scale, const void* query, const void* key,
    const void* value, void* output, void* packed, void* scores, int batch,
    int pack_tile, int pack_threads, std::uintptr_t stream_raw, int phase) {
    try {
        auto stream = reinterpret_cast<hipStream_t>(stream_raw);
        int blocks = rows / 32;
        int elements = 7 * blocks * 4096;
        void* reserved = nullptr;
        void* pack_args[] = {&value, &blocks, &packed, &rows, &elements, &reserved, &reserved};
        if (phase != 1) check(hipModuleLaunchKernel(kernels[2], (elements + pack_tile - 1) / pack_tile,
            1, 1, pack_threads, 1, 1, 0, stream, pack_args, nullptr));
        int qs = 896, ks = 896, vs = 2688, os = 896;
        const int total = 7 * ((rows + 127) / 128);
        for (int start = 0; start < total; start += batch) {
            void* args[] = {&query, &key, &value, &scale, &output, &rows, &qs, &rows, &ks,
                &rows, &vs, &rows, &os, &packed, &scores, &start};
            const int grid = std::min(batch, total - start);
            if (phase != 2) check(hipModuleLaunchKernel(kernels[0], grid, 1, 1, 256, 1, 1,
                16384, stream, args, nullptr));
            if (phase != 1) check(hipModuleLaunchKernel(kernels[1], grid, 1, 1, 256, 1, 1,
                16384, stream, args, nullptr));
        }
        return nullptr;
    } catch (const std::exception& error) { message = error.what(); return message.c_str(); }
}
