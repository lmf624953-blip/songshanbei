// 装载预排布 V 的实验注意力；九齿预处理通过原保留指针参数传入。
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<hipModule_t> modules;
std::vector<hipFunction_t> functions;
std::vector<hipModule_t> prepack_modules;
std::vector<hipFunction_t> prepack_functions;
int prepack_tile = 1024;
int prepack_threads = 256;
thread_local std::string message;

void check(hipError_t status) {
    if (status != hipSuccess) {
        throw std::runtime_error(hipGetErrorString(status));
    }
}

void dispatch_prepack(int device, int rows, const void* value, const void* packed, std::uintptr_t stream) {
    int blocks = rows / 32;
    int elements = 7 * blocks * 4096;
    void* reserved = nullptr;
    void* args[] = {&value, &blocks, &packed, &rows, &elements, &reserved, &reserved};
    check(hipModuleLaunchKernel(prepack_functions.at(device), (elements + prepack_tile - 1) / prepack_tile,
                               1, 1, prepack_threads, 1, 1, 0, reinterpret_cast<hipStream_t>(stream),
                               args, nullptr));
}
}

extern "C" const char* prepare_prepacked_attention(const char* path) {
    try {
        int count, prior;
        check(hipGetDeviceCount(&count));
        check(hipGetDevice(&prior));
        modules.resize(count);
        functions.resize(count);
        for (int device = 0; device < count; ++device) {
            check(hipSetDevice(device));
            check(hipModuleLoad(&modules[device], path));
            check(hipModuleGetFunction(&functions[device], modules[device], "dense_attention_aot_kernel"));
        }
        check(hipSetDevice(prior));
        return nullptr;
    } catch (const std::exception& error) {
        message = error.what();
        return message.c_str();
    }
}

extern "C" const char* prepare_value_prepack(const char* path, int tile, int warps) {
    try {
        if (tile < 256 || tile > 4096 || (tile & (tile - 1)) != 0 ||
            (warps != 2 && warps != 4 && warps != 8)) {
            throw std::runtime_error("Unsupported value prepack geometry");
        }
        prepack_tile = tile;
        prepack_threads = warps * 64;
        int count, prior;
        check(hipGetDeviceCount(&count));
        check(hipGetDevice(&prior));
        prepack_modules.resize(count);
        prepack_functions.resize(count);
        for (int device = 0; device < count; ++device) {
            check(hipSetDevice(device));
            check(hipModuleLoad(&prepack_modules[device], path));
            check(hipModuleGetFunction(&prepack_functions[device], prepack_modules[device],
                                       "h3_prepack_value_kernel"));
        }
        check(hipSetDevice(prior));
        return nullptr;
    } catch (const std::exception& error) {
        message = error.what();
        return message.c_str();
    }
}

extern "C" const char* bench_prepacked_attention(
    int device, int rows, float scale, const void* query, const void* key,
    const void* value, void* output, const void* packed, std::uintptr_t stream) {
    try {
        if (rows < 32 || packed == nullptr) {
            throw std::runtime_error("Prepacked experiment requires a complete K/V block");
        }
        if (!prepack_functions.empty()) {
            dispatch_prepack(device, rows, value, packed, stream);
        }
        int q_stride = 896, k_stride = 896, v_stride = 2688, out_stride = 896;
        void* reserved = nullptr;
        void* args[] = {&query, &key, &value, &scale, &output, &rows, &q_stride,
                        &rows, &k_stride, &rows, &v_stride, &rows, &out_stride,
                        &packed, &reserved};
        check(hipModuleLaunchKernel(functions.at(device), 7 * ((rows + 127) / 128), 1, 1,
                                   256, 1, 1, 16384, reinterpret_cast<hipStream_t>(stream),
                                   args, nullptr));
        return nullptr;
    } catch (const std::exception& error) {
        message = error.what();
        return message.c_str();
    }
}

extern "C" const char* bench_value_prepack(
    int device, int rows, const void* value, const void* packed, std::uintptr_t stream) {
    try {
        dispatch_prepack(device, rows, value, packed, stream);
        return nullptr;
    } catch (const std::exception& error) {
        message = error.what();
        return message.c_str();
    }
}
