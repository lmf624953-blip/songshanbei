// 与冻结的 attention_buffer_host.cc 合并编译。仅主机调度和工作区管理。
#include <map>
#include <mutex>

namespace {
std::vector<hipModule_t> value_modules, row_packed_modules;
std::vector<hipFunction_t> value_functions, row_packed_functions;
int value_tile = 0, value_threads = 0, row_packed_shared = 0;
struct ValueWorkspace {
    void *pointer = nullptr;
    std::size_t bytes = 0;
};
std::map<std::pair<int, std::uintptr_t>, ValueWorkspace> value_workspaces;
std::mutex value_mutex;

bool prepack_eligible(int rows, int qs, int ks, int vs, int os) {
    return rows >= 4096 && qs == 896 && ks == 896 && vs == 2688 && os == 896 &&
           buffer_eligible(rows, qs, ks, vs, os) && !value_functions.empty();
}

void dispatch_prepacked_or_reference(int device, int rows, float scale, const void *q,
    const void *k, const void *v, const void *out, int qs, int ks, int vs, int os,
    hipStream_t stream) {
    if (!prepack_eligible(rows, qs, ks, vs, os)) {
        dispatch_reference(device, rows, scale, q, k, v, out, qs, ks, vs, os, stream);
        return;
    }
    // 同一 stream 的预排布和主注意力成对提交；不同 stream 拥有独立工作区。
    std::lock_guard<std::mutex> lock(value_mutex);
    auto &workspace = value_workspaces[{device, reinterpret_cast<std::uintptr_t>(stream)}];
    int blocks = rows / 32;
    int elements = 7 * blocks * 4096;
    const std::size_t bytes = static_cast<std::size_t>(elements) * 2;
    if (workspace.bytes < bytes) {
        if (workspace.pointer != nullptr) {
            check(hipFree(workspace.pointer));
            workspace = {};
        }
        check(hipMalloc(&workspace.pointer, bytes));
        workspace.bytes = bytes;
    }
    void *reserved = nullptr;
    void *pack_args[] = {&v, &blocks, &workspace.pointer, &rows, &elements, &reserved, &reserved};
    check(hipModuleLaunchKernel(value_functions.at(device), (elements + value_tile - 1) / value_tile,
                               1, 1, value_threads, 1, 1, 0, stream, pack_args, nullptr));
    void *args[] = {&q, &k, &v, &scale, &out, &rows, &qs, &rows, &ks,
                    &rows, &vs, &rows, &os, &workspace.pointer, &reserved};
    check(hipModuleLaunchKernel(row_packed_functions.at(device), 7 * ((rows + 127) / 128),
                               1, 1, 256, 1, 1, row_packed_shared, stream, args, nullptr));
}
}

extern "C" const char *songshan_prepare_prepacked_value(const char *attention, const char *value,
    int tile, int warps, int lds) {
    int prior = 0;
    try {
        if (tile < 256 || tile > 4096 || (tile & (tile - 1)) != 0 ||
            (warps != 2 && warps != 4 && warps != 8) || query_block != 128 || lds != 16384) {
            throw std::runtime_error("Unsupported prepacked attention geometry");
        }
        int count = 0;
        check(hipGetDeviceCount(&count));
        check(hipGetDevice(&prior));
        value_modules.resize(count);
        value_functions.resize(count);
        row_packed_modules.resize(count);
        row_packed_functions.resize(count);
        for (int device = 0; device < count; ++device) {
            check(hipSetDevice(device));
            check(hipModuleLoad(&value_modules[device], value));
            check(hipModuleGetFunction(&value_functions[device], value_modules[device],
                                       "h3_prepack_value_kernel"));
            check(hipModuleLoad(&row_packed_modules[device], attention));
            check(hipModuleGetFunction(&row_packed_functions[device], row_packed_modules[device],
                                       "dense_attention_aot_kernel"));
        }
        check(hipSetDevice(prior));
        value_tile = tile;
        value_threads = warps * 64;
        row_packed_shared = lds;
        return nullptr;
    } catch (const std::exception &error) {
        static_cast<void>(hipSetDevice(prior));
        error_message = error.what();
        return error_message.c_str();
    }
}

extern "C" int songshan_attention_prepack_eligible(int rows, int qs, int ks, int vs, int os) {
    return prepack_eligible(rows, qs, ks, vs, os) ? 1 : 0;
}

extern "C" const char *songshan_bench_attention(int device, int rows, float scale,
    const void *q, const void *k, const void *v, const void *out,
    int qs, int ks, int vs, int os, std::uintptr_t stream) {
    try {
        dispatch_prepacked_or_reference(device, rows, scale, q, k, v, out, qs, ks, vs, os,
                                         reinterpret_cast<hipStream_t>(stream));
        return nullptr;
    } catch (const std::exception &error) {
        error_message = error.what();
        return error_message.c_str();
    }
}

extern "C" void songshan_launch_attention(const Tensor &out, const Tensor &q,
    const Tensor &k, const Tensor &v, float scale, std::uintptr_t stream) {
    dispatch_prepacked_or_reference(q->device().getIndex(), q->shape()[1], scale,
        q->data(), k->data(), v->data(), out->data(), q->strides()[1], k->strides()[1],
        v->strides()[1], out->strides()[1], reinterpret_cast<hipStream_t>(stream));
}
