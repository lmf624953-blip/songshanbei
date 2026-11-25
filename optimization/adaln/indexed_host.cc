// Dispatch the NineToothed indexed AdaLN and residual kernels on InfiniCore's stream.
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <hip/hip_runtime_api.h>
#include <infinicore/tensor.hpp>

using infinicore::Tensor;

#ifndef H3_INDEXED_TILE
#define H3_INDEXED_TILE 256
#endif

namespace {
constexpr int width = 5376;
constexpr int tile = H3_INDEXED_TILE;
struct DeviceKernels {
    std::array<hipModule_t, 2> modules{};
    std::array<hipFunction_t, 2> functions{};
};
std::vector<DeviceKernels> devices;
int thread_count = 0;
thread_local std::string error_message;

void check(hipError_t error) {
    if (error != hipSuccess)
        throw std::runtime_error(hipGetErrorString(error));
}

void launch_native(bool gate, int device, int rows, int table_rows, const void *first,
                   const void *second, const void *table, const void *indices,
                   const void *output, hipStream_t stream) {
    int second_rows = gate ? rows : table_rows;
    void *scratch = nullptr;
    void *params[] = {&first, &second, &table, &indices, &output, &rows, &second_rows,
                      &table_rows, &rows, &rows, &scratch, &scratch};
    check(hipModuleLaunchKernel(devices.at(device).functions[gate], rows * ((width + tile - 1) / tile), 1, 1,
                               thread_count, 1, 1, 0, stream, params, nullptr));
}
} // namespace

extern "C" int songshan_indexed_tile() { return tile; }

extern "C" const char *songshan_prepare_indexed(const char *adaln, const char *gate, int warps) {
    try {
        int count = 0;
        int prior_device = 0;
        check(hipGetDeviceCount(&count));
        check(hipGetDevice(&prior_device));
        devices.resize(count);
        const char *paths[] = {adaln, gate};
        const char *symbols[] = {"indexed_adaln_kernel", "indexed_gate_kernel"};
        thread_count = warps * 64;
        for (int i = 0; i < count; ++i) {
            check(hipSetDevice(i));
            for (int op = 0; op < 2; ++op) {
                check(hipModuleLoad(&devices[i].modules[op], paths[op]));
                check(hipModuleGetFunction(&devices[i].functions[op], devices[i].modules[op], symbols[op]));
            }
        }
        check(hipSetDevice(prior_device));
        return nullptr;
    } catch (const std::exception &error) {
        error_message = error.what();
        return error_message.c_str();
    }
}

extern "C" const char *songshan_bench_indexed(bool gate, int device, int rows, int table_rows,
                                             const void *first, const void *second, const void *table,
                                             const void *indices, const void *output, std::uintptr_t stream) {
    try {
        launch_native(gate, device, rows, table_rows, first, second, table, indices, output,
                      reinterpret_cast<hipStream_t>(stream));
        return nullptr;
    } catch (const std::exception &error) {
        error_message = error.what();
        return error_message.c_str();
    }
}

extern "C" void songshan_launch_indexed(bool gate, const Tensor &out, const Tensor &first,
                                        const Tensor &second, const Tensor &table,
                                        const Tensor &indices, std::uintptr_t stream) {
    launch_native(gate, out->device().getIndex(), out->shape()[1], table->shape()[1], first->data(),
                  second->data(), table->data(), indices->data(), out->data(),
                  reinterpret_cast<hipStream_t>(stream));
}
