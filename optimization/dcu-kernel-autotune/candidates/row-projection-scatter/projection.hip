// 对照原 rocBLAS 行并行投影、按 token 切分本地写入和对归约 owner 直接写入。
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {
constexpr int ranks = 8;
constexpr int width = 5376;
struct Rank {
    hipStream_t stream{};
    hipEvent_t start{}, done{};
    rocblas_handle blas{};
};
std::array<Rank, ranks> state{};
thread_local std::string error;

void check(hipError_t code, const char* call) {
    if (code != hipSuccess) throw std::runtime_error(std::string(call) + ": " + hipGetErrorString(code));
}

void check(rocblas_status code, const char* call) {
    if (code != rocblas_status_success)
        throw std::runtime_error(std::string(call) + ": rocBLAS status " + std::to_string(code));
}

#define CHECK(call) check((call), #call)

template<class Action>
const char* guarded(Action action) {
    try {
        action();
        return nullptr;
    } catch (const std::exception& exception) {
        error = exception.what();
        return error.c_str();
    }
}

void project(rocblas_handle handle, const uint16_t* weight, const uint16_t* input,
             uint16_t* output, int rows, int k) {
    const float alpha = 1.0f, beta = 0.0f;
    CHECK(rocblas_gemm_strided_batched_ex(handle, rocblas_operation_transpose,
        rocblas_operation_none, width, rows, k,
        &alpha, weight, rocblas_datatype_bf16_r, k, 0,
        input, rocblas_datatype_bf16_r, k, 0,
        &beta, output, rocblas_datatype_bf16_r, width, 0,
        output, rocblas_datatype_bf16_r, width, 0, 1,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
}
}

extern "C" const char* scatter_setup() {
    return guarded([] {
        int count = 0;
        CHECK(hipGetDeviceCount(&count));
        if (count != ranks || state[0].stream) throw std::runtime_error("requires eight devices and a fresh setup");
        for (int rank = 0; rank < ranks; ++rank) {
            CHECK(hipSetDevice(rank));
            auto& slot = state[rank];
            for (int peer = 0; peer < ranks; ++peer) {
                if (peer == rank) continue;
                const hipError_t status = hipDeviceEnablePeerAccess(peer, 0);
                if (status == hipErrorPeerAccessAlreadyEnabled) hipGetLastError();
                else check(status, "hipDeviceEnablePeerAccess");
            }
            CHECK(hipStreamCreateWithFlags(&slot.stream, hipStreamNonBlocking));
            CHECK(hipEventCreate(&slot.start));
            CHECK(hipEventCreate(&slot.done));
            CHECK(rocblas_create_handle(&slot.blas));
            CHECK(rocblas_set_stream(slot.blas, slot.stream));
        }
    });
}

extern "C" const char* scatter_run(int mode, int rows, int k, size_t plane_stride,
                                   const uint64_t* inputs, const uint64_t* weights,
                                   const uint64_t* outputs, float* milliseconds) {
    return guarded([&] {
        if (!state.back().blas || mode < 0 || mode > 2 || rows < ranks || k < 1 ||
            plane_stride < size_t((rows + ranks - 1) / ranks) * width)
            throw std::runtime_error("invalid row projection arguments");
        for (int rank = 0; rank < ranks; ++rank) {
            CHECK(hipSetDevice(rank));
            CHECK(hipEventRecord(state[rank].start, state[rank].stream));
        }
        for (int rank = 0; rank < ranks; ++rank) {
            CHECK(hipSetDevice(rank));
            auto& slot = state[rank];
            const auto* weight = reinterpret_cast<const uint16_t*>(weights[rank]);
            const auto* input = reinterpret_cast<const uint16_t*>(inputs[rank]);
            auto* output = reinterpret_cast<uint16_t*>(outputs[rank]);
            if (mode == 0) {
                project(slot.blas, weight, input, output, rows, k);
                continue;
            }
            for (int step = 0; step < ranks; ++step) {
                const int owner = (rank + step) % ranks;
                const size_t first = size_t(rows) * owner / ranks;
                const size_t end = size_t(rows) * (owner + 1) / ranks;
                auto* destination = mode == 1 ? output + first * width :
                    reinterpret_cast<uint16_t*>(outputs[owner]) + rank * plane_stride;
                project(slot.blas, weight, input + first * k, destination, int(end - first), k);
            }
        }
        for (int rank = 0; rank < ranks; ++rank) {
            CHECK(hipSetDevice(rank));
            CHECK(hipEventRecord(state[rank].done, state[rank].stream));
        }
        for (int rank = 0; rank < ranks; ++rank) {
            CHECK(hipSetDevice(rank));
            CHECK(hipEventSynchronize(state[rank].done));
            CHECK(hipEventElapsedTime(milliseconds + rank, state[rank].start, state[rank].done));
        }
    });
}

extern "C" const char* scatter_close() {
    return guarded([] {
        for (int rank = 0; rank < ranks; ++rank) {
            auto& slot = state[rank];
            CHECK(hipSetDevice(rank));
            if (slot.stream) CHECK(hipStreamSynchronize(slot.stream));
            if (slot.blas) CHECK(rocblas_destroy_handle(slot.blas));
            if (slot.start) CHECK(hipEventDestroy(slot.start));
            if (slot.done) CHECK(hipEventDestroy(slot.done));
            if (slot.stream) CHECK(hipStreamDestroy(slot.stream));
            slot = Rank{};
        }
    });
}
