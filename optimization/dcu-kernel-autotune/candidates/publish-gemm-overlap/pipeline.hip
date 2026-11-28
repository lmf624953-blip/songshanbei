// 九齿归一化和原 rocBLAS 负责计算。本文件安排行发布与各投影分块的依赖。
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {
constexpr int ranks = 8;
constexpr int width = 5376;
constexpr int row_bytes = width * sizeof(uint16_t);
using Norm = const char* (*)(int, int, int, const void*, const void*, const void*,
                            const void*, const void*, const void*, uintptr_t);

struct Destinations {
    uint16_t* data[ranks];
};

struct Rank {
    hipStream_t publish = nullptr;
    hipStream_t compute = nullptr;
    hipEvent_t begin = nullptr;
    hipEvent_t ready[2]{};
    hipEvent_t done = nullptr;
    rocblas_handle blas = nullptr;
};

std::array<Rank, ranks> state;
Norm normalize = nullptr;
thread_local std::string error;

void check(hipError_t status, const char* call) {
    if (status != hipSuccess)
        throw std::runtime_error(std::string(call) + ": " + hipGetErrorString(status));
}

#define HIP(call) check((call), #call)

void blas_check(rocblas_status status) {
    if (status != rocblas_status_success)
        throw std::runtime_error("rocBLAS status " + std::to_string(status));
}

template<class Function>
const char* run_checked(Function function) {
    try {
        function();
        return nullptr;
    } catch (const std::exception& exception) {
        error = exception.what();
        return error.c_str();
    }
}

__global__ void publish_rows(const uint16_t* source, Destinations destinations,
                             size_t first, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x) {
        const uint16_t value = source[i];
        #pragma unroll
        for (int destination = 0; destination < ranks; ++destination)
            destinations.data[destination][first + i] = value;
    }
}

template<class Word>
__global__ void publish_vector(const Word* source, Destinations destinations,
                               size_t first, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x) {
        const Word value = source[i];
        #pragma unroll
        for (int destination = 0; destination < ranks; ++destination)
            reinterpret_cast<Word*>(destinations.data[destination])[first + i] = value;
    }
}

template<class Word>
void launch_vector(const uint16_t* source, Destinations destinations,
                   size_t first, size_t count, int blocks, hipStream_t stream) {
    constexpr size_t elements = sizeof(Word) / sizeof(uint16_t);
    if (first % elements || count % elements || reinterpret_cast<uintptr_t>(source) % sizeof(Word))
        throw std::runtime_error("unaligned word publication");
    for (const auto* destination : destinations.data) {
        if (reinterpret_cast<uintptr_t>(destination) % sizeof(Word))
            throw std::runtime_error("unaligned publication destination");
    }
    hipLaunchKernelGGL((publish_vector<Word>), dim3(blocks), dim3(256), 0, stream,
        reinterpret_cast<const Word*>(source), destinations, first / elements, count / elements);
}

void project(Rank& rank, int rows, int columns, int first, const void* weight,
             const uint16_t* input, uint16_t* output) {
    const float alpha = 1.f;
    const float beta = 0.f;
    output += size_t(first) * columns;
    input += size_t(first) * width;
    blas_check(rocblas_gemm_strided_batched_ex(rank.blas, rocblas_operation_transpose,
        rocblas_operation_none, columns, rows, width,
        &alpha, weight, rocblas_datatype_bf16_r, width, 0,
        input, rocblas_datatype_bf16_r, width, 0,
        &beta, output, rocblas_datatype_bf16_r, columns, 0,
        output, rocblas_datatype_bf16_r, columns, 0, 1,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0));
}
}

extern "C" const char* pipeline_setup(void* norm) {
    return run_checked([&] {
        if (normalize || !norm) throw std::runtime_error("invalid pipeline setup");
        int count = 0;
        HIP(hipGetDeviceCount(&count));
        if (count != ranks) throw std::runtime_error("requires eight devices");
        normalize = reinterpret_cast<Norm>(norm);
        for (int i = 0; i < ranks; ++i) {
            HIP(hipSetDevice(i));
            for (int j = 0; j < ranks; ++j) {
                if (i == j) continue;
                const auto status = hipDeviceEnablePeerAccess(j, 0);
                if (status == hipErrorPeerAccessAlreadyEnabled) hipGetLastError();
                else check(status, "hipDeviceEnablePeerAccess");
            }
            auto& rank = state[i];
            HIP(hipStreamCreateWithFlags(&rank.publish, hipStreamNonBlocking));
            HIP(hipStreamCreateWithFlags(&rank.compute, hipStreamNonBlocking));
            HIP(hipEventCreate(&rank.begin));
            HIP(hipEventCreate(&rank.done));
            for (auto& event : rank.ready) HIP(hipEventCreateWithFlags(&event, hipEventDisableTiming));
            blas_check(rocblas_create_handle(&rank.blas));
            blas_check(rocblas_set_stream(rank.blas, rank.compute));
        }
    });
}

extern "C" const char* pipeline_run(int mode, int rows, int columns, int split, int blocks,
                                    const uint64_t* hidden, const uint64_t* norm_weight,
                                    const uint64_t* scale, const uint64_t* shift,
                                    const uint64_t* indices, const uint64_t* partial,
                                    const uint64_t* local, const uint64_t* weights,
                                    const uint64_t* outputs, float* elapsed) {
    return run_checked([&] {
        if (!normalize || mode < 0 || mode > 6 || rows <= 0 || columns <= 0 ||
            split <= 0 || split >= rows || blocks <= 0)
            throw std::runtime_error("invalid pipeline arguments");
        const int phases = mode == 1 || mode == 2 ? 2 : 1;
        const int boundary[3]{0, phases == 1 ? rows : split, rows};
        Destinations destinations{};
        for (int i = 0; i < ranks; ++i) {
            destinations.data[i] = reinterpret_cast<uint16_t*>(local[i]);
            HIP(hipSetDevice(i));
            const auto first = size_t(rows) * i / ranks;
            const auto last = size_t(rows) * (i + 1) / ranks;
            auto& rank = state[i];
            HIP(hipEventRecord(rank.begin, rank.publish));
            const char* status = normalize(i, int(last - first), 3,
                reinterpret_cast<void*>(hidden[i] + first * row_bytes),
                reinterpret_cast<void*>(norm_weight[i]), reinterpret_cast<void*>(scale[i]),
                reinterpret_cast<void*>(shift[i]), reinterpret_cast<void*>(indices[i] + first * 8),
                reinterpret_cast<void*>(partial[i]), reinterpret_cast<uintptr_t>(rank.publish));
            if (status) throw std::runtime_error(status);
        }
        for (int phase = 0; phase < phases; ++phase) {
            for (int i = 0; i < ranks; ++i) {
                const size_t owner_first = size_t(rows) * i / ranks;
                const size_t owner_last = size_t(rows) * (i + 1) / ranks;
                const size_t first = std::max(owner_first, size_t(boundary[phase]));
                const size_t last = std::min(owner_last, size_t(boundary[phase + 1]));
                if (last <= first) continue;
                HIP(hipSetDevice(i));
                const auto* base = reinterpret_cast<const uint16_t*>(partial[i]);
                auto stream = state[i].publish;
                const auto* source = base + (first - owner_first) * width;
                const size_t offset = first * width;
                const size_t count = (last - first) * width;
                if (mode == 3) launch_vector<uint32_t>(source, destinations, offset, count, blocks, stream);
                else if (mode == 4) launch_vector<uint2>(source, destinations, offset, count, blocks, stream);
                else if (mode == 5) launch_vector<uint4>(source, destinations, offset, count, blocks, stream);
                else if (mode == 6) {
                    for (int distance = 0; distance < ranks; ++distance) {
                        const int destination = (i + distance) % ranks;
                        auto* target = destinations.data[destination] + offset;
                        const size_t bytes = count * sizeof(uint16_t);
                        if (destination == i)
                            HIP(hipMemcpyAsync(target, source, bytes, hipMemcpyDeviceToDevice, stream));
                        else
                            HIP(hipMemcpyPeerAsync(target, destination, source, i, bytes, stream));
                    }
                }
                else hipLaunchKernelGGL(publish_rows, dim3(blocks), dim3(256), 0, stream,
                    source, destinations, offset, count);
                HIP(hipGetLastError());
                HIP(hipEventRecord(state[i].ready[phase], stream));
            }
        }
        for (int phase = 0; phase < phases; ++phase) {
            for (int i = 0; i < ranks; ++i) {
                HIP(hipSetDevice(i));
                auto& rank = state[i];
                if (!phase) HIP(hipStreamWaitEvent(rank.compute, rank.begin, 0));
                const int first_wait = mode == 1 && phase == 0 ? 0 : phase;
                const int last_wait = mode == 1 && phase == 0 ? phases : phase + 1;
                for (int slot = first_wait; slot < last_wait; ++slot) {
                    for (int producer = 0; producer < ranks; ++producer) {
                        const size_t first = size_t(rows) * producer / ranks;
                        const size_t last = size_t(rows) * (producer + 1) / ranks;
                        if (first < size_t(boundary[slot + 1]) && last > size_t(boundary[slot]))
                            HIP(hipStreamWaitEvent(rank.compute, state[producer].ready[slot], 0));
                    }
                }
                project(rank, boundary[phase + 1] - boundary[phase], columns, boundary[phase],
                        reinterpret_cast<const void*>(weights[i]),
                        destinations.data[i], reinterpret_cast<uint16_t*>(outputs[i]));
                if (phase == phases - 1) HIP(hipEventRecord(rank.done, rank.compute));
            }
        }
        for (int i = 0; i < ranks; ++i) {
            HIP(hipSetDevice(i));
            HIP(hipEventSynchronize(state[i].done));
            HIP(hipEventElapsedTime(elapsed + i, state[i].begin, state[i].done));
        }
    });
}

extern "C" const char* pipeline_close() {
    return run_checked([&] {
        for (int i = 0; i < ranks; ++i) {
            HIP(hipSetDevice(i));
            auto& rank = state[i];
            if (rank.publish) HIP(hipStreamSynchronize(rank.publish));
            if (rank.compute) HIP(hipStreamSynchronize(rank.compute));
            if (rank.blas) blas_check(rocblas_destroy_handle(rank.blas));
            for (auto event : rank.ready) if (event) HIP(hipEventDestroy(event));
            if (rank.begin) HIP(hipEventDestroy(rank.begin));
            if (rank.done) HIP(hipEventDestroy(rank.done));
            if (rank.publish) HIP(hipStreamDestroy(rank.publish));
            if (rank.compute) HIP(hipStreamDestroy(rank.compute));
            rank = Rank{};
        }
        normalize = nullptr;
    });
}
