#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr unsigned lanes = 64;
constexpr unsigned waves = 2;
constexpr unsigned lds_words_per_wave = 512;
constexpr unsigned input_words_per_wave = 256;
constexpr uint32_t sentinel = 0xdeadbeef;
using resource_t = uint32_t __attribute__((ext_vector_type(4)));
constexpr unsigned k_rows = 32;
constexpr unsigned k_stride_words = 448;
constexpr unsigned k_tile_words = 2048;

enum class Transfer { Global, Buffer, BufferWide, BufferControl };

template <Transfer transfer>
__global__ void copy_probe(const uint32_t* input, uint32_t* output) {
    extern __shared__ uint32_t scratch[];
    unsigned tid = threadIdx.x;
    unsigned wave = tid / lanes;
    unsigned lane = tid % lanes;
    for (unsigned i = tid; i < lds_words_per_wave * waves; i += blockDim.x) {
        scratch[i] = sentinel;
    }
    __syncthreads();

    constexpr unsigned width = transfer == Transfer::BufferWide ? 4 : 1;
    auto base = reinterpret_cast<uintptr_t>(input);
    unsigned offset = (wave * input_words_per_wave + lane * width) * sizeof(uint32_t);
    unsigned lds_start = wave * lds_words_per_wave * sizeof(uint32_t);
    unsigned m0_value = __builtin_amdgcn_readfirstlane(lds_start);
    if constexpr (transfer == Transfer::Global) {
        uint32_t unused;
        asm volatile(
            "s_mov_b32 m0, %2\n\t"
            "global_load_dword %0, %1, off lds\n\t"
            "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
            "s_mov_b32 m0, -1"
            : "=v"(unused) : "v"(base + offset), "s"(m0_value) : "memory");
    } else {
        resource_t descriptor = {
            static_cast<uint32_t>(base), static_cast<uint32_t>(base >> 32),
            input_words_per_wave * waves * sizeof(uint32_t), 0x00027000,
        };
        if constexpr (transfer == Transfer::BufferControl) {
            uint32_t value;
            asm volatile(
                "buffer_load_dword %0, %1, %2, 0 offen\n\t"
                "s_waitcnt vmcnt(0)"
                : "=v"(value) : "v"(offset), "s"(descriptor) : "memory");
            scratch[wave * lds_words_per_wave + lane] = value;
        } else if constexpr (transfer == Transfer::Buffer) {
            asm volatile(
                "s_mov_b32 m0, %2\n\t"
                "buffer_load_dword %0, %1, 0 offen lds\n\t"
                "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
                "s_mov_b32 m0, -1"
                : : "v"(offset), "s"(descriptor), "s"(m0_value) : "memory");
        } else {
            asm volatile(
                "s_mov_b32 m0, %2\n\t"
                "buffer_load_dwordx4 %0, %1, 0 offen lds\n\t"
                "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
                "s_mov_b32 m0, -1"
                : : "v"(offset), "s"(descriptor), "s"(m0_value) : "memory");
        }
    }
    __syncthreads();
    for (unsigned i = tid; i < lds_words_per_wave * waves; i += blockDim.x) {
        output[i] = scratch[i];
    }
}

void check(hipError_t status) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s\n", hipGetErrorString(status));
        std::exit(2);
    }
}

template <bool use_buffer>
__global__ void k_tile_probe(const uint32_t* input, uint32_t* output) {
    extern __shared__ uint32_t scratch[];
    unsigned tid = threadIdx.x;
    unsigned producer = tid ^ (tid >> 4);
    unsigned input_index = (producer >> 4) * k_stride_words + (producer & 15) * 4;
    unsigned lds_start = __builtin_amdgcn_readfirstlane((tid / lanes) * 1024);
    unsigned lds_second = lds_start + 4096;
    auto base = reinterpret_cast<uintptr_t>(input);
    constexpr unsigned second = 16 * k_stride_words * sizeof(uint32_t);
    if constexpr (use_buffer) {
        resource_t descriptor = {
            static_cast<uint32_t>(base), static_cast<uint32_t>(base >> 32),
            k_rows * k_stride_words * sizeof(uint32_t), 0x00027000,
        };
        unsigned byte_offset = input_index * sizeof(uint32_t);
        asm volatile(
            "s_mov_b32 m0, %3\n\t"
            "buffer_load_dwordx4 %0, %2, 0 offen lds\n\t"
            "s_mov_b32 m0, %4\n\t"
            "buffer_load_dwordx4 %1, %2, 0 offen lds\n\t"
            "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
            "s_mov_b32 m0, -1"
            : : "v"(byte_offset), "v"(byte_offset + second), "s"(descriptor),
                "s"(lds_start), "s"(lds_second) : "memory");
    } else {
        resource_t unused;
        uintptr_t address = base + input_index * sizeof(uint32_t);
        asm volatile(
            "s_mov_b32 m0, %3\n\t"
            "global_load_dwordx4 %0, %1, off lds\n\t"
            "s_mov_b32 m0, %4\n\t"
            "global_load_dwordx4 %0, %2, off lds\n\t"
            "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
            "s_mov_b32 m0, -1"
            : "=v"(unused) : "v"(address), "v"(address + second),
                "s"(lds_start), "s"(lds_second) : "memory");
    }
    __syncthreads();
    for (unsigned i = tid; i < k_tile_words; i += blockDim.x) {
        output[i] = scratch[i];
    }
}

int run_k_tile(bool use_buffer) {
    std::array<uint32_t, k_rows * k_stride_words> input;
    std::array<uint32_t, k_tile_words> output;
    for (unsigned i = 0; i < input.size(); ++i) {
        input[i] = 0x30000000 | i;
    }
    uint32_t *device_input, *device_output;
    check(hipMalloc(&device_input, sizeof(input)));
    check(hipMalloc(&device_output, sizeof(output)));
    check(hipMemcpy(device_input, input.data(), sizeof(input), hipMemcpyHostToDevice));
    if (use_buffer) {
        hipLaunchKernelGGL((k_tile_probe<true>), dim3(1), dim3(256), sizeof(output),
                           0, device_input, device_output);
    } else {
        hipLaunchKernelGGL((k_tile_probe<false>), dim3(1), dim3(256), sizeof(output),
                           0, device_input, device_output);
    }
    check(hipGetLastError());
    check(hipDeviceSynchronize());
    check(hipMemcpy(output.data(), device_output, sizeof(output), hipMemcpyDeviceToHost));
    check(hipFree(device_output));
    check(hipFree(device_input));
    unsigned mismatches = 0;
    for (unsigned tid = 0; tid < 256; ++tid) {
        unsigned base = ((tid << 4) ^ (tid & 0xf0)) / sizeof(uint32_t);
        unsigned global_index = (tid >> 4) * k_stride_words + (tid & 15) * 4;
        for (unsigned half = 0; half < 2; ++half) {
            for (unsigned element = 0; element < 4; ++element) {
                unsigned lds_index = base + half * 1024 + element;
                uint32_t expected = input[global_index + half * 16 * k_stride_words + element];
                if (output[lds_index] != expected && ++mismatches <= 8) {
                    std::printf("mismatch %u expected 0x%08x got 0x%08x\n", lds_index,
                                expected, output[lds_index]);
                }
            }
        }
    }
    std::printf("mode k-tile-%s words %u mismatches %u\n", use_buffer ? "buffer" : "global",
                k_tile_words, mismatches);
    return mismatches ? 1 : 0;
}

template <Transfer transfer>
void launch(const uint32_t* input, uint32_t* output) {
    hipLaunchKernelGGL((copy_probe<transfer>), dim3(1), dim3(lanes * waves),
                       lds_words_per_wave * waves * sizeof(uint32_t), 0, input, output);
    check(hipGetLastError());
    check(hipDeviceSynchronize());
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s {global|buffer|buffer-wide|control|k-tile-buffer|k-tile-global}\n", argv[0]);
        return 2;
    }
    const char* mode = argv[1];
    if (std::strcmp(mode, "k-tile-buffer") == 0 || std::strcmp(mode, "k-tile-global") == 0) {
        return run_k_tile(std::strcmp(mode, "k-tile-buffer") == 0);
    }
    std::array<uint32_t, input_words_per_wave * waves> input;
    std::array<uint32_t, lds_words_per_wave * waves> output;
    for (unsigned i = 0; i < input.size(); ++i) {
        input[i] = 0x10000000 | i;
    }
    uint32_t *device_input, *device_output;
    check(hipMalloc(&device_input, sizeof(input)));
    check(hipMalloc(&device_output, sizeof(output)));
    check(hipMemcpy(device_input, input.data(), sizeof(input), hipMemcpyHostToDevice));
    if (std::strcmp(mode, "global") == 0) {
        launch<Transfer::Global>(device_input, device_output);
    } else if (std::strcmp(mode, "buffer") == 0) {
        launch<Transfer::Buffer>(device_input, device_output);
    } else if (std::strcmp(mode, "buffer-wide") == 0) {
        launch<Transfer::BufferWide>(device_input, device_output);
    } else if (std::strcmp(mode, "control") == 0) {
        launch<Transfer::BufferControl>(device_input, device_output);
    } else {
        return 2;
    }
    check(hipMemcpy(output.data(), device_output, sizeof(output), hipMemcpyDeviceToHost));
    check(hipFree(device_output));
    check(hipFree(device_input));
    const unsigned width = std::strcmp(mode, "buffer-wide") == 0 ? 4 : 1;
    unsigned lane_major = 0;
    unsigned component_major = 0;
    unsigned changed = 0;
    for (unsigned wave = 0; wave < waves; ++wave) {
        for (unsigned i = 0; i < lds_words_per_wave; ++i) {
            uint32_t word = output[wave * lds_words_per_wave + i];
            if (word != sentinel) {
                ++changed;
                std::printf("word %u %u 0x%08x\n", wave, i, word);
            }
            if (i < width * lanes) {
                unsigned base_index = wave * input_words_per_wave;
                lane_major += word == input[base_index + i];
                component_major += word == input[base_index + (i % lanes) * width + i / lanes];
            }
        }
    }
    const unsigned expected = lanes * waves * width;
    std::printf("mode %s changed %u expected %u lane_major %u component_major %u\n",
                mode, changed, expected, lane_major, component_major);
    return changed == expected && (lane_major == expected || component_major == expected) ? 0 : 1;
}
