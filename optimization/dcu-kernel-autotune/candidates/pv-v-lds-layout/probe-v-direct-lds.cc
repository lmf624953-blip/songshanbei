#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using Resource = uint32_t __attribute__((ext_vector_type(4)));
constexpr unsigned stride = 2688;
constexpr unsigned rows = 32;
constexpr unsigned columns = 128;
constexpr unsigned tile_words = rows * columns / 2;
constexpr uint32_t sentinel = 0xdeadbeef;

void check(hipError_t status) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s\n", hipGetErrorString(status));
        std::exit(2);
    }
}

__global__ void halfword_probe(const uint16_t* input, uint32_t* output, int passes) {
    extern __shared__ uint32_t shared[];
    unsigned tid = threadIdx.x;
    shared[tid] = sentinel;
    __syncthreads();
    uintptr_t base = reinterpret_cast<uintptr_t>(input);
    Resource resource = {uint32_t(base), uint32_t(base >> 32), 256, 0x00027000};
    unsigned offset = tid * 2;
    asm volatile(
        "s_mov_b32 m0, 0\n\t"
        "buffer_load_ushort %0, %1, 0 offen lds\n\t"
        "s_waitcnt vmcnt(0) lgkmcnt(0)"
        : : "v"(offset), "s"(resource) : "memory");
    if (passes == 2) {
        unsigned upper = offset + 128;
        asm volatile(
            "s_mov_b32 m0, 2\n\t"
            "buffer_load_ushort %0, %1, 0 offen lds\n\t"
            "s_waitcnt vmcnt(0) lgkmcnt(0)"
            : : "v"(upper), "s"(resource) : "memory");
    }
    asm volatile("s_mov_b32 m0, -1" : : : "memory");
    __syncthreads();
    output[tid] = shared[tid];
}

__global__ void v_tile_probe(const uint16_t* input, uint32_t* output) {
    extern __shared__ uint32_t shared[];
    unsigned tid = threadIdx.x;
    for (unsigned i = tid; i < tile_words; i += blockDim.x) {
        shared[i] = sentinel;
    }
    __syncthreads();
    uintptr_t base = reinterpret_cast<uintptr_t>(input);
    Resource resource = {uint32_t(base), uint32_t(base >> 32), rows * stride * 2, 0x00027000};
    for (unsigned batch = 0; batch < 8; ++batch) {
        unsigned word = batch * 256 + tid;
        unsigned column = word >> 4;
        unsigned row = 2 * ((word & 15) ^ (((column >> 1) & 7) << 1));
        unsigned low_offset = (row * stride + column) * 2;
        unsigned high_offset = low_offset + stride * 2;
        unsigned lds_base = __builtin_amdgcn_readfirstlane((word & ~63u) * 4);
        unsigned lds_high = lds_base + 2;
        asm volatile(
            "s_mov_b32 m0, %3\n\t"
            "buffer_load_ushort %0, %2, 0 offen lds\n\t"
            "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
            "s_mov_b32 m0, %4\n\t"
            "buffer_load_ushort %1, %2, 0 offen lds\n\t"
            "s_waitcnt vmcnt(0) lgkmcnt(0)"
            : : "v"(low_offset), "v"(high_offset), "s"(resource),
                "s"(lds_base), "s"(lds_high) : "memory");
    }
    asm volatile("s_mov_b32 m0, -1" : : : "memory");
    __syncthreads();
    for (unsigned i = tid; i < tile_words; i += blockDim.x) {
        output[i] = shared[i];
    }
}

int main() {
    uint16_t* device_input;
    uint32_t* device_output;
    std::array<uint16_t, rows * stride> input;
    std::array<uint32_t, tile_words> output;
    for (unsigned i = 0; i < input.size(); ++i) {
        input[i] = uint16_t(i + 1);
    }
    check(hipMalloc(&device_input, sizeof(input)));
    check(hipMalloc(&device_output, sizeof(output)));
    check(hipMemcpy(device_input, input.data(), sizeof(input), hipMemcpyHostToDevice));
    for (int passes = 1; passes <= 2; ++passes) {
        hipLaunchKernelGGL(halfword_probe, dim3(1), dim3(64), 256, 0,
                           device_input, device_output, passes);
        check(hipGetLastError());
        check(hipDeviceSynchronize());
        check(hipMemcpy(output.data(), device_output, 256, hipMemcpyDeviceToHost));
        unsigned changed = 0, half_matches = 0, zero_matches = 0;
        for (unsigned i = 0; i < 64; ++i) {
            uint32_t expected_half = passes == 1
                ? (sentinel & 0xffff0000) | input[i]
                : uint32_t(input[i]) | (uint32_t(input[i + 64]) << 16);
            changed += output[i] != sentinel;
            half_matches += output[i] == expected_half;
            zero_matches += output[i] == input[i];
        }
        std::printf("halfword passes %d changed %u half_matches %u zero_matches %u first 0x%08x\n",
                    passes, changed, half_matches, zero_matches, output[0]);
    }
    hipLaunchKernelGGL(v_tile_probe, dim3(1), dim3(256), sizeof(output), 0,
                       device_input, device_output);
    check(hipGetLastError());
    check(hipDeviceSynchronize());
    check(hipMemcpy(output.data(), device_output, sizeof(output), hipMemcpyDeviceToHost));
    unsigned mismatches = 0;
    for (unsigned i = 0; i < tile_words; ++i) {
        unsigned column = i >> 4;
        unsigned row = 2 * ((i & 15) ^ (((column >> 1) & 7) << 1));
        uint32_t expected = uint32_t(input[row * stride + column])
            | (uint32_t(input[(row + 1) * stride + column]) << 16);
        if (output[i] != expected && ++mismatches <= 8) {
            std::printf("mismatch word %u got 0x%08x expected 0x%08x\n", i, output[i], expected);
        }
    }
    std::printf("V tile words %u mismatches %u\n", tile_words, mismatches);
    check(hipFree(device_input));
    check(hipFree(device_output));
    return mismatches ? 1 : 0;
}
