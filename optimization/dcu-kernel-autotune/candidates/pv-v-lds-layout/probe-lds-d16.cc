#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

static void check(hipError_t status) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s\n", hipGetErrorString(status));
        std::exit(1);
    }
}

__global__ void probe(std::uint32_t* output) {
    __shared__ std::uint32_t data[2];
    if (threadIdx.x == 0) {
        data[0] = 0x11223344;
        data[1] = 0x55667788;
    }
    __syncthreads();
    const auto address = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(data));
    std::uint32_t low = 0xaabbccdd;
    std::uint32_t high = 0xaabbccdd;
    std::uint32_t pair = 0xaabbccdd;
    std::uint32_t readback;
    asm volatile("ds_read_u16_d16 %0, %3\n\t"
                 "ds_read_u16_d16_hi %1, %3 offset:4\n\t"
                 "s_waitcnt lgkmcnt(0)\n\t"
                 "ds_read_u16_d16 %2, %3\n\t"
                 "s_waitcnt lgkmcnt(0)\n\t"
                 "ds_read_u16_d16_hi %2, %3 offset:4\n\t"
                 "s_waitcnt lgkmcnt(0)"
                 : "+&v"(low), "+&v"(high), "+&v"(pair) : "v"(address) : "memory");
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("ds_write_b16_d16_hi %1, %2\n\t"
                     "s_waitcnt lgkmcnt(0)\n\t"
                     "ds_read_b32 %0, %1\n\t"
                     "s_waitcnt lgkmcnt(0)"
                     : "=&v"(readback) : "v"(address), "v"(0x99aabbccu) : "memory");
        output[0] = low;
        output[1] = high;
        output[2] = pair;
        output[3] = readback;
    }
    return;
}

int main() {
    std::uint32_t* device = nullptr;
    std::uint32_t result[4] = {};
    check(hipMalloc(&device, sizeof(result)));
    hipLaunchKernelGGL(probe, dim3(1), dim3(64), 0, 0, device);
    check(hipGetLastError());
    check(hipMemcpy(result, device, sizeof(result), hipMemcpyDeviceToHost));
    check(hipFree(device));
    std::printf("{\"low\":\"%08x\",\"high\":\"%08x\",\"pair\":\"%08x\",\"write_high\":\"%08x\"}\n",
                result[0], result[1], result[2], result[3]);
}
