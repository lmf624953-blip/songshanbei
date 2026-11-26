#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

constexpr std::size_t count = 18;
using Values = std::array<std::uint32_t, count>;
using Pairs = std::array<std::uint32_t, count * count * 4>;

static bool signaling_nan(std::uint32_t value) {
    return (value & 0x7fc00000u) == 0x7f800000u && (value & 0x003fffffu) != 0;
}

static void check(hipError_t status) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "%s\n", hipGetErrorString(status));
        std::exit(1);
    }
}

__global__ void compare_max(const std::uint32_t* cases, std::uint32_t* output) {
    const auto i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < count * count) {
        const auto left = cases[i / count];
        const auto right = cases[i % count];
        std::uint32_t a, b, original, fused, dpp;
        asm volatile("v_max_f32_e32 %0, %4, %4\n\t"
                     "v_max_f32_e32 %1, %5, %5\n\t"
                     "v_max_f32_e32 %2, %0, %1\n\t"
                     "v_max_f32_e32 %3, %4, %5"
                     : "=&v"(a), "=&v"(b), "=&v"(original), "=&v"(fused) : "v"(left), "v"(right));
        asm volatile("v_max_f32_dpp %0, %1, %2 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf"
                     : "=&v"(dpp) : "v"(left), "v"(right));
        output[4 * i] = original;
        output[4 * i + 1] = fused;
        output[4 * i + 2] = a;
        output[4 * i + 3] = dpp;
    }
    return;
}

int main() {
    const Values cases = {0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff,
                          0x807fffff, 0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff,
                          0x7f800000, 0xff800000, 0x7fc00000, 0xffc01234, 0x7fffffff,
                          0x7f800001, 0xff800001, 0x7fbfffff};
    std::uint32_t *device_cases = nullptr, *device_output = nullptr;
    Pairs results = {};
    check(hipMalloc(&device_cases, sizeof(cases)));
    check(hipMalloc(&device_output, sizeof(results)));
    check(hipMemcpy(device_cases, cases.data(), sizeof(cases), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(compare_max, dim3(6), dim3(64), 0, 0, device_cases, device_output);
    check(hipGetLastError());
    check(hipMemcpy(results.data(), device_output, sizeof(results), hipMemcpyDeviceToHost));
    check(hipFree(device_output));
    check(hipFree(device_cases));
    unsigned mismatches = 0;
    unsigned without_signaling_nan = 0;
    unsigned signaling_outputs = 0;
    for (std::size_t i = 0; i < count * count; ++i) {
        signaling_outputs += signaling_nan(results[4 * i + 2]) + signaling_nan(results[4 * i + 3]);
        if (results[4 * i] != results[4 * i + 1]) {
            ++mismatches;
            if (i / count < 15 && i % count < 15) {
                ++without_signaling_nan;
            }
        }
    }
    std::printf("{\"pairs\":%zu,\"mismatches\":%u,\"pairs_without_signaling_nan\":225,"
                "\"mismatches_without_signaling_nan\":%u,\"signaling_producer_outputs\":%u}\n",
                count * count, mismatches, without_signaling_nan, signaling_outputs);
}
