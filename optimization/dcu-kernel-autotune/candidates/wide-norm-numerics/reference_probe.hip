// 仅用于诊断：按 InfiniCore rms_norm/cuda/kernel.cuh 与 reduce/cuda/reduce.cuh 的原算法导出 FP32 中间量。
// 不接入模型，不作为新增生产计算算子。
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hipcub/hipcub.hpp>
#include <cstdint>

__global__ __launch_bounds__(1024) void rms_reference_probe(const hip_bfloat16 *input, const hip_bfloat16 *weight,
                                  hip_bfloat16 *output, float *sums, float *inverse,
                                  int width, float epsilon) {
    const int row=blockIdx.x;
    const hip_bfloat16 *x=input+static_cast<std::int64_t>(row)*width;
    hip_bfloat16 *y=output+static_cast<std::int64_t>(row)*width;
    float ss=0;
    for(int i=threadIdx.x;i<width;i+=1024) ss+=float(x[i])*float(x[i]);
#ifdef PROBE_ORIGINAL_REDUCTION
    // 已部署 libinfiniop.so 的 ISA：wave 内 shuffle-down 1,2,4,8,16,32；
    // thread 0 按 wave 0..15 顺序相加，而非当前 DTK hipCUB 的树状跨 wave 归约。
    for(int offset=1;offset<64;offset*=2) {
        float other=__shfl_down(ss,offset,64);
        if((threadIdx.x&63)+offset<64)ss+=other;
    }
    __shared__ float wave_sums[16];
    if((threadIdx.x&63)==0)wave_sums[threadIdx.x/64]=ss;
    __syncthreads();
    if(threadIdx.x==0) {
        ss=wave_sums[0];
        for(int wave=1;wave<16;++wave)ss+=wave_sums[wave];
    }
#else
    using Reduce=hipcub::BlockReduce<float,1024>;
    __shared__ Reduce::TempStorage tmp;
    ss=Reduce(tmp).Sum(ss);
#endif
    __shared__ float inv;
    if(threadIdx.x==0){sums[row]=ss;inv=rsqrtf(ss/float(width)+epsilon);inverse[row]=inv;}
    __syncthreads();
    for(int i=threadIdx.x;i<width;i+=1024)y[i]=hip_bfloat16(float(x[i])*float(weight[i])*inv);
}
extern "C" int songshan_probe_rms(int rows,int width,const void *input,const void *weight,
                                 void *output,void *sums,void *inverse,std::uintptr_t stream){
    hipLaunchKernelGGL(rms_reference_probe,dim3(rows),dim3(1024),0,reinterpret_cast<hipStream_t>(stream),
        reinterpret_cast<const hip_bfloat16*>(input),reinterpret_cast<const hip_bfloat16*>(weight),
        reinterpret_cast<hip_bfloat16*>(output),static_cast<float*>(sums),static_cast<float*>(inverse),width,1e-5f);
    return static_cast<int>(hipGetLastError());
}
