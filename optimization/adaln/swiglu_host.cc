// 主机仅装载及发射九齿 SwiGLU 内核，数学计算由九齿生成。
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <hip/hip_runtime_api.h>
#include <infinicore/tensor.hpp>
using infinicore::Tensor;
namespace {
std::vector<hipModule_t> modules;
std::vector<hipFunction_t> functions;
int threads=0,shared=0;
thread_local std::string error_message;
void check(hipError_t value) {
    if(value!=hipSuccess) throw std::runtime_error(hipGetErrorString(value));
}
void dispatch(int device,int rows,const void *up,const void *gate,const void *output,hipStream_t stream) {
#if H3_SWIGLU_ROW
    void *scratch=nullptr;
    void *arguments[]={&up,&gate,&output,&rows,&scratch,&scratch};
    int blocks=rows;
#else
    int elements=rows*1792;
    void *scratch=nullptr;
    void *arguments[]={&up,&gate,&output,&rows,&elements,&scratch,&scratch};
    int blocks=(elements+1023)/1024;
#endif
    check(hipModuleLaunchKernel(functions.at(device),blocks,1,1,
          threads,1,1,shared,stream,arguments,nullptr));
}
}
extern "C" const char *songshan_prepare_swiglu(const char *path,int warps,int shared_memory) {
    try {
        if(warps!=(H3_SWIGLU_ROW ? 8 : 4) || shared_memory!=0)throw std::runtime_error("Unsupported SwiGLU geometry");
        int count=0,prior=0;
        check(hipGetDeviceCount(&count));check(hipGetDevice(&prior));
        modules.resize(count);functions.resize(count);threads=warps*64;shared=shared_memory;
        for(int device=0;device<count;++device) {
            check(hipSetDevice(device));check(hipModuleLoad(&modules[device],path));
            check(hipModuleGetFunction(&functions[device],modules[device],"swiglu_kernel"));
        }
        check(hipSetDevice(prior));return nullptr;
    } catch(const std::exception &error) {
        error_message=error.what();return error_message.c_str();
    }
}
extern "C" const char *songshan_bench_swiglu(int device,int rows,const void *up,const void *gate,
                                           const void *output,std::uintptr_t stream) {
    try {
        dispatch(device,rows,up,gate,output,reinterpret_cast<hipStream_t>(stream));return nullptr;
    } catch(const std::exception &error) {
        error_message=error.what();return error_message.c_str();
    }
}
extern "C" void songshan_launch_swiglu(const Tensor &output,const Tensor &up,const Tensor &gate,
                                     std::uintptr_t stream) {
    dispatch(up->device().getIndex(),up->shape()[1],up->data(),gate->data(),output->data(),
             reinterpret_cast<hipStream_t>(stream));
}
