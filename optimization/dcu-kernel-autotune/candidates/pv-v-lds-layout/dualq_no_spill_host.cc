// 同源单/双Q128的原生发射桥，只负责预排布和启动参数。
#include <hip/hip_runtime_api.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {
hipModule_t modules[2]; hipFunction_t functions[2];
thread_local std::string error_message;
void check(hipError_t code) { if(code!=hipSuccess)throw std::runtime_error(hipGetErrorString(code)); }
}
extern "C" const char* prepare_dual(const char* image,const char* prepack) {
    try {
        check(hipModuleLoad(&modules[0],image));check(hipModuleGetFunction(&functions[0],modules[0],"dense_attention_aot_kernel"));
        check(hipModuleLoad(&modules[1],prepack));check(hipModuleGetFunction(&functions[1],modules[1],"h3_prepack_value_kernel"));
        return nullptr;
    } catch(const std::exception& error){error_message=error.what();return error_message.c_str();}
}
extern "C" const char* bench_dual(int rows,float scale,const void* q,const void* k,const void* v,void* out,
    void* packed,int pack_tile,int pack_threads,int mode,std::uintptr_t raw_stream) {
    try {
        auto stream=reinterpret_cast<hipStream_t>(raw_stream);
        int blocks=rows/32,elements=7*blocks*4096;void* reserved=nullptr;
        void* prepack[]={&v,&blocks,&packed,&rows,&elements,&reserved,&reserved};
        check(hipModuleLaunchKernel(functions[1],(elements+pack_tile-1)/pack_tile,1,1,pack_threads,1,1,0,stream,prepack,nullptr));
        int qs=896,ks=896,vs=2688,os=896;
        void* parameters[]={&q,&k,&v,&scale,&out,&rows,&qs,&rows,&ks,&rows,&vs,&rows,&os,&packed,&reserved};
        const int tiles=(rows+127)/128;
        check(hipModuleLaunchKernel(functions[0],mode?((tiles+1)/2):7*tiles,mode?7:1,1,
                                   mode?512:256,1,1,mode?32768:16384,stream,parameters,nullptr));
        return nullptr;
    } catch(const std::exception& error){error_message=error.what();return error_message.c_str();}
}
