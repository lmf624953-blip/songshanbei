// 只实现内存映射、发布和流依赖；计算调用现用九齿norm和原rocBLAS。
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {
constexpr size_t width=5376,rowbytes=width*2;
using P=std::array<uint64_t,8>;
using Norm=const char*(*)(int,int,int,const void*,const void*,const void*,const void*,const void*,const void*,uintptr_t);
struct Addresses {uint16_t* p[8];};
struct State {
 size_t rows=0,total=0,gran=0;void* base=nullptr;Norm norm=nullptr;
 std::array<size_t,9> first{},offset{};std::array<hipMemGenericAllocationHandle_t,8> handles{};
 std::array<bool,8> created{},mapped{};std::array<hipStream_t,8> streams{};
 std::array<hipEvent_t,8> ready{},begin{},done{};std::array<rocblas_handle,8> blas{};
} state;
thread_local std::string error;
thread_local std::string phase;
void check_hip(hipError_t e,const char* call){if(e!=hipSuccess)throw std::runtime_error(phase+" "+call+": "+hipGetErrorString(e));}
#define ck(call) check_hip((call),#call)
void rb(rocblas_status e){if(e!=rocblas_status_success)throw std::runtime_error("rocblas status "+std::to_string(int(e)));}
template<class F>const char* wrap(F f){try{f();return nullptr;}catch(const std::exception& e){error=e.what();return error.c_str();}}
__global__ void publish_rows(const uint16_t* src,Addresses out,size_t rows,int rank){
 size_t first=rows*rank/8*width,local=(rows*(rank+1)/8-rows*rank/8)*width;
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<local;i+=size_t(gridDim.x)*blockDim.x){
  auto word=src[i];
  #pragma unroll
  for(int d=0;d<8;++d)out.p[d][first+i]=word;
 }
 return;
}
}

extern "C" const char* vm_setup(size_t rows,const uint64_t* streams,void* norm,uint64_t* address,size_t* boundaries,size_t* granularity){
 return wrap([&]{
  if(state.rows||rows<128)throw std::runtime_error("invalid setup state/rows");
  int count=0;ck(hipGetDeviceCount(&count));if(count!=8)throw std::runtime_error("requires eight devices");
  state.rows=rows;state.norm=reinterpret_cast<Norm>(norm);
  std::array<hipMemAllocationProp,8> props{};
  for(int r=0;r<8;++r){
   ck(hipSetDevice(r));int supported=0;ck(hipDeviceGetAttribute(&supported,hipDeviceAttributeVirtualMemoryManagementSupported,r));if(!supported)throw std::runtime_error("VMM unsupported");
   for(int p=0;p<8;++p)if(p!=r){auto e=hipDeviceEnablePeerAccess(p,0);if(e==hipErrorPeerAccessAlreadyEnabled)(void)hipGetLastError();else ck(e);}
   props[r].type=hipMemAllocationTypePinned;props[r].location={hipMemLocationTypeDevice,r};
   size_t g=0;ck(hipMemGetAllocationGranularity(&g,&props[r],hipMemAllocationGranularityMinimum));if(!g)throw std::runtime_error("zero granularity");state.gran=std::lcm(state.gran?state.gran:g,g);
   state.streams[r]=reinterpret_cast<hipStream_t>(streams[r]);
   ck(hipEventCreateWithFlags(&state.ready[r],hipEventDisableTiming));ck(hipEventCreate(&state.begin[r]));ck(hipEventCreate(&state.done[r]));
   rb(rocblas_create_handle(&state.blas[r]));rb(rocblas_set_stream(state.blas[r],state.streams[r]));
  }
  if(auto requested=std::getenv("H3_VMM_MAPPING_GRANULARITY")){
   size_t value=std::stoull(requested);if(value<state.gran||value%state.gran||(value&(value-1)))throw std::runtime_error("invalid requested mapping granularity");state.gran=value;
  }
  // 数学仍按完整行均分；物理页边界独立对齐，边界处允许norm写入邻卡页。
  size_t chunk=((rows*rowbytes+7)/8+state.gran-1)/state.gran*state.gran;
  for(int r=0;r<=8;++r){state.first[r]=rows*r/8;state.offset[r]=r*chunk;}
  state.total=8*chunk;
  phase="reserve bytes="+std::to_string(state.total)+" alignment="+std::to_string(state.gran);ck(hipMemAddressReserve(&state.base,state.total,state.gran,nullptr,0));
  std::array<hipMemAccessDesc,8> desc{};
  for(int r=0;r<8;++r){desc[r].location={hipMemLocationTypeDevice,r};desc[r].flags=hipMemAccessFlagsProtReadWrite;}
  for(int r=0;r<8;++r){
   ck(hipSetDevice(r));size_t start=state.offset[r],end=state.offset[r+1];if(end<=start)throw std::runtime_error("empty physical partition");
   phase="owner="+std::to_string(r)+" bytes="+std::to_string(end-start)+" offset="+std::to_string(start);
   ck(hipMemCreate(&state.handles[r],end-start,&props[r],0));state.created[r]=true;
   auto at=static_cast<char*>(state.base)+start;ck(hipMemMap(at,end-start,0,state.handles[r],0));state.mapped[r]=true;
  }
  for(int r=0;r<8;++r){
   ck(hipSetDevice(r));size_t start=state.offset[r],bytes=state.offset[r+1]-start;auto at=static_cast<char*>(state.base)+start;
   phase="all-mapped access owner="+std::to_string(r)+" bytes="+std::to_string(bytes)+" offset="+std::to_string(start);
   ck(hipMemSetAccess(at,bytes,desc.data(),8));ck(hipMemsetAsync(at,0x35,bytes,state.streams[r]));
  }
  *address=reinterpret_cast<uint64_t>(state.base);*granularity=state.gran;
  for(int r=0;r<=8;++r)boundaries[r]=state.first[r];
 });
}
extern "C" const char* vm_begin(){return wrap([&]{for(int r=0;r<8;++r){ck(hipSetDevice(r));ck(hipEventRecord(state.begin[r],state.streams[r]));}});}
extern "C" const char* vm_norm(int mode,const uint64_t* hidden,const uint64_t* weights,const uint64_t* scale,const uint64_t* shift,const uint64_t* indices,const uint64_t* partial,const uint64_t* local){
 return wrap([&]{
  Addresses out;for(int r=0;r<8;++r)out.p[r]=reinterpret_cast<uint16_t*>(local[r]);
  for(int r=0;r<8;++r){
   ck(hipSetDevice(r));size_t first=mode?state.first[r]:state.rows*r/8,last=mode?state.first[r+1]:state.rows*(r+1)/8;
   auto output=mode?static_cast<char*>(state.base)+first*rowbytes:reinterpret_cast<void*>(partial[r]);
   auto e=state.norm(r,int(last-first),3,reinterpret_cast<void*>(hidden[r]+first*rowbytes),reinterpret_cast<void*>(weights[r]),reinterpret_cast<void*>(scale[r]),reinterpret_cast<void*>(shift[r]),reinterpret_cast<void*>(indices[r]+first*8),output,reinterpret_cast<uintptr_t>(state.streams[r]));
   if(e)throw std::runtime_error(e);
  }
  for(int r=0;r<8;++r){ck(hipSetDevice(r));if(!mode){hipLaunchKernelGGL(publish_rows,dim3(5120),dim3(256),0,state.streams[r],reinterpret_cast<uint16_t*>(partial[r]),out,state.rows,r);ck(hipGetLastError());}ck(hipEventRecord(state.ready[r],state.streams[r]));}
  for(int r=0;r<8;++r){ck(hipSetDevice(r));for(int p=0;p<8;++p)ck(hipStreamWaitEvent(state.streams[r],state.ready[p],0));}
 });
}
extern "C" const char* vm_gemm(int mode,int rank,int m,const void* weight,const void* local,void* output){
 return wrap([&]{
  ck(hipSetDevice(rank));float alpha=1,beta=0;
  rb(rocblas_gemm_strided_batched_ex(state.blas[rank],rocblas_operation_transpose,rocblas_operation_none,m,int(state.rows),width,
      &alpha,weight,rocblas_datatype_bf16_r,width,0,mode?state.base:local,rocblas_datatype_bf16_r,width,0,
      &beta,output,rocblas_datatype_bf16_r,m,0,output,rocblas_datatype_bf16_r,m,0,1,
      rocblas_datatype_f32_r,rocblas_gemm_algo_standard,0,0));
  ck(hipEventRecord(state.done[rank],state.streams[rank]));
 });
}
extern "C" const char* vm_finish(float* milliseconds){return wrap([&]{for(int r=0;r<8;++r){ck(hipSetDevice(r));ck(hipEventSynchronize(state.done[r]));ck(hipEventElapsedTime(milliseconds+r,state.begin[r],state.done[r]));}});}
extern "C" const char* vm_copy_input(int rank,void* out){return wrap([&]{ck(hipSetDevice(rank));ck(hipMemcpyAsync(out,state.base,state.rows*rowbytes,hipMemcpyDeviceToDevice,state.streams[rank]));ck(hipStreamSynchronize(state.streams[rank]));});}
extern "C" const char* vm_close(){return wrap([&]{
 for(int r=0;r<8;++r){ck(hipSetDevice(r));ck(hipStreamSynchronize(state.streams[r]));if(state.blas[r])rb(rocblas_destroy_handle(state.blas[r]));for(auto event:{state.ready[r],state.begin[r],state.done[r]})if(event)ck(hipEventDestroy(event));}
 for(int r=0;r<8;++r){ck(hipSetDevice(r));size_t start=state.offset[r],end=state.offset[r+1];if(state.mapped[r])ck(hipMemUnmap(static_cast<char*>(state.base)+start,end-start));if(state.created[r])ck(hipMemRelease(state.handles[r]));}
 if(state.base)ck(hipMemAddressFree(state.base,state.total));state=State{};
 });}
