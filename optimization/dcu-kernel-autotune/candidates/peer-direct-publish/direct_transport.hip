// 仅负责peer内存交换、事件依赖与通信校准检查；模型计算由九齿镜像执行。
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstddef>
struct Addresses{uint16_t* p[8];};
struct Probe{uint16_t value[8][32];};
static hipEvent_t input_ready[8],reduced[8],published[8];
static bool initialized=false;
#define CHECK(call) do{auto e=(call);if(e!=hipSuccess)return int(e);}while(0)
__global__ void publish_shard(const uint16_t* source,Addresses outputs,size_t count,int rank){
 size_t shard=count/8;
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<shard;i+=size_t(gridDim.x)*blockDim.x){
  uint16_t word=source[i];
  #pragma unroll
  for(int dest=0;dest<8;++dest)outputs.p[dest][rank*shard+i]=word;
 }
}
__global__ void fill_probe(uint16_t* dst,size_t count,Probe values,int rank){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x)dst[i]=values.value[rank][i&31];
}
__global__ void make_map(uint8_t* dst,size_t count,unsigned channels,unsigned chunk,unsigned alignment){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count/8;i+=size_t(gridDim.x)*blockDim.x){
  size_t local=(i*8)%(count/channels),base=(local/(8*chunk))*(8*chunk);
  size_t rem=count/channels-base;if(rem>8*chunk)rem=8*chunk;
  size_t step=(((rem+7)/8+alignment-1)/alignment)*alignment;
  dst[i]=uint8_t((local-base)/step);
 }
}
__global__ void check_probe(const uint16_t* values,const uint8_t* owners,size_t count,Probe expected,unsigned long long* wrong){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count/8;i+=size_t(gridDim.x)*blockDim.x){
  unsigned errors=0;unsigned owner=owners[i];
  #pragma unroll
  for(int v=0;v<8;++v)errors+=(values[i*8+v]!=expected.value[owner][(i*8+v)&31]);
  if(errors)atomicAdd(wrong,static_cast<unsigned long long>(errors));
 }
}
extern "C" int transport_setup(){
 if(initialized)return 0;int n=0,prior=0;CHECK(hipGetDeviceCount(&n));if(n!=8)return -1;CHECK(hipGetDevice(&prior));
 for(int r=0;r<8;++r){CHECK(hipSetDevice(r));for(int p=0;p<8;++p)if(p!=r){auto e=hipDeviceEnablePeerAccess(p,0);if(e!=hipSuccess&&e!=hipErrorPeerAccessAlreadyEnabled)return int(e);if(e==hipErrorPeerAccessAlreadyEnabled)hipGetLastError();}
  CHECK(hipEventCreateWithFlags(&input_ready[r],hipEventDisableTiming));CHECK(hipEventCreateWithFlags(&reduced[r],hipEventDisableTiming));CHECK(hipEventCreateWithFlags(&published[r],hipEventDisableTiming));
 }
 CHECK(hipSetDevice(prior));initialized=true;return 0;
}
extern "C" int transport_inputs(uint64_t* streams){
 if(!initialized)return -1;int prior;CHECK(hipGetDevice(&prior));
 for(int r=0;r<8;++r){CHECK(hipSetDevice(r));CHECK(hipEventRecord(input_ready[r],reinterpret_cast<hipStream_t>(streams[r])));}
 for(int r=0;r<8;++r){CHECK(hipSetDevice(r));for(int p=0;p<8;++p)CHECK(hipStreamWaitEvent(reinterpret_cast<hipStream_t>(streams[r]),input_ready[p],0));}
 CHECK(hipSetDevice(prior));return 0;
}
extern "C" int transport_publish(uint64_t* parts,uint64_t* outputs,uint64_t* streams,size_t count,int blocks){
 if(!initialized||count%8||blocks<1)return -1;int prior;CHECK(hipGetDevice(&prior));Addresses out;
 for(int r=0;r<8;++r){out.p[r]=reinterpret_cast<uint16_t*>(outputs[r]);CHECK(hipSetDevice(r));CHECK(hipEventRecord(reduced[r],reinterpret_cast<hipStream_t>(streams[r])));}
 for(int r=0;r<8;++r){CHECK(hipSetDevice(r));auto s=reinterpret_cast<hipStream_t>(streams[r]);for(int p=0;p<8;++p)CHECK(hipStreamWaitEvent(s,reduced[p],0));
  hipLaunchKernelGGL(publish_shard,dim3(blocks),dim3(256),0,s,reinterpret_cast<uint16_t*>(parts[r]),out,count,r);CHECK(hipGetLastError());CHECK(hipEventRecord(published[r],s));
 }
 for(int r=0;r<8;++r){CHECK(hipSetDevice(r));for(int p=0;p<8;++p)CHECK(hipStreamWaitEvent(reinterpret_cast<hipStream_t>(streams[r]),published[p],0));}
 CHECK(hipSetDevice(prior));return 0;
}
extern "C" int transport_probe(uint64_t* buffers,uint64_t* maps,uint64_t* streams,const uint16_t* patterns,size_t count,unsigned channels,unsigned chunk,unsigned alignment){
 if(count%(channels*8))return -1;Probe values;for(int r=0;r<8;++r)for(int k=0;k<32;++k)values.value[r][k]=patterns[r*32+k];int prior;CHECK(hipGetDevice(&prior));
 for(int r=0;r<8;++r){CHECK(hipSetDevice(r));auto s=reinterpret_cast<hipStream_t>(streams[r]);hipLaunchKernelGGL(fill_probe,dim3(5120),dim3(256),0,s,reinterpret_cast<uint16_t*>(buffers[r]),count,values,r);CHECK(hipGetLastError());hipLaunchKernelGGL(make_map,dim3(5120),dim3(256),0,s,reinterpret_cast<uint8_t*>(maps[r]),count,channels,chunk,alignment);CHECK(hipGetLastError());}
 CHECK(hipSetDevice(prior));return 0;
}
extern "C" int transport_validate(uint64_t* buffers,uint64_t* maps,uint64_t* streams,const uint16_t* signatures,size_t count,uint64_t* errors){
 Probe expected;for(int r=0;r<8;++r)for(int k=0;k<32;++k)expected.value[r][k]=signatures[r*32+k];int prior;CHECK(hipGetDevice(&prior));unsigned long long* scratch[8]{};
 for(int r=0;r<8;++r){CHECK(hipSetDevice(r));auto s=reinterpret_cast<hipStream_t>(streams[r]);CHECK(hipMalloc(&scratch[r],8));CHECK(hipMemsetAsync(scratch[r],0,8,s));hipLaunchKernelGGL(check_probe,dim3(5120),dim3(256),0,s,reinterpret_cast<uint16_t*>(buffers[r]),reinterpret_cast<uint8_t*>(maps[r]),count,expected,scratch[r]);CHECK(hipGetLastError());}
 for(int r=0;r<8;++r){CHECK(hipSetDevice(r));CHECK(hipStreamSynchronize(reinterpret_cast<hipStream_t>(streams[r])));CHECK(hipMemcpy(errors+r,scratch[r],8,hipMemcpyDeviceToHost));CHECK(hipFree(scratch[r]));}
 CHECK(hipSetDevice(prior));return 0;
}
extern "C" int transport_cleanup(){
 if(!initialized)return 0;int prior;CHECK(hipGetDevice(&prior));for(int r=0;r<8;++r){CHECK(hipSetDevice(r));CHECK(hipEventDestroy(input_ready[r]));CHECK(hipEventDestroy(reduced[r]));CHECK(hipEventDestroy(published[r]));}CHECK(hipSetDevice(prior));initialized=false;return 0;
}
extern "C" int transport_map(uint64_t* maps,uint64_t* streams,size_t count,unsigned channels,unsigned chunk,unsigned alignment){
 if(count%(channels*8))return -1;int prior;CHECK(hipGetDevice(&prior));for(int r=0;r<8;++r){CHECK(hipSetDevice(r));auto s=reinterpret_cast<hipStream_t>(streams[r]);hipLaunchKernelGGL(make_map,dim3(5120),dim3(256),0,s,reinterpret_cast<uint8_t*>(maps[r]),count,channels,chunk,alignment);CHECK(hipGetLastError());}CHECK(hipSetDevice(prior));return 0;
}

// 每个九齿kernel已把当前分片写到全部输出；等待所有生产者完成后才允许消费。
extern "C" int transport_finish(uint64_t* streams){
 if(!initialized)return -1;int prior;CHECK(hipGetDevice(&prior));
 for(int r=0;r<8;++r){CHECK(hipSetDevice(r));CHECK(hipEventRecord(published[r],reinterpret_cast<hipStream_t>(streams[r])));}
 for(int r=0;r<8;++r){CHECK(hipSetDevice(r));for(int p=0;p<8;++p)CHECK(hipStreamWaitEvent(reinterpret_cast<hipStream_t>(streams[r]),published[p],0));}
 CHECK(hipSetDevice(prior));return 0;
}
