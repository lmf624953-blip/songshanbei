// 仅验证通信内存映射与逐字复制；不实现或替换任何模型计算。
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Error { hipError_t code; const char* phase; };
void check(hipError_t code,const char* phase){if(code!=hipSuccess)throw Error{code,phase};}
__global__ void copy_words(const uint32_t* input,uint32_t* output,size_t words){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<words;i+=size_t(gridDim.x)*blockDim.x)output[i]=input[i];
}
struct Memory {
 void* address=nullptr;size_t total=0,chunk=0;bool active=false;
 std::array<hipMemGenericAllocationHandle_t,8> handles{};
 std::array<bool,8> created{},mapped{};
 std::array<void*,8> outputs{};
 ~Memory(){
  if(!active)return;
  for(int r=0;r<8;++r){hipSetDevice(r);hipDeviceSynchronize();if(outputs[r])hipFree(outputs[r]);}
  for(int r=0;r<8;++r){hipSetDevice(r);if(mapped[r])hipMemUnmap(static_cast<char*>(address)+r*chunk,chunk);if(created[r])hipMemRelease(handles[r]);}
  if(address)hipMemAddressFree(address,total);
 }
};
}

int main(){
 int count=0,version=0;std::array<int,8> supported{};std::array<size_t,8> granularity{};Memory memory;
 try{
  check(hipGetDeviceCount(&count),"device-count");if(count!=8)throw std::runtime_error("requires-eight-devices");
  check(hipRuntimeGetVersion(&version),"runtime-version");
  for(int r=0;r<8;++r){
   check(hipDeviceGetAttribute(&supported[r],hipDeviceAttributeVirtualMemoryManagementSupported,r),"attribute-vmm");
  }
  std::cout<<"{\"hip_runtime_version\":"<<version<<",\"vmm_supported\":[";
  for(int r=0;r<8;++r){if(r)std::cout<<',';std::cout<<supported[r];}std::cout<<"]";
  if(*std::min_element(supported.begin(),supported.end())==0){std::cout<<",\"status\":\"unsupported-attribute\"}\n";return 0;}
  memory.active=true;
  std::array<hipMemAllocationProp,8> properties{};
  for(int r=0;r<8;++r){
   check(hipSetDevice(r),"set-device");
   for(int p=0;p<8;++p)if(p!=r){auto error=hipDeviceEnablePeerAccess(p,0);if(error==hipErrorPeerAccessAlreadyEnabled)hipGetLastError();else check(error,"enable-peer");}
   properties[r].type=hipMemAllocationTypePinned;properties[r].location={hipMemLocationTypeDevice,r};
   check(hipMemGetAllocationGranularity(&granularity[r],&properties[r],hipMemAllocationGranularityMinimum),"allocation-granularity");
   memory.chunk=std::max(memory.chunk,granularity[r]);
  }
  for(auto value:granularity)if(!value||memory.chunk%value)throw std::runtime_error("unexpected-granularity");
  memory.total=8*memory.chunk;
  check(hipMemAddressReserve(&memory.address,memory.total,memory.chunk,nullptr,0),"reserve-address");
  for(int r=0;r<8;++r){
   check(hipSetDevice(r),"set-owner-device");
   check(hipMemCreate(&memory.handles[r],memory.chunk,&properties[r],0),"create-physical");memory.created[r]=true;
   check(hipMemMap(static_cast<char*>(memory.address)+r*memory.chunk,memory.chunk,0,memory.handles[r],0),"map-owner");memory.mapped[r]=true;
  }
  std::array<hipMemAccessDesc,8> access{};
  for(int r=0;r<8;++r){access[r].location={hipMemLocationTypeDevice,r};access[r].flags=hipMemAccessFlagsProtReadWrite;}
  for(int owner=0;owner<8;++owner){
   check(hipMemSetAccess(static_cast<char*>(memory.address)+owner*memory.chunk,memory.chunk,access.data(),access.size()),"set-peer-access");
  }
  std::vector<uint32_t> expected(memory.total/4),observed(expected.size());
  for(size_t i=0;i<expected.size();++i)expected[i]=uint32_t((i/(memory.chunk/4)+1)*0x01010101U)^uint32_t(i*0x9e3779b9U);
  for(int r=0;r<8;++r){
   check(hipSetDevice(r),"set-write-owner");
   check(hipMemcpy(static_cast<char*>(memory.address)+r*memory.chunk,expected.data()+r*memory.chunk/4,memory.chunk,hipMemcpyHostToDevice),"write-owned-page");
  }
  for(int r=0;r<8;++r){
   check(hipSetDevice(r),"set-consumer-device");check(hipMalloc(&memory.outputs[r],memory.total),"allocate-local-output");
   hipLaunchKernelGGL(copy_words,dim3(256),dim3(256),0,0,static_cast<const uint32_t*>(memory.address),static_cast<uint32_t*>(memory.outputs[r]),expected.size());
   check(hipGetLastError(),"launch-memory-copy");check(hipDeviceSynchronize(),"synchronize-memory-copy");
   check(hipMemcpy(observed.data(),memory.outputs[r],memory.total,hipMemcpyDeviceToHost),"read-local-copy");
   if(observed!=expected)throw std::runtime_error("peer-virtual-copy-mismatch");
  }
  std::cout<<",\"granularity\":[";for(int r=0;r<8;++r){if(r)std::cout<<',';std::cout<<granularity[r];}
  std::cout<<"],\"mapped_bytes\":"<<memory.total<<",\"exact_consumer_devices\":8,\"status\":\"memory-copy-pass\"}\n";
  return 0;
 }catch(const Error& error){
  // 在完整JSON结束前失败时，诊断使用单独的stderr；调用方据退出码归类，不误当成收益。
  std::cerr<<"phase="<<error.phase<<" hip_error="<<int(error.code)<<" name="<<hipGetErrorName(error.code)<<" message="<<hipGetErrorString(error.code)<<'\n';return 2;
 }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 3;}
}
