// H3实验：原九齿连续分片计算不变；仅用InfiniCCL AllGather替代逐卡远端发布。
#include <hip/hip_runtime_api.h>
#include <infinicore/tensor.hpp>
#include <infiniccl.h>
#include <dlfcn.h>
#include <array>
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
using infinicore::Tensor;
namespace {
using U8=std::array<uint64_t,8>;
using Gate=const char*(*)(bool,int,int,int,const void*,const void*,const void*,const void*,const void*,uintptr_t);
using Setup=int(*)();using Inputs=int(*)(uint64_t*);using Publish=int(*)(uint64_t*,uint64_t*,uint64_t*,size_t,int);
using Probe=int(*)(uint64_t*,uint64_t*,uint64_t*,const uint16_t*,size_t,unsigned,unsigned,unsigned);
using Validate=int(*)(uint64_t*,uint64_t*,uint64_t*,const uint16_t*,size_t,uint64_t*);
using Map=int(*)(uint64_t*,uint64_t*,size_t,unsigned,unsigned,unsigned);
Setup setup=nullptr;Inputs input_wait=nullptr;Publish publish=nullptr;Probe probe=nullptr;Validate validate=nullptr;Map make_map=nullptr;Gate gate=nullptr;
void* transport=nullptr;std::array<hipModule_t,8> modules{};std::array<hipFunction_t,8> functions{};
int tile=0,warps=0;std::vector<int> abi;
struct Slot{const void *input,*residual,*table,*indices;const void* output;size_t count;int rows;infinicclComm_t comm;uint64_t stream;int device;};
std::array<Slot,8> slots{};std::mutex mutex;std::condition_variable cv;unsigned mask=0;uint64_t generation=0,batches=0,fused=0,fallback=0;std::string failure,stats,error;
struct Entry{size_t count=0;std::array<uintptr_t,8> comm{};U8 parts{},maps{};bool ok=false;unsigned ringbits=0,channels=0,chunk=0,alignment=0;std::array<uint64_t,8> errors{};};
std::vector<std::unique_ptr<Entry>> entries;
void hip_check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
void transport_check(int s){if(s)throw std::runtime_error("peer transport status "+std::to_string(s));}
void ccl_check(infiniStatus_t s){if(s!=INFINI_STATUS_SUCCESS)throw std::runtime_error("InfiniCCL status "+std::to_string(s));}
template<class T>T symbol(const char* name){auto p=dlsym(transport,name);if(!p)throw std::runtime_error(std::string("Missing transport symbol ")+name);return reinterpret_cast<T>(p);}
uint16_t add(uint16_t a,uint16_t b){uint32_t aa=uint32_t(a)<<16,bb=uint32_t(b)<<16,u;float x,y;std::memcpy(&x,&aa,4);std::memcpy(&y,&bb,4);float z=x+y;std::memcpy(&u,&z,4);return uint16_t((u+0x7fff+((u>>16)&1))>>16);}
void signatures(const std::array<int,8>& ring,const std::array<uint16_t,256>& patterns,std::array<uint16_t,256>& result){
 for(int o=0;o<8;++o)for(int k=0;k<32;++k){uint16_t x=patterns[ring[(o+1)%8]*32+k];for(int j=2;j<=8;++j)x=add(x,patterns[ring[(o+j)%8]*32+k]);result[o*32+k]=x;}
}
void allreduce(U8& buffers,U8& streams,size_t count){
 ccl_check(infinicclGroupStart(slots[0].comm));
 for(int r=0;r<8;++r){hip_check(hipSetDevice(r));ccl_check(infinicclAllReduce(reinterpret_cast<void*>(buffers[r]),reinterpret_cast<void*>(buffers[r]),count,INFINI_DTYPE_BF16,INFINICCL_SUM,slots[r].comm,reinterpret_cast<infinirtStream_t>(streams[r])));}
 ccl_check(infinicclGroupEnd(slots[0].comm));
}
void synchronize(U8& streams){for(int r=0;r<8;++r){hip_check(hipSetDevice(r));hip_check(hipStreamSynchronize(reinterpret_cast<hipStream_t>(streams[r])));}}
Entry& entry(U8& streams,size_t count){
 std::array<uintptr_t,8> comm{};for(int r=0;r<8;++r)comm[r]=reinterpret_cast<uintptr_t>(slots[r].comm);
 for(auto& e:entries)if(e->count==count&&e->comm==comm)return *e;
 auto item=std::make_unique<Entry>();item->count=count;item->comm=comm;Entry& e=*item;U8 temp{};
 for(int r=0;r<8;++r){hip_check(hipSetDevice(r));void *p=nullptr,*m=nullptr,*t=nullptr;hip_check(hipMalloc(&p,count/8*2));hip_check(hipMalloc(&m,count/8));hip_check(hipMalloc(&t,count*2));e.parts[r]=reinterpret_cast<uintptr_t>(p);e.maps[r]=reinterpret_cast<uintptr_t>(m);temp[r]=reinterpret_cast<uintptr_t>(t);}
 std::array<uint16_t,256> patterns{};std::mt19937 rng(823719);
 for(auto& x:patterns){auto v=rng();x=uint16_t(((v&1)<<15)|((125+(v%5))<<7)|((v>>8)&127));}
 transport_check(probe(temp.data(),e.maps.data(),streams.data(),patterns.data(),count,16,524288,1024));allreduce(temp,streams,count);synchronize(streams);
 for(unsigned channels:{16U,32U,8U})for(unsigned chunk:{524288U,262144U})for(unsigned alignment:{1024U,512U,2048U}){
  if(e.ok)continue;size_t length=count/channels,step=length<8*chunk?(((length+7)/8+alignment-1)/alignment)*alignment:chunk;
  if(7*step+32>length)continue;
  std::array<uint16_t,256> sampled{};hip_check(hipSetDevice(0));
  for(int o=0;o<8;++o)hip_check(hipMemcpy(sampled.data()+o*32,reinterpret_cast<uint16_t*>(temp[0])+o*step,64,hipMemcpyDeviceToHost));
  std::array<int,8> order{0,1,2,3,4,5,6,7},ring{};std::array<uint16_t,256> expected{};bool found=false;
  do{if(order[7]!=0)continue;bool match=true;
   for(int k=0;k<32&&match;++k){uint16_t x=patterns[order[0]*32+k];for(int j=1;j<8;++j)x=add(x,patterns[order[j]*32+k]);match=(x==sampled[k]);}
   if(!match)continue;ring[0]=0;for(int j=0;j<7;++j)ring[j+1]=order[j];signatures(ring,patterns,expected);
   if(expected==sampled){found=true;break;}
  }while(std::next_permutation(order.begin(),order.end()));
  if(!found)continue;
  transport_check(make_map(e.maps.data(),streams.data(),count,channels,chunk,alignment));transport_check(validate(temp.data(),e.maps.data(),streams.data(),expected.data(),count,e.errors.data()));
  if(std::all_of(e.errors.begin(),e.errors.end(),[](uint64_t n){return n==0;})){e.ok=true;e.channels=channels;e.chunk=chunk;e.alignment=alignment;for(int i=0;i<8;++i)e.ringbits|=unsigned(ring[i])<<(3*i);}
 }
 if(e.ok){
  std::mt19937 independent(823720);std::array<uint16_t,256> other{},expected{};std::array<int,8> ring{};
  for(auto& x:other){auto v=independent();x=uint16_t(((v&1)<<15)|((125+(v%5))<<7)|((v>>8)&127));}
  for(int i=0;i<8;++i)ring[i]=(e.ringbits>>(3*i))&7;signatures(ring,other,expected);
  transport_check(probe(temp.data(),e.maps.data(),streams.data(),other.data(),count,e.channels,e.chunk,e.alignment));allreduce(temp,streams,count);
  transport_check(validate(temp.data(),e.maps.data(),streams.data(),expected.data(),count,e.errors.data()));
  e.ok=std::all_of(e.errors.begin(),e.errors.end(),[](uint64_t n){return n==0;});
 }
 for(int r=0;r<8;++r){hip_check(hipSetDevice(r));hip_check(hipFree(reinterpret_cast<void*>(temp[r])));}
 std::cout<<"[peer_gate.calibration] {\"count\":"<<count<<",\"ok\":"<<(e.ok?"true":"false")<<",\"ringbits\":"<<e.ringbits<<",\"channels\":"<<e.channels<<",\"chunk\":"<<e.chunk<<",\"alignment\":"<<e.alignment<<"}"<<std::endl;
 entries.push_back(std::move(item));return e;
}
void execute(){
 int prior=0;hip_check(hipGetDevice(&prior));
 try{
  size_t count=slots[0].count;U8 input{},out{},streams{};
  for(int r=0;r<8;++r){if(slots[r].count!=count||slots[r].rows!=slots[0].rows)throw std::runtime_error("H3 peer ranks disagree on shape");input[r]=reinterpret_cast<uintptr_t>(slots[r].input);out[r]=reinterpret_cast<uintptr_t>(slots[r].output);streams[r]=slots[r].stream;}
  // 不覆盖任何尚被其他GPU读取的input/residual；限定真实H3独立输出路径。
  for(int r=0;r<8;++r)for(int p=0;p<8;++p){
   uintptr_t b=out[r],end=b+count*2;
   for(uintptr_t a:{input[p],reinterpret_cast<uintptr_t>(slots[p].residual)})
    if(b<a+count*2&&a<end)throw std::runtime_error("AllGather publish requires separate outputs");
  }
  transport_check(input_wait(streams.data()));Entry& e=entry(streams,count);
  if(!e.ok){allreduce(input,streams,count);for(int r=0;r<8;++r){hip_check(hipSetDevice(r));auto& s=slots[r];auto message=gate(true,r,int(count/5376),s.rows,s.residual,s.input,s.table,s.indices,s.output,s.stream);if(message)throw std::runtime_error(message);}++fallback;}
  else{
   for(int r=0;r<8;++r){hip_check(hipSetDevice(r));std::vector<uint64_t> storage(abi.size()+2);std::vector<void*> params(storage.size());
    for(size_t a=0;a<abi.size();++a){int code=abi[a];uint64_t value=0;if(code<8)value=input[code];else switch(code){case 8:value=e.maps[r];break;case 9:value=reinterpret_cast<uintptr_t>(slots[r].residual);break;case 10:value=reinterpret_cast<uintptr_t>(slots[r].table);break;case 11:value=reinterpret_cast<uintptr_t>(slots[r].indices);break;case 12:value=r;break;case 13:value=e.ringbits;break;case 14:value=e.parts[r];break;case 100:value=count;break;case 101:value=count/8;break;case 102:value=slots[r].rows;break;case 103:value=count/5376;break;default:throw std::runtime_error("Unknown peer kernel ABI code");}storage[a]=value;}
    for(size_t a=0;a<storage.size();++a)params[a]=&storage[a];hip_check(hipModuleLaunchKernel(functions[r],unsigned((count/8+tile-1)/tile),1,1,warps*64,1,1,0,reinterpret_cast<hipStream_t>(streams[r]),params.data(),nullptr));
   }
   // AllGather只复制各rank连续分片，不再做BF16求和；same-stream保证本卡partial已完成。
   ccl_check(infinicclGroupStart(slots[0].comm));
   for(int r=0;r<8;++r){hip_check(hipSetDevice(r));ccl_check(infinicclAllGather(
       reinterpret_cast<void*>(e.parts[r]),reinterpret_cast<void*>(out[r]),count/8,
       INFINI_DTYPE_BF16,slots[r].comm,reinterpret_cast<infinirtStream_t>(streams[r])));}
   ccl_check(infinicclGroupEnd(slots[0].comm));++fused;
  }
  ++batches;hip_check(hipSetDevice(prior));
 }catch(...){auto restored=hipSetDevice(prior);(void)restored;throw;}
}
}
extern "C" const char* h3_peer_bridge_prepare(const char* image,int kernel_tile,int kernel_warps,const int* codes,int code_count,const char* transport_path,void* gate_callback){
 try{std::lock_guard<std::mutex> lock(mutex);if(transport)throw std::runtime_error("Peer bridge already initialized");tile=kernel_tile;warps=kernel_warps;abi.assign(codes,codes+code_count);gate=reinterpret_cast<Gate>(gate_callback);if(!gate||tile<1||warps<1)throw std::runtime_error("Invalid peer bridge configuration");
  transport=dlopen(transport_path,RTLD_NOW|RTLD_LOCAL);if(!transport)throw std::runtime_error(dlerror());setup=symbol<Setup>("transport_setup");input_wait=symbol<Inputs>("transport_inputs");publish=symbol<Publish>("transport_publish");probe=symbol<Probe>("transport_probe");validate=symbol<Validate>("transport_validate");make_map=symbol<Map>("transport_map");transport_check(setup());
  int prior;hip_check(hipGetDevice(&prior));for(int r=0;r<8;++r){hip_check(hipSetDevice(r));hip_check(hipModuleLoad(&modules[r],image));hip_check(hipModuleGetFunction(&functions[r],modules[r],"h3_peer_gate_kernel"));}hip_check(hipSetDevice(prior));return nullptr;
 }catch(const std::exception& e){error=e.what();return error.c_str();}
}
namespace {
void arrive(int rank,int world,const Slot& value){
 std::unique_lock<std::mutex> lock(mutex);auto g=generation;
 try{if(!failure.empty())throw std::runtime_error(failure);if(world!=8||rank<0||rank>=8||value.device!=rank||(mask&(1U<<rank)))throw std::runtime_error("Unsupported/duplicate peer rank");
  if(value.count%5376||value.count%128||value.count>=2147483648ULL)throw std::runtime_error("Invalid H3 peer tensor shape");
  slots[rank]=value;mask|=1U<<rank;
  if(mask==255){execute();mask=0;++generation;lock.unlock();cv.notify_all();}
  else{cv.wait(lock,[&]{return generation!=g||!failure.empty();});if(!failure.empty())throw std::runtime_error(failure);}
 }catch(const std::exception& e){failure=e.what();if(lock.owns_lock())lock.unlock();cv.notify_all();throw;}
}
}
extern "C" void songshan_launch_peer_gate(const Tensor& output,const Tensor& residual,const Tensor& input,const Tensor& table,const Tensor& indices,int rank,int world,infinicclComm_t comm,uintptr_t stream){
 arrive(rank,world,Slot{input->data(),residual->data(),table->data(),indices->data(),output->data(),output->numel(),int(table->shape()[1]),comm,stream,int(output->device().getIndex())});
}
extern "C" const char* h3_peer_bridge_raw(int rank,uint64_t input,uint64_t output,uint64_t residual,uint64_t table,uint64_t indices,size_t count,int table_rows,uint64_t comm,uint64_t stream){
 int prior=0;
 try{hip_check(hipGetDevice(&prior));hip_check(hipSetDevice(rank));arrive(rank,8,Slot{reinterpret_cast<const void*>(input),reinterpret_cast<const void*>(residual),reinterpret_cast<const void*>(table),reinterpret_cast<const void*>(indices),reinterpret_cast<const void*>(output),count,table_rows,reinterpret_cast<infinicclComm_t>(comm),stream,rank});hip_check(hipSetDevice(prior));return nullptr;}
 catch(const std::exception& e){auto restored=hipSetDevice(prior);(void)restored;static thread_local std::string raw_error;raw_error=e.what();return raw_error.c_str();}
}
extern "C" const char* h3_peer_bridge_stats(){std::lock_guard<std::mutex> lock(mutex);std::ostringstream s;s<<"{\"batches\":"<<batches<<",\"fused\":"<<fused<<",\"fallback\":"<<fallback<<",\"calibrations\":[";bool first=true;for(auto& e:entries){if(!first)s<<',';first=false;s<<"{\"count\":"<<e->count<<",\"ok\":"<<(e->ok?"true":"false")<<",\"ringbits\":"<<e->ringbits<<",\"channels\":"<<e->channels<<",\"chunk\":"<<e->chunk<<",\"alignment\":"<<e->alignment<<"}";}s<<"]}";stats=s.str();return stats.c_str();}
