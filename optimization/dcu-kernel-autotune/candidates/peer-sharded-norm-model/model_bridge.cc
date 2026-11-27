// H3八个RankWorker的异步stream桥；计算来自九齿镜像，未校准条件回退原RCCL及九齿门控。
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
#include <optional>
#include <set>
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
 for(int r=0;r<8;++r){hip_check(hipSetDevice(r));void *p=nullptr,*m=nullptr,*t=nullptr;hip_check(hipMalloc(&p,(count/8+5376)*2));hip_check(hipMalloc(&m,count/8));hip_check(hipMalloc(&t,count*2));e.parts[r]=reinterpret_cast<uintptr_t>(p);e.maps[r]=reinterpret_cast<uintptr_t>(m);temp[r]=reinterpret_cast<uintptr_t>(t);}
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
// 模型接口与独立原型分开：gate返回partial hidden，norm返回完整输出。
using Norm=const char*(*)(int,int,int,const void*,const void*,const void*,const void*,const void*,const void*,uintptr_t);
using ReferenceGate=void(*)(const Tensor&,const Tensor&,const Tensor&,const Tensor&,const Tensor&,int,int,infinicclComm_t,uintptr_t);
Norm norm=nullptr;ReferenceGate reference_gate=nullptr;bool enabled=true;
struct Pending{Tensor tensor;Entry* entry=nullptr;uint64_t stream=0;bool normalized=false;};
struct NormSlot{const void* input=nullptr;const void* output=nullptr;const void* weight=nullptr;const void* scale=nullptr;const void* shift=nullptr;const void* indices=nullptr;int rows=0;uint64_t stream=0;};
std::array<Pending,8> pending{};std::array<Tensor,8> new_hidden{};
std::array<NormSlot,8> norm_slots{};
int phase=0;uint64_t norm_batches=0,materialized=0,replica_checks=0,replica_fallbacks=0,full_norm_calls=0;
struct ReplicaRecord{bool ok;std::array<Tensor,8> weights;};
std::array<Tensor,8> norm_weight_hold{};
std::map<std::pair<uintptr_t,U8>,ReplicaRecord> replicas;
struct DeviceRestore{int prior;DeviceRestore(){hip_check(hipGetDevice(&prior));}~DeviceRestore(){auto e=hipSetDevice(prior);(void)e;}};
size_t row_first(size_t rows,int rank){return rows*rank/8;}
bool has_pending(){return std::any_of(pending.begin(),pending.end(),[](const Pending& p){return !p.tensor.empty();});}
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void fail(const std::string& message){std::lock_guard<std::mutex> lock(mutex);if(failure.empty())failure=message;cv.notify_all();}
void checked_state(){if(!failure.empty())throw std::runtime_error(failure);}
void materialize_pending(){
 DeviceRestore restore;require(has_pending(),"No partial hidden to materialize");
 Entry* e=pending[0].entry;require(e!=nullptr,"Partial hidden has no calibration");size_t rows=e->count/5376;U8 parts{},out{},streams{};
 for(int r=0;r<8;++r){auto& p=pending[r];require(!p.tensor.empty()&&p.entry==e&&p.tensor->numel()==e->count,"Partial ranks disagree during materialization");out[r]=reinterpret_cast<uintptr_t>(p.tensor->data());parts[r]=out[r]+row_first(rows,r)*5376*2;streams[r]=p.stream;}
 transport_check(publish(parts.data(),out.data(),streams.data(),e->count,5120));
 for(auto& p:pending)p=Pending{};++materialized;
}
bool verify_replicas(Entry& e){
 U8 weights{};for(int r=0;r<8;++r)weights[r]=reinterpret_cast<uintptr_t>(norm_slots[r].weight);
 auto key=std::make_pair(reinterpret_cast<uintptr_t>(&e),weights);auto found=replicas.find(key);if(found!=replicas.end())return found->second.ok;
 DeviceRestore restore;std::vector<unsigned char> expected;bool ok=true;
 for(int r=0;r<8;++r){hip_check(hipSetDevice(r));auto& n=norm_slots[r];hip_check(hipStreamSynchronize(reinterpret_cast<hipStream_t>(n.stream)));
  size_t width=5376*2,table=width*n.rows,idx=e.count/5376*8;std::vector<unsigned char> data(width+2*table+idx);
  hip_check(hipMemcpy(data.data(),n.weight,width,hipMemcpyDeviceToHost));
  hip_check(hipMemcpy2D(data.data()+width,width,n.scale,6*width,width,n.rows,hipMemcpyDeviceToHost));
  hip_check(hipMemcpy2D(data.data()+width+table,width,n.shift,6*width,width,n.rows,hipMemcpyDeviceToHost));
  hip_check(hipMemcpy(data.data()+width+2*table,n.indices,idx,hipMemcpyDeviceToHost));
  if(r==0)expected=std::move(data);else ok=ok&&(data==expected);
 }
 replicas.emplace(key,ReplicaRecord{ok,norm_weight_hold});++replica_checks;return ok;
}
void execute_gate(){
 DeviceRestore restore;size_t count=slots[0].count;U8 input{},streams{};
 bool partial=has_pending();
 for(int r=0;r<8;++r){auto& s=slots[r];require(s.count==count&&s.rows==slots[0].rows,"Gate rank shape mismatch");input[r]=reinterpret_cast<uintptr_t>(s.input);streams[r]=s.stream;
  if(partial){auto& p=pending[r];require(!p.tensor.empty()&&p.tensor->data()==s.residual&&p.tensor->numel()==count&&p.stream==s.stream&&p.normalized,"Gate residual identity/lifecycle mismatch");}
 }
 for(int r=0;r<8;++r)for(int p=0;p<8;++p){uintptr_t out=reinterpret_cast<uintptr_t>(slots[r].output);
  for(uintptr_t src:{input[p],reinterpret_cast<uintptr_t>(slots[p].residual)})require(!(out<src+count*2&&src<out+count*2),"New hidden aliases live input/residual");
 }
 transport_check(input_wait(streams.data()));Entry& e=entry(streams,count);
 if(!e.ok){
  if(partial)materialize_pending();allreduce(input,streams,count);
  for(int r=0;r<8;++r){hip_check(hipSetDevice(r));auto& s=slots[r];auto message=gate(true,r,int(count/5376),s.rows,s.residual,s.input,s.table,s.indices,s.output,s.stream);if(message)throw std::runtime_error(message);new_hidden[r].reset();}
  ++fallback;
 }else{
  size_t rows=count/5376;
  for(int r=0;r<8;++r){hip_check(hipSetDevice(r));auto& s=slots[r];size_t first=row_first(rows,r)*5376,local=(row_first(rows,r+1)-row_first(rows,r))*5376;
   std::vector<uint64_t> storage(abi.size()+2);std::vector<void*> params(storage.size());
   for(size_t a=0;a<abi.size();++a){int code=abi[a];uint64_t value=0;if(code<8)value=input[code];else switch(code){case 8:value=e.maps[r];break;case 9:value=reinterpret_cast<uintptr_t>(s.residual);break;case 10:value=reinterpret_cast<uintptr_t>(s.table);break;case 11:value=reinterpret_cast<uintptr_t>(s.indices);break;case 13:value=e.ringbits;break;case 14:value=reinterpret_cast<uintptr_t>(s.output)+first*2;break;case 104:value=first;break;case 105:value=local;break;case 100:value=count;break;case 101:value=count/8;break;case 102:value=s.rows;break;case 103:value=rows;break;default:throw std::runtime_error("Unknown model gate ABI");}storage[a]=value;}
   for(size_t a=0;a<storage.size();++a)params[a]=&storage[a];hip_check(hipModuleLaunchKernel(functions[r],unsigned((local+tile-1)/tile),1,1,warps*64,1,1,0,reinterpret_cast<hipStream_t>(s.stream),params.data(),nullptr));
  }
  for(int r=0;r<8;++r){pending[r]=Pending{new_hidden[r],&e,streams[r],false};new_hidden[r].reset();}++fused;
 }
 ++batches;
}
void execute_norm(){
 DeviceRestore restore;Entry* e=pending[0].entry;require(e!=nullptr,"Missing norm calibration");size_t rows=e->count/5376;
 for(int r=0;r<8;++r){auto& p=pending[r];auto& n=norm_slots[r];require(!p.tensor.empty()&&p.entry==e&&p.tensor->data()==n.input&&p.stream==n.stream&&n.rows==norm_slots[0].rows,"Norm partial identity mismatch");
  for(int j=0;j<8;++j){uintptr_t out=reinterpret_cast<uintptr_t>(n.output),src=reinterpret_cast<uintptr_t>(pending[j].tensor->data());require(!(out<src+e->count*2&&src<out+e->count*2),"Norm output aliases hidden");}
 }
 if(!verify_replicas(*e)){
  materialize_pending();
  for(int r=0;r<8;++r){hip_check(hipSetDevice(r));auto& n=norm_slots[r];auto message=norm(r,int(rows),n.rows,n.input,n.weight,n.scale,n.shift,n.indices,n.output,n.stream);if(message)throw std::runtime_error(message);}++replica_fallbacks;return;
 }
 U8 out{},streams{};
 for(int r=0;r<8;++r){hip_check(hipSetDevice(r));auto& n=norm_slots[r];size_t first=row_first(rows,r),local=row_first(rows,r+1)-first;
  auto message=norm(r,int(local),n.rows,reinterpret_cast<const char*>(n.input)+first*5376*2,n.weight,n.scale,n.shift,reinterpret_cast<const char*>(n.indices)+first*8,reinterpret_cast<void*>(e->parts[r]),n.stream);if(message)throw std::runtime_error(message);out[r]=reinterpret_cast<uintptr_t>(n.output);streams[r]=n.stream;
 }
 transport_check(publish(e->parts.data(),out.data(),streams.data(),e->count,5120));for(auto& p:pending)p.normalized=true;++norm_batches;
}
template<class Fill>void gather(int kind,int rank,Fill fill){
 std::unique_lock<std::mutex> lock(mutex);auto observed=generation;
 try{checked_state();require(rank>=0&&rank<8&&!(mask&(1U<<rank)),"Invalid/duplicate model rank");require(phase==0||phase==kind,"Model ranks reached different phases");phase=kind;fill();mask|=1U<<rank;
  if(mask==255){if(kind==1)execute_gate();else if(kind==2)execute_norm();else materialize_pending();mask=0;phase=0;++generation;lock.unlock();cv.notify_all();}
  else{cv.wait(lock,[&]{return generation!=observed||!failure.empty();});checked_state();}
 }catch(const std::exception& e){failure=e.what();if(lock.owns_lock())lock.unlock();cv.notify_all();throw;}
}
bool mode(){std::lock_guard<std::mutex> lock(mutex);checked_state();return enabled;}
void validate_tensor(const Tensor& tensor,int rank){require(!tensor.empty()&&tensor->dtype()==infinicore::DataType::BF16&&tensor->device().getType()==infinicore::Device::Type::HYGON&&tensor->device().getIndex()==size_t(rank),"Invalid model BF16 tensor");}
} // namespace
extern "C" const char* h3_peer_bridge_prepare(const char* image,int kernel_tile,int kernel_warps,const int* codes,int code_count,const char* transport_path,void* gate_callback){
 try{std::lock_guard<std::mutex> lock(mutex);require(!transport,"Model bridge already prepared");tile=kernel_tile;warps=kernel_warps;abi.assign(codes,codes+code_count);gate=reinterpret_cast<Gate>(gate_callback);require(gate&&tile>0&&warps>0,"Invalid model bridge configuration");
  transport=dlopen(transport_path,RTLD_NOW|RTLD_LOCAL);if(!transport)throw std::runtime_error(dlerror());setup=symbol<Setup>("transport_setup");input_wait=symbol<Inputs>("transport_inputs");publish=symbol<Publish>("transport_publish_rows");probe=symbol<Probe>("transport_probe");validate=symbol<Validate>("transport_validate");make_map=symbol<Map>("transport_map");transport_check(setup());
  DeviceRestore restore;for(int r=0;r<8;++r){hip_check(hipSetDevice(r));hip_check(hipModuleLoad(&modules[r],image));hip_check(hipModuleGetFunction(&functions[r],modules[r],"h3_peer_gate_kernel"));}return nullptr;
 }catch(const std::exception& e){error=e.what();return error.c_str();}
}
extern "C" const char* h3_model_prepare(void* norm_callback,void* reference_callback){
 try{std::lock_guard<std::mutex> lock(mutex);require(!norm&&!reference_gate&&norm_callback&&reference_callback,"Invalid/repeated model callbacks");norm=reinterpret_cast<Norm>(norm_callback);reference_gate=reinterpret_cast<ReferenceGate>(reference_callback);return nullptr;}catch(const std::exception& e){error=e.what();return error.c_str();}
}
extern "C" const char* h3_model_set_enabled(int value){
 try{std::lock_guard<std::mutex> lock(mutex);checked_state();require(mask==0&&!has_pending(),"Mode change requires completed forward and materialized hidden");enabled=value!=0;return nullptr;}catch(const std::exception& e){static thread_local std::string message;message=e.what();return message.c_str();}
}
extern "C" void h3_model_gate(const Tensor& output,const Tensor& residual,const Tensor& input,const Tensor& table,const Tensor& indices,int rank,int world,infinicclComm_t comm,uintptr_t stream){
 try{require(norm&&reference_gate,"Model callbacks not prepared");if(!mode()){reference_gate(output,residual,input,table,indices,rank,world,comm,stream);return;}
  require(world==8&&comm,"Model gate requires TP8 communicator");validate_tensor(output,rank);validate_tensor(input,rank);validate_tensor(residual,rank);size_t count=output->numel();require(count%5376==0&&count%128==0&&count<2147483648ULL&&input->numel()==count&&residual->numel()==count,"Invalid model gate shape");
  gather(1,rank,[&]{slots[rank]=Slot{input->data(),residual->data(),table->data(),indices->data(),output->data(),count,int(table->shape()[1]),comm,stream,rank};new_hidden[rank]=output;});
 }catch(const std::exception& e){fail(e.what());throw;}
}
extern "C" void h3_model_norm(const Tensor& output,const Tensor& input,const Tensor& weight,const Tensor& scale,const Tensor& shift,const Tensor& indices,uintptr_t stream){
 try{int rank=int(input->device().getIndex());require(norm&&rank>=0&&rank<8,"Invalid model norm context");bool partial=false;
  {std::lock_guard<std::mutex> lock(mutex);checked_state();auto& p=pending[rank];partial=!p.tensor.empty();if(partial)require(enabled&&p.tensor->data()==input->data()&&p.tensor->numel()==input->numel()&&p.stream==stream,"Norm reads unexpected partial Tensor");}
  if(!partial){auto message=norm(rank,int(input->numel()/5376),int(scale->shape()[1]),input->data(),weight->data(),scale->data(),shift->data(),indices->data(),output->data(),stream);if(message)throw std::runtime_error(message);std::lock_guard<std::mutex> lock(mutex);++full_norm_calls;return;}
  validate_tensor(output,rank);validate_tensor(weight,rank);require(input->is_contiguous()&&output->shape()==input->shape()&&weight->numel()==5376,"Invalid partial norm layout");
  gather(2,rank,[&]{norm_weight_hold[rank]=weight;norm_slots[rank]=NormSlot{input->data(),output->data(),weight->data(),scale->data(),shift->data(),indices->data(),int(scale->shape()[1]),stream};});
 }catch(const std::exception& e){fail(e.what());throw;}
}
extern "C" void h3_model_materialize(const Tensor& input,int rank,int world,infinicclComm_t comm,uintptr_t stream){
 try{bool partial;{std::lock_guard<std::mutex> lock(mutex);checked_state();partial=!pending.at(rank).tensor.empty();if(partial)require(world==8&&pending[rank].tensor->data()==input->data()&&pending[rank].tensor->numel()==input->numel()&&pending[rank].stream==stream&&pending[rank].entry->comm[rank]==reinterpret_cast<uintptr_t>(comm),"Materialization identity mismatch");}
  if(partial)gather(3,rank,[]{});
 }catch(const std::exception& e){fail(e.what());throw;}
}
extern "C" void h3_model_abort(){fail("Native forward aborted; partial state retained and subsequent calls rejected");}
extern "C" const char* h3_peer_bridge_stats(){
 std::lock_guard<std::mutex> lock(mutex);std::ostringstream s;s<<"{\"batches\":"<<batches<<",\"fused\":"<<fused<<",\"fallback\":"<<fallback<<",\"norm_batches\":"<<norm_batches<<",\"materialized\":"<<materialized<<",\"replica_checks\":"<<replica_checks<<",\"replica_fallbacks\":"<<replica_fallbacks<<",\"full_norm_calls\":"<<full_norm_calls<<",\"pending_ranks\":"<<std::count_if(pending.begin(),pending.end(),[](const Pending& p){return !p.tensor.empty();})<<",\"failed\":"<<(!failure.empty()?"true":"false")<<",\"calibrations\":[";bool first=true;for(auto& e:entries){if(!first)s<<',';first=false;s<<"{\"count\":"<<e->count<<",\"ok\":"<<(e->ok?"true":"false")<<",\"ringbits\":"<<e->ringbits<<",\"channels\":"<<e->channels<<",\"chunk\":"<<e->chunk<<",\"alignment\":"<<e->alignment<<"}";}s<<"]}";stats=s.str();return stats.c_str();
}
// 原始指针适配只供独立闸门；真正模型路径传递拥有生命周期的Tensor。
namespace {
Tensor raw(uint64_t ptr,const infinicore::Shape& shape,int rank,infinicore::DataType dtype=infinicore::DataType::BF16){return Tensor::from_blob(reinterpret_cast<void*>(ptr),shape,dtype,infinicore::Device(infinicore::Device::Type::HYGON,rank));}
Tensor raw_table(uint64_t ptr,int rows,int rank){return Tensor::strided_from_blob(reinterpret_cast<void*>(ptr),{1,size_t(rows),5376},{rows*6*5376,6*5376,1},infinicore::DataType::BF16,infinicore::Device(infinicore::Device::Type::HYGON,rank));}
template<class F>const char* raw_call(int rank,F fn){try{DeviceRestore restore;hip_check(hipSetDevice(rank));fn();return nullptr;}catch(const std::exception& e){static thread_local std::string message;message=e.what();return message.c_str();}}
}
extern "C" const char* h3_model_gate_raw(int rank,uint64_t input,uint64_t output,uint64_t residual,uint64_t table,uint64_t indices,size_t count,int table_rows,uint64_t comm,uint64_t stream){return raw_call(rank,[&]{infinicore::Shape shape{1,count/5376,5376};h3_model_gate(raw(output,shape,rank),raw(residual,shape,rank),raw(input,shape,rank),raw_table(table,table_rows,rank),raw(indices,{count/5376},rank,infinicore::DataType::I64),rank,8,reinterpret_cast<infinicclComm_t>(comm),stream);});}
extern "C" const char* h3_model_norm_raw(int rank,size_t count,int table_rows,uint64_t input,uint64_t weight,uint64_t scale,uint64_t shift,uint64_t indices,uint64_t output,uint64_t stream){return raw_call(rank,[&]{infinicore::Shape shape{1,count/5376,5376};h3_model_norm(raw(output,shape,rank),raw(input,shape,rank),raw(weight,{5376},rank),raw_table(scale,table_rows,rank),raw_table(shift,table_rows,rank),raw(indices,{count/5376},rank,infinicore::DataType::I64),stream);});}
extern "C" const char* h3_model_materialize_raw(int rank,size_t count,uint64_t input,uint64_t comm,uint64_t stream){return raw_call(rank,[&]{h3_model_materialize(raw(input,{1,count/5376,5376},rank),rank,8,reinterpret_cast<infinicclComm_t>(comm),stream);});}
