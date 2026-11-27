// 八卡peer分片可行性原型；未接入正式推理，顺序映射必须另外验证。
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstddef>
struct Pointers { uint16_t* data[8]; };
static hipEvent_t ready[8], gathered[8];
static bool initialized=false;
__device__ __forceinline__ uint16_t add_bf16(uint16_t a,uint16_t b){
 float value=__uint_as_float(uint32_t(a)<<16)+__uint_as_float(uint32_t(b)<<16);
 uint32_t bits=__float_as_uint(value);
 // 有限输入的逐步RNE；NaN行为在专项验证前不声明与RCCL相同。
 return uint16_t((bits+0x7fff+((bits>>16)&1))>>16);
}
__device__ __forceinline__ uint16_t cast_ninetoothed(float value){
 uint32_t bits=__float_as_uint(value);
 return (bits&0x7fffffff)>0x7f800000?uint16_t(0x7fff):uint16_t((bits+0x7fff+((bits>>16)&1))>>16);
}
template<int V> struct Type;
template<> struct Type<1>{using Word=uint16_t;};
template<> struct Type<2>{using Word=uint32_t;};
template<> struct Type<4>{using Word=uint2;};
template<> struct Type<8>{using Word=uint4;};
template<int V> union Pack{typename Type<V>::Word word;uint16_t value[V];};
#ifdef PEER_PACKED_RING
__device__ __forceinline__ unsigned packed_rank(unsigned i){
 constexpr unsigned bits=(1U<<3)|(3U<<6)|(2U<<9)|(4U<<12)|(6U<<15)|(5U<<18)|(7U<<21);
 return (bits>>(3*(i&7)))&7;
}
#define RING_RANK(I) packed_rank(I)
#else
#define RING_RANK(I) ring[(I)&7]
#endif
template<int Owner,int V> __device__ __forceinline__ Pack<V> reduce_fixed(Pointers src,size_t idx){
 constexpr int ring[8]={0,1,3,2,4,6,5,7};
 Pack<V> acc;acc.word=*reinterpret_cast<typename Type<V>::Word*>(src.data[ring[(Owner+1)%8]]+idx);
 #pragma unroll
 for(int j=2;j<=8;++j){
  Pack<V> next;next.word=*reinterpret_cast<typename Type<V>::Word*>(src.data[ring[(Owner+j)%8]]+idx);
  #pragma unroll
  for(int v=0;v<V;++v)acc.value[v]=add_bf16(acc.value[v],next.value[v]);
 }
 return acc;
}
template<int V> __global__ __launch_bounds__(256) void peer_reduce(Pointers src,uint16_t* out,size_t count,int rank,const uint8_t* order_map){
 const int ring[8]={0,1,3,2,4,6,5,7};size_t shard=count/8;
 for(size_t local=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)*V;local<shard;local+=size_t(gridDim.x)*blockDim.x*V){
  size_t idx=size_t(rank)*shard+local;
  // nullptr只用于性能可行性；完整顺序映射通过数值闸门前不得接入模型。
  int owner=order_map?order_map[idx/8]:rank;
#ifdef PEER_SPECIALIZED_OWNER
  Pack<V> acc;
  #define OWNER_CASE(O) case O:acc=reduce_fixed<O,V>(src,idx);break
  switch(owner){OWNER_CASE(0);OWNER_CASE(1);OWNER_CASE(2);OWNER_CASE(3);OWNER_CASE(4);OWNER_CASE(5);OWNER_CASE(6);default:acc=reduce_fixed<7,V>(src,idx);break;}
  #undef OWNER_CASE
#else
  Pack<V> acc;acc.word=*reinterpret_cast<typename Type<V>::Word*>(src.data[RING_RANK(owner+1)]+idx);
  #pragma unroll
  for(int j=2;j<=8;++j){
   Pack<V> next;next.word=*reinterpret_cast<typename Type<V>::Word*>(src.data[RING_RANK(owner+j)]+idx);
   #pragma unroll
   for(int v=0;v<V;++v)acc.value[v]=add_bf16(acc.value[v],next.value[v]);
  }
#endif
  *reinterpret_cast<typename Type<V>::Word*>(out+local)=acc.word;
 }
}
template<int V> __global__ __launch_bounds__(256) void peer_reduce_gate(Pointers src,uint16_t* out,size_t count,int rank,const uint8_t* order_map,const uint16_t* residual,const uint16_t* gate,const int64_t* indices){
 const int ring[8]={0,1,3,2,4,6,5,7};size_t shard=count/8;
 for(size_t local=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)*V;local<shard;local+=size_t(gridDim.x)*blockDim.x*V){
  size_t idx=size_t(rank)*shard+local;
  // nullptr只用于性能可行性；完整顺序映射通过数值闸门前不得接入模型。
  int owner=order_map?order_map[idx/8]:rank;
#ifdef PEER_SPECIALIZED_OWNER
  Pack<V> acc;
  #define OWNER_CASE(O) case O:acc=reduce_fixed<O,V>(src,idx);break
  switch(owner){OWNER_CASE(0);OWNER_CASE(1);OWNER_CASE(2);OWNER_CASE(3);OWNER_CASE(4);OWNER_CASE(5);OWNER_CASE(6);default:acc=reduce_fixed<7,V>(src,idx);break;}
  #undef OWNER_CASE
#else
  Pack<V> acc;acc.word=*reinterpret_cast<typename Type<V>::Word*>(src.data[RING_RANK(owner+1)]+idx);
  #pragma unroll
  for(int j=2;j<=8;++j){
   Pack<V> next;next.word=*reinterpret_cast<typename Type<V>::Word*>(src.data[RING_RANK(owner+j)]+idx);
   #pragma unroll
   for(int v=0;v<V;++v)acc.value[v]=add_bf16(acc.value[v],next.value[v]);
  }
#endif
  constexpr int width=5376,table_stride=6*width;
  int row=int(idx)/width,column=int(idx)%width;
  Pack<V> old,g;old.word=*reinterpret_cast<const typename Type<V>::Word*>(residual+idx);
  g.word=*reinterpret_cast<const typename Type<V>::Word*>(gate+indices[row]*table_stride+column);
  #pragma unroll
  for(int v=0;v<V;++v){
   float product=__uint_as_float(uint32_t(acc.value[v])<<16)*__uint_as_float(uint32_t(g.value[v])<<16);
   uint16_t scaled=cast_ninetoothed(product);
   acc.value[v]=cast_ninetoothed(__uint_as_float(uint32_t(old.value[v])<<16)+__uint_as_float(uint32_t(scaled)<<16));
  }
  *reinterpret_cast<typename Type<V>::Word*>(out+local)=acc.word;
 }
}
template<int V> __global__ __launch_bounds__(256) void peer_gather(Pointers shards,uint16_t* out,size_t count){
 size_t shard=count/8;
 for(size_t idx=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)*V;idx<count;idx+=size_t(gridDim.x)*blockDim.x*V){
  *reinterpret_cast<typename Type<V>::Word*>(out+idx)=*reinterpret_cast<typename Type<V>::Word*>(shards.data[idx/shard]+idx%shard);
 }
}
template<int V> __global__ __launch_bounds__(256) void peer_broadcast(uint16_t* local, Pointers outputs,size_t count,int rank){
 size_t shard=count/8;
 for(size_t offset=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)*V;offset<shard;offset+=size_t(gridDim.x)*blockDim.x*V){
  auto word=*reinterpret_cast<typename Type<V>::Word*>(local+offset);
  #pragma unroll
  for(int dest=0;dest<8;++dest)*reinterpret_cast<typename Type<V>::Word*>(outputs.data[dest]+size_t(rank)*shard+offset)=word;
 }
}
#define HIP_CHECK(call) do{auto e=(call);if(e!=hipSuccess)return int(e);}while(0)
extern "C" int peer_setup(){
 if(initialized)return 0;
 for(int rank=0;rank<8;++rank){HIP_CHECK(hipSetDevice(rank));
  for(int peer=0;peer<8;++peer)if(peer!=rank){auto e=hipDeviceEnablePeerAccess(peer,0);if(e!=hipSuccess&&e!=hipErrorPeerAccessAlreadyEnabled)return int(e);}
  HIP_CHECK(hipEventCreateWithFlags(&ready[rank],hipEventDisableTiming));HIP_CHECK(hipEventCreateWithFlags(&gathered[rank],hipEventDisableTiming));
 }
 initialized=true;return 0;
}
extern "C" int peer_execute_mode(uint64_t* inputs,uint64_t* shards,uint64_t* outputs,uint64_t* streams,uint64_t* maps,size_t count,int vec,int blocks,int push){
 if(!initialized||count%64||!count||blocks<1||(vec!=1&&vec!=2&&vec!=4&&vec!=8))return -1;
 Pointers src,parts,dest;for(int i=0;i<8;++i){src.data[i]=reinterpret_cast<uint16_t*>(inputs[i]);parts.data[i]=reinterpret_cast<uint16_t*>(shards[i]);dest.data[i]=reinterpret_cast<uint16_t*>(outputs[i]);}
 for(int rank=0;rank<8;++rank){HIP_CHECK(hipSetDevice(rank));auto stream=reinterpret_cast<hipStream_t>(streams[rank]);auto map=maps?reinterpret_cast<uint8_t*>(maps[rank]):nullptr;
  #define LAUNCH_REDUCE(V) hipLaunchKernelGGL((peer_reduce<V>),dim3(blocks),dim3(256),0,stream,src,parts.data[rank],count,rank,map)
  switch(vec){case 1:LAUNCH_REDUCE(1);break;case 2:LAUNCH_REDUCE(2);break;case 4:LAUNCH_REDUCE(4);break;case 8:LAUNCH_REDUCE(8);break;}
  HIP_CHECK(hipGetLastError());HIP_CHECK(hipEventRecord(ready[rank],stream));
 }
 for(int rank=0;rank<8;++rank){HIP_CHECK(hipSetDevice(rank));auto stream=reinterpret_cast<hipStream_t>(streams[rank]);
  for(int peer=0;peer<8;++peer)HIP_CHECK(hipStreamWaitEvent(stream,ready[peer],0));
  auto out=reinterpret_cast<uint16_t*>(outputs[rank]);
  #define LAUNCH_GATHER(V) hipLaunchKernelGGL((peer_gather<V>),dim3(blocks),dim3(256),0,stream,parts,out,count)
  #define LAUNCH_PUSH(V) hipLaunchKernelGGL((peer_broadcast<V>),dim3(blocks),dim3(256),0,stream,parts.data[rank],dest,count,rank)
  if(push){switch(vec){case 1:LAUNCH_PUSH(1);break;case 2:LAUNCH_PUSH(2);break;case 4:LAUNCH_PUSH(4);break;case 8:LAUNCH_PUSH(8);break;}}
  else{switch(vec){case 1:LAUNCH_GATHER(1);break;case 2:LAUNCH_GATHER(2);break;case 4:LAUNCH_GATHER(4);break;case 8:LAUNCH_GATHER(8);break;}}
  HIP_CHECK(hipGetLastError());
  if(push)HIP_CHECK(hipEventRecord(gathered[rank],stream));
 }
 // 推送写入由其他卡发起：本卡消费者之前必须等待全部发布完成。
 if(push)for(int rank=0;rank<8;++rank){HIP_CHECK(hipSetDevice(rank));auto stream=reinterpret_cast<hipStream_t>(streams[rank]);for(int peer=0;peer<8;++peer)HIP_CHECK(hipStreamWaitEvent(stream,gathered[peer],0));}
 return 0;
}
extern "C" int peer_execute_gate(uint64_t* inputs,uint64_t* shards,uint64_t* outputs,uint64_t* streams,uint64_t* maps,uint64_t* residuals,uint64_t* tables,uint64_t* indices,size_t count,int vec,int blocks){
 int push=1;
 if(count%5376||count>=2147483648ULL)return -2;
 if(!initialized||count%64||!count||blocks<1||(vec!=1&&vec!=2&&vec!=4&&vec!=8))return -1;
 Pointers src,parts,dest;for(int i=0;i<8;++i){src.data[i]=reinterpret_cast<uint16_t*>(inputs[i]);parts.data[i]=reinterpret_cast<uint16_t*>(shards[i]);dest.data[i]=reinterpret_cast<uint16_t*>(outputs[i]);}
 for(int rank=0;rank<8;++rank){HIP_CHECK(hipSetDevice(rank));auto stream=reinterpret_cast<hipStream_t>(streams[rank]);auto map=maps?reinterpret_cast<uint8_t*>(maps[rank]):nullptr;
 auto residual=reinterpret_cast<const uint16_t*>(residuals[rank]);auto table=reinterpret_cast<const uint16_t*>(tables[rank]);auto index=reinterpret_cast<const int64_t*>(indices[rank]);
  #define LAUNCH_REDUCE(V) hipLaunchKernelGGL((peer_reduce_gate<V>),dim3(blocks),dim3(256),0,stream,src,parts.data[rank],count,rank,map,residual,table,index)
  switch(vec){case 1:LAUNCH_REDUCE(1);break;case 2:LAUNCH_REDUCE(2);break;case 4:LAUNCH_REDUCE(4);break;case 8:LAUNCH_REDUCE(8);break;}
  HIP_CHECK(hipGetLastError());HIP_CHECK(hipEventRecord(ready[rank],stream));
 }
 for(int rank=0;rank<8;++rank){HIP_CHECK(hipSetDevice(rank));auto stream=reinterpret_cast<hipStream_t>(streams[rank]);
  for(int peer=0;peer<8;++peer)HIP_CHECK(hipStreamWaitEvent(stream,ready[peer],0));
  auto out=reinterpret_cast<uint16_t*>(outputs[rank]);
  #define LAUNCH_GATHER(V) hipLaunchKernelGGL((peer_gather<V>),dim3(blocks),dim3(256),0,stream,parts,out,count)
  #define LAUNCH_PUSH(V) hipLaunchKernelGGL((peer_broadcast<V>),dim3(blocks),dim3(256),0,stream,parts.data[rank],dest,count,rank)
  if(push){switch(vec){case 1:LAUNCH_PUSH(1);break;case 2:LAUNCH_PUSH(2);break;case 4:LAUNCH_PUSH(4);break;case 8:LAUNCH_PUSH(8);break;}}
  else{switch(vec){case 1:LAUNCH_GATHER(1);break;case 2:LAUNCH_GATHER(2);break;case 4:LAUNCH_GATHER(4);break;case 8:LAUNCH_GATHER(8);break;}}
  HIP_CHECK(hipGetLastError());
  if(push)HIP_CHECK(hipEventRecord(gathered[rank],stream));
 }
 // 推送写入由其他卡发起：本卡消费者之前必须等待全部发布完成。
 if(push)for(int rank=0;rank<8;++rank){HIP_CHECK(hipSetDevice(rank));auto stream=reinterpret_cast<hipStream_t>(streams[rank]);for(int peer=0;peer<8;++peer)HIP_CHECK(hipStreamWaitEvent(stream,gathered[peer],0));}
 return 0;
}
extern "C" int peer_execute(uint64_t* inputs,uint64_t* shards,uint64_t* outputs,uint64_t* streams,uint64_t* maps,size_t count,int vec,int blocks){return peer_execute_mode(inputs,shards,outputs,streams,maps,count,vec,blocks,0);}
extern "C" int peer_cleanup(){
 if(!initialized)return 0;
 for(int rank=0;rank<8;++rank){HIP_CHECK(hipSetDevice(rank));HIP_CHECK(hipEventDestroy(ready[rank]));HIP_CHECK(hipEventDestroy(gathered[rank]));}
 initialized=false;return 0;
}
