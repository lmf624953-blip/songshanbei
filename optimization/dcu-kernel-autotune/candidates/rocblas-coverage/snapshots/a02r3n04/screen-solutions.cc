// 按现用Tensile solution index筛选真实BF16 TN形状；不修改公共库或正式配置。
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
static void hip_ok(hipError_t x){if(x!=hipSuccess)throw std::runtime_error(hipGetErrorString(x));}
static void roc_ok(rocblas_status x){if(x!=rocblas_status_success)throw std::runtime_error("rocBLAS status "+std::to_string(x));}
static float median(std::vector<float> v){std::sort(v.begin(),v.end());return v[v.size()/2];}

#if defined(H3_SCREEN_DEVICE_COMPARE)
// 仅用于独立筛选的位模式检查，不注册到模型推理。
__global__ void compare_bits(const uint16_t* reference, const uint16_t* candidate,
                             size_t count, unsigned long long* mismatch) {
 unsigned long long errors = 0;
 for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
      index += size_t(gridDim.x) * blockDim.x) {
  errors += reference[index] != candidate[index];
 }
 if (errors) atomicAdd(mismatch, errors);
}
#endif

int main(int argc,char**argv){try{
 if(argc!=7)throw std::runtime_error("用法: screen-solutions shapes.tsv candidates.tsv warmup repeats pairs output.jsonl");
 const int warm=std::stoi(argv[3]),repeat=std::stoi(argv[4]),pairs=std::stoi(argv[5]);
 if(warm<1||repeat<1||pairs<1)throw std::runtime_error("计时参数必须为正");
 std::ifstream shapes(argv[1]),candidates(argv[2]);if(!shapes||!candidates)throw std::runtime_error("无法读取输入");
 std::vector<std::pair<int,std::string>> sols;int index;std::string name;
 while(candidates>>index>>name)sols.push_back({index,name});
 std::ofstream out(argv[6]);if(!out)throw std::runtime_error("无法写入结果");
 rocblas_handle handle;roc_ok(rocblas_create_handle(&handle));hipStream_t stream;hip_ok(hipStreamCreate(&stream));roc_ok(rocblas_set_stream(handle,stream));
 hipEvent_t start,stop;hip_ok(hipEventCreate(&start));hip_ok(hipEventCreate(&stop));
 int m,n,k;while(shapes>>m>>n>>k){
  const size_t ac=size_t(m)*k,bc=size_t(n)*k,cc=size_t(m)*n;
  std::vector<uint16_t>a(ac),b(bc),base(cc),cand(cc);std::mt19937 rng(42);
  // 通过原始BF16位构造正负、不同指数的有限值；不让host FP转换掩盖位差异。
  for(auto* v:{&a,&b})for(auto&x:*v){uint32_t r=rng();x=uint16_t(((r&1)<<15)|((123+(r%7))<<7)|((r>>8)&127));}
  void *da,*db,*dc,*dd;hip_ok(hipMalloc(&da,ac*2));hip_ok(hipMalloc(&db,bc*2));hip_ok(hipMalloc(&dc,cc*2));hip_ok(hipMalloc(&dd,cc*2));
  hip_ok(hipMemcpy(da,a.data(),ac*2,hipMemcpyHostToDevice));hip_ok(hipMemcpy(db,b.data(),bc*2,hipMemcpyHostToDevice));hip_ok(hipMemset(dc,0,cc*2));
  float alpha=1,beta=0;
  auto launch=[&](int solution){return rocblas_gemm_strided_batched_ex(handle,rocblas_operation_transpose,rocblas_operation_none,m,n,k,&alpha,
   da,rocblas_datatype_bf16_r,k,0,db,rocblas_datatype_bf16_r,k,0,&beta,dc,rocblas_datatype_bf16_r,m,0,
   dd,rocblas_datatype_bf16_r,m,0,1,rocblas_datatype_f32_r,solution<0?rocblas_gemm_algo_standard:rocblas_gemm_algo_solution_index,solution<0?0:solution+1,0);};
  auto timing=[&](int solution){hip_ok(hipEventRecord(start,stream));for(int i=0;i<repeat;++i)roc_ok(launch(solution));hip_ok(hipEventRecord(stop,stream));hip_ok(hipEventSynchronize(stop));float ms;hip_ok(hipEventElapsedTime(&ms,start,stop));return ms/repeat;};
  for(int t=0;t<warm;++t)roc_ok(launch(-1));hip_ok(hipStreamSynchronize(stream));hip_ok(hipMemcpy(base.data(),dd,cc*2,hipMemcpyDeviceToHost));
#if defined(H3_SCREEN_DEVICE_COMPARE)
  uint16_t* device_reference = nullptr;
  unsigned long long* device_mismatches = nullptr;
  hip_ok(hipMalloc(&device_reference, cc * sizeof(uint16_t)));
  hip_ok(hipMalloc(&device_mismatches, sizeof(unsigned long long)));
  hip_ok(hipMemcpy(device_reference, dd, cc * sizeof(uint16_t), hipMemcpyDeviceToDevice));
  int checks_cross_validated = 0;
#endif
  for(auto const& sol:sols){
   auto status=launch(sol.first);if(status!=rocblas_status_success){out<<"{\"m\":"<<m<<",\"n\":"<<n<<",\"k\":"<<k<<",\"index\":"<<sol.first<<",\"name\":\""<<sol.second<<"\",\"status\":"<<status<<"}\n";out.flush();continue;}
   size_t mismatch = 0;
#if defined(H3_SCREEN_DEVICE_COMPARE)
   hip_ok(hipMemsetAsync(device_mismatches, 0, sizeof(unsigned long long), stream));
   hipLaunchKernelGGL(compare_bits, dim3(4096), dim3(256), 0, stream,
                     device_reference, static_cast<const uint16_t*>(dd), cc, device_mismatches);
   hip_ok(hipGetLastError());
   hip_ok(hipStreamSynchronize(stream));
   unsigned long long device_count = 0;
   hip_ok(hipMemcpy(&device_count, device_mismatches, sizeof(device_count), hipMemcpyDeviceToHost));
   mismatch = size_t(device_count);
   if (checks_cross_validated < 3) {
    hip_ok(hipMemcpy(cand.data(), dd, cc * sizeof(uint16_t), hipMemcpyDeviceToHost));
    size_t host_count = 0;
    for (size_t i = 0; i < cc; ++i) host_count += base[i] != cand[i];
    if (host_count != mismatch) throw std::runtime_error("设备位差计数与CPU完整数组核对不一致");
    ++checks_cross_validated;
   }
#else
   hip_ok(hipStreamSynchronize(stream));
   hip_ok(hipMemcpy(cand.data(), dd, cc * sizeof(uint16_t), hipMemcpyDeviceToHost));
   for (size_t i = 0; i < cc; ++i) mismatch += base[i] != cand[i];
#endif
   // 位差异候选只归档数值结果，不进入性能选择。
   std::vector<float> before,after;
   if(!mismatch){for(int t=0;t<warm;++t)roc_ok(launch(sol.first));for(int t=0;t<pairs;++t){float x,y;if(t%2){y=timing(sol.first);x=timing(-1);}else{x=timing(-1);y=timing(sol.first);}before.push_back(x);after.push_back(y);}}
   out<<"{\"m\":"<<m<<",\"n\":"<<n<<",\"k\":"<<k<<",\"index\":"<<sol.first<<",\"name\":\""<<sol.second<<"\",\"status\":0,\"mismatches\":"<<mismatch<<",\"elements\":"<<cc<<",\"baseline_ms\":[";
   for(size_t i=0;i<before.size();++i){if(i)out<<',';out<<before[i];}out<<"],\"candidate_ms\":[";
   for(size_t i=0;i<after.size();++i){if(i)out<<',';out<<after[i];}out<<"]";
   if(!mismatch)out<<",\"baseline_median_ms\":"<<median(before)<<",\"candidate_median_ms\":"<<median(after)<<",\"gain_percent\":"<<100*(1-median(after)/median(before));
   out<<"}\n";out.flush();std::cout<<m<<' '<<n<<' '<<k<<' '<<sol.first<<" mismatches="<<mismatch<<std::endl;
  }
#if defined(H3_SCREEN_DEVICE_COMPARE)
  if (checks_cross_validated < std::min(size_t(3), sols.size()))
   throw std::runtime_error("设备检查缺少CPU交叉验证样本");
  hip_ok(hipFree(device_reference));
  hip_ok(hipFree(device_mismatches));
#endif
  for(void*p:{da,db,dc,dd})hip_ok(hipFree(p));
 }
 hip_ok(hipEventDestroy(start));hip_ok(hipEventDestroy(stop));roc_ok(rocblas_destroy_handle(handle));hip_ok(hipStreamDestroy(stream));return 0;
 }catch(std::exception const&e){std::cerr<<e.what()<<std::endl;return 1;}}
