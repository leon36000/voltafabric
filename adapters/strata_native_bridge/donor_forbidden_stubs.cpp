// Test-only stubs for the full donor's non-opt-in branches. Every such path
// aborts, including CUDA entry points: this binary cannot launch GPU work.
#include "strata/core/expert_cache.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
namespace strata::kernels::cpu {
const ExpertLayout& expert_layout(){std::abort();}
bool cpu_avx2_ok(){std::abort();}
void native_quant_act(const NativeFmt&,const float*,void*){std::abort();}
void act_quant_q8_1(const float*,int,ActQ&){std::abort();}
void act_quant_any(const float*,int,ActQ&){std::abort();}
void ExpertPool::run_split_multi(ExpertJobMulti*,int){std::abort();}
void ExpertPool::run_split_multi_native(const NativeFmt&,ExpertJobMulti*,int){std::abort();}
}
namespace strata::core {
int32_t ExpertCache::slot_of(int64_t,int64_t) const {std::abort();}
bool PeerExperts::launch(int64_t,const float*,const int32_t*,int64_t,int64_t,const int32_t*,std::string&,float*){std::abort();}
bool PeerExperts::finish(float*,std::string&){std::abort();}
bool RemoteExperts::begin(int64_t,const float*,const int32_t*,int64_t,int64_t,const int32_t*,const int32_t*,std::string&){std::abort();}
bool RemoteExperts::finish(float*,std::string&){std::abort();}
}
extern "C" {
cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t){std::abort();}
cudaError_t CUDARTAPI cudaGetLastError(){std::abort();}
cudaError_t CUDARTAPI cudaMemcpy(void*,const void*,size_t,cudaMemcpyKind){std::abort();}
cudaError_t CUDARTAPI cudaSetDevice(int){std::abort();}
cudaError_t CUDARTAPI cudaGetDevice(int*){std::abort();}
}
