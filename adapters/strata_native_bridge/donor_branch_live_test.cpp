// Exercises the exact opt-in branch extracted from the patched donor source;
// the donor's ordinary path and GPU verifier are not present in this executable.
#include "strata/core/expert_source.hpp"
#include "pool_hook.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#ifdef VF_FULL_DONOR_TEST
#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif
int main(int argc,char** argv) {
    if(argc!=2) return 1;
    std::ifstream file(argv[1],std::ios::binary); std::vector<float> expected(2560);
    if(!file.read(reinterpret_cast<char*>(expected.data()),expected.size()*sizeof(float))) return 2;
    vf::strata_bridge::Client client(0,1,30000,0);
    vf::strata_bridge::PoolContext ctx{client,0,40,false,{}, {}};
    strata::core::ExpertDispatch dispatch;
    dispatch.vf_cpu_pool=&vf::strata_bridge::pool_cpu_only; dispatch.vf_cpu_context=&ctx; dispatch.n_expert=512;
    std::vector<float> x(2560,0.001f),out(2*2560,9.f); int32_t ids[2]={0,1};
    for(int i=0;i<2;++i) {
        dispatch.layers=0; // the actual driver binds the layer before every call
        strata::core::expert_pool_dispatch_multi(dispatch,x.data(),ids,1,2,out.data());
        if(dispatch.failed || dispatch.layers!=1) return 3;
        for(size_t j=0;j<2560;++j) if(std::memcmp(&out[j],&expected[j],4)||std::memcmp(&out[2560+j],&expected[j],4)) return 4;
    }
    std::fill(out.begin(),out.end(),9.f);
    strata::core::ExpertDispatch gpu_owner;
    gpu_owner.vf_cpu_pool=&vf::strata_bridge::pool_cpu_only; gpu_owner.vf_cpu_context=&ctx; gpu_owner.n_expert=512;
    std::vector<int32_t> residency(512,-1); residency[1]=0; gpu_owner.host_res=residency.data();
    strata::core::expert_pool_dispatch_multi(gpu_owner,x.data(),ids,1,2,out.data());
    if(!gpu_owner.failed || ctx.next_batch!=42 || ctx.terminal) return 5;
    for(float v:out) if(v!=9.f) return 6;
    strata::core::ExpertDispatch fail;
    fail.vf_cpu_pool=&vf::strata_bridge::pool_cpu_only; fail.vf_cpu_context=&ctx; fail.n_expert=512;
    ctx.layer=1;
    strata::core::expert_pool_dispatch_multi(fail,x.data(),ids,1,2,out.data());
    if(!fail.failed || !ctx.terminal || fail.layers!=0) return 7;
    for(float v:out) if(v!=9.f) return 8;
#ifdef VF_FULL_DONOR_TEST
    // The complete donor's unqualified fallback must hit an aborting test stub,
    // not an actual CUDA or peer implementation. Avoid leaving core files.
    pid_t pid=fork(); if(pid<0) return 10;
    if(pid==0) {
        rlimit limit{0,0}; setrlimit(RLIMIT_CORE,&limit);
        strata::core::ExpertDispatch ordinary;
        strata::core::expert_pool_dispatch_multi(ordinary,x.data(),ids,1,2,out.data());
        _exit(11);
    }
    int status=0; if(waitpid(pid,&status,0)!=pid || !WIFSIGNALED(status) || WTERMSIG(status)!=SIGABRT) return 12;
#endif
    std::string error; return client.shutdown(error)?0:9;
}
