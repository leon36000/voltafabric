#include "pool_hook.hpp"
#include <algorithm>
#include <limits>
#include <vector>
namespace vf::strata_bridge {
bool pool_cpu_only(void* user,const float* x,const int32_t* ids,int64_t tokens,int64_t k,float* out,int64_t layer) noexcept {
    if (!user) return false;
    auto& ctx=*static_cast<PoolContext*>(user);
    try {
        std::lock_guard<std::mutex> lock(ctx.mutex);
        if(ctx.terminal) return false;
        ctx.terminal=true;
        if(!x || !ids || !out || layer!=ctx.layer || tokens<1 || tokens>8 || k<1 || k>64 || tokens*k>128 || ctx.next_batch==std::numeric_limits<uint64_t>::max()) {
            ctx.terminal=true; ctx.error="invalid native pool input"; return false;
        }
        const size_t n=size_t(tokens*k), width=2560;
        std::vector<float> input(x,x+size_t(tokens)*width), staged(n*width);
        std::vector<int32_t> expert(ids,ids+n), kinds(n,-1);
        if(!ctx.client.run(ctx.next_batch,int(tokens),int(k),input,expert,kinds,staged,ctx.error)) { ctx.terminal=true; return false; }
        std::copy(staged.begin(),staged.end(),out);
        ++ctx.next_batch;
        ctx.terminal=false;
        return true;
    } catch (...) { return false; }
}
}
