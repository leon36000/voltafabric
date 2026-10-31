#include "pool_hook.hpp"
#include <algorithm>
#include <limits>
#include <new>
#include <utility>
#include <vector>
namespace vf::strata_bridge {
bool pool_cpu_only(void* user,const float* x,const int32_t* ids,int64_t tokens,int64_t k,float* out,int64_t layer) noexcept {
    if (!user) return false;
    auto& ctx=*static_cast<PoolContext*>(user);
    try {
        std::lock_guard<std::mutex> lock(ctx.mutex);
        if(ctx.terminal) return false;
        ctx.terminal=true;
        if(!x || !ids || !out || layer!=ctx.layer || layer!=int64_t(ctx.client.layer()) || tokens<1 || tokens>8 || k<1 || k>64 || tokens*k>128 || ctx.next_batch==std::numeric_limits<uint64_t>::max()) {
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
bool pool_cpu_by_layer(void* user,const float* x,const int32_t* ids,int64_t tokens,int64_t k,float* out,int64_t layer) noexcept {
    if (!user) return false;
    auto& fleet=*static_cast<PoolFleet*>(user);
    try {
        std::lock_guard<std::mutex> lock(fleet.mutex_);
        if(fleet.terminal_) return false;
        fleet.terminal_=true;
        if(layer<0 || layer>=int64_t(fleet.contexts_.size())) return false;
        auto* ctx=fleet.contexts_[size_t(layer)];
        if(!ctx || ctx->layer!=layer || ctx->client.layer()!=uint32_t(layer)) return false;
        if(!pool_cpu_only(ctx,x,ids,tokens,k,out,layer)) return false;
        fleet.terminal_=false;
        return true;
    } catch (...) { return false; }
}

PoolFleetOwner::Contexts PoolFleetOwner::make_contexts(const Clients& clients, uint64_t first_batch) {
    Contexts contexts{};
    for (size_t layer = 0; layer < clients.size(); ++layer) {
        contexts[layer].reset(new PoolContext{*clients[layer], static_cast<int64_t>(layer), first_batch,
                                               false, {}, {}});
    }
    return contexts;
}

std::array<PoolContext*, 48> PoolFleetOwner::context_map(const Contexts& contexts) noexcept {
    std::array<PoolContext*, 48> map{};
    for (size_t layer = 0; layer < contexts.size(); ++layer) map[layer] = contexts[layer].get();
    return map;
}

PoolFleetOwner::PoolFleetOwner(Clients clients, uint64_t first_batch)
    : clients_(std::move(clients)), contexts_(make_contexts(clients_, first_batch)), fleet_(context_map(contexts_)) {}

std::unique_ptr<PoolFleetOwner> PoolFleetOwner::create(Clients clients, uint64_t first_batch, std::string& error) {
    error.clear();
    if (first_batch == std::numeric_limits<uint64_t>::max()) {
        error = "initial batch id cannot be advanced";
        return nullptr;
    }
    for (size_t layer = 0; layer < clients.size(); ++layer) {
        if (!clients[layer]) {
            error = "missing CPU session for layer " + std::to_string(layer);
            return nullptr;
        }
        if (!clients[layer]->usable() || clients[layer]->layer() != layer) {
            error = "CPU session is invalid or mapped to the wrong layer " + std::to_string(layer);
            return nullptr;
        }
    }
    try {
        return std::unique_ptr<PoolFleetOwner>(new PoolFleetOwner(std::move(clients), first_batch));
    } catch (const std::bad_alloc&) {
        error = "cannot allocate CPU session contexts";
        return nullptr;
    }
}

bool PoolFleetOwner::shutdown_all(std::string& error) {
    error.clear();
    {
        // Drain a callback already in flight, then reject future callbacks before
        // issuing one terminal request to each independent layer session.
        std::lock_guard<std::mutex> lock(fleet_.mutex_);
        if (shutdown_attempted_) {
            error = "session shutdown already attempted";
            return false;
        }
        shutdown_attempted_ = true;
        fleet_.terminal_ = true;
    }

    bool all_closed = true;
    for (size_t layer = 0; layer < clients_.size(); ++layer) {
        std::string layer_error;
        if (!clients_[layer]->shutdown(layer_error)) {
            all_closed = false;
            if (error.empty()) error = "layer " + std::to_string(layer) + ": " + layer_error;
        }
    }
    return all_closed;
}

}
