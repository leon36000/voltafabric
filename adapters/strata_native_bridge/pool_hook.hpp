#pragma once
#include "pipe_client.hpp"
#include <cstdint>
#include <mutex>
#include <string>
namespace vf::strata_bridge {
// One fixed layer and run/plan/epoch, bound by the preceding bridge Init.
// This opt-in hook serves CPU-only experts; no GPU, peer or helper may be active.
struct PoolContext {
    Client& client;
    int64_t layer;
    uint64_t next_batch;
    bool terminal = false;
    std::string error;
    std::mutex mutex;
};
// The caller must guarantee the pointer extents follow Strata's pinned callback
// signature. On failure this hook writes nothing and latches terminal.
bool pool_cpu_only(void* user, const float* x, const int32_t* ids, int64_t tokens,
                   int64_t top_k, float* out, int64_t layer) noexcept;
}
