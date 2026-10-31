#pragma once
#include "pipe_client.hpp"
#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <memory>
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
// Immutable mapping established before use. Contexts and clients must outlive
// the fleet; callers retain ownership and close every session.
class PoolFleet {
public:
    explicit PoolFleet(std::array<PoolContext*, 48> contexts) noexcept : contexts_(contexts) {}
    PoolFleet(const PoolFleet&) = delete;
    PoolFleet& operator=(const PoolFleet&) = delete;
    friend bool pool_cpu_by_layer(void*, const float*, const int32_t*, int64_t,
                                  int64_t, float*, int64_t) noexcept;
    friend class PoolFleetOwner;
private:
    const std::array<PoolContext*, 48> contexts_;
    std::mutex mutex_;
    bool terminal_ = false;
};
// Any failure latches the fleet so an incomplete model step is refused.
bool pool_cpu_by_layer(void* user, const float* x, const int32_t* ids, int64_t tokens,
                       int64_t top_k, float* out, int64_t layer) noexcept;

 // Owns the per-layer callback contexts and Client objects for one generation.
 // File descriptors remain borrowed by Client and must outlive this owner.
 // shutdown_all() is explicit and one-shot; the destructor never performs I/O.
 class PoolFleetOwner final {
 public:
     using Clients = std::array<std::unique_ptr<Client>, 48>;

     static std::unique_ptr<PoolFleetOwner> create(Clients clients, uint64_t first_batch,
                                                    std::string& error);
     PoolFleetOwner(const PoolFleetOwner&) = delete;
     PoolFleetOwner& operator=(const PoolFleetOwner&) = delete;

     PoolFleet& fleet() noexcept { return fleet_; }
     void* callback_user() noexcept { return &fleet_; }
     bool shutdown_all(std::string& error);

 private:
     using Contexts = std::array<std::unique_ptr<PoolContext>, 48>;
     PoolFleetOwner(Clients clients, uint64_t first_batch);
     static Contexts make_contexts(const Clients& clients, uint64_t first_batch);
     static std::array<PoolContext*, 48> context_map(const Contexts& contexts) noexcept;

     // Destruction is reverse declaration order: fleet, contexts, then clients.
     Clients clients_;
     Contexts contexts_;
     PoolFleet fleet_;
     bool shutdown_attempted_ = false;
 };

}
