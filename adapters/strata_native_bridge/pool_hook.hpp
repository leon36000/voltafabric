#pragma once
#include "pipe_client.hpp"
#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <memory>
namespace vf::strata_bridge {
// One fixed layer and run/plan/epoch, bound by the preceding bridge Init.
// Batch and attempt advance independently after each complete CPU callback.
struct PoolContext {
    Client& client;
    int64_t layer;
    uint64_t next_batch;
    bool terminal = false;
    std::string error;
    std::mutex mutex;
    uint64_t next_attempt = 1;
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
    friend bool pool_cpu_misses_by_layer(void*, const float*, const int32_t*, const int32_t*,
                                         int64_t, int64_t, float*, int64_t) noexcept;
    friend class PoolFleetOwner;
private:
    const std::array<PoolContext*, 48> contexts_;
    std::mutex mutex_;
    bool terminal_ = false;
};
// Any failure latches the fleet so an incomplete model step is refused.
bool pool_cpu_by_layer(void* user, const float* x, const int32_t* ids, int64_t tokens,
                       int64_t top_k, float* out, int64_t layer) noexcept;

// Mixed-owner variant for Strata's final per-route ownership decision:
// -1 rows are computed by VoltaFabric; 0..2 rows remain owned by Strata's
// VRAM/PCIe/peer path. A successful call writes only the -1 rows.
bool pool_cpu_misses_by_layer(void* user, const float* x, const int32_t* ids,
                              const int32_t* kinds, int64_t tokens,
                              int64_t top_k, float* out, int64_t layer) noexcept;

 // Owns the per-layer callback contexts and Client objects for one generation.
 // File descriptors remain borrowed by Client and must outlive this owner.
 // shutdown_all() is explicit and one-shot; the destructor never performs I/O.
 class PoolFleetOwner final {
 public:
     using Clients = std::array<std::unique_ptr<Client>, 48>;

     static std::unique_ptr<PoolFleetOwner> create(Clients clients, uint64_t first_batch,
                                                    std::string& error, uint64_t first_attempt = 1);
     // Adopts a launcher-supplied set of already initialized AF_UNIX sockets.
     // Descriptor ownership remains with the launcher; each session identity
     // is checked by the Rust coordinator on every VFB3 request.
     static std::unique_ptr<PoolFleetOwner> create_from_fds(const std::array<int, 48>& fds,
                                                            int timeout_ms,
                                                            SessionIdentity identity,
                                                            uint64_t first_batch,
                                                            std::string& error,
                                                            uint64_t first_attempt = 1);
     PoolFleetOwner(const PoolFleetOwner&) = delete;
     PoolFleetOwner& operator=(const PoolFleetOwner&) = delete;

     PoolFleet& fleet() noexcept { return fleet_; }
     void* callback_user() noexcept { return &fleet_; }
     bool shutdown_all(std::string& error);

 private:
     using Contexts = std::array<std::unique_ptr<PoolContext>, 48>;
     PoolFleetOwner(Clients clients, uint64_t first_batch, uint64_t first_attempt);
     static Contexts make_contexts(const Clients& clients, uint64_t first_batch, uint64_t first_attempt);
     static std::array<PoolContext*, 48> context_map(const Contexts& contexts) noexcept;

     // Destruction is reverse declaration order: fleet, contexts, then clients.
     Clients clients_;
     Contexts contexts_;
     PoolFleet fleet_;
     bool shutdown_attempted_ = false;
 };

}
