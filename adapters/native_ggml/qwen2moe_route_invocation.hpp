#pragma once

#include "route_graph_transaction.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace vf::ggml {

// Per-process_ubatch adapter for the pinned Qwen2MoE graph seam. The caller
// supplies the run/batch identity and expert->owner registry projection; each
// routed layer gets a fresh, single-use RouteGraphTransaction.
class Qwen2MoERouteInvocation {
public:
    Qwen2MoERouteInvocation(RouteBatch identity_and_owners, size_t expected_layers,
                            Collector collector, void *collector_data);
    Qwen2MoERouteInvocation(const Qwen2MoERouteInvocation &) = delete;
    Qwen2MoERouteInvocation &operator=(const Qwen2MoERouteInvocation &) = delete;

    // Mirrors the pinned build_moe_ffn data seam: original activation, dynamic
    // selected expert IDs, final post-gating weights, and raw unweighted down
    // expert outputs. Returns the VoltaFabric aggregate node or nullptr.
    ggml_tensor *append_layer(ggml_context *ctx, uint32_t layer,
                              ggml_tensor *activations, ggml_tensor *selected_experts,
                              ggml_tensor *final_weights, ggml_tensor *raw_expert_outputs);

    // Called by the host after graph_compute. A false result means the graph
    // output must not be published, even if GGML itself reported success.
    bool accepts(ggml_status graph_status);
    bool graph_reuse_allowed() const;
    bool ready() const { return !invalid_ && host_gate_ != nullptr; }
    size_t registered_layers() const { return transactions_.size(); }

private:
    RouteBatch identity_and_owners_;
    size_t expected_layers_ = 0;
    Collector collector_ = nullptr;
    void *collector_data_ = nullptr;
    std::unique_ptr<RouteGraphInvocation> host_gate_;
    std::vector<std::unique_ptr<RouteGraphTransaction>> transactions_;
    std::vector<bool> seen_layers_;
    bool invalid_ = false;
    bool finalized_ = false;
};

} // namespace vf::ggml
