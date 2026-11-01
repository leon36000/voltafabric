#include "qwen2moe_route_invocation.hpp"

#include <utility>

namespace vf::ggml {

Qwen2MoERouteInvocation::Qwen2MoERouteInvocation(RouteBatch identity_and_owners,
                                                 size_t expected_layers,
                                                 Collector collector,
                                                 void *collector_data)
    : identity_and_owners_(std::move(identity_and_owners)),
      expected_layers_(expected_layers),
      collector_(collector),
      collector_data_(collector_data) {
    const auto &identity = identity_and_owners_.identity;
    if (identity.run.empty() || identity.model.empty() || identity.plan.empty() ||
        identity.epoch == 0 || identity.batch == 0 || identity.attempt == 0 ||
        identity_and_owners_.owner_by_expert.empty() ||
        !identity_and_owners_.owner_kinds.empty() || expected_layers_ == 0 ||
        expected_layers_ > 4096 || collector_ == nullptr) {
        invalid_ = true;
        return;
    }
    host_gate_ = std::make_unique<RouteGraphInvocation>(expected_layers_);
    seen_layers_.resize(expected_layers_, false);
    transactions_.reserve(expected_layers_);
}

ggml_tensor *Qwen2MoERouteInvocation::append_layer(ggml_context *ctx, uint32_t layer,
                                                   ggml_tensor *activations,
                                                   ggml_tensor *selected_experts,
                                                   ggml_tensor *final_weights,
                                                   ggml_tensor *raw_expert_outputs) {
    if (invalid_ || finalized_ || host_gate_ == nullptr || layer >= expected_layers_ ||
        seen_layers_[layer] || ctx == nullptr || activations == nullptr ||
        selected_experts == nullptr || final_weights == nullptr || raw_expert_outputs == nullptr ||
        activations->ne[0] <= 0 || activations->ne[1] <= 0 ||
        selected_experts->ne[0] <= 0 || selected_experts->ne[1] != activations->ne[1] ||
        raw_expert_outputs->ne[0] != activations->ne[0] ||
        raw_expert_outputs->ne[1] != selected_experts->ne[0] ||
        raw_expert_outputs->ne[2] != activations->ne[1]) {
        invalid_ = true;
        return nullptr;
    }

    RouteBatch batch = identity_and_owners_;
    batch.identity.layer = layer;
    batch.hidden = static_cast<size_t>(activations->ne[0]);
    batch.slots = static_cast<size_t>(selected_experts->ne[0]);
    batch.tokens = static_cast<size_t>(activations->ne[1]);
    batch.experts = batch.owner_by_expert.size();
    batch.expert_ids.clear();
    batch.owner_kinds.clear();
    batch.weights.clear();

    auto transaction = std::make_unique<RouteGraphTransaction>(std::move(batch), collector_, collector_data_);
    ggml_tensor *result = transaction->append_dynamic(ctx, raw_expert_outputs, activations,
                                                      selected_experts, final_weights);
    if (result == nullptr || !host_gate_->add(*transaction)) {
        invalid_ = true;
        return nullptr;
    }

    // The custom GGML node stores a pointer to the transaction. Keep it stable
    // until the host gate has accepted or rejected this invocation.
    seen_layers_[layer] = true;
    transactions_.push_back(std::move(transaction));
    return result;
}

bool Qwen2MoERouteInvocation::accepts(ggml_status graph_status) {
    if (finalized_) return false;
    finalized_ = true;
    if (invalid_ || host_gate_ == nullptr) return false;
    return host_gate_->accepts(graph_status);
}

bool Qwen2MoERouteInvocation::graph_reuse_allowed() const {
    return host_gate_ != nullptr && host_gate_->graph_reuse_allowed();
}

} // namespace vf::ggml
