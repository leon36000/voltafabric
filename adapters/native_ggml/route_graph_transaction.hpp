#pragma once

#include "ggml-cpu.h"
#include "ggml.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vf::ggml {

enum class RouteStatus : uint8_t {
    pending,
    ready,
    provider_failure,
    invalid_batch,
    invalid_result,
    unsupported_tasking,
};

struct BatchIdentity {
    std::string run;
    std::string model;
    std::string plan;
    uint32_t layer = 0;
    uint64_t epoch = 0;
    uint64_t batch = 0;
    uint64_t attempt = 0;
};

struct RouteBatch {
    BatchIdentity identity;
    size_t hidden = 0;
    size_t slots = 0;
    size_t tokens = 0;
    size_t experts = 0;
    std::vector<int32_t> expert_ids; // [slot, token]
    std::vector<int32_t> owner_kinds; // optional VFB3 owner kinds, same route order
    // Optional registry projection for dynamic routes: one owner kind per expert
    // (-1 remote, 0..2 local), resolved after GGML provides each route ID.
    std::vector<int32_t> owner_by_expert;
    // Optional checkpoint-index -> VoltaFabric expert-ID projection for donor
    // graph routes. Empty preserves the original identity mapping.
    std::vector<int32_t> checkpoint_expert_to_registry_expert;
    std::vector<float> weights;      // [slot, token], forwarded unchanged
};

using Collector = RouteStatus (*)(const RouteBatch &, const float *, size_t, const float *, size_t,
                                   float *, size_t, void *);

// A single-threaded GGML graph boundary for a complete route batch. The caller must
// keep this object alive through ggml_graph_compute and accept output only when
// accepts(status) is true. The Collector callback is the seam for the Rust Session.
class RouteGraphTransaction {
public:
    RouteGraphTransaction(RouteBatch batch, Collector collector, void *collector_data);
    RouteGraphTransaction(const RouteGraphTransaction &) = delete;
    RouteGraphTransaction &operator=(const RouteGraphTransaction &) = delete;

    ggml_tensor *append(ggml_context *ctx, ggml_tensor *slot_outputs);
    // Optional activation source is a graph dependency for VFB3 dispatch. Slot
    // outputs contain raw local-owner results and placeholders for remote owners.
    ggml_tensor *append(ggml_context *ctx, ggml_tensor *slot_outputs, ggml_tensor *activations);
    // Capture the donor's already-selected route IDs and final post-gating weights
    // as graph inputs. IDs use [slot, token]; weights use [1, slot, token].
    ggml_tensor *append_dynamic(ggml_context *ctx, ggml_tensor *slot_outputs,
                                ggml_tensor *activations, ggml_tensor *selected_experts,
                                ggml_tensor *final_weights);
    void attach_abort(ggml_cplan &plan);
    bool accepts(ggml_status graph_status) const;
    RouteStatus status() const;
    const BatchIdentity &identity() const { return batch_.identity; }

private:
    static void compute(ggml_tensor *dst, int ith, int nth, void *userdata);
    static bool abort_after_failed_route(void *userdata);
    ggml_tensor *append_impl(ggml_context *ctx, ggml_tensor *slot_outputs,
                             ggml_tensor *activations, ggml_tensor *selected_experts,
                             ggml_tensor *final_weights, bool dynamic_route_inputs);

    RouteBatch batch_;
    Collector collector_ = nullptr;
    void *collector_data_ = nullptr;
    size_t output_count_ = 0;
    std::vector<float> staging_;
    bool appended_ = false;
    bool requires_activations_ = false;
    bool dynamic_route_inputs_ = false;
    std::atomic<bool> compute_started_{false};
    std::atomic<RouteStatus> status_{RouteStatus::pending};
};

// Host-side gate for one graph execution. Routed graphs are not reusable because
// every RouteGraphTransaction carries one complete, single-use Stamp. Construct
// the gate with the number of route transactions expected in this graph, register
// them before compute, and expose the graph result only after accepts() succeeds.
class RouteGraphInvocation {
public:
    explicit RouteGraphInvocation(size_t expected_transactions)
        : expected_transactions_(expected_transactions) {}

    bool add(const RouteGraphTransaction &transaction) {
        if (finalized_ || transactions_.size() >= expected_transactions_ ||
            transaction.status() != RouteStatus::pending) {
            return false;
        }
        const auto &identity = transaction.identity();
        for (const auto *existing : transactions_) {
            const auto &prior = existing->identity();
            if (prior.run == identity.run && prior.model == identity.model &&
                prior.plan == identity.plan && prior.layer == identity.layer &&
                prior.epoch == identity.epoch && prior.batch == identity.batch &&
                prior.attempt == identity.attempt) {
                return false;
            }
        }
        transactions_.push_back(&transaction);
        return true;
    }

    bool graph_reuse_allowed() const {
        return expected_transactions_ == 0 && transactions_.empty() && !finalized_;
    }

    bool accepts(ggml_status graph_status) {
        if (finalized_) return false;
        finalized_ = true;
        if (transactions_.size() != expected_transactions_ ||
            graph_status != GGML_STATUS_SUCCESS) {
            return false;
        }
        for (const auto *transaction : transactions_) {
            if (transaction == nullptr || transaction->status() != RouteStatus::ready) {
                return false;
            }
        }
        return true;
    }

    size_t expected_transactions() const { return expected_transactions_; }
    size_t registered_transactions() const { return transactions_.size(); }

private:
    size_t expected_transactions_ = 0;
    std::vector<const RouteGraphTransaction *> transactions_;
    bool finalized_ = false;
};

} // namespace vf::ggml

