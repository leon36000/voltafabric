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
    std::vector<float> weights;      // [slot, token], forwarded unchanged
};

using Collector = RouteStatus (*)(const RouteBatch &, const float *, size_t, float *, size_t, void *);

// A single-threaded GGML graph boundary for a complete route batch. The caller must
// keep this object alive through ggml_graph_compute and accept output only when
// accepts(status) is true. The Collector callback is the seam for the Rust Session.
class RouteGraphTransaction {
public:
    RouteGraphTransaction(RouteBatch batch, Collector collector, void *collector_data);
    RouteGraphTransaction(const RouteGraphTransaction &) = delete;
    RouteGraphTransaction &operator=(const RouteGraphTransaction &) = delete;

    ggml_tensor *append(ggml_context *ctx, ggml_tensor *slot_outputs);
    void attach_abort(ggml_cplan &plan);
    bool accepts(ggml_status graph_status) const;
    RouteStatus status() const;

private:
    static void compute(ggml_tensor *dst, int ith, int nth, void *userdata);
    static bool abort_after_failed_route(void *userdata);

    RouteBatch batch_;
    Collector collector_ = nullptr;
    void *collector_data_ = nullptr;
    size_t output_count_ = 0;
    std::vector<float> staging_;
    bool appended_ = false;
    std::atomic<bool> compute_started_{false};
    std::atomic<RouteStatus> status_{RouteStatus::pending};
};

} // namespace vf::ggml
