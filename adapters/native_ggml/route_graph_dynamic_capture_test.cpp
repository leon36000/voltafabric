#include "route_graph_transaction.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {
using vf::ggml::BatchIdentity;
using vf::ggml::RouteBatch;
using vf::ggml::RouteGraphInvocation;
using vf::ggml::RouteGraphTransaction;
using vf::ggml::RouteStatus;

constexpr size_t kHidden = 2;
constexpr size_t kSlots = 2;
constexpr size_t kTokens = 2;
constexpr size_t kExperts = 8;

struct Capture {
    std::vector<int32_t> ids;
    std::vector<float> weights;
    size_t calls = 0;
};

RouteStatus capture_collector(const RouteBatch &batch, const float *activations,
                              size_t activation_count, const float *slot_outputs,
                              size_t slot_output_count, float *output,
                              size_t output_count, void *userdata) {
    auto &capture = *static_cast<Capture *>(userdata);
    if (batch.identity.run != "r673-dynamic-route" ||
        batch.identity.model != "qwen2moe-route-fixture" ||
        batch.identity.plan != "r673-final-gate-weights" || batch.identity.layer != 4 ||
        batch.identity.epoch != 6730261101ULL || batch.identity.batch != 673554953ULL ||
        batch.identity.attempt != 1 || activations == nullptr || activation_count != 4 ||
        slot_outputs == nullptr || slot_output_count != 8 || output == nullptr || output_count != 4 ||
        batch.expert_ids != std::vector<int32_t>{3, 1, 4, 7} || batch.weights.size() != 4 ||
        std::fabs(batch.weights[0] - 0.375f) > 1e-6f ||
        std::fabs(batch.weights[1] - 0.125f) > 1e-6f ||
        std::fabs(batch.weights[2] - 0.300f) > 1e-6f ||
        std::fabs(batch.weights[3] - 0.200f) > 1e-6f) {
        return RouteStatus::invalid_batch;
    }
    ++capture.calls;
    capture.ids = batch.expert_ids;
    capture.weights = batch.weights;
    for (size_t token = 0; token < batch.tokens; ++token) {
        for (size_t hidden = 0; hidden < batch.hidden; ++hidden) {
            double sum = 0.0;
            for (size_t slot = 0; slot < batch.slots; ++slot) {
                const size_t route = slot + batch.slots * token;
                const size_t input = hidden + batch.hidden * route;
                sum += static_cast<double>(batch.weights[route]) * slot_outputs[input];
            }
            output[hidden + batch.hidden * token] = static_cast<float>(sum);
        }
    }
    return RouteStatus::ready;
}

void copy_downstream(ggml_tensor *dst, const ggml_tensor *src, int ith, int nth, void *userdata) {
    auto &calls = *static_cast<size_t *>(userdata);
    if (ith != 0 || nth != 1 || dst->type != GGML_TYPE_F32 || src->type != GGML_TYPE_F32 ||
        dst->data == nullptr || src->data == nullptr || ggml_nelements(dst) != ggml_nelements(src)) {
        return;
    }
    std::memcpy(dst->data, src->data, static_cast<size_t>(ggml_nelements(src)) * sizeof(float));
    ++calls;
}

struct FreshCapture {
    size_t calls = 0;
};

RouteStatus capture_fresh_invocation(const RouteBatch &batch, const float *, size_t,
                                     const float *slot_outputs, size_t slot_output_count,
                                     float *output, size_t output_count, void *userdata) {
    auto &capture = *static_cast<FreshCapture *>(userdata);
    if (batch.identity.run != "r674-rebuilt-graph" ||
        batch.identity.model != "qwen2moe-route-fixture" ||
        batch.identity.plan != "r674-fresh-stamp" || batch.identity.layer != 4 ||
        batch.identity.epoch != 6740000001ULL || batch.identity.batch != 6740001ULL ||
        batch.identity.attempt != 1 || slot_outputs == nullptr || slot_output_count != 8 ||
        output == nullptr || output_count != 4 ||
        batch.expert_ids != std::vector<int32_t>{2, 5, 6, 0} || batch.weights.size() != 4 ||
        std::fabs(batch.weights[0] - 0.375f) > 1e-6f ||
        std::fabs(batch.weights[1] - 0.125f) > 1e-6f ||
        std::fabs(batch.weights[2] - 0.375f) > 1e-6f ||
        std::fabs(batch.weights[3] - 0.125f) > 1e-6f) {
        return RouteStatus::invalid_batch;
    }
    ++capture.calls;
    for (size_t token = 0; token < batch.tokens; ++token) {
        for (size_t hidden = 0; hidden < batch.hidden; ++hidden) {
            double sum = 0.0;
            for (size_t slot = 0; slot < batch.slots; ++slot) {
                const size_t route = slot + batch.slots * token;
                const size_t input = hidden + batch.hidden * route;
                sum += static_cast<double>(batch.weights[route]) * slot_outputs[input];
            }
            output[hidden + batch.hidden * token] = static_cast<float>(sum);
        }
    }
    return RouteStatus::ready;
}

struct Graph {
    ggml_context *ctx = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_cplan plan{};
    std::vector<uint8_t> work;
    ggml_tensor *downstream = nullptr;
    ~Graph() { if (ctx != nullptr) ggml_free(ctx); }
};

bool run_case(bool invalid_expert) {
    RouteBatch batch;
    batch.identity = BatchIdentity{"r673-dynamic-route", "qwen2moe-route-fixture",
                                   "r673-final-gate-weights", 4, 6730261101ULL,
                                   673554953ULL, 1};
    batch.hidden = kHidden;
    batch.slots = kSlots;
    batch.tokens = kTokens;
    batch.experts = 8;
    batch.owner_kinds = {0, 0, 0, 0};

    Capture capture;
    RouteGraphTransaction transaction(batch, capture_collector, &capture);
    size_t downstream_calls = 0;
    Graph g;
    ggml_init_params params{};
    params.mem_size = 1U << 20;
    g.ctx = ggml_init(params);
    if (g.ctx == nullptr) return false;

    ggml_tensor *slot_outputs = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, kHidden, kSlots, kTokens);
    ggml_tensor *activations = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, kHidden, kTokens);
    if (slot_outputs == nullptr || activations == nullptr || slot_outputs->data == nullptr ||
        activations->data == nullptr || !ggml_is_contiguous(slot_outputs) ||
        !ggml_is_contiguous(activations)) {
        std::cerr << "dynamic fixture input allocation failed\n";
        return false;
    }

    const float raw_slots[] = {1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 20.0f, 30.0f, 40.0f};
    const float activation_values[] = {0.5f, 1.0f, 2.0f, 3.0f};
    std::memcpy(slot_outputs->data, raw_slots, sizeof(raw_slots));
    std::memcpy(activations->data, activation_values, sizeof(activation_values));

    ggml_tensor *selected = nullptr;
    ggml_tensor *final_weights = nullptr;
    ggml_tensor *router_logits = nullptr;
    if (invalid_expert) {
        selected = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, kSlots, kTokens);
        final_weights = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 1, kSlots, kTokens);
        if (selected == nullptr || final_weights == nullptr || selected->data == nullptr ||
            final_weights->data == nullptr) return false;
        const int32_t invalid_ids[] = {3, 1, static_cast<int32_t>(kExperts), 7};
        const float weights[] = {0.375f, 0.125f, 0.300f, 0.200f};
        std::memcpy(selected->data, invalid_ids, sizeof(invalid_ids));
        std::memcpy(final_weights->data, weights, sizeof(weights));
    } else {
        router_logits = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, kExperts, kTokens);
        if (router_logits == nullptr || router_logits->data == nullptr) {
            std::cerr << "router logits allocation failed\n";
            return false;
        }
        auto *values = static_cast<float *>(router_logits->data);
        std::fill(values, values + kExperts * kTokens, -100.0f);
        values[3] = 1.0986123f;
        values[1] = 0.0f;
        values[kExperts + 4] = 0.4054651f;
        values[kExperts + 7] = 0.0f;
        ggml_tensor *probs = ggml_soft_max(g.ctx, router_logits);
        selected = ggml_argsort_top_k(g.ctx, probs, kSlots);
        ggml_tensor *probs_3d = ggml_reshape_3d(g.ctx, probs, 1, kExperts, kTokens);
        ggml_tensor *weights = ggml_get_rows(g.ctx, probs_3d, selected);
        final_weights = ggml_scale(g.ctx, weights, 0.5f);
        if (probs == nullptr || selected == nullptr || weights == nullptr || final_weights == nullptr ||
            !ggml_is_contiguous(final_weights)) {
            std::cerr << "GGML route graph construction failed: probs=" << (probs != nullptr)
                      << " selected=" << (selected != nullptr) << " weights=" << (weights != nullptr)
                      << " final=" << (final_weights != nullptr)
                      << " weights_contiguous=" << (final_weights != nullptr && ggml_is_contiguous(final_weights))
                      << "\n";
            return false;
        }
    }

    ggml_tensor *routed = transaction.append_dynamic(g.ctx, slot_outputs, activations,
                                                     selected, final_weights);
    g.downstream = routed == nullptr
        ? nullptr : ggml_map_custom1(g.ctx, routed, copy_downstream, 1, &downstream_calls);
    g.graph = ggml_new_graph(g.ctx);
    if (g.downstream == nullptr || g.graph == nullptr) {
        std::cerr << "dynamic GGML transaction graph construction failed\n";
        return false;
    }
    ggml_build_forward_expand(g.graph, g.downstream);
    g.plan = ggml_graph_plan(g.graph, 1, nullptr);
    g.work.resize(g.plan.work_size);
    g.plan.work_data = g.work.empty() ? nullptr : g.work.data();
    transaction.attach_abort(g.plan);

    const ggml_status status = ggml_graph_compute(g.graph, &g.plan);
    if (invalid_expert) {
        if (status != GGML_STATUS_ABORTED || transaction.status() != RouteStatus::invalid_batch ||
            capture.calls != 0 || downstream_calls != 0) {
            std::cerr << "invalid runtime expert was not rejected atomically\n";
            return false;
        }
        return true;
    }

    const bool values_match = transaction.accepts(status) && capture.calls == 1 &&
        capture.ids == std::vector<int32_t>{3, 1, 4, 7} && capture.weights.size() == 4 &&
        std::fabs(capture.weights[0] - 0.375f) <= 1e-6f &&
        std::fabs(capture.weights[1] - 0.125f) <= 1e-6f &&
        std::fabs(capture.weights[2] - 0.300f) <= 1e-6f &&
        std::fabs(capture.weights[3] - 0.200f) <= 1e-6f &&
        downstream_calls == 1;
    if (!values_match) {
        std::cerr << "dynamic route capture failed: status=" << ggml_status_to_string(status)
                  << " transaction=" << static_cast<int>(transaction.status())
                  << " collector_calls=" << capture.calls
                  << " downstream_calls=" << downstream_calls << "\n";
        return false;
    }
    const auto *out = static_cast<const float *>(g.downstream->data);
    if (std::fabs(out[0] - 0.75f) > 1e-6f || std::fabs(out[1] - 1.25f) > 1e-6f ||
        std::fabs(out[2] - 9.0f) > 1e-6f || std::fabs(out[3] - 14.0f) > 1e-6f) return false;

    // llama.cpp can reuse a graph object while replacing its input tensors. Make
    // the routes different and verify this single-use transaction does not
    // accidentally collect the next batch under the previous Stamp.
    auto *next_logits = static_cast<float *>(router_logits->data);
    std::fill(next_logits, next_logits + kExperts * kTokens, -100.0f);
    next_logits[2] = 1.0986123f;
    next_logits[5] = 0.0f;
    next_logits[kExperts + 6] = 1.0986123f;
    next_logits[kExperts + 0] = 0.0f;

    const ggml_status replay = ggml_graph_compute(g.graph, &g.plan);
    int32_t changed_ids[kSlots * kTokens]{};
    const auto *selected_bytes = static_cast<const uint8_t *>(selected->data);
    const auto *changed_weight_bytes = static_cast<const uint8_t *>(final_weights->data);
    float changed_weights[kSlots * kTokens]{};
    for (size_t token = 0; token < kTokens; ++token) {
        for (size_t slot = 0; slot < kSlots; ++slot) {
            const size_t route = slot + kSlots * token;
            std::memcpy(&changed_ids[route],
                        selected_bytes + slot * selected->nb[0] + token * selected->nb[1],
                        sizeof(int32_t));
            std::memcpy(&changed_weights[route],
                        changed_weight_bytes + slot * final_weights->nb[1] + token * final_weights->nb[2],
                        sizeof(float));
        }
    }
    if (replay != GGML_STATUS_ABORTED || transaction.status() != RouteStatus::invalid_batch ||
        changed_ids[0] != 2 || changed_ids[1] != 5 || changed_ids[2] != 6 || changed_ids[3] != 0 ||
        std::fabs(changed_weights[0] - 0.375f) > 1e-6f ||
        std::fabs(changed_weights[1] - 0.125f) > 1e-6f ||
        capture.calls != 1 || downstream_calls != 1) {
        std::cerr << "reused graph did not refuse a fresh dynamic route without a new transaction\n";
        return false;
    }
    if (ggml_graph_compute(g.graph, &g.plan) != GGML_STATUS_ABORTED ||
        transaction.status() != RouteStatus::invalid_batch || capture.calls != 1 || downstream_calls != 1) {
        std::cerr << "dynamic route repeated replay was not rejected\n";
        return false;
    }
    return true;
}

bool run_fresh_invocation() {
    RouteBatch batch;
    batch.identity = BatchIdentity{"r674-rebuilt-graph", "qwen2moe-route-fixture",
                                   "r674-fresh-stamp", 4, 6740000001ULL, 6740001ULL, 1};
    batch.hidden = kHidden;
    batch.slots = kSlots;
    batch.tokens = kTokens;
    batch.experts = kExperts;
    batch.owner_kinds = {0, 0, 0, 0};

    FreshCapture capture;
    RouteGraphTransaction transaction(batch, capture_fresh_invocation, &capture);
    RouteBatch pending_batch = batch;
    pending_batch.identity.batch++;
    RouteGraphTransaction pending_transaction(pending_batch, capture_fresh_invocation, &capture);
    RouteGraphInvocation invocation(1);
    RouteGraphInvocation incomplete_invocation(2);
    RouteGraphInvocation duplicate_stamp_invocation(2);
    if (!invocation.add(transaction) || invocation.graph_reuse_allowed() ||
        !incomplete_invocation.add(transaction) || !incomplete_invocation.add(pending_transaction) ||
        !duplicate_stamp_invocation.add(transaction) || duplicate_stamp_invocation.add(transaction)) {
        std::cerr << "per-invocation route registration/reuse gate failed\\n";
        return false;
    }
    size_t downstream_calls = 0;
    Graph g;
    ggml_init_params params{};
    params.mem_size = 1U << 20;
    g.ctx = ggml_init(params);
    if (g.ctx == nullptr) return false;

    ggml_tensor *slot_outputs = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, kHidden, kSlots, kTokens);
    ggml_tensor *activations = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, kHidden, kTokens);
    ggml_tensor *selected = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, kSlots, kTokens);
    ggml_tensor *weights = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 1, kSlots, kTokens);
    if (slot_outputs == nullptr || activations == nullptr || selected == nullptr || weights == nullptr ||
        slot_outputs->data == nullptr || activations->data == nullptr || selected->data == nullptr ||
        weights->data == nullptr) return false;

    const float raw_slots[] = {1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 20.0f, 30.0f, 40.0f};
    const float activation_values[] = {0.5f, 1.0f, 2.0f, 3.0f};
    const int32_t route_ids[] = {2, 5, 6, 0};
    const float route_weights[] = {0.375f, 0.125f, 0.375f, 0.125f};
    std::memcpy(slot_outputs->data, raw_slots, sizeof(raw_slots));
    std::memcpy(activations->data, activation_values, sizeof(activation_values));
    std::memcpy(selected->data, route_ids, sizeof(route_ids));
    std::memcpy(weights->data, route_weights, sizeof(route_weights));

    ggml_tensor *routed = transaction.append_dynamic(g.ctx, slot_outputs, activations, selected, weights);
    g.downstream = routed == nullptr
        ? nullptr : ggml_map_custom1(g.ctx, routed, copy_downstream, 1, &downstream_calls);
    g.graph = ggml_new_graph(g.ctx);
    if (g.downstream == nullptr || g.graph == nullptr) return false;
    ggml_build_forward_expand(g.graph, g.downstream);
    g.plan = ggml_graph_plan(g.graph, 1, nullptr);
    g.work.resize(g.plan.work_size);
    g.plan.work_data = g.work.empty() ? nullptr : g.work.data();
    transaction.attach_abort(g.plan);

    const ggml_status status = ggml_graph_compute(g.graph, &g.plan);
    const auto *out = static_cast<const float *>(g.downstream->data);
    const bool complete_batch_accepted = invocation.accepts(status);
    const bool incomplete_batch_rejected = !incomplete_invocation.accepts(status);
    return complete_batch_accepted && !invocation.accepts(status) && incomplete_batch_rejected &&
           transaction.status() == RouteStatus::ready &&
           capture.calls == 1 && downstream_calls == 1 &&
           std::fabs(out[0] - 0.75f) <= 1e-6f && std::fabs(out[1] - 1.25f) <= 1e-6f &&
           std::fabs(out[2] - 7.5f) <= 1e-6f && std::fabs(out[3] - 12.5f) <= 1e-6f;
}
} // namespace

int main() {
    if (!run_case(false) || !run_case(true) || !run_fresh_invocation()) return 1;
    std::cout << "dynamic GGML route capture: pinned softmax/top-k/gather/scale inputs, single weighting, atomic invalid-ID rejection, replay refusal, and fresh-graph/new-Stamp invocation passed\n";
    return 0;
}
