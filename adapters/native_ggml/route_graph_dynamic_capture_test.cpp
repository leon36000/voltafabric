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
        ggml_tensor *logits = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, kExperts, kTokens);
        if (logits == nullptr || logits->data == nullptr) {
            std::cerr << "router logits allocation failed\n";
            return false;
        }
        auto *values = static_cast<float *>(logits->data);
        std::fill(values, values + kExperts * kTokens, -100.0f);
        values[3] = 1.0986123f;
        values[1] = 0.0f;
        values[kExperts + 4] = 0.4054651f;
        values[kExperts + 7] = 0.0f;
        ggml_tensor *probs = ggml_soft_max(g.ctx, logits);
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

    if (ggml_graph_compute(g.graph, &g.plan) != GGML_STATUS_ABORTED ||
        transaction.status() != RouteStatus::invalid_batch || capture.calls != 1 || downstream_calls != 1) {
        std::cerr << "dynamic route replay was not rejected\n";
        return false;
    }
    return true;
}
} // namespace

int main() {
    if (!run_case(false) || !run_case(true)) return 1;
    std::cout << "dynamic GGML route capture: pinned softmax/top-k/gather/scale inputs, single weighting, atomic invalid-ID rejection, replay refusal passed\n";
    return 0;
}
