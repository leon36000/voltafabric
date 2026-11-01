#include "qwen2moe_llama_route_bridge.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {
using namespace vf::ggml;

struct Captured {
    std::vector<uint32_t> layers;
    std::vector<BatchIdentity> identities;
    std::vector<std::vector<int32_t>> experts;
    std::vector<std::vector<int32_t>> owners;
    std::vector<std::vector<float>> weights;
};

RouteStatus collect(const RouteBatch &batch, const float *, size_t,
                    const float *slots, size_t slot_count,
                    float *output, size_t output_count, void *userdata) {
    auto &seen = *static_cast<Captured *>(userdata);
    const size_t routes = batch.slots * batch.tokens;
    if (slots == nullptr || output == nullptr ||
        slot_count != batch.hidden * routes || output_count != batch.hidden * batch.tokens ||
        batch.expert_ids.size() != routes || batch.owner_kinds.size() != routes ||
        batch.weights.size() != routes) {
        return RouteStatus::invalid_batch;
    }
    seen.layers.push_back(batch.identity.layer);
    seen.identities.push_back(batch.identity);
    seen.experts.push_back(batch.expert_ids);
    seen.owners.push_back(batch.owner_kinds);
    seen.weights.push_back(batch.weights);
    std::fill(output, output + output_count, 0.0f);
    for (size_t token = 0; token < batch.tokens; ++token) {
        for (size_t slot = 0; slot < batch.slots; ++slot) {
            const size_t route = slot + batch.slots * token;
            for (size_t channel = 0; channel < batch.hidden; ++channel) {
                output[channel + batch.hidden * token] +=
                    batch.weights[route] * slots[channel + batch.hidden * route];
            }
        }
    }
    return RouteStatus::ready;
}

bool make_batch(void *, const llama_ubatch &ubatch, llm_graph_type graph_type,
                uint32_t expected_layers, RouteBatch &batch) {
    if (ubatch.n_tokens != 1 || graph_type != LLM_GRAPH_TYPE_DEFAULT || expected_layers != 2) {
        return false;
    }
    batch.identity = BatchIdentity{"bridge-run", "qwen-model", "qwen-plan", 0, 31, 12, 4};
    batch.owner_by_expert = {0, -1, 1, 2};
    return true;
}

bool near(float actual, float expected) {
    return std::isfinite(actual) && std::fabs(actual - expected) <= 1.0e-5f;
}

bool identity_matches(const BatchIdentity &identity, uint32_t layer) {
    return identity.run == "bridge-run" && identity.model == "qwen-model" &&
        identity.plan == "qwen-plan" && identity.layer == layer &&
        identity.epoch == 31 && identity.batch == 12 && identity.attempt == 4;
}

ggml_tensor *append_layer(const llm_vf_route_hooks &hooks, ggml_context *ctx,
                          int32_t layer, float raw0, float raw1,
                          std::array<int32_t, 2> ids,
                          std::array<float, 2> weights) {
    auto *activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 1);
    auto *selected = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 1);
    auto *final_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 2, 1);
    auto *raw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 1);
    if (activation == nullptr || selected == nullptr || final_weights == nullptr || raw == nullptr) return nullptr;
    auto *a = static_cast<float *>(activation->data);
    a[0] = 0.5f;
    a[1] = -0.25f;
    std::copy(ids.begin(), ids.end(), static_cast<int32_t *>(selected->data));
    std::copy(weights.begin(), weights.end(), static_cast<float *>(final_weights->data));
    auto *r = static_cast<float *>(raw->data);
    r[0] = raw0;
    r[1] = raw0 + 1.0f;
    r[2] = raw1;
    r[3] = raw1 + 1.0f;
    return hooks.append_qwen2moe(hooks.userdata, ctx, layer, activation, selected, final_weights, raw);
}

bool test_complete_callback_invocation() {
    Captured captured;
    Qwen2MoELlamaRouteBridge bridge(make_batch, nullptr, collect, &captured);
    const auto *hooks = bridge.hooks();
    llama_ubatch ubatch{};
    ubatch.n_tokens = 1;
    if (hooks == nullptr || hooks->begin == nullptr || hooks->append_qwen2moe == nullptr ||
        hooks->finish == nullptr || hooks->abort == nullptr ||
        !hooks->begin(hooks->userdata, ubatch, LLM_GRAPH_TYPE_DEFAULT, 2)) return false;

    ggml_init_params params{};
    params.mem_size = 1U << 20;
    ggml_context *ctx = ggml_init(params);
    if (ctx == nullptr) return false;
    ggml_tensor *first = append_layer(*hooks, ctx, 0, 2.0f, 6.0f, {1, 3}, {0.25f, 0.75f});
    ggml_tensor *second = append_layer(*hooks, ctx, 1, 10.0f, 14.0f, {2, 0}, {0.4f, 0.6f});
    ggml_cgraph *graph = ggml_new_graph(ctx);
    if (first == nullptr || second == nullptr || graph == nullptr) {
        hooks->abort(hooks->userdata);
        ggml_free(ctx);
        return false;
    }
    ggml_tensor *combined = ggml_add(ctx, first, second);
    if (combined == nullptr) {
        hooks->abort(hooks->userdata);
        ggml_free(ctx);
        return false;
    }
    ggml_build_forward_expand(graph, combined);
    ggml_cplan plan = ggml_graph_plan(graph, 1, nullptr);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.empty() ? nullptr : work.data();
    const ggml_status status = ggml_graph_compute(graph, &plan);
    const bool accepted = hooks->finish(hooks->userdata, status == GGML_STATUS_SUCCESS);
    const auto *actual = static_cast<const float *>(combined->data);
    const bool result_ok = status == GGML_STATUS_SUCCESS && accepted &&
        near(actual[0], 17.4f) && near(actual[1], 19.4f) &&
        captured.layers == std::vector<uint32_t>({0, 1}) &&
        captured.identities.size() == 2 &&
        identity_matches(captured.identities[0], 0) && identity_matches(captured.identities[1], 1) &&
        captured.experts == std::vector<std::vector<int32_t>>({{1, 3}, {2, 0}}) &&
        captured.owners == std::vector<std::vector<int32_t>>({{-1, 2}, {1, 0}}) &&
        captured.weights == std::vector<std::vector<float>>({{0.25f, 0.75f}, {0.4f, 0.6f}});
    if (!accepted) hooks->abort(hooks->userdata);
    ggml_free(ctx);
    return result_ok;
}

bool test_missing_layer_refuses_host_result() {
    Captured captured;
    Qwen2MoELlamaRouteBridge bridge(make_batch, nullptr, collect, &captured);
    const auto *hooks = bridge.hooks();
    llama_ubatch ubatch{};
    ubatch.n_tokens = 1;
    if (!hooks->begin(hooks->userdata, ubatch, LLM_GRAPH_TYPE_DEFAULT, 2)) return false;
    ggml_init_params params{};
    params.mem_size = 1U << 20;
    ggml_context *ctx = ggml_init(params);
    if (ctx == nullptr) return false;
    ggml_tensor *only = append_layer(*hooks, ctx, 0, 2.0f, 6.0f, {1, 3}, {0.25f, 0.75f});
    ggml_cgraph *graph = ggml_new_graph(ctx);
    if (only == nullptr || graph == nullptr) {
        hooks->abort(hooks->userdata);
        ggml_free(ctx);
        return false;
    }
    ggml_build_forward_expand(graph, only);
    ggml_cplan plan = ggml_graph_plan(graph, 1, nullptr);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.empty() ? nullptr : work.data();
    const ggml_status status = ggml_graph_compute(graph, &plan);
    const bool accepted = hooks->finish(hooks->userdata, status == GGML_STATUS_SUCCESS);
    if (accepted) hooks->abort(hooks->userdata);
    ggml_free(ctx);
    return status == GGML_STATUS_SUCCESS && !accepted;
}

} // namespace

int main() {
    if (!test_complete_callback_invocation()) {
        std::cerr << "complete hook bridge fixture failed\n";
        return 1;
    }
    if (!test_missing_layer_refuses_host_result()) {
        std::cerr << "incomplete hook bridge was accepted\n";
        return 1;
    }
    std::cout << "Qwen2MoE llama hook bridge: complete oracle pass; missing layer refused\n";
    return 0;
}
