#include "qwen2moe_route_invocation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {
using vf::ggml::BatchIdentity;
using vf::ggml::Qwen2MoERouteInvocation;
using vf::ggml::RouteBatch;
using vf::ggml::RouteStatus;

struct SeenRoute {
    uint32_t layer = 0;
    std::vector<int32_t> experts;
    std::vector<int32_t> owners;
    std::vector<float> weights;
};

struct Capture {
    std::vector<SeenRoute> routes;
};

RouteStatus collect(const RouteBatch &batch, const float *, size_t,
                    const float *slots, size_t slot_count,
                    float *output, size_t output_count, void *userdata) {
    auto &capture = *static_cast<Capture *>(userdata);
    const size_t route_count = batch.slots * batch.tokens;
    if (slots == nullptr || output == nullptr ||
        slot_count != batch.hidden * route_count ||
        output_count != batch.hidden || batch.owner_kinds.size() != route_count ||
        batch.expert_ids.size() != route_count || batch.weights.size() != route_count) {
        return RouteStatus::invalid_batch;
    }
    SeenRoute seen;
    seen.layer = batch.identity.layer;
    seen.experts = batch.expert_ids;
    seen.owners = batch.owner_kinds;
    seen.weights = batch.weights;
    capture.routes.push_back(std::move(seen));
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

bool near(float actual, float expected) {
    return std::isfinite(actual) && std::fabs(actual - expected) <= 1.0e-5f;
}

struct Fixture {
    ggml_context *ctx = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_cplan plan{};
    std::vector<uint8_t> work;
    ggml_tensor *activation = nullptr;
    ggml_tensor *ids = nullptr;
    ggml_tensor *weights = nullptr;
    ggml_tensor *raw = nullptr;
    ggml_tensor *routed = nullptr;
};

bool build_layer(Fixture &f, Qwen2MoERouteInvocation &invocation,
                 uint32_t layer, const std::array<float, 2> &raw_values,
                 const std::array<int32_t, 2> &ids, const std::array<float, 2> &weights) {
    f.activation = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, 2, 1);
    f.ids = ggml_new_tensor_2d(f.ctx, GGML_TYPE_I32, 2, 1);
    f.weights = ggml_new_tensor_3d(f.ctx, GGML_TYPE_F32, 1, 2, 1);
    f.raw = ggml_new_tensor_3d(f.ctx, GGML_TYPE_F32, 2, 2, 1);
    if (f.activation == nullptr || f.ids == nullptr || f.weights == nullptr || f.raw == nullptr) return false;
    auto *activation_data = static_cast<float *>(f.activation->data);
    activation_data[0] = 0.5f;
    activation_data[1] = -0.25f;
    std::copy(ids.begin(), ids.end(), static_cast<int32_t *>(f.ids->data));
    std::copy(weights.begin(), weights.end(), static_cast<float *>(f.weights->data));
    auto *raw_data = static_cast<float *>(f.raw->data);
    raw_data[0] = raw_values[0];
    raw_data[1] = raw_values[0] + 1.0f;
    raw_data[2] = raw_values[1];
    raw_data[3] = raw_values[1] + 1.0f;
    f.routed = invocation.append_layer(f.ctx, layer, f.activation, f.ids, f.weights, f.raw);
    return f.routed != nullptr;
}

bool test_two_layer_invocation() {
    ggml_init_params params{};
    params.mem_size = 1U << 20;
    Fixture f;
    f.ctx = ggml_init(params);
    if (f.ctx == nullptr) return false;
    RouteBatch identity;
    identity.identity = BatchIdentity{"qwen-run", "qwen-model", "qwen-plan", 0, 19, 7, 2};
    identity.owner_by_expert = {0, -1, 1, 2};
    Capture capture;
    Qwen2MoERouteInvocation invocation(identity, 2, collect, &capture);
    if (invocation.graph_reuse_allowed()) return false;

    Fixture first, second;
    first.ctx = f.ctx;
    second.ctx = f.ctx;
    if (!build_layer(first, invocation, 0, {2.0f, 6.0f}, {1, 3}, {0.25f, 0.75f}) ||
        !build_layer(second, invocation, 1, {10.0f, 14.0f}, {2, 0}, {0.4f, 0.6f})) {
        std::cerr << "layer graph append failed; registered=" << invocation.registered_layers() << '\n';
        return false;
    }
    f.graph = ggml_new_graph(f.ctx);
    ggml_tensor *combined = ggml_add(f.ctx, first.routed, second.routed);
    if (f.graph == nullptr || combined == nullptr || invocation.registered_layers() != 2) return false;
    ggml_build_forward_expand(f.graph, combined);
    f.plan = ggml_graph_plan(f.graph, 1, nullptr);
    f.work.resize(f.plan.work_size);
    f.plan.work_data = f.work.empty() ? nullptr : f.work.data();
    const ggml_status status = ggml_graph_compute(f.graph, &f.plan);
    if (status != GGML_STATUS_SUCCESS || !invocation.accepts(status)) {
        std::cerr << "host gate refused graph status=" << ggml_status_to_string(status)
                  << " routes=" << capture.routes.size() << '\n';
        return false;
    }
    if (capture.routes.size() != 2 || capture.routes[0].layer != 0 || capture.routes[1].layer != 1 ||
        capture.routes[0].experts != std::vector<int32_t>({1, 3}) ||
        capture.routes[0].owners != std::vector<int32_t>({-1, 2}) ||
        capture.routes[1].experts != std::vector<int32_t>({2, 0}) ||
        capture.routes[1].owners != std::vector<int32_t>({1, 0}) ||
        capture.routes[0].weights != std::vector<float>({0.25f, 0.75f}) ||
        capture.routes[1].weights != std::vector<float>({0.4f, 0.6f})) {
        std::cerr << "route capture mismatch; calls=" << capture.routes.size() << '\n';
        return false;
    }
    const auto *actual = static_cast<const float *>(combined->data);
    if (!near(actual[0], 17.4f) || !near(actual[1], 19.4f)) {
        std::cerr << "oracle mismatch: [" << actual[0] << ", " << actual[1] << "]\n";
        return false;
    }
    return true;
}

bool test_missing_layer_refuses_publication() {
    ggml_init_params params{};
    params.mem_size = 1U << 20;
    Fixture f;
    f.ctx = ggml_init(params);
    if (f.ctx == nullptr) return false;
    RouteBatch identity;
    identity.identity = BatchIdentity{"qwen-run", "qwen-model", "qwen-plan", 0, 20, 8, 1};
    identity.owner_by_expert = {0, -1, 1, 2};
    Capture capture;
    Qwen2MoERouteInvocation invocation(identity, 2, collect, &capture);
    if (!build_layer(f, invocation, 0, {1.0f, 2.0f}, {1, 3}, {0.5f, 0.5f})) return false;
    f.graph = ggml_new_graph(f.ctx);
    if (f.graph == nullptr) return false;
    ggml_build_forward_expand(f.graph, f.routed);
    f.plan = ggml_graph_plan(f.graph, 1, nullptr);
    f.work.resize(f.plan.work_size);
    f.plan.work_data = f.work.empty() ? nullptr : f.work.data();
    const ggml_status status = ggml_graph_compute(f.graph, &f.plan);
    return status == GGML_STATUS_SUCCESS && !invocation.accepts(status) && capture.routes.size() == 1;
}
} // namespace

int main() {
    if (!test_two_layer_invocation()) {
        std::cerr << "Qwen2MoE two-layer route invocation failed\n";
        return 1;
    }
    if (!test_missing_layer_refuses_publication()) {
        std::cerr << "Qwen2MoE incomplete host invocation was not refused\n";
        return 1;
    }
    return 0;
}
