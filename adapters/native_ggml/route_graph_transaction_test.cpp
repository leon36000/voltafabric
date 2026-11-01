#include "route_graph_transaction.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using vf::ggml::BatchIdentity;
using vf::ggml::RouteBatch;
using vf::ggml::RouteGraphTransaction;
using vf::ggml::RouteStatus;

namespace {

void downstream_copy(ggml_tensor *dst, const ggml_tensor *src, int ith, int nth, void *userdata) {
    auto &calls = *static_cast<int *>(userdata);
    if (ith != 0 || nth != 1 || src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        src->data == nullptr || dst->data == nullptr || ggml_nelements(src) != ggml_nelements(dst)) {
        return;
    }
    std::memcpy(dst->data, src->data, static_cast<size_t>(ggml_nelements(src)) * sizeof(float));
    ++calls;
}

RouteStatus fixture_collector(const RouteBatch &batch, const float *input, size_t input_count,
                              float *output, size_t output_count, void *userdata) {
    auto &calls = *static_cast<int *>(userdata);
    if (batch.identity.run != "run-r669" || batch.identity.model != "qwen2moe-fixture" ||
        batch.identity.plan != "plan-r669" || batch.identity.layer != 2 ||
        batch.identity.epoch != 41 || batch.identity.batch != 7 || batch.identity.attempt != 1 ||
        batch.expert_ids != std::vector<int32_t>{1, 0, 2, 3} ||
        batch.weights != std::vector<float>{0.25f, 0.75f, 0.60f, 0.40f} ||
        input_count != 8 || output_count != 4) {
        return RouteStatus::invalid_batch;
    }
    ++calls;
    for (size_t token = 0; token < batch.tokens; ++token) {
        for (size_t hidden = 0; hidden < batch.hidden; ++hidden) {
            double sum = 0.0;
            for (size_t slot = 0; slot < batch.slots; ++slot) {
                const size_t route = slot + batch.slots * token;
                const size_t value = hidden + batch.hidden * route;
                sum += static_cast<double>(input[value]) * batch.weights[route];
            }
            output[hidden + batch.hidden * token] = static_cast<float>(sum);
        }
    }
    return RouteStatus::ready;
}

RouteStatus partial_failure_collector(const RouteBatch &, const float *, size_t,
                                      float *output, size_t output_count, void *userdata) {
    auto &calls = *static_cast<int *>(userdata);
    ++calls;
    if (output_count != 4) return RouteStatus::invalid_batch;
    output[0] = 12345.0f;
    return RouteStatus::provider_failure;
}

RouteBatch make_batch() {
    RouteBatch batch;
    batch.identity = BatchIdentity{"run-r669", "qwen2moe-fixture", "plan-r669", 2, 41, 7, 1};
    batch.hidden = 2;
    batch.slots = 2;
    batch.tokens = 2;
    batch.experts = 4;
    batch.expert_ids = {1, 0, 2, 3};
    batch.weights = {0.25f, 0.75f, 0.60f, 0.40f};
    return batch;
}

struct FixtureGraph {
    ggml_context *ctx = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_cplan plan{};
    std::vector<uint8_t> work;
    ggml_tensor *source = nullptr;
    ggml_tensor *routed = nullptr;
    ggml_tensor *downstream = nullptr;

    ~FixtureGraph() {
        if (ctx != nullptr) ggml_free(ctx);
    }
};

bool make_graph(RouteGraphTransaction &transaction, const RouteBatch &batch,
                int &downstream_calls, FixtureGraph &fixture) {
    ggml_init_params params{};
    params.mem_size = 1U << 20;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    fixture.ctx = ggml_init(params);
    if (fixture.ctx == nullptr) return false;

    fixture.source = ggml_new_tensor_3d(fixture.ctx, GGML_TYPE_F32,
        static_cast<int64_t>(batch.hidden), static_cast<int64_t>(batch.slots),
        static_cast<int64_t>(batch.tokens));
    if (fixture.source == nullptr || fixture.source->data == nullptr || ggml_nelements(fixture.source) != 8) {
        return false;
    }
    const float values[] = {1, 2, 3, 4, 10, 20, 30, 40};
    std::memcpy(fixture.source->data, values, sizeof(values));

    fixture.routed = transaction.append(fixture.ctx, fixture.source);
    if (fixture.routed == nullptr) return false;
    fixture.downstream = ggml_map_custom1(fixture.ctx, fixture.routed, downstream_copy, 1, &downstream_calls);
    fixture.graph = ggml_new_graph(fixture.ctx);
    if (fixture.downstream == nullptr || fixture.graph == nullptr) return false;
    ggml_build_forward_expand(fixture.graph, fixture.downstream);
    fixture.plan = ggml_graph_plan(fixture.graph, 1, nullptr);
    fixture.work.resize(fixture.plan.work_size);
    fixture.plan.work_data = fixture.work.empty() ? nullptr : fixture.work.data();
    transaction.attach_abort(fixture.plan);
    return true;
}

int run_case(vf::ggml::Collector collector, bool should_succeed) {
    RouteBatch batch = make_batch();
    int collector_calls = 0;
    int downstream_calls = 0;
    RouteGraphTransaction transaction(batch, collector, &collector_calls);
    FixtureGraph fixture;
    if (!make_graph(transaction, batch, downstream_calls, fixture)) {
        std::cerr << "fixture graph construction failed\n";
        return 1;
    }

    const ggml_status graph_status = ggml_graph_compute(fixture.graph, &fixture.plan);
    const bool accepted = transaction.accepts(graph_status);
    bool values_match = false;
    if (accepted) {
        const float *out = static_cast<const float *>(fixture.downstream->data);
        values_match = std::fabs(out[0] - 2.5f) < 1e-6f && std::fabs(out[1] - 3.5f) < 1e-6f &&
                       std::fabs(out[2] - 18.0f) < 1e-6f && std::fabs(out[3] - 28.0f) < 1e-6f;
    }
    const bool failed_atomically = !accepted && graph_status == GGML_STATUS_ABORTED &&
        transaction.status() == RouteStatus::provider_failure && downstream_calls == 0;
    if (collector_calls != 1 || downstream_calls != (should_succeed ? 1 : 0) ||
        (should_succeed ? (!accepted || !values_match) : !failed_atomically)) {
        std::cerr << "route graph transaction failed: status=" << ggml_status_to_string(graph_status)
                  << " transaction=" << static_cast<int>(transaction.status())
                  << " collector_calls=" << collector_calls
                  << " downstream_calls=" << downstream_calls << "\n";
        return 1;
    }
    if (should_succeed) {
        const ggml_status replay_status = ggml_graph_compute(fixture.graph, &fixture.plan);
        if (replay_status != GGML_STATUS_ABORTED || transaction.status() != RouteStatus::invalid_batch ||
            collector_calls != 1 || downstream_calls != 1) {
            std::cerr << "route graph transaction replay was not rejected\n";
            return 1;
        }
    }
    return 0;
}

} // namespace

int main() {
    if (run_case(fixture_collector, true) != 0) return 1;
    if (run_case(partial_failure_collector, false) != 0) return 1;
    std::cout << "GGML route graph transaction: success and injected failure passed\n";
    return 0;
}
