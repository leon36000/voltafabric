#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

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
    std::vector<int32_t> expert_ids; // [slot, token], no renormalization here
    std::vector<float> weights;      // [slot, token], applied once by the collector
};

using Collector = RouteStatus (*)(const RouteBatch &, const float *, size_t, float *, size_t, void *);

struct GraphTransaction {
    RouteBatch batch;
    Collector collector = nullptr;
    void *collector_data = nullptr;
    std::vector<float> staging;
    std::atomic<RouteStatus> status{RouteStatus::pending};
    std::atomic<int> downstream_calls{0};
};

bool valid_batch(const RouteBatch &batch, size_t input_count, size_t output_count) {
    if (batch.identity.run.empty() || batch.identity.model.empty() || batch.identity.plan.empty() ||
        batch.identity.epoch == 0 || batch.identity.batch == 0 || batch.identity.attempt == 0 ||
        batch.hidden == 0 || batch.hidden > 8192 || batch.slots == 0 || batch.slots > 64 ||
        batch.tokens == 0 || batch.tokens > 4096 || batch.experts == 0 || batch.experts > 4096) {
        return false;
    }
    if (batch.hidden > std::numeric_limits<size_t>::max() / batch.slots ||
        batch.hidden * batch.slots > std::numeric_limits<size_t>::max() / batch.tokens ||
        batch.hidden * batch.slots * batch.tokens != input_count ||
        batch.hidden > std::numeric_limits<size_t>::max() / batch.tokens ||
        batch.hidden * batch.tokens != output_count ||
        batch.slots > std::numeric_limits<size_t>::max() / batch.tokens) {
        return false;
    }
    const size_t route_count = batch.slots * batch.tokens;
    if (batch.expert_ids.size() != route_count || batch.weights.size() != route_count) {
        return false;
    }
    for (size_t token = 0; token < batch.tokens; ++token) {
        for (size_t slot = 0; slot < batch.slots; ++slot) {
            const size_t route = slot + batch.slots * token;
            if (batch.expert_ids[route] < 0 || static_cast<size_t>(batch.expert_ids[route]) >= batch.experts ||
                !std::isfinite(batch.weights[route]) || batch.weights[route] < 0.0f) {
                return false;
            }
            for (size_t earlier = 0; earlier < slot; ++earlier) {
                if (batch.expert_ids[earlier + batch.slots * token] == batch.expert_ids[route]) {
                    return false;
                }
            }
        }
    }
    return true;
}

void route_graph_op(ggml_tensor *dst, int ith, int nth, void *userdata) {
    auto &transaction = *static_cast<GraphTransaction *>(userdata);
    if (ith != 0 || nth != 1) {
        transaction.status.store(RouteStatus::unsupported_tasking, std::memory_order_release);
        return;
    }
    const ggml_tensor *slot_outputs = dst->src[0];
    if (transaction.batch.hidden == 0 || transaction.batch.tokens == 0 ||
        transaction.batch.hidden > std::numeric_limits<size_t>::max() / transaction.batch.tokens) {
        transaction.status.store(RouteStatus::invalid_batch, std::memory_order_release);
        return;
    }
    const size_t input_count = slot_outputs == nullptr ? 0 : static_cast<size_t>(ggml_nelements(slot_outputs));
    const size_t output_count = transaction.batch.hidden * transaction.batch.tokens;
    if (slot_outputs == nullptr || slot_outputs->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        !transaction.collector || !valid_batch(transaction.batch, input_count, output_count) ||
        slot_outputs->data == nullptr || dst->data == nullptr || transaction.staging.size() != output_count) {
        transaction.status.store(RouteStatus::invalid_batch, std::memory_order_release);
        return;
    }

    // The donor graph sees no partial collector output. The external callback may
    // write this preallocated staging buffer and then fail; publication happens only
    // after full status, length, and finiteness validation.
    std::fill(transaction.staging.begin(), transaction.staging.end(),
              std::numeric_limits<float>::quiet_NaN());
    const auto *input = static_cast<const float *>(slot_outputs->data);
    for (size_t i = 0; i < input_count; ++i) {
        if (!std::isfinite(input[i])) {
            transaction.status.store(RouteStatus::invalid_batch, std::memory_order_release);
            return;
        }
    }
    RouteStatus result = transaction.collector(transaction.batch, input, input_count,
                                               transaction.staging.data(), transaction.staging.size(),
                                               transaction.collector_data);
    if (result != RouteStatus::ready) {
        transaction.status.store(result == RouteStatus::invalid_batch ? result : RouteStatus::provider_failure,
                                 std::memory_order_release);
        return;
    }
    for (float value : transaction.staging) {
        if (!std::isfinite(value)) {
            transaction.status.store(RouteStatus::invalid_result, std::memory_order_release);
            return;
        }
    }
    std::memcpy(dst->data, transaction.staging.data(), transaction.staging.size() * sizeof(float));
    transaction.status.store(RouteStatus::ready, std::memory_order_release);
}

bool abort_after_failed_route(void *userdata) {
    const auto &transaction = *static_cast<const GraphTransaction *>(userdata);
    const RouteStatus status = transaction.status.load(std::memory_order_acquire);
    return status != RouteStatus::pending && status != RouteStatus::ready;
}

void downstream_op(ggml_tensor *dst, const ggml_tensor *src, int ith, int nth, void *userdata) {
    auto &transaction = *static_cast<GraphTransaction *>(userdata);
    if (ith != 0 || nth != 1 || src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        src->data == nullptr || dst->data == nullptr || ggml_nelements(src) != ggml_nelements(dst)) {
        transaction.status.store(RouteStatus::invalid_result, std::memory_order_release);
        return;
    }
    const size_t bytes = static_cast<size_t>(ggml_nelements(src)) * sizeof(float);
    std::memcpy(dst->data, src->data, bytes);
    transaction.downstream_calls.fetch_add(1, std::memory_order_relaxed);
}

struct FixtureContext {
    ggml_context *ctx = nullptr;
    ggml_cgraph *graph = nullptr;
    ggml_cplan plan{};
    std::vector<uint8_t> work;
    ggml_tensor *source = nullptr;
    ggml_tensor *route_output = nullptr;
    ggml_tensor *downstream = nullptr;
};

bool make_graph(GraphTransaction &transaction, FixtureContext &fixture) {
    ggml_init_params params{};
    params.mem_size = 1U << 20;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    fixture.ctx = ggml_init(params);
    if (fixture.ctx == nullptr) return false;

    fixture.source = ggml_new_tensor_3d(fixture.ctx, GGML_TYPE_F32,
                                        static_cast<int64_t>(transaction.batch.hidden),
                                        static_cast<int64_t>(transaction.batch.slots),
                                        static_cast<int64_t>(transaction.batch.tokens));
    if (fixture.source == nullptr || fixture.source->data == nullptr) return false;
    float *source = static_cast<float *>(fixture.source->data);
    const float values[] = {1, 2, 3, 4, 10, 20, 30, 40};
    if (ggml_nelements(fixture.source) != 8) return false;
    std::memcpy(source, values, sizeof(values));

    ggml_tensor *args[] = {fixture.source};
    fixture.route_output = ggml_custom_4d(fixture.ctx, GGML_TYPE_F32,
        static_cast<int64_t>(transaction.batch.hidden), static_cast<int64_t>(transaction.batch.tokens), 1, 1,
        args, 1, route_graph_op, 1, &transaction);
    fixture.downstream = ggml_map_custom1(fixture.ctx, fixture.route_output, downstream_op, 1, &transaction);
    fixture.graph = ggml_new_graph(fixture.ctx);
    if (fixture.route_output == nullptr || fixture.downstream == nullptr || fixture.graph == nullptr) return false;
    ggml_build_forward_expand(fixture.graph, fixture.downstream);

    fixture.plan = ggml_graph_plan(fixture.graph, 1, nullptr);
    fixture.work.resize(fixture.plan.work_size);
    fixture.plan.work_data = fixture.work.empty() ? nullptr : fixture.work.data();
    fixture.plan.abort_callback = abort_after_failed_route;
    fixture.plan.abort_callback_data = &transaction;
    return true;
}

RouteStatus fixture_collector(const RouteBatch &batch, const float *input, size_t input_count,
                              float *output, size_t output_count, void *userdata) {
    auto &provider = *static_cast<int *>(userdata);
    if (input_count != 8 || output_count != 4 || batch.expert_ids != std::vector<int32_t>{1, 0, 2, 3} ||
        batch.weights != std::vector<float>{0.25f, 0.75f, 0.60f, 0.40f}) {
        return RouteStatus::invalid_batch;
    }
    ++provider;
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
    auto &provider = *static_cast<int *>(userdata);
    ++provider;
    if (output_count != 4) return RouteStatus::invalid_batch;
    output[0] = 12345.0f; // A deliberately partial staging write must never be published.
    return RouteStatus::provider_failure;
}

RouteBatch make_batch() {
    RouteBatch batch;
    batch.identity = {"run-r669", "qwen2moe-fixture", "plan-r669", 2, 41, 7, 1};
    batch.hidden = 2;
    batch.slots = 2;
    batch.tokens = 2;
    batch.experts = 4;
    batch.expert_ids = {1, 0, 2, 3};
    batch.weights = {0.25f, 0.75f, 0.60f, 0.40f};
    return batch;
}

int run_case(Collector collector, bool should_succeed) {
    GraphTransaction transaction;
    transaction.batch = make_batch();
    transaction.staging.resize(transaction.batch.hidden * transaction.batch.tokens);
    int provider_calls = 0;
    transaction.collector = collector;
    transaction.collector_data = &provider_calls;
    FixtureContext fixture;
    if (!make_graph(transaction, fixture)) {
        if (fixture.ctx != nullptr) ggml_free(fixture.ctx);
        std::cerr << "fixture graph construction failed\n";
        return 1;
    }
    const ggml_status status = ggml_graph_compute(fixture.graph, &fixture.plan);
    const bool success = status == GGML_STATUS_SUCCESS &&
                         transaction.status.load(std::memory_order_acquire) == RouteStatus::ready &&
                         transaction.downstream_calls.load(std::memory_order_relaxed) == 1;
    const bool failed_atomically = status == GGML_STATUS_ABORTED &&
                         transaction.status.load(std::memory_order_acquire) == RouteStatus::provider_failure &&
                         transaction.downstream_calls.load(std::memory_order_relaxed) == 0;
    bool values_match = false;
    if (success) {
        const float *out = static_cast<const float *>(fixture.downstream->data);
        values_match = std::fabs(out[0] - 2.5f) < 1e-6f && std::fabs(out[1] - 3.5f) < 1e-6f &&
                       std::fabs(out[2] - 18.0f) < 1e-6f && std::fabs(out[3] - 28.0f) < 1e-6f;
    }
    ggml_free(fixture.ctx);
    if (provider_calls != 1 || (should_succeed ? (!success || !values_match) : !failed_atomically)) {
        std::cerr << "route graph transaction contract failed: status=" << ggml_status_to_string(status)
                  << " provider_calls=" << provider_calls << " downstream_calls="
                  << transaction.downstream_calls.load(std::memory_order_relaxed) << "\n";
        return 1;
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
