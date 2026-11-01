#include "route_graph_transaction.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace vf::ggml {
namespace {

constexpr size_t kMaxInputFloats = 64U * 1024U * 1024U;
constexpr size_t kMaxOutputFloats = 16U * 1024U * 1024U;

bool checked_product(size_t a, size_t b, size_t &result) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b) return false;
    result = a * b;
    return true;
}

bool valid_identity(const BatchIdentity &identity) {
    return !identity.run.empty() && !identity.model.empty() && !identity.plan.empty() &&
           identity.epoch != 0 && identity.batch != 0 && identity.attempt != 0;
}

bool valid_batch(const RouteBatch &batch, size_t input_count, size_t output_count) {
    if (!valid_identity(batch.identity) || batch.hidden == 0 || batch.hidden > 8192 ||
        batch.slots == 0 || batch.slots > 64 || batch.tokens == 0 || batch.tokens > 4096 ||
        batch.experts == 0 || batch.experts > 4096 || input_count > kMaxInputFloats ||
        output_count > kMaxOutputFloats) {
        return false;
    }

    size_t expected_input = 0;
    size_t hidden_slots = 0;
    size_t expected_output = 0;
    size_t route_count = 0;
    if (!checked_product(batch.hidden, batch.slots, hidden_slots) ||
        !checked_product(hidden_slots, batch.tokens, expected_input) ||
        !checked_product(batch.hidden, batch.tokens, expected_output) ||
        !checked_product(batch.slots, batch.tokens, route_count) ||
        expected_input != input_count || expected_output != output_count ||
        batch.expert_ids.size() != route_count || batch.weights.size() != route_count) {
        return false;
    }

    for (size_t token = 0; token < batch.tokens; ++token) {
        for (size_t slot = 0; slot < batch.slots; ++slot) {
            const size_t route = slot + batch.slots * token;
            const int32_t expert = batch.expert_ids[route];
            if (expert < 0 || static_cast<size_t>(expert) >= batch.experts ||
                !std::isfinite(batch.weights[route]) || batch.weights[route] < 0.0f) {
                return false;
            }
            for (size_t earlier = 0; earlier < slot; ++earlier) {
                if (batch.expert_ids[earlier + batch.slots * token] == expert) return false;
            }
        }
    }
    return true;
}

} // namespace

RouteGraphTransaction::RouteGraphTransaction(RouteBatch batch, Collector collector, void *collector_data)
    : batch_(std::move(batch)), collector_(collector), collector_data_(collector_data) {
    if (!checked_product(batch_.hidden, batch_.tokens, output_count_) ||
        output_count_ == 0 || output_count_ > kMaxOutputFloats || collector_ == nullptr) {
        output_count_ = 0;
        status_.store(RouteStatus::invalid_batch, std::memory_order_relaxed);
        return;
    }
    staging_.resize(output_count_);
}

ggml_tensor *RouteGraphTransaction::append(ggml_context *ctx, ggml_tensor *slot_outputs) {
    if (appended_ || status_.load(std::memory_order_acquire) != RouteStatus::pending || ctx == nullptr ||
        slot_outputs == nullptr || slot_outputs->type != GGML_TYPE_F32 || slot_outputs->data == nullptr ||
        static_cast<size_t>(ggml_nelements(slot_outputs)) > kMaxInputFloats ||
        !valid_batch(batch_, static_cast<size_t>(ggml_nelements(slot_outputs)), output_count_)) {
        status_.store(RouteStatus::invalid_batch, std::memory_order_release);
        return nullptr;
    }
    appended_ = true;

    ggml_tensor *args[] = {slot_outputs};
    ggml_tensor *result = ggml_custom_4d(ctx, GGML_TYPE_F32,
        static_cast<int64_t>(batch_.hidden), static_cast<int64_t>(batch_.tokens), 1, 1,
        args, 1, compute, 1, this);
    if (result == nullptr) status_.store(RouteStatus::invalid_batch, std::memory_order_release);
    return result;
}

void RouteGraphTransaction::attach_abort(ggml_cplan &plan) {
    plan.abort_callback = abort_after_failed_route;
    plan.abort_callback_data = this;
}

bool RouteGraphTransaction::accepts(ggml_status graph_status) const {
    return graph_status == GGML_STATUS_SUCCESS && status_.load(std::memory_order_acquire) == RouteStatus::ready;
}

RouteStatus RouteGraphTransaction::status() const {
    return status_.load(std::memory_order_acquire);
}

void RouteGraphTransaction::compute(ggml_tensor *dst, int ith, int nth, void *userdata) {
    auto &transaction = *static_cast<RouteGraphTransaction *>(userdata);
    if (transaction.status_.load(std::memory_order_acquire) != RouteStatus::pending ||
        transaction.compute_started_.exchange(true, std::memory_order_acq_rel)) {
        transaction.status_.store(RouteStatus::invalid_batch, std::memory_order_release);
        return;
    }
    if (ith != 0 || nth != 1) {
        transaction.status_.store(RouteStatus::unsupported_tasking, std::memory_order_release);
        return;
    }
    const ggml_tensor *slot_outputs = dst->src[0];
    const size_t input_count = slot_outputs == nullptr ? 0 : static_cast<size_t>(ggml_nelements(slot_outputs));
    if (slot_outputs == nullptr || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        slot_outputs->type != GGML_TYPE_F32 || slot_outputs->data == nullptr ||
        !valid_batch(transaction.batch_, input_count, transaction.output_count_)) {
        transaction.status_.store(RouteStatus::invalid_batch, std::memory_order_release);
        return;
    }

    std::fill(transaction.staging_.begin(), transaction.staging_.end(),
              std::numeric_limits<float>::quiet_NaN());
    const auto *input = static_cast<const float *>(slot_outputs->data);
    for (size_t i = 0; i < input_count; ++i) {
        if (!std::isfinite(input[i])) {
            transaction.status_.store(RouteStatus::invalid_batch, std::memory_order_release);
            return;
        }
    }

    RouteStatus result = RouteStatus::provider_failure;
    try {
        result = transaction.collector_(transaction.batch_, input, input_count,
            transaction.staging_.data(), transaction.staging_.size(), transaction.collector_data_);
    } catch (...) {
        transaction.status_.store(RouteStatus::provider_failure, std::memory_order_release);
        return;
    }
    if (result != RouteStatus::ready) {
        transaction.status_.store(result == RouteStatus::pending ? RouteStatus::provider_failure : result,
                                  std::memory_order_release);
        return;
    }
    for (float value : transaction.staging_) {
        if (!std::isfinite(value)) {
            transaction.status_.store(RouteStatus::invalid_result, std::memory_order_release);
            return;
        }
    }
    std::memcpy(dst->data, transaction.staging_.data(), transaction.output_count_ * sizeof(float));
    transaction.status_.store(RouteStatus::ready, std::memory_order_release);
}

bool RouteGraphTransaction::abort_after_failed_route(void *userdata) {
    const auto &transaction = *static_cast<const RouteGraphTransaction *>(userdata);
    const RouteStatus status = transaction.status_.load(std::memory_order_acquire);
    return status != RouteStatus::pending && status != RouteStatus::ready;
}

} // namespace vf::ggml
