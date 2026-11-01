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

bool valid_batch(const RouteBatch &batch, size_t input_count, size_t output_count,
                 bool dynamic_route_values = false) {
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
        (!dynamic_route_values &&
         (batch.expert_ids.size() != route_count || batch.weights.size() != route_count)) ||
        (dynamic_route_values &&
         !((batch.owner_kinds.size() == route_count && batch.owner_by_expert.empty()) ||
           (batch.owner_kinds.empty() && batch.owner_by_expert.size() == batch.experts))) ||
        (!batch.owner_kinds.empty() && batch.owner_kinds.size() != route_count) ||
        (!batch.owner_by_expert.empty() && batch.owner_by_expert.size() != batch.experts)) {
        return false;
    }

    for (const int32_t owner_kind : batch.owner_by_expert) {
        if (owner_kind < -1 || owner_kind > 2) return false;
    }

    for (size_t token = 0; token < batch.tokens; ++token) {
        for (size_t slot = 0; slot < batch.slots; ++slot) {
            const size_t route = slot + batch.slots * token;
            if (!batch.owner_kinds.empty() &&
                (batch.owner_kinds[route] < -1 || batch.owner_kinds[route] > 2)) {
                return false;
            }
            if (dynamic_route_values) continue;
            const int32_t expert = batch.expert_ids[route];
            if (expert < 0 || static_cast<size_t>(expert) >= batch.experts ||
                !std::isfinite(batch.weights[route]) || batch.weights[route] < 0.0f) return false;
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
    return append_impl(ctx, slot_outputs, nullptr, nullptr, nullptr, false);
}

ggml_tensor *RouteGraphTransaction::append(ggml_context *ctx, ggml_tensor *slot_outputs,
                                           ggml_tensor *activations) {
    return append_impl(ctx, slot_outputs, activations, nullptr, nullptr, false);
}

ggml_tensor *RouteGraphTransaction::append_dynamic(ggml_context *ctx, ggml_tensor *slot_outputs,
                                                   ggml_tensor *activations,
                                                   ggml_tensor *selected_experts,
                                                   ggml_tensor *final_weights) {
    return append_impl(ctx, slot_outputs, activations, selected_experts, final_weights, true);
}

ggml_tensor *RouteGraphTransaction::append_impl(ggml_context *ctx, ggml_tensor *slot_outputs,
                                                ggml_tensor *activations,
                                                ggml_tensor *selected_experts,
                                                ggml_tensor *final_weights,
                                                bool dynamic_route_inputs) {
    const bool valid_slot_source = slot_outputs != nullptr && slot_outputs->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(slot_outputs) &&
        slot_outputs->ne[0] == static_cast<int64_t>(batch_.hidden) &&
        slot_outputs->ne[1] == static_cast<int64_t>(batch_.slots) &&
        slot_outputs->ne[2] == static_cast<int64_t>(batch_.tokens) && slot_outputs->ne[3] == 1 &&
        static_cast<size_t>(ggml_nelements(slot_outputs)) <= kMaxInputFloats;
    const bool valid_activation_source = activations == nullptr ||
        (activations->type == GGML_TYPE_F32 &&
         ggml_is_contiguous(activations) &&
         activations->ne[0] == static_cast<int64_t>(batch_.hidden) &&
         activations->ne[1] == static_cast<int64_t>(batch_.tokens) &&
         activations->ne[2] == 1 && activations->ne[3] == 1 &&
         (!batch_.owner_kinds.empty() || !batch_.owner_by_expert.empty()));
    const bool valid_dynamic_route_sources = !dynamic_route_inputs ||
        (activations != nullptr && selected_experts != nullptr &&
         selected_experts->type == GGML_TYPE_I32 &&
         selected_experts->ne[0] == static_cast<int64_t>(batch_.slots) &&
         selected_experts->ne[1] == static_cast<int64_t>(batch_.tokens) &&
         selected_experts->ne[2] == 1 && selected_experts->ne[3] == 1 &&
         final_weights != nullptr && final_weights->type == GGML_TYPE_F32 &&
         final_weights->ne[0] == 1 &&
         final_weights->ne[1] == static_cast<int64_t>(batch_.slots) &&
         final_weights->ne[2] == static_cast<int64_t>(batch_.tokens) &&
         final_weights->ne[3] == 1);
    const bool valid_batch_inputs = valid_batch(batch_, slot_outputs == nullptr ? 0 :
        static_cast<size_t>(ggml_nelements(slot_outputs)), output_count_, dynamic_route_inputs);
    if (appended_ || status_.load(std::memory_order_acquire) != RouteStatus::pending ||
        ctx == nullptr || !valid_slot_source || !valid_activation_source ||
        !valid_dynamic_route_sources ||
        !valid_batch_inputs) {
        status_.store(RouteStatus::invalid_batch, std::memory_order_release);
        return nullptr;
    }
    appended_ = true;
    requires_activations_ = activations != nullptr;
    dynamic_route_inputs_ = dynamic_route_inputs;

    ggml_tensor *args[] = {slot_outputs, activations, selected_experts, final_weights};
    ggml_tensor *result = ggml_custom_4d(ctx, GGML_TYPE_F32,
        static_cast<int64_t>(batch_.hidden), static_cast<int64_t>(batch_.tokens), 1, 1,
        args, dynamic_route_inputs ? 4 : (activations == nullptr ? 1 : 2), compute, 1, this);
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
    const ggml_tensor *activations = dst->src[1];
    const ggml_tensor *selected_experts = transaction.dynamic_route_inputs_ ? dst->src[2] : nullptr;
    const ggml_tensor *final_weights = transaction.dynamic_route_inputs_ ? dst->src[3] : nullptr;
    const size_t input_count = slot_outputs == nullptr ? 0 : static_cast<size_t>(ggml_nelements(slot_outputs));
    const size_t activation_count = activations == nullptr ? 0 : static_cast<size_t>(ggml_nelements(activations));
    const bool slot_shape_ok = slot_outputs != nullptr && slot_outputs->type == GGML_TYPE_F32 &&
        slot_outputs->data != nullptr && ggml_is_contiguous(slot_outputs) &&
        slot_outputs->ne[0] == static_cast<int64_t>(transaction.batch_.hidden) &&
        slot_outputs->ne[1] == static_cast<int64_t>(transaction.batch_.slots) &&
        slot_outputs->ne[2] == static_cast<int64_t>(transaction.batch_.tokens) && slot_outputs->ne[3] == 1;
    const bool activation_shape_ok = transaction.requires_activations_
        ? (activations != nullptr && activations->type == GGML_TYPE_F32 && activations->data != nullptr &&
           ggml_is_contiguous(activations) &&
           activations->ne[0] == static_cast<int64_t>(transaction.batch_.hidden) &&
           activations->ne[1] == static_cast<int64_t>(transaction.batch_.tokens) &&
           activations->ne[2] == 1 && activations->ne[3] == 1)
        : activations == nullptr;
    const bool dynamic_route_shape_ok = !transaction.dynamic_route_inputs_ ||
        (selected_experts != nullptr && selected_experts->type == GGML_TYPE_I32 &&
         selected_experts->data != nullptr &&
         selected_experts->ne[0] == static_cast<int64_t>(transaction.batch_.slots) &&
         selected_experts->ne[1] == static_cast<int64_t>(transaction.batch_.tokens) &&
         selected_experts->ne[2] == 1 && selected_experts->ne[3] == 1 &&
         final_weights != nullptr && final_weights->type == GGML_TYPE_F32 &&
         final_weights->data != nullptr &&
         final_weights->ne[0] == 1 &&
         final_weights->ne[1] == static_cast<int64_t>(transaction.batch_.slots) &&
         final_weights->ne[2] == static_cast<int64_t>(transaction.batch_.tokens) &&
         final_weights->ne[3] == 1);
    if (dst->type != GGML_TYPE_F32 || dst->data == nullptr || !slot_shape_ok ||
        !activation_shape_ok || !dynamic_route_shape_ok ||
        !valid_batch(transaction.batch_, input_count, transaction.output_count_,
                     transaction.dynamic_route_inputs_)) {
        transaction.status_.store(RouteStatus::invalid_batch, std::memory_order_release);
        return;
    }

    if (transaction.dynamic_route_inputs_) {
        const size_t route_count = transaction.batch_.slots * transaction.batch_.tokens;
        if (selected_experts->nb[0] < sizeof(int32_t) ||
            selected_experts->nb[1] < selected_experts->nb[0] * transaction.batch_.slots ||
            final_weights->nb[0] < sizeof(float) ||
            final_weights->nb[1] < final_weights->nb[0] ||
            final_weights->nb[2] < final_weights->nb[1] * transaction.batch_.slots) {
            transaction.status_.store(RouteStatus::invalid_batch, std::memory_order_release);
            return;
        }
        transaction.batch_.expert_ids.resize(route_count);
        transaction.batch_.weights.resize(route_count);
        const auto *id_bytes = static_cast<const uint8_t *>(selected_experts->data);
        const auto *weight_bytes = static_cast<const uint8_t *>(final_weights->data);
        for (size_t token = 0; token < transaction.batch_.tokens; ++token) {
            for (size_t slot = 0; slot < transaction.batch_.slots; ++slot) {
                const size_t route = slot + transaction.batch_.slots * token;
                std::memcpy(&transaction.batch_.expert_ids[route],
                    id_bytes + slot * selected_experts->nb[0] + token * selected_experts->nb[1],
                    sizeof(int32_t));
                std::memcpy(&transaction.batch_.weights[route],
                    weight_bytes + slot * final_weights->nb[1] + token * final_weights->nb[2],
                    sizeof(float));
            }
        }
        if (!transaction.batch_.owner_by_expert.empty()) {
            transaction.batch_.owner_kinds.resize(route_count);
            for (size_t route = 0; route < route_count; ++route) {
                const int32_t expert = transaction.batch_.expert_ids[route];
                if (expert < 0 || static_cast<size_t>(expert) >= transaction.batch_.owner_by_expert.size()) {
                    transaction.status_.store(RouteStatus::invalid_batch, std::memory_order_release);
                    return;
                }
                transaction.batch_.owner_kinds[route] = transaction.batch_.owner_by_expert[expert];
            }
        }
        if (!valid_batch(transaction.batch_, input_count, transaction.output_count_)) {
            transaction.status_.store(RouteStatus::invalid_batch, std::memory_order_release);
            return;
        }
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

    if (activations != nullptr) {
        const auto *activation_data = static_cast<const float *>(activations->data);
        for (size_t i = 0; i < activation_count; ++i) {
            if (!std::isfinite(activation_data[i])) {
                transaction.status_.store(RouteStatus::invalid_batch, std::memory_order_release);
                return;
            }
        }
    }

    RouteStatus result = RouteStatus::provider_failure;
    try {
        result = transaction.collector_(transaction.batch_,
            activations == nullptr ? nullptr : static_cast<const float *>(activations->data),
            activation_count, input, input_count, transaction.staging_.data(),
            transaction.staging_.size(), transaction.collector_data_);
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
