#include "session_vfr3_collector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace vf::ggml {
namespace {
constexpr size_t kVfr3Hidden = 2560;
constexpr size_t kMaxVfr3Rows = 128;

bool same_session(const RouteBatch &batch, const vf::strata_bridge::SessionIdentity &identity,
                  uint32_t layer) {
    return batch.identity.run == identity.run_id && batch.identity.model == identity.model_revision &&
        batch.identity.plan == identity.plan_id && batch.identity.epoch == identity.epoch &&
        batch.identity.layer == layer;
}
}

RouteStatus collect_vfr3(const RouteBatch &batch, const float *activations,
                         size_t activation_count, const float *slot_outputs,
                         size_t slot_output_count, float *output,
                         size_t output_count, void *userdata) {
    if (userdata == nullptr) return RouteStatus::invalid_batch;
    auto &context = *static_cast<Vfr3CollectorContext *>(userdata);
    ++context.calls;
    context.error.clear();
    if (context.client == nullptr || activations == nullptr || slot_outputs == nullptr || output == nullptr ||
        batch.hidden != kVfr3Hidden || batch.tokens == 0 || batch.tokens > 8 ||
        batch.slots == 0 || batch.slots > 64 || batch.tokens * batch.slots > kMaxVfr3Rows ||
        batch.owner_kinds.size() != batch.tokens * batch.slots ||
        batch.identity.epoch == 0 || batch.identity.batch == 0 || batch.identity.attempt == 0 ||
        batch.experts == 0 || batch.expert_ids.size() != batch.tokens * batch.slots ||
        batch.owner_kinds.size() != batch.tokens * batch.slots ||
        batch.weights.size() != batch.tokens * batch.slots ||
        activation_count != batch.tokens * kVfr3Hidden ||
        slot_output_count != batch.tokens * batch.slots * kVfr3Hidden ||
        output_count != batch.tokens * kVfr3Hidden) {
        context.error = "invalid VFR3 graph inputs";
        return RouteStatus::invalid_batch;
    }
    const auto *identity = context.client->session_identity();
    if (identity == nullptr || !context.client->usable() ||
        !same_session(batch, *identity, context.client->layer())) {
        context.error = "VFR3 session identity differs from graph transaction";
        return RouteStatus::invalid_batch;
    }

    for (size_t token = 0; token < batch.tokens; ++token) {
        for (size_t slot = 0; slot < batch.slots; ++slot) {
            const size_t route = slot + batch.slots * token;
            const int32_t expert = batch.expert_ids[route];
            if (expert < 0 || static_cast<size_t>(expert) >= batch.experts ||
                batch.owner_kinds[route] < -1 || batch.owner_kinds[route] > 2 ||
                !std::isfinite(batch.weights[route]) || batch.weights[route] < 0.0f) {
                context.error = "invalid VFR3 route or coefficient";
                return RouteStatus::invalid_batch;
            }
            for (size_t earlier = 0; earlier < slot; ++earlier) {
                if (batch.expert_ids[earlier + batch.slots * token] == expert) {
                    context.error = "duplicate VFR3 expert in token route";
                    return RouteStatus::invalid_batch;
                }
            }
        }
    }

    try {
        std::vector<float> input(activations, activations + activation_count);
        std::vector<float> merged(slot_outputs, slot_outputs + slot_output_count);
        std::vector<float> staged_output(output_count,
            std::numeric_limits<float>::quiet_NaN());
        const bool has_remote_owner = batch.owner_kinds.end() !=
            std::find(batch.owner_kinds.begin(), batch.owner_kinds.end(), -1);
        if (has_remote_owner &&
            !context.client->run(batch.identity.batch, batch.identity.attempt,
                static_cast<int>(batch.tokens), static_cast<int>(batch.slots),
                input, batch.expert_ids, batch.owner_kinds, merged, context.error)) {
            return RouteStatus::provider_failure;
        }

        for (size_t token = 0; token < batch.tokens; ++token) {
            for (size_t hidden = 0; hidden < kVfr3Hidden; ++hidden) {
                double sum = 0.0;
                for (size_t slot = 0; slot < batch.slots; ++slot) {
                    const size_t route = slot + batch.slots * token;
                    const float raw = merged[route * kVfr3Hidden + hidden];
                    const float decoded = ggml_fp16_to_fp32(ggml_fp32_to_fp16(raw));
                    const double term = static_cast<double>(batch.weights[route]) *
                        static_cast<double>(decoded);
                    sum += term;
                    if (!std::isfinite(decoded) || !std::isfinite(term) || !std::isfinite(sum)) {
                        context.error = "non-finite FP16/f64 Collector intermediate";
                        return RouteStatus::invalid_result;
                    }
                }
                const float result = static_cast<float>(sum);
                if (!std::isfinite(result)) {
                    context.error = "non-finite VFR3 Collector result";
                    return RouteStatus::invalid_result;
                }
                staged_output[hidden + kVfr3Hidden * token] = result;
            }
        }
        std::copy(staged_output.begin(), staged_output.end(), output);
        return RouteStatus::ready;
    } catch (...) {
        context.error = "VFR3 graph collector failure";
        return RouteStatus::provider_failure;
    }
}

} // namespace vf::ggml
