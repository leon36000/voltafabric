#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {
struct VFR697MemoryAdvice {
    std::uint32_t version;
    std::uint32_t status;
    std::uint32_t refusal_code;
    std::uint32_t layer;
    std::uint32_t rank;
    std::uint32_t reserved;
    std::uint64_t epoch;
    std::uint64_t batch;
    std::uint64_t attempt;
    std::uint64_t capacity_bytes;
    std::uint64_t resident_expert_bytes;
    std::uint64_t candidate_expert_bytes;
    std::uint64_t loading_peak_bytes;
    std::uint64_t headroom_bytes;
};

void *vf_r697_open(const char *source_gguf, const char *rank_session_binary,
                   const char *ledger_root);
int32_t vf_r697_set_advisory_capacity(void *context, std::uint64_t capacity_bytes);
int32_t vf_r697_get_memory_advice(
    void *context, const char *run_id, const char *model_revision,
    const char *plan_id, std::uint32_t layer, std::uint64_t epoch,
    std::uint64_t batch, std::uint64_t attempt, VFR697MemoryAdvice *out);
int32_t vf_r697_run_routed_layer(
    void *context, const char *run_id, const char *model_revision,
    const char *plan_id, std::uint32_t layer, std::uint64_t epoch,
    std::uint64_t batch, std::uint64_t attempt, std::size_t hidden,
    std::size_t slots, std::size_t tokens, const std::int32_t *expert_ids,
    std::size_t expert_ids_len, const float *route_weights,
    std::size_t route_weights_len, const float *activations,
    std::size_t activations_len, double *output, std::size_t output_len);
void vf_r697_close(void *context);
}
