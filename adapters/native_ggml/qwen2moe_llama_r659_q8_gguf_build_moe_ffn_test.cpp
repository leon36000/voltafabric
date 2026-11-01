#include "llama-context.h"
#include "gguf.h"
#include "llama-graph.h"
#include "llama-model.h"
#include "qwen2moe_llama_route_bridge.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace vf::ggml;
constexpr size_t kHidden = 32;
constexpr size_t kExperts = 2;
constexpr size_t kQ8RowBytes = 34;
constexpr size_t kExpertBytes = kHidden * kQ8RowBytes;

std::vector<uint8_t> g_gate;
std::vector<uint8_t> g_up;
std::vector<uint8_t> g_down;

void append_u32(std::vector<uint8_t> &out, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<uint8_t>(value >> shift));
    }
}

void append_u64(std::vector<uint8_t> &out, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        out.push_back(static_cast<uint8_t>(value >> shift));
    }
}

void append_string(std::vector<uint8_t> &out, const std::string &value) {
    append_u64(out, value.size());
    out.insert(out.end(), value.begin(), value.end());
}

// Byte-for-byte layout recipe from R659's runtime fixture(false, 32): F16
// scale 1.0 and a single row-indexed signed Q8_0 code in each row.
std::vector<uint8_t> r659_q8_payload(size_t rows, size_t columns,
                                     size_t experts, int8_t first_code) {
    const size_t row_bytes = columns / 32 * kQ8RowBytes;
    std::vector<uint8_t> payload(rows * experts * row_bytes, 0);
    for (size_t expert = 0; expert < experts; ++expert) {
        for (size_t row = 0; row < rows; ++row) {
            const size_t row_offset = (expert * rows + row) * row_bytes;
            for (size_t block = 0; block < columns / 32; ++block) {
                const size_t block_offset = row_offset + block * kQ8RowBytes;
                payload[block_offset] = 0x00;
                payload[block_offset + 1] = 0x3c; // F16 1.0
                if (row < columns && row / 32 == block) {
                    payload[block_offset + 2 + row % 32] =
                        static_cast<uint8_t>(first_code + static_cast<int8_t>(expert));
                }
            }
        }
    }
    return payload;
}

struct FixtureTensor {
    std::string name;
    std::array<uint64_t, 3> dims;
    std::vector<uint8_t> payload;
};

std::vector<uint8_t> make_r659_fixture_bytes() {
    const std::vector<FixtureTensor> tensors{
        {"blk.0.ffn_gate_exps.weight", {32, 32, 2}, r659_q8_payload(32, 32, 2, 1)},
        {"blk.0.ffn_up_exps.weight",   {32, 32, 2}, r659_q8_payload(32, 32, 2, 2)},
        {"blk.0.ffn_down_exps.weight", {32, 32, 2}, r659_q8_payload(32, 32, 2, 1)},
    };
    std::vector<uint8_t> out{'G', 'G', 'U', 'F'};
    append_u32(out, 3);
    append_u64(out, tensors.size());
    append_u64(out, 5);

    append_string(out, "general.architecture");
    append_u32(out, 8); // GGUF_TYPE_STRING
    append_string(out, "qwen2moe");
    append_string(out, "qwen2moe.block_count");
    append_u32(out, 4); // GGUF_TYPE_UINT32
    append_u32(out, 1);
    append_string(out, "qwen2moe.embedding_length");
    append_u32(out, 4);
    append_u32(out, 32);
    append_string(out, "qwen2moe.expert_count");
    append_u32(out, 4);
    append_u32(out, 2);
    append_string(out, "qwen2moe.expert_feed_forward_length");
    append_u32(out, 4);
    append_u32(out, 32);

    uint64_t relative = 0;
    for (const auto &tensor : tensors) {
        append_string(out, tensor.name);
        append_u32(out, 3);
        for (const uint64_t dim : tensor.dims) append_u64(out, dim);
        append_u32(out, 8); // GGML_TYPE_Q8_0
        append_u64(out, relative);
        relative += tensor.payload.size();
    }
    while (out.size() % 32 != 0) out.push_back(0);
    for (const auto &tensor : tensors) {
        out.insert(out.end(), tensor.payload.begin(), tensor.payload.end());
    }
    return out;
}

bool load_r659_fixture(const std::string &directory) {
    if (directory.empty()) return false;
    std::filesystem::create_directories(directory);
    const auto path = std::filesystem::path(directory) / "r659-qwen2moe-q8-gguf-v3.gguf";
    const auto expected = make_r659_fixture_bytes();
    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file || !file.write(reinterpret_cast<const char *>(expected.data()),
                                 static_cast<std::streamsize>(expected.size()))) {
            std::cerr << "could not materialize the R659 Q8_0 GGUF fixture\n";
            return false;
        }
        file.flush();
        if (!file) return false;
    }

    ggml_context *tensor_context = nullptr;
    gguf_init_params params{false, &tensor_context};
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
        gguf_init_from_file(path.c_str(), params), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(tensor_context, ggml_free);
    if (!metadata || !tensors || gguf_get_version(metadata.get()) != 3 ||
        gguf_get_n_tensors(metadata.get()) != 3 ||
        std::filesystem::file_size(path) != expected.size()) {
        std::cerr << "R659 fixture was not parsed as the expected GGUF v3\n";
        return false;
    }
    const int64_t arch = gguf_find_key(metadata.get(), "general.architecture");
    const int64_t hidden = gguf_find_key(metadata.get(), "qwen2moe.embedding_length");
    const int64_t experts = gguf_find_key(metadata.get(), "qwen2moe.expert_count");
    const int64_t ffn = gguf_find_key(metadata.get(), "qwen2moe.expert_feed_forward_length");
    if (arch < 0 || hidden < 0 || experts < 0 || ffn < 0 ||
        std::string(gguf_get_val_str(metadata.get(), arch)) != "qwen2moe" ||
        gguf_get_val_u32(metadata.get(), hidden) != kHidden ||
        gguf_get_val_u32(metadata.get(), experts) != kExperts ||
        gguf_get_val_u32(metadata.get(), ffn) != kHidden) {
        std::cerr << "R659 GGUF metadata did not round-trip through the donor parser\n";
        return false;
    }

    auto load_tensor = [&](const char *name, std::vector<uint8_t> &destination,
                           size_t expected_offset) {
        const int64_t id = gguf_find_tensor(metadata.get(), name);
        if (id < 0 || gguf_get_tensor_type(metadata.get(), id) != GGML_TYPE_Q8_0 ||
            gguf_get_tensor_size(metadata.get(), id) != kExpertBytes * kExperts) {
            return false;
        }
        const int64_t *dims = gguf_get_tensor_ne(metadata.get(), id);
        if (dims == nullptr || dims[0] != 32 || dims[1] != 32 || dims[2] != 2 ||
            gguf_get_tensor_offset(metadata.get(), id) != expected_offset) {
            return false;
        }
        const ggml_tensor *tensor = ggml_get_tensor(tensors.get(), name);
        if (tensor == nullptr || tensor->type != GGML_TYPE_Q8_0 ||
            !ggml_is_contiguous(tensor) || ggml_nbytes(tensor) != kExpertBytes * kExperts) {
            return false;
        }
        const auto *begin = static_cast<const uint8_t *>(tensor->data);
        destination.assign(begin, begin + ggml_nbytes(tensor));
        const size_t absolute = gguf_get_data_offset(metadata.get()) + expected_offset;
        return absolute <= expected.size() && destination.size() <= expected.size() - absolute &&
               std::equal(destination.begin(), destination.end(), expected.begin() + absolute);
    };
    if (!load_tensor("blk.0.ffn_gate_exps.weight", g_gate, 0) ||
        !load_tensor("blk.0.ffn_up_exps.weight", g_up, kExpertBytes * kExperts) ||
        !load_tensor("blk.0.ffn_down_exps.weight", g_down, 2 * kExpertBytes * kExperts)) {
        std::cerr << "R659 Q8_0 tensor names, dimensions, spans, or bytes did not round-trip\n";
        return false;
    }
    std::cout << "R659 GGUF v3 donor load: 3 Q8_0 tensors, " << expected.size()
              << " bytes; exact R659 payload layout\n";
    return true;
}

struct RouteCapture {
    uint64_t batch_id = 0;
    int32_t checkpoint_expert = -1;
    int32_t registry_expert = -1;
    float route_weight = 0.0f;
    bool reject = false;
    bool duplicate_projection = false;
    size_t calls = 0;
    BatchIdentity identity;
    std::vector<int32_t> experts;
    std::vector<int32_t> owners;
    std::vector<float> activations;
    std::vector<float> weights;
    std::vector<float> raw_slots;
    std::vector<float> output;
};

class R659MoEInputs final : public llm_graph_input_i {
public:
    R659MoEInputs(ggml_tensor *activations, ggml_tensor *logits,
                  ggml_tensor *gate, ggml_tensor *up, ggml_tensor *down,
                  int32_t checkpoint_expert)
        : activations_(activations), logits_(logits), gate_(gate),
          up_(up), down_(down), checkpoint_expert_(checkpoint_expert) {}

    void set_input(const llama_ubatch *ubatch) override {
        if (ubatch == nullptr || ubatch->n_tokens != 1) return;
        std::array<float, kHidden> activations{};
        activations[0] = 1.0f;
        const std::array<float, 2> logits = checkpoint_expert_ == 0
            ? std::array<float, 2>{std::log(0.25f / 0.75f), std::log(0.20f / 0.80f)}
            : std::array<float, 2>{std::log(0.20f / 0.80f), std::log(0.75f / 0.25f)};
        ggml_backend_tensor_set(activations_, activations.data(), 0, sizeof(activations));
        ggml_backend_tensor_set(logits_, logits.data(), 0, sizeof(logits));
        ggml_backend_tensor_set(gate_, g_gate.data(), 0, g_gate.size());
        ggml_backend_tensor_set(up_, g_up.data(), 0, g_up.size());
        ggml_backend_tensor_set(down_, g_down.data(), 0, g_down.size());
    }

private:
    ggml_tensor *activations_;
    ggml_tensor *logits_;
    ggml_tensor *gate_;
    ggml_tensor *up_;
    ggml_tensor *down_;
    int32_t checkpoint_expert_;
};

class R659Q8MoEGraph final : public llm_graph_context {
public:
    R659Q8MoEGraph(const llm_graph_params &params, int32_t checkpoint_expert)
        : llm_graph_context(params) {
        auto *activations = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, kHidden, 1);
        auto *logits = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2, 1);
        auto *gate = ggml_new_tensor_3d(ctx0, GGML_TYPE_Q8_0, 32, 32, 2);
        auto *up = ggml_new_tensor_3d(ctx0, GGML_TYPE_Q8_0, 32, 32, 2);
        auto *down = ggml_new_tensor_3d(ctx0, GGML_TYPE_Q8_0, 32, 32, 2);
        if (!activations || !logits || !gate || !up || !down) {
            throw std::runtime_error("could not allocate Q8_0 donor graph tensors");
        }
        for (auto *input : {activations, logits, gate, up, down}) ggml_set_input(input);
        res->add_input(std::make_unique<R659MoEInputs>(
            activations, logits, gate, up, down, checkpoint_expert));
        ggml_tensor *routed = build_moe_ffn(
            activations, nullptr, up, gate, down, nullptr,
            2, 1, LLM_FFN_SILU, false, 1.0f,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, 0,
            logits, nullptr, nullptr, nullptr, nullptr, nullptr);
        if (routed == nullptr) throw std::runtime_error("donor build_moe_ffn returned null");
        res->t_embd = routed;
        ggml_build_forward_expand(gf, routed);
    }
};

class R659Qwen2MoEModel final : public llama_model {
public:
    explicit R659Qwen2MoEModel(int32_t checkpoint_expert)
        : llama_model(llama_model_default_params()), checkpoint_expert_(checkpoint_expert) {
        arch = LLM_ARCH_QWEN2MOE;
        hparams.vocab_only = false;
        hparams.n_ctx_train = 256;
        hparams.n_embd = kHidden;
        hparams.n_embd_out_impl = kHidden;
        hparams.n_layer_all = 1;
        hparams.n_layer_nextn = 0;
        hparams.n_expert = 2;
        hparams.n_expert_used_arr[0] = 1;
        hparams.n_head_arr[0] = 1;
        hparams.n_head_kv_arr[0] = 1;
        hparams.n_layer_kv_from_start = 0;
        hparams.causal_attn = true;
        layers.resize(1);
    }

    void load_stats(llama_model_loader &) override {}
    void load_hparams(llama_model_loader &) override {}
    void load_vocab(llama_model_loader &) override {}
    bool load_tensors(llama_model_loader &) override { return false; }
    void load_arch_hparams(llama_model_loader &) override {}
    void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(
            const llm_graph_params &params) const override {
        return std::make_unique<R659Q8MoEGraph>(params, checkpoint_expert_);
    }

private:
    int32_t checkpoint_expert_;
};

bool make_batch(void *userdata, const llama_ubatch &ubatch, llm_graph_type graph_type,
                uint32_t expected_layers, RouteBatch &batch) {
    auto &capture = *static_cast<RouteCapture *>(userdata);
    if (ubatch.n_tokens != 1 || graph_type != LLM_GRAPH_TYPE_DEFAULT || expected_layers != 1) {
        return false;
    }
    batch.identity = BatchIdentity{"r685-r659-q8-gguf", "r659-qwen2moe-q8-fixture",
                                   "checkpoint-explicit-map-cpu", 0, 685,
                                   capture.batch_id, capture.batch_id};
    batch.checkpoint_expert_to_registry_expert = capture.duplicate_projection
        ? std::vector<int32_t>{3, 3} : std::vector<int32_t>{3, 40};
    batch.owner_by_expert.assign(41, -1);
    batch.owner_by_expert[3] = 0;
    batch.owner_by_expert[40] = 0;
    return true;
}

RouteStatus collect(const RouteBatch &batch, const float *activations, size_t activation_count,
                    const float *slots, size_t slot_count, float *output, size_t output_count,
                    void *userdata) {
    auto &capture = *static_cast<RouteCapture *>(userdata);
    ++capture.calls;
    capture.identity = batch.identity;
    capture.experts = batch.expert_ids;
    capture.owners = batch.owner_kinds;
    capture.weights = batch.weights;
    if (activations) capture.activations.assign(activations, activations + activation_count);
    if (slots) capture.raw_slots.assign(slots, slots + slot_count);
    if (activations == nullptr || slots == nullptr || output == nullptr ||
        batch.hidden != kHidden || batch.slots != 1 || batch.tokens != 1 ||
        activation_count != kHidden || slot_count != kHidden || output_count != kHidden ||
        batch.expert_ids != std::vector<int32_t>{capture.registry_expert} ||
        batch.owner_kinds != std::vector<int32_t>{0} || batch.weights.size() != 1 ||
        std::fabs(batch.weights[0] - capture.route_weight) > 1.0e-6f ||
        capture.activations[0] != 1.0f ||
        std::any_of(capture.activations.begin() + 1, capture.activations.end(),
                    [](float value) { return value != 0.0f; })) {
        return RouteStatus::invalid_batch;
    }
    if (capture.reject) return RouteStatus::provider_failure;
    for (size_t i = 0; i < output_count; ++i) output[i] = slots[i] * batch.weights[0];
    capture.output.assign(output, output + output_count);
    return RouteStatus::ready;
}

bool near_oracle(float actual, float expected) {
    const float tolerance = 1.0e-3f * std::max(1.0f, std::fabs(expected));
    return std::isfinite(actual) && std::fabs(actual - expected) <= tolerance;
}

bool run_graph_case(uint64_t batch_id, int32_t checkpoint_expert, int32_t registry_expert,
                    float route_weight, bool reject, bool duplicate_projection = false) {
    R659Qwen2MoEModel model(checkpoint_expert);
    auto params = llama_context_default_params();
    params.n_ctx = 256;
    params.n_batch = 1;
    params.n_ubatch = 1;
    params.n_threads = 1;
    params.n_threads_batch = 1;
    params.embeddings = true;
    params.offload_kqv = false;
    params.op_offload = false;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context context(model, params);

    RouteCapture capture;
    capture.batch_id = batch_id;
    capture.checkpoint_expert = checkpoint_expert;
    capture.registry_expert = registry_expert;
    capture.route_weight = route_weight;
    capture.reject = reject;
    capture.duplicate_projection = duplicate_projection;
    Qwen2MoELlamaRouteBridge bridge(make_batch, &capture, collect, &capture);
    if (!bridge.attach(context) || context.get_vf_route_hooks() != bridge.hooks()) return false;

    llama_ubatch ubatch{};
    ubatch.n_tokens = 1;
    ggml_status status = GGML_STATUS_FAILED;
    llm_graph_result *result = context.process_ubatch(
        ubatch, LLM_GRAPH_TYPE_DEFAULT, nullptr, status);
    const bool detached = bridge.detach();
    if (duplicate_projection) {
        return result == nullptr && status == GGML_STATUS_FAILED && detached &&
               context.get_vf_route_hooks() == nullptr && capture.calls == 0 &&
               capture.output.empty();
    }
    if (reject) {
        const bool refused = result == nullptr && status == GGML_STATUS_FAILED && detached &&
               context.get_vf_route_hooks() == nullptr && capture.calls == 1 &&
               capture.experts == std::vector<int32_t>{registry_expert} && capture.output.empty();
        if (!refused) {
            std::cerr << "R659 rejected-route mismatch: result=" << (result != nullptr)
                      << " status=" << status << " detached=" << detached
                      << " callbacks=" << capture.calls << " published="
                      << !capture.output.empty() << '\n';
        }
        return refused;
    }
    if (result == nullptr || status != GGML_STATUS_SUCCESS || !detached ||
        context.get_vf_route_hooks() != nullptr || capture.calls != 1 ||
        capture.identity.run != "r685-r659-q8-gguf" || capture.identity.model != "r659-qwen2moe-q8-fixture" ||
        capture.identity.plan != "checkpoint-explicit-map-cpu" || capture.identity.layer != 0 ||
        capture.identity.epoch != 685 || capture.identity.batch != batch_id ||
        capture.identity.attempt != batch_id ||
        capture.experts != std::vector<int32_t>{registry_expert} ||
        capture.weights.size() != 1 || std::fabs(capture.weights[0] - route_weight) > 1.0e-6f ||
        capture.raw_slots.size() != kHidden || capture.output.size() != kHidden) {
        std::cerr << "R659 route mismatch: result=" << (result != nullptr) << " status=" << status
                  << " detached=" << detached << " callbacks=" << capture.calls
                  << " checkpoint=" << checkpoint_expert << " mapped=" << registry_expert
                  << " captured_ids=" << (capture.experts.empty() ? -1 : capture.experts[0])
                  << " weight=" << (capture.weights.empty() ? -1.0f : capture.weights[0])
                  << " raw_n=" << capture.raw_slots.size()
                  << " out_n=" << capture.output.size() << '\n';
        return false;
    }

    const float gate = checkpoint_expert == 0 ? 1.0f : 2.0f;
    const float up = checkpoint_expert == 0 ? 2.0f : 3.0f;
    const float down = checkpoint_expert == 0 ? 1.0f : 2.0f;
    const float expected_raw = (gate / (1.0f + std::exp(-gate))) * up * down;
    for (size_t i = 0; i < kHidden; ++i) {
        const float expected = i == 0 ? expected_raw : 0.0f;
        const float expected_weighted = capture.raw_slots[i] * capture.weights[0];
        if (!near_oracle(capture.raw_slots[i], expected) ||
            std::fabs(capture.output[i] - expected_weighted) > 1.0e-6f ||
            !near_oracle(capture.output[i], expected * route_weight)) {
            std::cerr << "R659 numeric mismatch at " << i << ": raw=" << capture.raw_slots[i]
                      << " expected_raw=" << expected << " out=" << capture.output[i]
                      << " expected_out=" << expected * route_weight << '\n';
            return false;
        }
    }
    return true;
}
} // namespace

int main() {
    const char *output_directory = std::getenv("VF_TEST_OUTPUT_DIR");
    if (output_directory == nullptr || !load_r659_fixture(output_directory)) return 1;
    // The donor's actual top-1 gate chooses each checkpoint ID. R659 resident
    // order [expert 40, expert 3] maps to checkpoint [1, 0]; the route boundary
    // must translate the donor ID exactly once.
    if (!run_graph_case(1, 0, 3, 0.25f, false) ||
        !run_graph_case(2, 1, 40, 0.75f, false) ||
        !run_graph_case(3, 1, 40, 0.75f, true) ||
        !run_graph_case(4, 1, 40, 0.75f, false, true)) {
        std::cerr << "R659 Q8_0 GGUF donor MoE bridge failed\n";
        return 1;
    }
    std::cout << "R659 Q8_0 GGUF -> donor build_moe_ffn: explicit checkpoint mapping, "
                 "SiLU(gate)*up, route weight once, atomic refusal PASS\n";
    return 0;
}
