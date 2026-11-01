#include "llama-context.h"
#include "gguf.h"
#include "llama-graph.h"
#include "llama-model.h"
#include "qwen2moe_llama_route_bridge.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <memory>
#include <vector>

namespace {
using namespace vf::ggml;


std::vector<float> g_up_weights;
std::vector<float> g_down_weights;

bool load_tiny_qwen2moe_gguf(const std::string & output_directory) {
    if (output_directory.empty()) {
        std::cerr << "VF_TEST_OUTPUT_DIR is required\n";
        return false;
    }
    std::filesystem::create_directories(output_directory);
    const auto path = std::filesystem::path(output_directory) / "qwen2moe-tiny-v3.gguf";

    ggml_init_params source_params{1u << 20, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> source_ctx(
        ggml_init(source_params), ggml_free);
    if (!source_ctx) {
        std::cerr << "could not allocate tiny GGUF tensor context\n";
        return false;
    }
    ggml_tensor * up = ggml_new_tensor_3d(source_ctx.get(), GGML_TYPE_F32, 2, 2, 2);
    ggml_tensor * down = ggml_new_tensor_3d(source_ctx.get(), GGML_TYPE_F32, 2, 2, 2);
    if (up == nullptr || down == nullptr) {
        std::cerr << "could not allocate tiny GGUF expert tensors\n";
        return false;
    }
    ggml_set_name(up, "blk.0.ffn_up_exps.weight");
    ggml_set_name(down, "blk.0.ffn_down_exps.weight");

    const std::array<float, 8> up_data{
        0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 1.0f
    };
    const std::array<float, 8> down_data{
        0.0f, 0.0f, 0.0f, 0.0f,
        2.0f, 0.0f, 0.0f, 5.0f
    };
    std::memcpy(up->data, up_data.data(), sizeof(up_data));
    std::memcpy(down->data, down_data.data(), sizeof(down_data));

    std::unique_ptr<gguf_context, decltype(&gguf_free)> output(gguf_init_empty(), gguf_free);
    if (!output) {
        std::cerr << "could not create GGUF metadata context\n";
        return false;
    }
    gguf_set_val_str(output.get(), "general.architecture", "qwen2moe");
    gguf_set_val_str(output.get(), "general.name", "vf-r684-tiny-qwen2moe-fixture");
    gguf_set_val_u32(output.get(), "qwen2moe.block_count", 1);
    gguf_set_val_u32(output.get(), "qwen2moe.embedding_length", 2);
    gguf_set_val_u32(output.get(), "qwen2moe.feed_forward_length", 2);
    gguf_set_val_u32(output.get(), "qwen2moe.expert_count", 2);
    gguf_set_val_u32(output.get(), "qwen2moe.expert_used_count", 1);
    gguf_add_tensor(output.get(), up);
    gguf_add_tensor(output.get(), down);
    if (!gguf_write_to_file(output.get(), path.c_str(), false)) {
        std::cerr << "could not write tiny GGUF v3 fixture\n";
        return false;
    }

    ggml_context * loaded_tensor_ctx = nullptr;
    gguf_init_params read_params{false, &loaded_tensor_ctx};
    std::unique_ptr<gguf_context, decltype(&gguf_free)> loaded(
        gguf_init_from_file(path.c_str(), read_params), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> loaded_ctx(
        loaded_tensor_ctx, ggml_free);
    if (!loaded || !loaded_ctx || gguf_get_version(loaded.get()) != 3 ||
        gguf_get_n_tensors(loaded.get()) != 2) {
        std::cerr << "tiny fixture did not round-trip through donor GGUF v3 parser\n";
        return false;
    }

    auto value_is_u32 = [&](const char * key, uint32_t expected) {
        const int64_t id = gguf_find_key(loaded.get(), key);
        return id >= 0 && gguf_get_kv_type(loaded.get(), id) == GGUF_TYPE_UINT32 &&
               gguf_get_val_u32(loaded.get(), id) == expected;
    };
    const int64_t arch_id = gguf_find_key(loaded.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(loaded.get(), arch_id) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(loaded.get(), arch_id)) != "qwen2moe" ||
        !value_is_u32("qwen2moe.block_count", 1) ||
        !value_is_u32("qwen2moe.embedding_length", 2) ||
        !value_is_u32("qwen2moe.feed_forward_length", 2) ||
        !value_is_u32("qwen2moe.expert_count", 2) ||
        !value_is_u32("qwen2moe.expert_used_count", 1)) {
        std::cerr << "tiny GGUF metadata did not round-trip with expected Qwen2MoE dimensions\n";
        return false;
    }

    const auto file_size = std::filesystem::file_size(path);
    const size_t expected_bytes = 8 * sizeof(float);
    const size_t data_offset = gguf_get_data_offset(loaded.get());
    auto tensor_is_valid = [&](const char * name) {
        const int64_t id = gguf_find_tensor(loaded.get(), name);
        if (id < 0 || gguf_get_tensor_type(loaded.get(), id) != GGML_TYPE_F32 ||
            gguf_get_tensor_size(loaded.get(), id) != expected_bytes) {
            return false;
        }
        const int64_t * ne = gguf_get_tensor_ne(loaded.get(), id);
        if (ne == nullptr || ne[0] != 2 || ne[1] != 2 || ne[2] != 2) {
            return false;
        }
        const size_t relative = gguf_get_tensor_offset(loaded.get(), id);
        if (relative > file_size || data_offset > file_size - relative) {
            return false;
        }
        const size_t absolute = data_offset + relative;
        return expected_bytes <= file_size - absolute;
    };
    if (!tensor_is_valid("blk.0.ffn_up_exps.weight") ||
        !tensor_is_valid("blk.0.ffn_down_exps.weight")) {
        std::cerr << "GGUF tensor type, shape, or file span was invalid\n";
        return false;
    }

    const ggml_tensor * loaded_up =
        ggml_get_tensor(loaded_ctx.get(), "blk.0.ffn_up_exps.weight");
    const ggml_tensor * loaded_down =
        ggml_get_tensor(loaded_ctx.get(), "blk.0.ffn_down_exps.weight");
    if (loaded_up == nullptr || loaded_down == nullptr ||
        loaded_up->type != GGML_TYPE_F32 || loaded_down->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(loaded_up) || !ggml_is_contiguous(loaded_down) ||
        ggml_nbytes(loaded_up) != expected_bytes || ggml_nbytes(loaded_down) != expected_bytes) {
        std::cerr << "donor GGUF tensor allocation was absent or incompatible\n";
        return false;
    }

    const auto * up_values = static_cast<const float *>(loaded_up->data);
    const auto * down_values = static_cast<const float *>(loaded_down->data);
    g_up_weights.assign(up_values, up_values + 8);
    g_down_weights.assign(down_values, down_values + 8);
    if (g_up_weights != std::vector<float>(up_data.begin(), up_data.end()) ||
        g_down_weights != std::vector<float>(down_data.begin(), down_data.end())) {
        std::cerr << "GGUF tensor bytes changed across donor writer/parser round-trip\n";
        return false;
    }

    std::cout << "GGUF v3 fixture read by pinned donor parser: 2 tensors, "
              << file_size << " bytes\n";
    return true;
}

struct RouteCapture {
    size_t calls = 0;
    uint64_t batch_id = 1;
    bool reject = false;
    BatchIdentity identity;
    std::vector<int32_t> experts;
    std::vector<int32_t> owners;
    std::vector<float> activations;
    std::vector<float> weights;
    std::vector<float> raw_slots;
    std::vector<float> output;
};

class SyntheticMoEInputs final : public llm_graph_input_i {
public:
    SyntheticMoEInputs(ggml_tensor * activations, ggml_tensor * logits,
                       ggml_tensor * selected, ggml_tensor * up,
                       ggml_tensor * down)
        : activations_(activations), logits_(logits), selected_(selected),
          up_(up), down_(down) {}

    void set_input(const llama_ubatch * ubatch) override {
        if (ubatch == nullptr || ubatch->n_tokens != 1) return;
        const std::array<float, 2> activations{3.0f, 4.0f};
        const std::array<float, 2> logits{0.0f, 0.0f};
        const int32_t selected = 1;
        if (g_up_weights.size() != 8 || g_down_weights.size() != 8) {
            throw std::runtime_error("GGUF weights were not loaded before graph construction");
        }
        ggml_backend_tensor_set(activations_, activations.data(), 0, sizeof(activations));
        ggml_backend_tensor_set(logits_, logits.data(), 0, sizeof(logits));
        ggml_backend_tensor_set(selected_, &selected, 0, sizeof(selected));
        ggml_backend_tensor_set(up_, g_up_weights.data(), 0, g_up_weights.size() * sizeof(float));
        ggml_backend_tensor_set(down_, g_down_weights.data(), 0, g_down_weights.size() * sizeof(float));
    }

private:
    ggml_tensor * activations_;
    ggml_tensor * logits_;
    ggml_tensor * selected_;
    ggml_tensor * up_;
    ggml_tensor * down_;
};

class ProductionMoEGraph final : public llm_graph_context {
public:
    explicit ProductionMoEGraph(const llm_graph_params & params)
        : llm_graph_context(params) {
        auto * activations = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2, 1);
        auto * logits = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2, 1);
        auto * selected = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, 1);
        auto * up = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 2, 2, 2);
        auto * down = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 2, 2, 2);
        if (activations == nullptr || logits == nullptr || selected == nullptr ||
            up == nullptr || down == nullptr) {
            throw std::runtime_error("synthetic MoE tensors were not created");
        }
        ggml_set_input(activations);
        ggml_set_input(logits);
        ggml_set_input(selected);
        ggml_set_input(up);
        ggml_set_input(down);
        res->add_input(std::make_unique<SyntheticMoEInputs>(
            activations, logits, selected, up, down));

        // This deliberately calls the donor implementation, including its
        // expert gather, activation and down projection before the VF hook.
        ggml_tensor * routed = build_moe_ffn(
            activations, nullptr, up, nullptr, down, nullptr,
            /*n_expert=*/2, /*n_expert_used=*/1,
            LLM_FFN_RELU, /*norm_w=*/false, /*w_scale=*/1.0f,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, /*il=*/0,
            logits, nullptr, nullptr, nullptr, nullptr, selected);
        if (routed == nullptr) {
            throw std::runtime_error("donor build_moe_ffn returned no route output");
        }
        res->t_embd = routed;
        ggml_build_forward_expand(gf, routed);
    }
};

class SyntheticQwen2MoEModel final : public llama_model {
public:
    SyntheticQwen2MoEModel() : llama_model(llama_model_default_params()) {
        arch = LLM_ARCH_QWEN2MOE;
        hparams.vocab_only = false;
        hparams.n_ctx_train = 256;
        hparams.n_embd = 2;
        hparams.n_embd_out_impl = 2;
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
            const llm_graph_params & params) const override {
        return std::make_unique<ProductionMoEGraph>(params);
    }
};

bool make_batch(void * userdata, const llama_ubatch & ubatch, llm_graph_type graph_type,
                uint32_t expected_layers, RouteBatch & batch) {
    if (ubatch.n_tokens != 1 || graph_type != LLM_GRAPH_TYPE_DEFAULT || expected_layers != 1) {
        return false;
    }
    const auto & capture = *static_cast<RouteCapture *>(userdata);
    batch.identity = BatchIdentity{"r684-gguf-build-moe-ffn", "synthetic-qwen2moe-gguf-v3",
                                   "cpu-route-plan", 0, 684, capture.batch_id, capture.batch_id};
    batch.owner_by_expert = {2, 0};
    return true;
}

RouteStatus collect(const RouteBatch & batch, const float * activations,
                    size_t activation_count, const float * slots, size_t slot_count,
                    float * output, size_t output_count, void * userdata) {
    auto & capture = *static_cast<RouteCapture *>(userdata);
    ++capture.calls;
    capture.identity = batch.identity;
    capture.experts = batch.expert_ids;
    capture.owners = batch.owner_kinds;
    capture.weights = batch.weights;
    if (activations != nullptr) capture.activations.assign(activations, activations + activation_count);
    if (slots != nullptr) capture.raw_slots.assign(slots, slots + slot_count);
    if (slots == nullptr || output == nullptr || batch.hidden != 2 || batch.slots != 1 ||
        batch.tokens != 1 || activation_count != 2 || slot_count != 2 || output_count != 2 ||
        batch.expert_ids != std::vector<int32_t>{1} || batch.owner_kinds != std::vector<int32_t>{0} ||
        batch.weights != std::vector<float>{0.5f} ||
        capture.activations != std::vector<float>({3.0f, 4.0f}) ||
        capture.raw_slots != std::vector<float>({6.0f, 20.0f})) {
        return RouteStatus::invalid_batch;
    }
    if (capture.reject) {
        return RouteStatus::provider_failure;
    }
    output[0] = slots[0] * batch.weights[0];
    output[1] = slots[1] * batch.weights[0];
    capture.output.assign(output, output + output_count);
    return RouteStatus::ready;
}

bool near(float actual, float expected) {
    return std::isfinite(actual) && std::fabs(actual - expected) <= 1.0e-6f;
}

} // namespace

bool run_graph_case(bool reject) {
    SyntheticQwen2MoEModel model;
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
    capture.batch_id = reject ? 2 : 1;
    capture.reject = reject;
    Qwen2MoELlamaRouteBridge bridge(make_batch, &capture, collect, &capture);
    if (!bridge.attach(context) || context.get_vf_route_hooks() != bridge.hooks()) {
        std::cerr << "route bridge did not attach to live llama_context\n";
        return false;
    }

    llama_ubatch ubatch{};
    ubatch.n_tokens = 1;
    ggml_status status = GGML_STATUS_FAILED;
    llm_graph_result * result = context.process_ubatch(
        ubatch, LLM_GRAPH_TYPE_DEFAULT, nullptr, status);
    const bool detached = bridge.detach();
    if (reject) {
        const bool refused_atomically = result == nullptr && status == GGML_STATUS_FAILED &&
            detached && context.get_vf_route_hooks() == nullptr && capture.calls == 1 &&
            capture.batch_id == 2 && capture.output.empty();
        if (!refused_atomically) {
            std::cerr << "rejected collector output escaped graph gate: status=" << status
                      << " result=" << (result != nullptr)
                      << " calls=" << capture.calls
                      << " published=" << !capture.output.empty() << '\n';
        }
        return refused_atomically;
    }

    const bool passed = result != nullptr && status == GGML_STATUS_SUCCESS && detached &&
        context.get_vf_route_hooks() == nullptr && capture.calls == 1 &&
        capture.identity.run == "r684-gguf-build-moe-ffn" &&
        capture.identity.model == "synthetic-qwen2moe-gguf-v3" &&
        capture.identity.plan == "cpu-route-plan" && capture.identity.layer == 0 &&
        capture.identity.epoch == 684 && capture.identity.batch == 1 &&
        capture.identity.attempt == 1 && capture.experts == std::vector<int32_t>{1} &&
        capture.owners == std::vector<int32_t>{0} &&
        capture.weights == std::vector<float>{0.5f} &&
        capture.raw_slots == std::vector<float>({6.0f, 20.0f}) &&
        capture.output.size() == 2 && near(capture.output[0], 3.0f) &&
        near(capture.output[1], 10.0f);
    if (!passed) {
        std::cerr << "donor GGUF build_moe_ffn route hook failed: status=" << status
                  << " result=" << (result != nullptr)
                  << " callback_calls=" << capture.calls << '\n';
    }
    return passed;
}

int main() {
    const char * output_dir = std::getenv("VF_TEST_OUTPUT_DIR");
    if (output_dir == nullptr || !load_tiny_qwen2moe_gguf(output_dir)) return 1;
    if (!run_graph_case(false)) return 1;
    if (!run_graph_case(true)) return 1;
    std::cout << "donor GGUF v3 -> build_moe_ffn CPU hook pass; raw [6,20], weight once, output [3,10], rejected output withheld\n";
    return 0;
}

