#include "llama-context.h"
#include "llama-graph.h"
#include "llama-model.h"
#include "qwen2moe_llama_route_bridge.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace {
using namespace vf::ggml;

struct RouteCapture {
    size_t calls = 0;
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
        const std::array<float, 8> up{
            0.0f, 0.0f, 0.0f, 0.0f, // expert 0
            1.0f, 0.0f, 0.0f, 1.0f  // expert 1: identity
        };
        const std::array<float, 8> down{
            0.0f, 0.0f, 0.0f, 0.0f, // expert 0
            2.0f, 0.0f, 0.0f, 5.0f  // expert 1: [3,4] -> [6,20]
        };
        ggml_backend_tensor_set(activations_, activations.data(), 0, sizeof(activations));
        ggml_backend_tensor_set(logits_, logits.data(), 0, sizeof(logits));
        ggml_backend_tensor_set(selected_, &selected, 0, sizeof(selected));
        ggml_backend_tensor_set(up_, up.data(), 0, sizeof(up));
        ggml_backend_tensor_set(down_, down.data(), 0, sizeof(down));
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

bool make_batch(void *, const llama_ubatch & ubatch, llm_graph_type graph_type,
                uint32_t expected_layers, RouteBatch & batch) {
    if (ubatch.n_tokens != 1 || graph_type != LLM_GRAPH_TYPE_DEFAULT || expected_layers != 1) {
        return false;
    }
    batch.identity = BatchIdentity{"r683-build-moe-ffn", "synthetic-qwen2moe", "cpu-route-plan", 0, 683, 1, 1};
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
    output[0] = slots[0] * batch.weights[0];
    output[1] = slots[1] * batch.weights[0];
    capture.output.assign(output, output + output_count);
    return RouteStatus::ready;
}

bool near(float actual, float expected) {
    return std::isfinite(actual) && std::fabs(actual - expected) <= 1.0e-6f;
}

} // namespace

int main() {
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
    Qwen2MoELlamaRouteBridge bridge(make_batch, nullptr, collect, &capture);
    if (!bridge.attach(context) || context.get_vf_route_hooks() != bridge.hooks()) {
        std::cerr << "route bridge did not attach to live llama_context\n";
        return 1;
    }

    llama_ubatch ubatch{};
    ubatch.n_tokens = 1;
    ggml_status status = GGML_STATUS_FAILED;
    llm_graph_result * result = context.process_ubatch(
        ubatch, LLM_GRAPH_TYPE_DEFAULT, nullptr, status);
    const bool detached = bridge.detach();
    const bool passed = result != nullptr && status == GGML_STATUS_SUCCESS && detached &&
        context.get_vf_route_hooks() == nullptr && capture.calls == 1 &&
        capture.identity.run == "r683-build-moe-ffn" &&
        capture.identity.model == "synthetic-qwen2moe" &&
        capture.identity.plan == "cpu-route-plan" && capture.identity.layer == 0 &&
        capture.identity.epoch == 683 && capture.identity.batch == 1 &&
        capture.identity.attempt == 1 && capture.experts == std::vector<int32_t>{1} &&
        capture.owners == std::vector<int32_t>{0} &&
        capture.weights == std::vector<float>{0.5f} &&
        capture.raw_slots == std::vector<float>({6.0f, 20.0f}) &&
        capture.output.size() == 2 && near(capture.output[0], 3.0f) &&
        near(capture.output[1], 10.0f);
    if (!passed) {
        std::cerr << "donor build_moe_ffn route hook failed: status=" << status
                  << " result=" << (result != nullptr)
                  << " callback_calls=" << capture.calls << '\n';
        return 1;
    }
    std::cout << "live donor build_moe_ffn CPU hook pass; raw [6,20], weight once, output [3,10]\n";
    return 0;
}
