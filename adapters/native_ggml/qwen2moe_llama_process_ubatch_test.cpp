#include "llama-context.h"
#include "llama-graph.h"
#include "llama-model.h"
#include "qwen2moe_llama_route_bridge.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace {
using namespace vf::ggml;

struct RouteCapture {
    size_t collector_calls = 0;
    size_t activation_count = 0;
    size_t output_count = 0;
    BatchIdentity identity;
    std::vector<int32_t> experts;
    std::vector<int32_t> owners;
    std::vector<float> activations;
    std::vector<float> weights;
    std::vector<float> output;
};

class SyntheticRouteInput final : public llm_graph_input_i {
public:
    SyntheticRouteInput(ggml_tensor * activations, ggml_tensor * experts,
                        ggml_tensor * weights, ggml_tensor * raw)
        : activations_(activations), experts_(experts), weights_(weights), raw_(raw) {}

    void set_input(const llama_ubatch * ubatch) override {
        if (ubatch == nullptr || ubatch->n_tokens != 1) return;
        const std::array<float, 2> activations{3.0f, 4.0f};
        const int32_t expert = 1;
        const float weight = 0.5f;
        const std::array<float, 2> raw{10.0f, 20.0f};
        ggml_backend_tensor_set(activations_, activations.data(), 0, sizeof(activations));
        ggml_backend_tensor_set(experts_, &expert, 0, sizeof(expert));
        ggml_backend_tensor_set(weights_, &weight, 0, sizeof(weight));
        ggml_backend_tensor_set(raw_, raw.data(), 0, sizeof(raw));
    }

private:
    ggml_tensor * activations_;
    ggml_tensor * experts_;
    ggml_tensor * weights_;
    ggml_tensor * raw_;
};

class SyntheticQwen2MoEGraph final : public llm_graph_context {
public:
    explicit SyntheticQwen2MoEGraph(const llm_graph_params & params)
        : llm_graph_context(params) {
        auto * activations = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2, 1);
        auto * experts = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, 1);
        auto * weights = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, 1);
        auto * raw = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 2, 1, 1);
        if (activations == nullptr || experts == nullptr || weights == nullptr || raw == nullptr) {
            throw std::runtime_error("synthetic route tensors were not created");
        }
        ggml_set_input(activations);
        ggml_set_input(experts);
        ggml_set_input(weights);
        ggml_set_input(raw);

        res->add_input(std::make_unique<SyntheticRouteInput>(activations, experts, weights, raw));
        ggml_tensor * routed = raw;
        if (vf_route_hooks != nullptr) {
            routed = vf_route_hooks->append_qwen2moe(
                vf_route_hooks->userdata, ctx0, 0, activations, experts, weights, raw);
            if (routed == nullptr) {
                throw std::runtime_error("VoltaFabric Qwen2MoE hook rejected synthetic layer");
            }
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
        return std::make_unique<SyntheticQwen2MoEGraph>(params);
    }
};

bool make_route_batch(void *, const llama_ubatch & ubatch, llm_graph_type graph_type,
                      uint32_t expected_layers, RouteBatch & batch) {
    if (ubatch.n_tokens != 1 || graph_type != LLM_GRAPH_TYPE_DEFAULT || expected_layers != 1) {
        return false;
    }
    batch.identity = BatchIdentity{"r682-process-ubatch", "synthetic-qwen2moe", "cpu-route-plan", 0, 682, 1, 1};
    batch.owner_by_expert = {2, 0};
    return true;
}

RouteStatus collect_route(const RouteBatch & batch, const float * activations,
                          size_t activation_count, const float * slots, size_t slot_count,
                          float * output, size_t output_count, void * userdata) {
    auto & capture = *static_cast<RouteCapture *>(userdata);
    ++capture.collector_calls;
    capture.identity = batch.identity;
    capture.experts = batch.expert_ids;
    capture.owners = batch.owner_kinds;
    capture.weights = batch.weights;
    capture.activation_count = activation_count;
    capture.output_count = output_count;
    if (activations != nullptr) capture.activations.assign(activations, activations + activation_count);
    if (slots == nullptr || output == nullptr || batch.hidden != 2 || batch.slots != 1 ||
        batch.tokens != 1 || activation_count != 2 || slot_count != 2 || output_count != 2 ||
        batch.expert_ids != std::vector<int32_t>{1} || batch.owner_kinds != std::vector<int32_t>{0} ||
        batch.weights != std::vector<float>{0.5f}) {
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
    Qwen2MoELlamaRouteBridge bridge(make_route_batch, nullptr, collect_route, &capture);
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
        context.get_vf_route_hooks() == nullptr && capture.collector_calls == 1 &&
        capture.identity.run == "r682-process-ubatch" && capture.identity.model == "synthetic-qwen2moe" &&
        capture.identity.plan == "cpu-route-plan" && capture.identity.layer == 0 &&
        capture.identity.epoch == 682 && capture.identity.batch == 1 && capture.identity.attempt == 1 &&
        capture.experts == std::vector<int32_t>{1} && capture.owners == std::vector<int32_t>{0} &&
        capture.activations == std::vector<float>({3.0f, 4.0f}) &&
        capture.weights == std::vector<float>{0.5f} &&
        capture.output.size() == 2 && near(capture.output[0], 5.0f) && near(capture.output[1], 10.0f);
    if (!passed) {
        std::cerr << "live process_ubatch route callback failed: status=" << status
                  << " result=" << (result != nullptr)
                  << " collector_calls=" << capture.collector_calls << '\n';
        return 1;
    }
    std::cout << "live llama_context::process_ubatch CPU hook pass; one dynamic route, one weight application, output [5,10]\n";
    return 0;
}
