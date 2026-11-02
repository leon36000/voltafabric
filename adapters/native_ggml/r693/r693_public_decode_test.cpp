#include "llama-context.h"
#include "llama.h"
#include "gguf.h"
#include "qwen2moe_llama_route_bridge.hpp"
#include "r693/rust-loader/r693_session_loader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace vf::ggml;
constexpr int64_t kHidden = 32;
constexpr int64_t kVocab = 4;
constexpr int64_t kExperts = 41;
constexpr size_t kLayers = 2;
constexpr int64_t kLocalExperts = 2;
constexpr size_t kQ8RowBytes = 34;
constexpr size_t kExpertBytes = kHidden * kQ8RowBytes;
constexpr std::array<int32_t, 2> kRegistryIds{40, 3};

struct FixtureWeights {
    std::array<std::vector<uint8_t>, kLayers> gate;
    std::array<std::vector<uint8_t>, kLayers> up;
    std::array<std::vector<uint8_t>, kLayers> down;
    std::array<float, kHidden * kVocab> output{};
    bool callback_failed = false;
    std::string callback_error;
};

struct FixtureMetadata {
    ggml_context * tensor_context = nullptr;
    gguf_context * gguf = nullptr;

    ~FixtureMetadata() {
        gguf_free(gguf);
        ggml_free(tensor_context);
    }
};

bool has_suffix(const std::string & value, const char * suffix) {
    const size_t suffix_size = std::char_traits<char>::length(suffix);
    return value.size() >= suffix_size &&
           value.compare(value.size() - suffix_size, suffix_size, suffix) == 0;
}

struct RunCapture {
    void * rust_session = nullptr;
    uint64_t batch = 0;
    uint64_t attempt = 0;
    int32_t reject_layer = -1;
    size_t collect_calls = 0;
    std::array<std::vector<int32_t>, kLayers> experts;
    std::array<std::vector<float>, kLayers> weights;
    std::array<std::vector<float>, kLayers> router_input;
    std::array<std::vector<float>, kLayers> output;
    std::array<int32_t, kLayers> rust_status{-1, -1};
};

void set_metadata(FixtureMetadata &meta) {
    meta.gguf = gguf_init_empty();
    if (meta.gguf == nullptr) throw std::runtime_error("gguf_init_empty failed");
    gguf_set_val_str(meta.gguf, "general.architecture", "qwen2moe");
    gguf_set_val_u32(meta.gguf, "qwen2moe.context_length", 32);
    gguf_set_val_u32(meta.gguf, "qwen2moe.embedding_length", kHidden);
    gguf_set_val_u32(meta.gguf, "qwen2moe.block_count", static_cast<uint32_t>(kLayers));
    gguf_set_val_u32(meta.gguf, "qwen2moe.vocab_size", kVocab);
    gguf_set_val_u32(meta.gguf, "qwen2moe.attention.head_count", 1);
    gguf_set_val_u32(meta.gguf, "qwen2moe.attention.head_count_kv", 1);
    gguf_set_val_u32(meta.gguf, "qwen2moe.attention.key_length", kHidden);
    gguf_set_val_u32(meta.gguf, "qwen2moe.attention.value_length", kHidden);
    gguf_set_val_u32(meta.gguf, "qwen2moe.rope.dimension_count", kHidden);
    gguf_set_val_f32(meta.gguf, "qwen2moe.rope.freq_base", 10000.0f);
    gguf_set_val_bool(meta.gguf, "qwen2moe.attention.causal", true);
    gguf_set_val_f32(meta.gguf, "qwen2moe.attention.layer_norm_rms_epsilon", 1.0e-5f);
    gguf_set_val_u32(meta.gguf, "qwen2moe.expert_count", kExperts);
    gguf_set_val_u32(meta.gguf, "qwen2moe.expert_used_count", 2);
    gguf_set_val_u32(meta.gguf, "qwen2moe.expert_feed_forward_length", kHidden);
    gguf_set_val_u32(meta.gguf, "qwen2moe.expert_shared_feed_forward_length", kHidden);
    gguf_set_val_str(meta.gguf, "tokenizer.ggml.model", "no_vocab");

    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * (3 * kLayers + 2);
    params.no_alloc = true;
    meta.tensor_context = ggml_init(params);
    if (meta.tensor_context == nullptr) throw std::runtime_error("ggml_init failed");
    for (size_t layer = 0; layer < kLayers; ++layer) {
        for (const char * projection : {"ffn_gate_exps.weight", "ffn_up_exps.weight",
                                        "ffn_down_exps.weight"}) {
            auto * tensor = ggml_new_tensor_3d(meta.tensor_context, GGML_TYPE_Q8_0,
                                                kHidden, kHidden, kExperts);
            if (tensor == nullptr) throw std::runtime_error("GGUF tensor metadata allocation failed");
            const std::string name = "blk." + std::to_string(layer) + "." + projection;
            ggml_set_name(tensor, name.c_str());
            gguf_add_tensor(meta.gguf, tensor);
        }
    }
}

std::vector<uint8_t> make_q8_tensor(int8_t code_expert_40, int8_t code_expert_3) {
    std::vector<uint8_t> bytes(static_cast<size_t>(kLocalExperts) * kExpertBytes, 0);
    for (int64_t expert = 0; expert < kLocalExperts; ++expert) {
        const int8_t code = expert == 0 ? code_expert_40 : code_expert_3;
        for (int64_t row = 0; row < kHidden; ++row) {
            const size_t offset = static_cast<size_t>(expert) * kExpertBytes +
                                  static_cast<size_t>(row) * kQ8RowBytes;
            bytes[offset] = 0;
            bytes[offset + 1] = 0x3c; // IEEE F16 one
            bytes[offset + 2 + static_cast<size_t>(row)] =
                static_cast<uint8_t>(code);
        }
    }
    return bytes;
}

FixtureWeights make_weights() {
    FixtureWeights weights;
    // Resident order [expert 40, expert 3] maps to source indices [1, 0].
    for (size_t layer = 0; layer < kLayers; ++layer) {
        const int8_t base = static_cast<int8_t>(1 + layer);
        weights.gate[layer] = make_q8_tensor(static_cast<int8_t>(base + 1), base);
        weights.up[layer] = make_q8_tensor(static_cast<int8_t>(base + 2),
                                            static_cast<int8_t>(base + 1));
        weights.down[layer] = make_q8_tensor(static_cast<int8_t>(base + 1), base);
    }
    for (int64_t token = 0; token < kVocab; ++token) {
        for (int64_t input = 0; input < kHidden; ++input) {
            const float token_scale = static_cast<float>(token + 1) * 0.015625f;
            const float input_scale = static_cast<float>(input % 5 + 1) * 0.03125f;
            weights.output[static_cast<size_t>(token * kHidden + input)] =
                token_scale * input_scale;
        }
    }
    return weights;
}

void copy_experts(uint8_t * dst, const std::vector<uint8_t> & src) {
    if (dst == nullptr || src.size() != 2 * kExpertBytes) {
        throw std::runtime_error("synthetic Q8 expert fixture has an invalid extent");
    }
    for (size_t local = 0; local < kRegistryIds.size(); ++local) {
        const size_t registry = static_cast<size_t>(kRegistryIds[local]);
        std::memcpy(dst + registry * kExpertBytes, src.data() + local * kExpertBytes,
                    kExpertBytes);
    }
}

void fill_tensor(ggml_tensor * tensor, void * userdata) {
    if (userdata == nullptr) return;
    auto & weights = *static_cast<FixtureWeights *>(userdata);
    try {
        if (tensor == nullptr || userdata == nullptr || tensor->data == nullptr) {
            throw std::runtime_error("llama.cpp supplied an unallocated synthetic tensor");
        }
        const size_t bytes = ggml_nbytes(tensor);
        const std::string name = ggml_get_name(tensor);
        std::memset(tensor->data, 0, bytes);

        for (size_t layer = 0; layer < kLayers; ++layer) {
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            const std::string gate_name = prefix + "ffn_gate_exps.weight";
            const std::string up_name = prefix + "ffn_up_exps.weight";
            const std::string down_name = prefix + "ffn_down_exps.weight";
            if (name == gate_name || name == up_name || name == down_name) {
                if (tensor->type != GGML_TYPE_Q8_0 || tensor->ne[0] != kHidden ||
                    tensor->ne[1] != kHidden || tensor->ne[2] != kExperts) {
                    throw std::runtime_error("Qwen2MoE expert tensor type/shape changed");
                }
                const auto & src = name == gate_name ? weights.gate[layer]
                                 : name == up_name ? weights.up[layer]
                                                   : weights.down[layer];
                copy_experts(static_cast<uint8_t *>(tensor->data), src);
                return;
            }
        }

        if (tensor->type != GGML_TYPE_F32 || bytes % sizeof(float) != 0) {
            throw std::runtime_error("unexpected non-Q8 tensor type in CPU fixture");
        }
        auto * out = static_cast<float *>(tensor->data);
        const size_t count = bytes / sizeof(float);
        if (name == "token_embd.weight") {
            if (tensor->ne[0] != kHidden || tensor->ne[1] != kVocab) {
                throw std::runtime_error("token embedding shape mismatch");
            }
            std::fill(out, out + count, 1.0f);
        } else if (name == "output.weight") {
            if (count != weights.output.size()) throw std::runtime_error("LM head shape mismatch");
            std::memcpy(out, weights.output.data(), bytes);
        } else if (name == "output_norm.weight" || has_suffix(name, ".attn_norm.weight") ||
                   has_suffix(name, ".ffn_norm.weight")) {
            std::fill(out, out + count, 1.0f);
        } else if (name == "blk.0.ffn_gate_inp.weight" ||
                   name == "blk.1.ffn_gate_inp.weight") {
            const size_t layer = name[4] == '0' ? 0 : 1;
            if (tensor->ne[0] != kHidden || tensor->ne[1] != kExperts) {
                throw std::runtime_error("Qwen2MoE router shape mismatch");
            }
            for (int64_t expert = 0; expert < kExperts; ++expert) {
                out[static_cast<size_t>(expert * kHidden)] = -1.0f;
            }
            out[3 * kHidden] = layer == 0 ? 0.0f : 0.1f;
            out[40 * kHidden] = layer == 0 ? 0.1f : 0.0f;
        } else if (name == "blk.0.ffn_gate_inp_shexp.weight" ||
                   name == "blk.1.ffn_gate_inp_shexp.weight") {
            std::fill(out, out + count, 1.0f); // keep silu(x)/x away from its 0/0 point
        }
    } catch (const std::exception & e) {
        weights.callback_failed = true;
        weights.callback_error = e.what();
        if (tensor != nullptr && tensor->data != nullptr) {
            std::memset(tensor->data, 0, ggml_nbytes(tensor));
        }
    }
}

bool make_batch(void * userdata, const llama_ubatch & ubatch,
                llm_graph_type graph_type, uint32_t expected_layers,
                RouteBatch & batch) {
    const auto & capture = *static_cast<const RunCapture *>(userdata);
    if (ubatch.n_tokens != 1 || graph_type != LLM_GRAPH_TYPE_DEFAULT ||
        expected_layers != kLayers) return false;
    batch.identity = BatchIdentity{"r693-public-decode-v10", "qwen2moe-synthetic-32x41",
                                   "r693-routed-layer-plan-v10", 0, 1,
                                   capture.batch, capture.attempt};
    batch.owner_by_expert.assign(kExperts, -1);
    batch.owner_by_expert[3] = 0;
    batch.owner_by_expert[40] = 0;
    return true;
}

RouteStatus collect(const RouteBatch & batch, const float * activations,
                    size_t activation_count, const float * slots, size_t slot_count,
                    float * output, size_t output_count, void * userdata) {
    auto & capture = *static_cast<RunCapture *>(userdata);
    ++capture.collect_calls;
    const uint32_t layer = batch.identity.layer;
    if (activations == nullptr || slots == nullptr || output == nullptr ||
        capture.rust_session == nullptr || batch.hidden != kHidden || batch.slots != 2 ||
        batch.tokens != 1 || activation_count != kHidden || slot_count != 2 * kHidden ||
        output_count != kHidden || batch.expert_ids.size() != 2 || batch.weights.size() != 2 ||
        batch.owner_kinds != std::vector<int32_t>{0, 0} ||
        batch.identity.run != "r693-public-decode-v10" || layer >= kLayers ||
        batch.identity.batch != capture.batch || batch.identity.attempt != capture.attempt) {
        return RouteStatus::invalid_batch;
    }
    capture.experts[layer] = batch.expert_ids;
    capture.weights[layer] = batch.weights;
    capture.router_input[layer].assign(activations, activations + activation_count);
    if (capture.reject_layer == static_cast<int32_t>(layer)) return RouteStatus::provider_failure;
    std::vector<double> staged(output_count, std::numeric_limits<double>::quiet_NaN());
    capture.rust_status[layer] = vf_r693_run_routed_layer(
        capture.rust_session, batch.identity.run.c_str(), batch.identity.model.c_str(),
        batch.identity.plan.c_str(), batch.identity.layer, batch.identity.epoch,
        batch.identity.batch, batch.identity.attempt, batch.hidden, batch.slots,
        batch.tokens, batch.expert_ids.data(), batch.expert_ids.size(),
        batch.weights.data(), batch.weights.size(), activations, activation_count,
        staged.data(), staged.size());
    if (capture.rust_status[layer] != 0 ||
        std::any_of(staged.begin(), staged.end(), [](double x) { return !std::isfinite(x); })) {
        return RouteStatus::provider_failure;
    }
    capture.output[layer].resize(output_count);
    for (size_t i = 0; i < output_count; ++i) {
        capture.output[layer][i] = static_cast<float>(staged[i]);
        if (!std::isfinite(capture.output[layer][i])) return RouteStatus::invalid_result;
        output[i] = capture.output[layer][i];
    }
    return RouteStatus::ready;
}

struct ModelOwner {
    FixtureMetadata metadata;
    FixtureWeights weights;
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    std::unique_ptr<Qwen2MoELlamaRouteBridge> bridge;

    explicit ModelOwner(bool use_vf_hooks, RunCapture * capture) {
        set_metadata(metadata);
        weights = make_weights();
        auto model_params = llama_model_default_params();
        ggml_backend_dev_t devices[] = {ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr};
        if (devices[0] == nullptr) throw std::runtime_error("CPU backend device is unavailable");
        model_params.devices = devices;
        model_params.n_gpu_layers = 0;
        model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        model_params.progress_callback = nullptr;
        model = llama_model_init_from_user(metadata.gguf, fill_tensor, &weights, model_params);
        if (model == nullptr) throw std::runtime_error("pinned llama_model_init_from_user failed");
        if (weights.callback_failed) {
            throw std::runtime_error("synthetic model tensor callback failed: " + weights.callback_error);
        }

        auto context_params = llama_context_default_params();
        context_params.n_ctx = 32;
        context_params.n_batch = 1;
        context_params.n_ubatch = 1;
        context_params.n_seq_max = 1;
        context_params.n_threads = 1;
        context_params.n_threads_batch = 1;
        context_params.embeddings = false;
        context_params.offload_kqv = false;
        context_params.op_offload = false;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        context_params.type_k = GGML_TYPE_F32;
        context_params.type_v = GGML_TYPE_F32;
        context_params.cb_eval = nullptr;
        context = llama_init_from_model(model, context_params);
        if (context == nullptr) throw std::runtime_error("llama_init_from_model failed");

        if (use_vf_hooks) {
            bridge = std::make_unique<Qwen2MoELlamaRouteBridge>(make_batch, capture, collect, capture);
            auto * internal = reinterpret_cast<llama_context *>(context);
            if (!bridge->attach(*internal)) throw std::runtime_error("VoltaFabric route hook attach failed");
        }
    }

    ~ModelOwner() {
        if (bridge != nullptr) bridge->detach();
        if (context != nullptr) llama_free(context);
        if (model != nullptr) llama_model_free(model);
    }
};

int32_t decode_one(llama_context * context, llama_token token, llama_pos position) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    if (batch.token == nullptr || batch.pos == nullptr || batch.n_seq_id == nullptr ||
        batch.seq_id == nullptr || batch.logits == nullptr) {
        llama_batch_free(batch);
        throw std::runtime_error("llama_batch_init failed");
    }
    batch.n_tokens = 1;
    batch.token[0] = token;
    batch.pos[0] = position;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
    const int32_t status = llama_decode(context, batch);
    llama_batch_free(batch);
    return status;
}

void append_u32(std::vector<uint8_t> & out, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<uint8_t>(value >> shift));
    }
}

void append_u64(std::vector<uint8_t> & out, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        out.push_back(static_cast<uint8_t>(value >> shift));
    }
}

void append_string(std::vector<uint8_t> & out, const std::string & value) {
    append_u64(out, value.size());
    out.insert(out.end(), value.begin(), value.end());
}

std::vector<uint8_t> source_q8_payload(size_t experts, int8_t first_code) {
    std::vector<uint8_t> bytes(experts * kExpertBytes, 0);
    for (size_t expert = 0; expert < experts; ++expert) {
        for (size_t row = 0; row < static_cast<size_t>(kHidden); ++row) {
            const size_t offset = expert * kExpertBytes + row * kQ8RowBytes;
            bytes[offset] = 0;
            bytes[offset + 1] = 0x3c;
            bytes[offset + 2 + row] = static_cast<uint8_t>(first_code + expert);
        }
    }
    return bytes;
}

std::vector<uint8_t> source_gguf_bytes() {
    std::vector<std::string> names;
    std::vector<std::vector<uint8_t>> tensors;
    for (size_t layer = 0; layer < kLayers; ++layer) {
        const int8_t base = static_cast<int8_t>(1 + layer);
        names.push_back("blk." + std::to_string(layer) + ".ffn_gate_exps.weight");
        tensors.push_back(source_q8_payload(4, base));
        names.push_back("blk." + std::to_string(layer) + ".ffn_up_exps.weight");
        tensors.push_back(source_q8_payload(4, static_cast<int8_t>(base + 1)));
        names.push_back("blk." + std::to_string(layer) + ".ffn_down_exps.weight");
        tensors.push_back(source_q8_payload(4, base));
    }
    std::vector<uint8_t> out{'G', 'G', 'U', 'F'};
    append_u32(out, 3);
    append_u64(out, tensors.size());
    append_u64(out, 5);
    append_string(out, "general.architecture");
    append_u32(out, 8); // GGUF_TYPE_STRING
    append_string(out, "qwen2moe");
    append_string(out, "qwen2moe.block_count");
    append_u32(out, 4); append_u32(out, static_cast<uint32_t>(kLayers));
    append_string(out, "qwen2moe.embedding_length");
    append_u32(out, 4); append_u32(out, static_cast<uint32_t>(kHidden));
    append_string(out, "qwen2moe.expert_count");
    append_u32(out, 4); append_u32(out, 4);
    append_string(out, "qwen2moe.expert_feed_forward_length");
    append_u32(out, 4); append_u32(out, static_cast<uint32_t>(kHidden));
    uint64_t relative = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        append_string(out, names[i]);
        append_u32(out, 3);
        append_u64(out, kHidden); append_u64(out, kHidden); append_u64(out, 4);
        append_u32(out, 8); // GGML_TYPE_Q8_0
        append_u64(out, relative);
        relative += tensors[i].size();
    }
    while (out.size() % 32 != 0) out.push_back(0);
    for (const auto & tensor : tensors) out.insert(out.end(), tensor.begin(), tensor.end());
    return out;
}

bool write_file(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file || !file.write(reinterpret_cast<const char *>(bytes.data()),
                             static_cast<std::streamsize>(bytes.size()))) return false;
    file.flush();
    return static_cast<bool>(file);
}

bool route_is_expected(const RunCapture & capture, size_t layer) {
    if (layer >= kLayers || capture.experts[layer].size() != 2 ||
        capture.weights[layer].size() != 2 || capture.router_input[layer].size() != kHidden ||
        !std::isfinite(capture.weights[layer][0]) || !std::isfinite(capture.weights[layer][1]) ||
        !std::all_of(capture.router_input[layer].begin(), capture.router_input[layer].end(),
                     [](float value) { return std::isfinite(value); })) return false;
    const double gate_input = capture.router_input[layer][0];
    const double other = std::exp(-gate_input);
    const double expert_three = std::exp(layer == 0 ? 0.0 : 0.1 * gate_input);
    const double expert_forty = std::exp(layer == 0 ? 0.1 * gate_input : 0.0);
    const double denominator = 39.0 * other + expert_three + expert_forty;
    for (size_t slot = 0; slot < capture.experts[layer].size(); ++slot) {
        const double expected = capture.experts[layer][slot] == 3 ? expert_three / denominator
                               : capture.experts[layer][slot] == 40 ? expert_forty / denominator
                                                                    : -1.0;
        if (expected < 0.0 || std::fabs(capture.weights[layer][slot] - expected) > 2.0e-6) {
            return false;
        }
    }
    return capture.experts[layer][0] != capture.experts[layer][1] &&
           ((capture.experts[layer][0] == 40 && capture.experts[layer][1] == 3) ||
            (capture.experts[layer][0] == 3 && capture.experts[layer][1] == 40));
}

std::vector<float> last_logits(llama_context * context);
llama_token sample_greedy(llama_sampler * sampler, llama_context * context);

int32_t run_public_generation(void * rust_session, bool use_hooks, int32_t reject_layer,
                              size_t token_steps, std::vector<llama_token> & generated,
                              std::vector<std::vector<float>> & logits_history) {
    RunCapture capture;
    capture.rust_session = rust_session;
    capture.reject_layer = reject_layer;
    ModelOwner owner(use_hooks, &capture);
    llama_sampler * sampler = llama_sampler_init_greedy();
    if (sampler == nullptr) throw std::runtime_error("llama greedy sampler init failed");
    std::unique_ptr<llama_sampler, void (*)(llama_sampler *)> sampler_guard(sampler, llama_sampler_free);

    llama_token input = 0;
    for (size_t step = 0; step < token_steps; ++step) {
        capture.batch = (reject_layer >= 0 ? 9901 : 901) + step;
        capture.attempt = (reject_layer >= 0 ? 99001 : 9001) + step;
        capture.rust_status.fill(-1);
        const int32_t status = decode_one(owner.context, input, static_cast<llama_pos>(step));
        if (reject_layer >= 0) {
            // A refused call must never be sampled or published, including stale logits.
            if (status == 0 || capture.collect_calls != kLayers ||
                capture.rust_status[0] != 0 || capture.rust_status[1] != -1 ||
                !generated.empty() || !logits_history.empty()) {
                throw std::runtime_error("late-layer refusal exposed a partial decode or sample");
            }
            return status;
        }
        if (status != 0) throw std::runtime_error("public llama_decode returned " + std::to_string(status));
        bool layers_ready = !use_hooks || capture.collect_calls == (step + 1) * kLayers;
        for (size_t layer = 0; use_hooks && layer < kLayers; ++layer) {
            layers_ready = layers_ready && route_is_expected(capture, layer) &&
                           capture.rust_status[layer] == 0 &&
                           capture.output[layer].size() == kHidden;
        }
        if (use_hooks && !layers_ready) {
            std::cerr << "R693 route check failed: collector_calls=" << capture.collect_calls
                      << " expected_calls=" << (step + 1) * kLayers;
            for (size_t layer = 0; layer < kLayers; ++layer) {
                std::cerr << " layer=" << layer << " rust_status=" << capture.rust_status[layer]
                          << " output_count=" << capture.output[layer].size() << " experts=";
                for (const int32_t expert : capture.experts[layer]) std::cerr << expert << ',';
                std::cerr << " weights=";
                for (const float weight : capture.weights[layer]) std::cerr << weight << ',';
            }
            std::cerr << '\n';
            throw std::runtime_error("VoltaFabric Sessions did not commit every Qwen2MoE layer route");
        }
        auto logits = last_logits(owner.context);
        if (!std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); })) {
            std::cerr << "R693 non-finite logits:";
            for (float x : logits) std::cerr << ' ' << x;
            std::cerr << '\n';
            throw std::runtime_error("public llama_decode produced non-finite logits");
        }
        logits_history.push_back(logits);
        const llama_token token = sample_greedy(sampler, owner.context);
        if (token < 0 || token >= kVocab) throw std::runtime_error("greedy sampler returned an invalid ID");
        llama_sampler_accept(sampler, token);
        generated.push_back(token);
        input = token;
        if (use_hooks) {
            std::cout << "R693 route step=" << step << " batch=" << capture.batch
                      << " attempt=" << capture.attempt;
            for (size_t layer = 0; layer < kLayers; ++layer) {
                std::cout << " layer" << layer << "_experts="
                          << capture.experts[layer][0] << ',' << capture.experts[layer][1]
                          << "_weights_sum=" << capture.weights[layer][0] + capture.weights[layer][1];
            }
            std::cout << '\n';
        }
    }
    return 0;
}

std::vector<float> last_logits(llama_context * context) {
    float * logits = llama_get_logits_ith(context, -1);
    if (logits == nullptr) throw std::runtime_error("public logits output is unavailable");
    return {logits, logits + kVocab};
}

llama_token sample_greedy(llama_sampler * sampler, llama_context * context) {
    return llama_sampler_sample(sampler, context, -1);
}

} // namespace

int main() {
    try {
        const char * output_directory = std::getenv("VF_TEST_OUTPUT_DIR");
        const char * rank_session_binary = std::getenv("VF_RANK_SESSION_BIN");
        if (output_directory == nullptr || rank_session_binary == nullptr ||
            *rank_session_binary == '\0') {
            std::cerr << "R693 needs VF_TEST_OUTPUT_DIR and VF_RANK_SESSION_BIN\n";
            return 2;
        }
        std::filesystem::create_directories(output_directory);
        const auto source_path = (std::filesystem::path(output_directory) /
                                  "r693-source-qwen2moe-two-layer-4experts.gguf").string();
        const auto ledger_root = (std::filesystem::path(output_directory) / "rust-ledger").string();
        std::filesystem::create_directories(ledger_root);
        const auto source_bytes = source_gguf_bytes();
        if (!write_file(source_path, source_bytes)) throw std::runtime_error("could not write R693 Session GGUF");
        llama_backend_init();
        std::unique_ptr<void, void (*)(void *)> rust_session(
            vf_r693_open(source_path.c_str(), rank_session_binary, ledger_root.c_str()), vf_r693_close);
        if (!rust_session) throw std::runtime_error("could not initialize R693 Rust rank Sessions");

        std::vector<llama_token> donor_tokens;
        std::vector<llama_token> vf_tokens;
        std::vector<std::vector<float>> donor_logits;
        std::vector<std::vector<float>> vf_logits;
        if (run_public_generation(rust_session.get(), false, -1, 3, donor_tokens, donor_logits) != 0 ||
            run_public_generation(rust_session.get(), true, -1, 3, vf_tokens, vf_logits) != 0) {
            throw std::runtime_error("public generation fixture did not finish");
        }
        double max_logit_error = 0.0;
        if (donor_tokens != vf_tokens || donor_tokens.size() != 3 ||
            donor_logits.size() != vf_logits.size()) {
            throw std::runtime_error("pinned donor and VoltaFabric token sequences differ");
        }
        for (size_t step = 0; step < donor_logits.size(); ++step) {
            if (donor_logits[step].size() != vf_logits[step].size()) {
                throw std::runtime_error("public logits extent differs");
            }
            for (size_t i = 0; i < donor_logits[step].size(); ++i) {
                max_logit_error = std::max(max_logit_error,
                    std::fabs(static_cast<double>(donor_logits[step][i]) - vf_logits[step][i]));
            }
        }
        if (max_logit_error > 1.0e-6) {
            throw std::runtime_error("pinned donor/VoltaFabric public logits exceed 1e-6 tolerance");
        }

        // Use a fresh Session identity and model context; on refusal do not read logits or sample.
        std::vector<llama_token> refused_tokens;
        std::vector<std::vector<float>> refused_logits;
        const int32_t rejected_status = run_public_generation(
            rust_session.get(), true, 1, 1, refused_tokens, refused_logits);
        if (rejected_status == 0 || !refused_tokens.empty() || !refused_logits.empty()) {
            throw std::runtime_error("public llama_decode exposed a partial/refused route");
        }
        std::cout << "R693 public llama_decode CPU fixture passed\n"
                  << "steps=" << donor_tokens.size() << " donor_tokens=";
        for (auto token : donor_tokens) std::cout << token << ',';
        std::cout << " vf_tokens=";
        for (auto token : vf_tokens) std::cout << token << ',';
        std::cout << " max_abs_logit_error=" << max_logit_error
                  << " layers_per_decode=" << kLayers
                  << " decode_reject_status=" << rejected_status
                  << " sampled_after_late_layer_reject=false\n";
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "R693 exception: " << e.what() << '\n';
        return 1;
    }
}
