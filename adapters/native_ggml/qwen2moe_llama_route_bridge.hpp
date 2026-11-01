#pragma once

#include "llama-context.h"
#include "llama-graph.h"
#include "qwen2moe_route_invocation.hpp"

#include <memory>

namespace vf::ggml {

using Qwen2MoERouteBatchFactory = bool (*)(void * userdata,
                                           const llama_ubatch & ubatch,
                                           llm_graph_type graph_type,
                                           uint32_t expected_layers,
                                           RouteBatch & batch);

// Adapts llama.cpp's pinned internal hook ABI to the VoltaFabric per-layer
// invocation. The factory must return the complete run/model/plan/epoch/batch/
// attempt identity plus owner_by_expert for each real process_ubatch call.
// Keep this bridge alive while attached; detach (or destroy it) before the
// llama_context so its hook pointer cannot dangle.
class Qwen2MoELlamaRouteBridge {
public:
    Qwen2MoELlamaRouteBridge(Qwen2MoERouteBatchFactory factory,
                             void * factory_userdata,
                             Collector collector,
                             void * collector_userdata);
    ~Qwen2MoELlamaRouteBridge();
    Qwen2MoELlamaRouteBridge(const Qwen2MoELlamaRouteBridge &) = delete;
    Qwen2MoELlamaRouteBridge & operator=(const Qwen2MoELlamaRouteBridge &) = delete;

    bool attach(llama_context & context);
    bool detach();
    const llm_vf_route_hooks * hooks() const { return &hooks_; }

private:
    static bool begin(void * userdata, const llama_ubatch & ubatch,
                      llm_graph_type graph_type, uint32_t expected_layers);
    static ggml_tensor * append_qwen2moe(void * userdata, ggml_context * ctx,
                                         int32_t layer, ggml_tensor * activations,
                                         ggml_tensor * selected_experts,
                                         ggml_tensor * final_weights,
                                         ggml_tensor * raw_expert_outputs);
    static bool finish(void * userdata, bool ggml_success);
    static void abort(void * userdata);

    Qwen2MoERouteBatchFactory factory_ = nullptr;
    void * factory_userdata_ = nullptr;
    Collector collector_ = nullptr;
    void * collector_userdata_ = nullptr;
    std::unique_ptr<Qwen2MoERouteInvocation> invocation_;
    llama_context * attached_context_ = nullptr;
    llm_vf_route_hooks hooks_{};
    bool failed_ = false;
};

} // namespace vf::ggml
