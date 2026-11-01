#include "qwen2moe_llama_route_bridge.hpp"

#include <utility>

namespace vf::ggml {

Qwen2MoELlamaRouteBridge::Qwen2MoELlamaRouteBridge(
        Qwen2MoERouteBatchFactory factory, void * factory_userdata,
        Collector collector, void * collector_userdata)
    : factory_(factory), factory_userdata_(factory_userdata),
      collector_(collector), collector_userdata_(collector_userdata) {
    hooks_.userdata = this;
    hooks_.begin = &Qwen2MoELlamaRouteBridge::begin;
    hooks_.append_qwen2moe = &Qwen2MoELlamaRouteBridge::append_qwen2moe;
    hooks_.finish = &Qwen2MoELlamaRouteBridge::finish;
    hooks_.abort = &Qwen2MoELlamaRouteBridge::abort;
}

Qwen2MoELlamaRouteBridge::~Qwen2MoELlamaRouteBridge() {
    if (attached_context_ != nullptr) {
        detach();
    }
}

bool Qwen2MoELlamaRouteBridge::attach(llama_context & context) {
    if (attached_context_ != nullptr || context.get_vf_route_hooks() != nullptr) {
        return false;
    }
    context.set_vf_route_hooks(&hooks_);
    attached_context_ = &context;
    return true;
}

bool Qwen2MoELlamaRouteBridge::detach() {
    if (attached_context_ == nullptr) {
        return true;
    }
    if (attached_context_->get_vf_route_hooks() != &hooks_) {
        attached_context_ = nullptr;
        return false;
    }
    attached_context_->set_vf_route_hooks(nullptr);
    attached_context_ = nullptr;
    return true;
}

bool Qwen2MoELlamaRouteBridge::begin(void * userdata, const llama_ubatch & ubatch,
                                     llm_graph_type graph_type, uint32_t expected_layers) {
    auto & self = *static_cast<Qwen2MoELlamaRouteBridge *>(userdata);
    if (self.invocation_ != nullptr) {
        self.failed_ = true;
        return false;
    }
    self.failed_ = false;
    if (self.factory_ == nullptr || self.collector_ == nullptr || expected_layers == 0) {
        self.failed_ = true;
        return false;
    }
    try {
        RouteBatch batch;
        if (!self.factory_(self.factory_userdata_, ubatch, graph_type, expected_layers, batch)) {
            self.failed_ = true;
            return false;
        }
        auto invocation = std::make_unique<Qwen2MoERouteInvocation>(
            std::move(batch), expected_layers, self.collector_, self.collector_userdata_);
        if (!invocation->ready()) {
            self.failed_ = true;
            return false;
        }
        self.invocation_ = std::move(invocation);
        return true;
    } catch (...) {
        self.failed_ = true;
        self.invocation_.reset();
        return false;
    }
}

ggml_tensor * Qwen2MoELlamaRouteBridge::append_qwen2moe(
        void * userdata, ggml_context * ctx, int32_t layer,
        ggml_tensor * activations, ggml_tensor * selected_experts,
        ggml_tensor * final_weights, ggml_tensor * raw_expert_outputs) {
    auto & self = *static_cast<Qwen2MoELlamaRouteBridge *>(userdata);
    if (self.failed_ || self.invocation_ == nullptr || layer < 0) {
        self.failed_ = true;
        return nullptr;
    }
    try {
        ggml_tensor * result = self.invocation_->append_layer(
            ctx, static_cast<uint32_t>(layer), activations, selected_experts,
            final_weights, raw_expert_outputs);
        if (result == nullptr) {
            self.failed_ = true;
        }
        return result;
    } catch (...) {
        self.failed_ = true;
        self.invocation_.reset();
        return nullptr;
    }
}

bool Qwen2MoELlamaRouteBridge::finish(void * userdata, bool ggml_success) {
    auto & self = *static_cast<Qwen2MoELlamaRouteBridge *>(userdata);
    if (self.failed_ || self.invocation_ == nullptr) {
        self.invocation_.reset();
        self.failed_ = true;
        return false;
    }
    const bool accepted = self.invocation_->accepts(
        ggml_success ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED);
    self.invocation_.reset();
    self.failed_ = !accepted;
    return accepted;
}

void Qwen2MoELlamaRouteBridge::abort(void * userdata) {
    auto & self = *static_cast<Qwen2MoELlamaRouteBridge *>(userdata);
    self.invocation_.reset();
    self.failed_ = true;
}

} // namespace vf::ggml
