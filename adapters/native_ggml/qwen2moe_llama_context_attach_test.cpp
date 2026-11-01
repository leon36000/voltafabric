#include "llama-context.h"
#include "llama-graph.h"
#include "llama-model.h"
#include "llama.h"
#include "qwen2moe_llama_route_bridge.hpp"

#include <iostream>

namespace {
using namespace vf::ggml;

bool make_batch(void *, const llama_ubatch &, llm_graph_type, uint32_t, RouteBatch &) {
    return false;
}

RouteStatus collect(const RouteBatch &, const float *, size_t, const float *, size_t,
                    float *, size_t, void *) {
    return RouteStatus::provider_failure;
}

struct ContextFixture {
    struct TestModel final : llama_model {
        explicit TestModel(const llama_model_params &params) : llama_model(params) {}
        void load_stats(llama_model_loader &) override {}
        void load_hparams(llama_model_loader &) override {}
        void load_vocab(llama_model_loader &) override {}
        bool load_tensors(llama_model_loader &) override { return false; }
        void load_arch_hparams(llama_model_loader &) override {}
        void load_arch_tensors(llama_model_loader &) override {}
        std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override {
            return nullptr;
        }
    } model;
    llama_context context;

    explicit ContextFixture(llm_arch arch)
        : model(llama_model_default_params()),
          context(init_model(model, arch), llama_context_default_params()) {}

private:
    static const llama_model & init_model(llama_model & model, llm_arch arch) {
        model.arch = arch;
        model.hparams.vocab_only = true;
        model.hparams.n_ctx_train = 256;
        model.hparams.n_layer_all = 1;
        return model;
    }
};

bool test_architecture_and_configuration_gate() {
    ContextFixture wrong_arch(LLM_ARCH_LLAMA);
    Qwen2MoELlamaRouteBridge bridge(make_batch, nullptr, collect, nullptr);
    if (bridge.attach(wrong_arch.context) || wrong_arch.context.get_vf_route_hooks() != nullptr) {
        return false;
    }

    Qwen2MoELlamaRouteBridge invalid(nullptr, nullptr, nullptr, nullptr);
    ContextFixture qwen(LLM_ARCH_QWEN2MOE);
    return !invalid.attach(qwen.context) && qwen.context.get_vf_route_hooks() == nullptr;
}

bool test_hook_ownership_and_detachment() {
    ContextFixture qwen(LLM_ARCH_QWEN2MOE);
    llm_vf_route_hooks foreign{};
    {
        Qwen2MoELlamaRouteBridge bridge(make_batch, nullptr, collect, nullptr);
        if (!bridge.attach(qwen.context) || qwen.context.get_vf_route_hooks() != bridge.hooks() ||
            bridge.attach(qwen.context)) {
            return false;
        }
        Qwen2MoELlamaRouteBridge contender(make_batch, nullptr, collect, nullptr);
        if (contender.attach(qwen.context)) return false;

        qwen.context.set_vf_route_hooks(&foreign);
        if (bridge.detach() || qwen.context.get_vf_route_hooks() != &foreign) return false;
        qwen.context.set_vf_route_hooks(nullptr);
        if (!bridge.attach(qwen.context)) return false;
    }
    return qwen.context.get_vf_route_hooks() == nullptr;
}
} // namespace

int main() {
    if (!test_architecture_and_configuration_gate()) {
        std::cerr << "Qwen2MoE context architecture/configuration gate failed\n";
        return 1;
    }
    if (!test_hook_ownership_and_detachment()) {
        std::cerr << "Qwen2MoE context hook ownership/detachment failed\n";
        return 1;
    }
    std::cout << "Qwen2MoE live llama_context: arch/config gate and hook ownership pass\n";
    return 0;
}
