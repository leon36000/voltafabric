# R682 — live `process_ubatch` route-hook fixture

## Scope

R682 makes a narrow CPU integration seam reproducible through the pinned donor's live `llama_context::process_ubatch`. It extends the R681 attachment/callback work; it does not replay R681's terminal coordinator/worker receipt.

## Reused and adapted

- Donor: `ggml-org/llama.cpp` revision `3cf03257f219afbe7334045ff7c6a06ac68c627d`, MIT, with the private R679/R680 hook and bridge adaptations already recorded on this branch.
- The new test constructs a minimal internal `llama_model` reporting `LLM_ARCH_QWEN2MOE`, a live donor `llama_context`, and a one-layer synthetic `llm_graph_context`.
- The synthetic graph creates input tensors, invokes the existing `Qwen2MoELlamaRouteBridge` hook, and is dispatched by the donor's actual `llama_context::process_ubatch` implementation. The callback checks complete run/model/plan/layer/epoch/batch/attempt identity, the dynamic expert-to-owner projection, original activations, final route weight, and one collector call.
- The collector fixture applies weight `0.5` once to raw output `[10,20]` and checks `[5,10]`.
- A bounded runner checks the exact donor revision, an unchanged `ggml` dependency and the expected built shared libraries before compiling the fixture. An opt-in CTest entry registers this runner when the R680 bridge is enabled.

## Verification

- Direct runner: compile with `g++ -std=c++17 -Wall -Wextra -Werror`; exit 0.
- CMake configure with CPU-only pinned GGML settings: passed.
- Focused CTest `qwen2moe_llama_process_ubatch`: 1/1 passed, 1.61 s.
- Assertion printed by the test: live `process_ubatch` callback once, one dynamic route, one weight application, output `[5,10]`.
- Source and test artifacts are listed in `SHA256SUMS`; GitLab readback is required after publication.

## Limits

This proves the hook callback can be reached while the pinned donor's live `llama_context::process_ubatch` drives a synthetic Qwen2MoE graph. The graph's test subclass invokes the hook; this does not execute the donor's production `build_moe_ffn` routed-expert graph. No model checkpoint, tokenizer, token loop, actual generation, Rust coordinator/worker process, production R53, GPU/CUDA graph, inter-PC/RDMA or performance was tested. The previous R681 one-shot receipt remains terminal and was not replayed.

## Next

R683: exercise the hook at the donor's production Qwen2MoE `build_moe_ffn` seam with a bounded CPU fixture, or document a concrete blocker and identify the smallest compatible checkpoint/model path. Keep route identity complete, coefficients single-applied, shared/residual donor paths intact, and host publication atomic. No automatic relaunch.
