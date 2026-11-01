# R679 — llama.cpp Qwen2MoE route-hook seam

## Donor and scope

Donor: `ggml-org/llama.cpp`, pinned commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`, MIT. The patch is based on that exact source tree and is stored here as `llama.cpp-r679.patch`; it is a proposed source adaptation for VoltaFabric, not a vendored or independently licensed llama.cpp tree.

The patch adds an internal opt-in callback at `llama_context::process_ubatch` and `llm_graph_context::build_moe_ffn`. It opens one route transaction per actual ubatch, disables graph reuse for the routed invocation, passes Qwen2MoE's original activation, dynamic selected expert IDs, final post-gating coefficients, and raw per-expert down outputs to the callback, and accepts only an F32 `[n_embd, n_tokens, 1, 1]` routed aggregate. Qwen2MoE's shared expert and residual remain on the donor path. Callback completion is a host-side publication gate; failed/missing layers or graph-compute failures call idempotent abort and return no successful graph result.

The route hook is not used by graph reservation/warmup (`graph_params` defaults to no hook) and any active route invocation bypasses graph reuse. This keeps per-batch identity and dynamic routing outside captured/reused graph state.

## Verification

- `git diff --check`: passed on the pinned donor worktree.
- CPU-only CMake configuration succeeded with CUDA, Vulkan, Metal, tests, examples, server and curl disabled; HTTPS support is unavailable because OpenSSL is absent.
- `cmake --build .runs/llama-vf-route-r679-build --target llama -j2`: passed on AArch64 CPU; the `llama` library target compiled with the patch.
- `BUILD_TEST.txt` records the final incremental build exit status; `SHA256SUMS` covers the report, patch, and build record.

## Limits and next integration step

This is a compile-qualified callback seam, not a runtime integration test. It is not yet wired to the VoltaFabric Rust session or `Qwen2MoERouteInvocation` from R678, and no checkpoint or model generation was run. The hook ABI remains an internal C++ interface; the next step is a narrow bridge that owns the complete run/model/plan/layer/batch/epoch/attempt identity, maps the callback payload into the R678 per-layer invocation, and is tested with a small real donor graph/checkpoint before making generation claims. No GPU, CUDA graph, inter-PC/RDMA, production R53, or performance claim is made.
