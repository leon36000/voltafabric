# R680 — bind llama.cpp route hooks to the VoltaFabric invocation

## Scope

R680 adapts the opt-in internal hook seam from R679 to the existing `Qwen2MoERouteInvocation` from R678. A per-ubatch factory supplies the full run/model/plan/epoch/batch/attempt identity and owner-by-expert projection. Each donor `begin` creates one fresh invocation; each routed Qwen2MoE layer forwards the donor activation, dynamic selected expert IDs, final post-gating coefficients, and raw expert outputs exactly once. The host `finish` accepts output only if every expected layer transaction is READY and GGML reports success. `abort` discards the invocation.

Attachment refuses to overwrite another context hook. Detachment clears only this bridge's hook. The bridge must be detached or destroyed before its `llama_context` so the callback userdata cannot outlive it.

## Reuse and adaptation

- Reuses the pinned llama.cpp internal callback ABI introduced by R679 and the VoltaFabric `Qwen2MoERouteInvocation` and collector seam introduced by R678.
- Adds a small context hook getter in `llama.cpp-r680.patch`, applied after `llama.cpp-r679.patch`; it supports safe hook ownership checks.
- Adds an optional CMake bridge target. It checks the exact llama.cpp commit, a clean GGML dependency, and the presence of the R679 hook declarations. It does not vendor or install llama.cpp.
- The CPU fixture invokes the same callback table and runs actual GGML route nodes for two layers. It checks complete identity propagation, selected IDs, dynamic owner mapping, unchanged final coefficients, aggregate `[17.4, 19.4]`, and refusal when a layer is missing.

## Verification

- AArch64 CPU build of the optional bridge and fixture succeeded with C++17, `-Wall -Wextra -Werror`; CUDA, Vulkan and Metal were disabled.
- Six focused CTests passed: native provider, route graph transaction, dynamic route capture, Qwen2MoE route invocation, llama hook bridge, and VFR3 route graph. The process-level VFR3 bridge test was skipped because no external process fixture paths were supplied.
- The pinned llama.cpp `llama` library rebuilt after the R680 getter was added; donor `git diff --check` passed.
- The R680 getter patch was validated against the exact R679-patched header state.

## Limits and next step

This qualifies a CPU callback-table fixture linked against the pinned donor headers, not a real `llama_context::process_ubatch` execution. Attachment/detachment through a live context, runtime callback delivery from a real model graph, the Rust Session/collector path, any checkpoint, complete generation, GPU/CUDA graph, inter-PC/RDMA, production R53, and performance remain unqualified. R681 should exercise attachment and callback delivery through a donor-side `llama_context`/Qwen2MoE runtime fixture before loading a checkpoint.
