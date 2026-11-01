# R684 — GGUF v3 weights through pinned Qwen2MoE build_moe_ffn

## Change

Adds an opt-in CPU CTest that creates a 544-byte, one-layer Qwen2MoE-shaped GGUF v3 file with two F32 expert tensors, then reads it back through the pinned llama.cpp GGUF writer/parser. The fixture includes typed architecture, layer, hidden, FFN and expert-count metadata. Before graph construction, the test checks tensor names, F32 types, exact [2,2,2] dimensions, 32-byte tensor sizes, and that both absolute tensor spans remain within the file. It then copies the parser-loaded tensor bytes into the synthetic donor graph inputs.

This drives the actual llama.cpp `llm_graph_context::build_moe_ffn` inside the live `llama_context::process_ubatch` path added in R683. The VoltaFabric route bridge receives expert 1, the donor's raw down result, and its final post-gating coefficient. A second identified call forces the collector to return provider failure and verifies that llama_context returns no result and a failed status.

## CPU evidence

- Pinned llama.cpp revision: `3cf03257f219afbe7334045ff7c6a06ac68c627d` (MIT).
- C++17 compile with `-Wall -Wextra -Werror` against the pinned shared libraries: PASS.
- Direct isolated runner: PASS.
- CMake registration probe and CTest: 1/1 PASS (about 1.7 s).
- Parsed GGUF v3: 2 tensors, 544 bytes; the up/down tensor metadata and file spans were accepted.
- Successful donor/collector path: raw donor result `[6,20]`, coefficient `0.5` exactly once, final output `[3,10]`.
- Rejection path: collector failure yielded a null graph result, `GGML_STATUS_FAILED`, detached hooks, and no published output.

## Reuse and boundary

This uses the pinned llama.cpp GGUF writer/reader and its existing `build_moe_ffn`/live `process_ubatch` graph seam. The test is checked in as an optional CPU test; it does not vendor the donor. The existing VoltaFabric transaction/route bridge applies the final coefficient once and rejects a failed collector transaction.

R659's separate Rust test generates a Q8_0 GGUF fixture at runtime with hidden/FFN width 32. R684 uses a new, smaller F32 GGUF v3 fixture with hidden/FFN width 2; it does not claim to reuse the R659 Q8_0 payload or prove Q8_0 execution at the donor seam. The test supplies logits and the selected expert directly, so architecture top-k selection and the gate/up/SwiGLU composition are also outside this result.

## Limits

No production checkpoint, tokenizer, real Qwen2MoE model loader, full model layer, token loop, complete generation, Rust coordinator/worker integration, GPU/CUDA graph, inter-PC/RDMA, shared production R53 reservation, or performance comparison was exercised. The fixture is diagnostic evidence for the GGUF-to-donor-graph boundary only.

## Next

R685 should use the existing R659 Q8_0 GGUF fixture generator and connect its named gate/up/down tensor layout to the pinned donor graph, keeping the route identity and single-coefficient/atomic-refusal checks. Keep the fixture small and CPU-only; do not download a production checkpoint.

