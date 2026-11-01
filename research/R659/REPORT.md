# R659 — GGUF MoE layout and rank-local CPU FFN fixture

## Change

Added a metadata-only resolver for the pinned llama.cpp GGUF expert-FFN tensor convention. It reads the architecture name and typed `block_count`, `embedding_length`, `expert_count`, and `expert_feed_forward_length` metadata, then validates every layer's exact gate/up/down tensor names, physical dimensions, and file spans. It accepts either separate gate and up tensors or one fused gate/up tensor, and rejects incomplete or ambiguous combinations. The fused form is resolved as a single tensor; this round does not split its halves for execution.

The R658 GGUF reader now exposes typed metadata and tensor lookup through its owned `no_alloc` index. Rank-local checkpoint loading uses a separate explicit map from VoltaFabric `(expert, local)` residents to checkpoint expert indices. The loader seeks and streams only the selected expert slices, in resident order, into identified CPU Sessions; it does not stage the full GGUF tensor.

## CPU evidence

The new integration fixture is synthetic Qwen2Moe-shaped GGUF v3 with Q8_0 weights, one layer, hidden width 32, FFN width 32, and two checkpoint experts. Three local rank Sessions load the gate, up, and down tensors using the explicit checkpoint mapping `[expert 40 -> file expert 1, expert 3 -> file expert 0]`. Gate and up projections are combined with the fixture's SiLU activation; the route weights 0.25 and 0.75 are applied once at the down projection. The resulting output exactly matches a separate scalar Q8_0 decoder/matvec oracle with FP16 round/decode at the VoltaFabric Collector boundary. A separate metadata fixture verifies that the resolver accepts rectangular hidden/FFN shapes; Session execution with those shapes is not tested.

Verification on the isolated Linux CPU harness:

- `gguf_moe_layout`: 4/4 passed (split names/shapes, fused name/shape, rectangular shape resolution, rank-local gate/up/down Session path and oracle).
- `native_matrix_session`: 5/5 passed, including explicit non-contiguous expert selection, resident order, oversized streaming, and atomic short-source refusal.
- The repository-wide test target is not available in this source snapshot because historical included fixtures and binaries are absent. The two focused targets were built offline with a temporary narrowed Cargo manifest; `Cargo.toml` and `Cargo.lock` were restored byte-for-byte. The only compiler warning is the existing unused `CapturedQsa` fields in `src/sc117_layer_chain.rs`.

## Reuse and provenance

This round reuses the pinned llama.cpp/GGML GGUF metadata parser at commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` (MIT) for typed metadata and tensor span discovery. It also composes the existing VoltaFabric expert-slice loader, resident grouped-matrix Session, CPU routing, and ordinary FP16-decoded/f64 Collector. No llama.cpp generation loop, scheduler, router, or checkpoint runtime is used. No code or runtime from vLLM, TensorRT-LLM, SGLang, Maya, or DeepInfer is newly integrated in this round; donor quality is not inherited.

## Limits

The checkpoint and all weights are synthetic Q8_0 fixtures, not a downloaded or production model. The shape fixture intentionally uses `hidden == expert_ffn_size`; the current Session grouped projection requires `binding.rows == problem.hidden`, so a rectangular MoE FFN cannot yet be composed through this API. The fused gate/up form is recognized but not split for execution. Router semantics, top-k selection, shared experts, model-specific biases, real checkpoint compatibility, full-token generation, CUDA/V100, inter-PC transport, RDMA, production R53, and performance remain unqualified.

## Next

R660 should add an identified per-slot projection interface whose output width can differ from the dispatch hidden width, preserving the existing route identity, one-time coefficient application, and atomic Collector contract. Then adapt separate gate/up activation and down projection for the explicit resident map. Keep fused tensors unresolved for execution until a format-checked half-view loader exists. Do not infer checkpoint expert ids from VoltaFabric expert ids.
