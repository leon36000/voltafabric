# R689 — Live Qwen2MoE route replacement with VoltaFabric FP32 slot collection

## Result

The CPU fixture now replaces the routed FFN output inside the pinned llama.cpp Qwen2MoE graph with a result produced by resident VoltaFabric `Session`s. For a synthetic one-token, one-layer, top-2 Q8_0 fixture, the unhooked donor graph, the hooked graph using VoltaFabric, and an independent scalar Q8_0 oracle agree exactly in FP32 (`max_abs = 0`). The fixture verifies rank-local source slicing `[1, 0] -> [40, 3]`, route identity and weights, full-slot completion, and refusal without output after provider failure or a repeated completed identity.

## Arithmetic boundary

The prior ordinary collector intentionally decodes each returned row through FP16 and combines in FP64. R689 showed that this arithmetic is not equivalent to llama.cpp's Qwen2MoE graph, which multiplies each FP32 down-projection row by its route coefficient and adds slots in FP32. R689 adds a distinct `qwen2moe_ffn_postdown_preweighted_fp32_slot_sum_v1` contract. It binds the attempt in the contract digest, validates the complete `(token, slot, expert, rank, local, weight)` dispatch, applies each coefficient once after down projection, and performs ordered FP32 slot additions. Its format remains separate from R9/MXFP4 preweighted FP32.

The R187 warning against numerically non-equivalent partial aggregation remains in force: this path only publishes after the complete routed batch has been accepted, and aborts the whole output on refusal.

## Reused component and provenance

- Donor: llama.cpp / GGML at commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`; its checked-in `LICENSE` identifies MIT License.
- Adapted boundary: llama.cpp's existing Qwen2MoE `build_moe_ffn` graph and route hook, retained at the graph's post-down routed aggregate boundary. VoltaFabric continues to own route validation, resident expert lookup, execution, and atomic publication.
- Fixture: synthetic GGUF Q8_0 checkpoint, 32-wide FFN, one layer, one token, two selected experts. Only 6,528 of 13,056 source tensor bytes are read for resident experts `[40, 3]` from checkpoint IDs `[1, 0]`.

## Verification

- `cargo test --offline --test preweighted_fp32`: **14 passed**.
- `adapters/native_ggml/run_qwen2moe_r659_live_session_r689_test.sh`: **exit 0**, CPU only. Donor vs VoltaFabric graph, donor vs independent Q8_0/FP32 oracle, and donor vs captured-slot arithmetic each report `max_abs = 0`.
- The script also checks output withholding after an injected provider refusal and after reuse of a completed identity.
- Attempted `cargo test --offline --test r12_mxfp4_session`: could not compile in this scratch checkout because the two included local fixtures `fixtures/preweighted_mxfp4_tiny_golden.json` and `fixtures/preweighted_mxfp4_tiny.safetensors` are absent. No fixture was fetched or substituted. The native CPU runner compiled the changed library and R9 completion construction successfully.

## Limits

This establishes one synthetic CPU graph boundary only. It does not establish a compatible production checkpoint, full-model generation, GPU execution, inter-machine transport, performance improvement, or production RDMA. No CUDA, network worker, or performance comparison was run.
