# R690 — Qwen2MoE routed output to llama.cpp token-selection seam

## Result

Extended the pinned CPU fixture after the VoltaFabric-routed Qwen2MoE FFN. A small F32
`lm_head` now uses GGML `ggml_mul_mat` and is exposed through
`llm_graph_result::get_logits()`. The fixture passes those logits through llama.cpp's
public `llama_sampler_init_greedy` / `llama_sampler_apply` / `llama_sampler_accept` API.

For the synthetic 32-wide, one-token, top-2 Q8_0 case, donor and VoltaFabric logits
match exactly; an independent double-accumulation scalar `lm_head` oracle also has
`max_abs = 0`. The donor sampler, VoltaFabric sampler, and scalar argmax all select
token ID 3. A refused Session request publishes no logits, and the sampler emits no
token. The duplicate completed identity also withholds both graph output and logits.

## Reuse and adaptation

- Donor: `ggml-org/llama.cpp` commit
  `3cf03257f219afbe7334045ff7c6a06ac68c627d`; checked-in `LICENSE` identifies MIT.
- Reused interfaces: pinned Qwen2MoE `build_moe_ffn`, the existing internal
  `process_ubatch` graph boundary, GGML `ggml_mul_mat`, `llm_graph_result::get_logits`,
  and the public llama.cpp greedy sampler API. No upstream source file was copied.
- Adaptation: a synthetic F32 vocabulary head follows the routed FFN output; a scalar
  oracle independently computes the output projection and argmax.
- R689's attempt-bound route contract and FP32 slot sum remain the only place applying
  the two route coefficients. The new head does not reapply routing or weights.

## Verification

The exact CPU build and direct runner command, result, tolerances, and output are in
`BUILD_TEST.txt`; hashes are in `SHA256SUMS`. The test ran the R689 Rust loader/worker
and compiled the modified C++ runner against the pinned donor and static GGML build.
R689's 14 Rust unit tests are inherited evidence and were not rerun in this C++-only
tranche.

One additional manual invocation of the generated binary omitted required
`VF_TEST_OUTPUT_DIR`, exited 1, and left an empty log. It is excluded from the passing
result. The local synthetic ledger receipts remain `RESERVED_UNKNOWN_NO_REPLAY`; no
attempt was repeated or repaired. The passing result above came from the complete
R689 runner script with all required environment variables set.

## Limits

This is only a synthetic **logits-to-token-selection seam**. It still calls internal
`process_ubatch` on a custom graph, not public `llama_decode`; it has no real token
embedding, attention, KV state, tokenizer, production checkpoint, GPU, inter-PC worker,
RDMA, or performance comparison. It is not a full generation result. A five-paper
targeted screen is recorded in `ARXIV_SCREEN.md`, with a bounded 24-paper survey and
engineering priorities in `ARXIV_SURVEY_24.md`. Neither qualifies donor code or
hardware compatibility.

## Next step

R691 should assess whether a public `llama_decode` path can be exercised safely with a
bounded synthetic model and independent token oracle. If the required model/vocabulary
contracts cannot be constructed without pretending the custom FFN graph is a full
decoder, stop that seam and prioritize locating an owner-admitted compatible checkpoint;
no model download, GPU, or production network is authorized by this CPU tranche.
