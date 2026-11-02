# R691 — public llama_decode to VoltaFabric Qwen2MoE Session

## Result

The pinned CPU fixture now loads a tiny one-layer Qwen2MoE through
`llama_model_init_from_user`, executes public `llama_decode` for three token steps,
and samples with llama.cpp's public greedy sampler. The same synthetic resident
experts are routed through VoltaFabric's Rust Session adapter.

The donor and VoltaFabric runs produced tokens `3,3,3`; maximum absolute logit
difference was `0`. An independent softmax calculation over the fixture's 41 router
scores confirms the selected experts and final top-2 weights. The top-2 weights sum
to about `0.127954`, as expected when probabilities are selected from a 41-way
softmax without renormalizing them. VoltaFabric applies those donor weights once in
its FP32 preweighted collector.

The failure path returns `llama_decode == -3`; it publishes no VoltaFabric output,
does not call the Rust Session, does not read logits, and does not sample a token.

## Reuse and adaptation

- Donor: `ggml-org/llama.cpp` commit
  `3cf03257f219afbe7334045ff7c6a06ac68c627d`, MIT license.
- The local donor checkout has the R679 hook adaptation in four `src/llama-*` files;
  this is a local seam patch, not an unmodified upstream build. No upstream file was
  copied into the VoltaFabric source tree.
- Reused public boundaries: `llama_model_init_from_user`, `llama_decode`, and the
  llama.cpp greedy sampler. Reused the existing R688 Rust Session loader design,
  narrowed to a fresh R691 run identity and the existing Qwen2MoE donor-route and
  atomic-Collector path.
- The C++ fixture provides complete tensor metadata and synthetic CPU weights for a
  32-wide, one-layer, 41-expert, top-2 model. A generated four-expert Q8_0 GGUF source
  feeds checkpoint slices `[1,0]` into resident experts `[40,3]` through the actual
  Rust loader. Tokenizer metadata is `no_vocab`; the test uses token IDs directly.
- The comparison includes the public attention/KV decode path but not trained model
  weights. It validates a generation boundary, not a usable model or a speed gain.

## Verification

`BUILD_TEST.txt` records the exact runner result. The final clean-source run passed
with C++17 `-Wall -Wextra -Werror`, offline Rust builds, `rustfmt --check`, shell
syntax validation, and scoped Clippy for the FFI crate. The repository library emits
two pre-existing warnings in `session.rs` and `sc117_layer_chain.rs` during build;
they are outside the R691 files.

Three unique R691 identities in the final run completed all steps. The local durable
ledger intentionally leaves reservations marked `RESERVED_UNKNOWN_NO_REPLAY`; the
runner output is the completion evidence. Earlier R691 reservations for
`r691-public-decode` batch `300` and `r691-public-decode-v3` batch `501` are retained
as uncertain and were never retried. Other runs used distinct run/plan/batch/attempt
identities. The reject path creates no reservation because it refuses before the
Session call.

## Limits

CPU only; synthetic one-layer weights, a four-ID dummy vocabulary, no production
checkpoint or text tokenizer, no GPU, no inter-PC/RDMA worker, no production R53
lease, and no performance comparison. This does not qualify a full-model generation
or compatibility with any checkpoint. The local ledger is not the shared R53
hardware namespace.

## Next step

R692 should inventory only already-present, owner-admitted checkpoint candidates and
their actual GGUF architecture, tensor formats, tokenizer metadata, and license.
Then choose one compatible CPU-only layer to replace from synthetic data with a
checkpoint-backed source. Do not download a model, run a GPU, use production network,
or restart automatic work as part of that inventory.
