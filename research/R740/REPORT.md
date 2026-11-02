# R740 — DeepSeek V4.1 prompt encoding boundary

Date: 2026-11-02

GitLab source commit: `3381700ba4804f0271cdce68d514b89fd83295fc` on private branch `r740-deepseek-recipe-prompt-boundary-20261102`. Pipeline #2599 passed Gitleaks, Semgrep, Trivy and the security fallback; the eight source/report/manifest files were read back exactly from GitLab.

The first receipt update, commit `1618be41f956502846b55630feca77872e26dbb1`, triggered pipeline #2600. Gitleaks classified the public tokenizer artifact digest's handoff metadata as a generic API-key finding. Inspection of the redacted report showed one finding at `handoff/PRODUCTIVITY_CURRENT.json`, line 4545; its value was the fixed public artifact fingerprint, not a credential. The handoff field is being replaced with a reference to `research/R740/RESULT.json`, where the integrity digest remains documented. This remediation is pending a fresh pipeline.

## Decision

Integrate DeepSeek's own prompt-rendering and token-encoding crates as an **optional** VoltaFabric feature. The wrapper accepts tokenizer bytes from its caller, checks the exact tokenizer artifact pinned to the selected model revision, encodes one plain-text user turn, and rejects a prompt whose completion reservation would exceed the caller's context limit.

This reuses the upstream crates through dependencies; no donor source file is copied. It advances the prompt-to-token seam, but it is not an inference backend, model loader, sampler, stream parser, or complete generation path.

## Source and adaptation

- Model: `deepseek-ai/DeepSeek-V4.1-Flash`, revision `314113f8a7dadb388215396ba5fbee2c610a04d6`.
- Prompt code: `deepseek-recipe-core = 0.2.0` and `deepseek-recipe-encoding = 0.2.0`, upstream MIT.
- Tokenizer code: Hugging Face `tokenizers = 0.23.2`, Apache-2.0.
- Pinned `tokenizer.json`: 6,367,257 bytes, SHA-256 `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`. The bytes were fetched for digest verification only and are not stored in this repository or cache.
- The adapter lives in `src/deepseek_v41_prompt.rs` and is gated by the `deepseek-v41-recipe` Cargo feature. The default feature set does not compile this dependency surface.
- At construction, the caller-supplied bytes must match both the fixed byte count and pinned SHA-256 before tokenizer parsing. No downloader or model cache is added.
- At encoding, only one `InputMessage::User` with plain text and no image sources is admitted. Input bytes, context size, token count and completion reservation have explicit bounds; arithmetic overflow or a prompt-plus-reservation beyond context is rejected.

The root `src/lib.rs` was updated to expose the adapter only under the feature. The project `Cargo.lock` adds 176 package entries relative to R739; keeping the feature optional prevents unrelated/default builds from enabling the runtime path, but the dependency and supply-chain surface still deserves review. Upstream encoding currently brings tokenizers' HTTP/Hugging Face support into the feature's resolved graph; this wrapper itself performs no network requests.

## Scope exclusions

This first adapter intentionally refuses to claim support for multi-message history, tool calls or namespaces, assistant `action`, multimodal inputs, JSON/schema modes, output streaming, stop-reason parsing, sampling, or logits. A public SGLang report identifies output differences for some tool and reasoning cases with `deepseek-recipe 0.2.0`; therefore this adapter is limited to a single plain user turn until the upstream behavior has an independent conformance comparison. See [SGLang issue 46348](https://github.com/sgl-project/sglang/issues/46348). The issue is a report, not a substitute for a pinned reference oracle.

The tokenizer asset is not embedded and no checkpoint payload is included. No output token was generated in this tranche. No V100, GPU, inter-PC transport, R53 production route, end-to-end generation, or performance result is qualified.

## Verification

- `rustfmt` on the new Rust module and scratch crate entry point: pass.
- `cargo check --locked --lib` with the default feature set in an isolated scratch crate using the R739 root manifest and updated lockfile: pass.
- `cargo check --locked --lib --features deepseek-v41-recipe` in that same scoped scratch crate: pass.
- `cargo clippy --locked --lib --features deepseek-v41-recipe -- -D warnings` in that same scoped scratch crate: pass.
- No test was added or run. No full VoltaFabric workspace build was run: the scratch crate contains the new module and the existing SHA-256 module, with workspace members represented by compile-only placeholders. Full integration and runtime conformance remain unverified.
- GitLab CI is the existing R739 security/fallback pipeline: its fallback job prints that local security scans completed; it is not a compile or model-generation test.

## Next

R741 should produce an independent, pinned prompt-token oracle for admitted plain-text inputs using the official DeepSeek reference and the exact tokenizer digest, then add bounded output-token decoding/stop handling only after reference parity is established. Keep tool/history/multimodal protocol support separate. Do not load a checkpoint or use a GPU as part of this prompt-only step.

## References

- [DeepSeek Recipe repository and MIT license](https://github.com/deepseek-ai/deepseek-recipe)
- [DeepSeek Recipe tokenizer documentation](https://github.com/deepseek-ai/deepseek-recipe/blob/main/docs/tokenizer.md)
- [DeepSeek V4.1 tokenizer provenance notes](https://github.com/deepseek-ai/deepseek-recipe/blob/main/static/tokenizers/v41/README.md)
- [Pinned DeepSeek V4.1 Flash tokenizer artifact](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/tokenizer.json)
- [Hugging Face tokenizers repository](https://github.com/huggingface/tokenizers)
