# R739 — transactional staging for the DeepSeek window KV ring

## Change

The R738 token block cloned the complete sliding KV ring at three levels: the outer block transaction, the attention sublayer transaction, and the cache append transaction. R739 adds a private, identity-bound `DeepSeekV41WindowCacheUpdate`. Attention now reads the existing ring while overlaying the new key/value row at the current slot; it owns only that one row until commit.

The block commits the row only after the route receipt, shared-expert join and both HC residual operations succeed. A failed or UNKNOWN routed execution still leaves the caller's KV ring unchanged. The direct cache and attention APIs retain their existing immediate-commit behavior through the same staged operation.

At the pinned 128-slot × 512-value × FP32 shape, the previous three ring copies represent 768 KiB of KV-ring copying per token per layer. R739 stages one 512-value row (2 KiB) and does not copy the ring. This is a byte-count derived from pinned dimensions, not a measured latency or throughput gain. The attention score and numerator loops still visit the same context slots and keep their accumulation order; this code review is not an independent numerical oracle.

## Provenance and reuse

This tranche changes VoltaFabric's existing cache transaction. It copies no code from vLLM, TensorRT-LLM, llama.cpp, SGLang, Maya, Strata or DeepInfer. The pinned DeepSeek source remains semantic reference only. The broader external-component extraction requested by the owner is still outstanding and must be advanced in a later, source-pinned tranche.

## Verification

The four modified Rust files were based on exact R738 GitLab source readback. The bounded staging directory replaced those four files and reused the existing local R737 support-module/dependency mirror; it was not a fresh full clone.

- `cargo fmt --all -- --check`: passed.
- `cargo check --locked --offline --lib`: passed.
- `cargo clippy --locked --offline --lib -- -D warnings`: passed.
- No tests were added or run.
- GitLab source pipeline #2595 succeeded on commit `6486e1010accefd314df7e83967b004837961a0f`; Gitleaks, Semgrep, Trivy and fallback all passed. This is a scan/fallback result, not a compile, oracle or generation proof.

## Receipt publication

The source receipt commit `80515d0e1790579dbd8a6bad3f58aed011dbfe5a` passed GitLab pipeline #2596; Gitleaks, Semgrep, Trivy and fallback all passed. The four receipt files were read back exactly from that commit. A final additive reconciliation commit records this receipt state; history remains unrewritten.

## Publication correction

The first R739 source commit accidentally supplied Base64 text as file contents. It was superseded by an additive text-content commit on the same branch, without rewriting history. The pipeline for the malformed intermediate commit was cancelled. The corrected commit is the R739 source revision and was re-read from GitLab exactly before receipt publication.

## Limits

This remains a CPU reference/operator slice. No checkpoint payload, numerical oracle, complete layer, tokenizer/logits/sampling, multi-token generation, GPU or inter-PC execution was run. No external engine code was integrated in this round.
