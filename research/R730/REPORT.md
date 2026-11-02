# R730 — CPU HC-post after routed plus shared expert

## Result

The four-file source manifest covers the code, integrated CPU test, this report and the
local build record. Mutable publication receipts and live root handoffs remain separate.

R730 adds a CPU scalar boundary for the pinned DeepSeek-V4.1-Flash `Block.hc_post`
formula and composes it after the R729 result: a real `Completed` receipt from five
VoltaFabric CPU Sessions (six routed slots), then the R728 shared FP8 expert result,
then HC post into four residual copies. Stamp and attempt are checked at every boundary.
The route and shared join retain their existing single application of coefficients; HC
post applies only the caller-provided `post` and `comb` coefficients once. The output is
staged locally and returned only after all four full rows pass finite checks and BF16
rounding.

The pinned formula is `post[i] * x + sum_j(comb[i,j] * residual[j])`, followed by a cast
back to the sublayer activation dtype. The implementation accepts validated, caller-
supplied HC coefficients and exact BF16 residual rows. It deliberately does not calculate
the learned `hc_mixes` projection or Sinkhorn, or the preceding `hc_pre` operation. Those
remain independent boundaries.

The new R712 integration test carries the R729 CPU Session receipt through the shared
join and HC-post. It compares every resulting element against a separate f64 formula and
bit-level BF16 RNE oracle, using non-dyadic coefficients, and separately exercises an
exact BF16 midpoint to check ties-to-even. Invalid shape, nonfinite values, model/layer
identity and attempt mismatches fail closed. This is a CPU formula oracle, not an
execution of the donor's PyTorch runtime.

## Reuse and source provenance

- Donor semantics: private pinned DeepSeek-V4.1-Flash revision
  `314113f8a7dadb388215396ba5fbee2c610a04d6`, MIT. The consulted local source snapshot
  is `evidence/continuation_a97_model_metadata/upstream_314113f8/model.py`, SHA-256
  `e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d`; `Block.hc_post`
  is at lines 964–968. No donor implementation was copied and no dependency was added.
- R730 reuses VoltaFabric's R712 grouped CPU Sessions, R728 resident shared-expert
  operator and identity-bound join, and R729's real completed Session receipt.
- R727's focused open-source/arXiv screen remains the source-selection record. The
  existing R721 adaptation uses vLLM's Apache-2.0 checkpoint-index convention in
  VoltaFabric's own bounded Safetensors reader. This HC-post tranche imports no code
  from Strata, Maya, vLLM runtime, llama.cpp, TensorRT-LLM, or SGLang; no suitable
  component from them was established for this specific contract.

## Verification

- `cargo test --locked --offline --lib r730_hc_post_tests`: 3/3 passed in debug and release.
- `cargo test --locked --offline --test r712_deepseek_session_owner`: 12/12 passed
  (debug, 123.99 s) and 12/12 passed (release, 7.70 s).
- `cargo clippy --locked --offline --test r712_deepseek_session_owner -- -D warnings`:
  passed; library Clippy with `-D warnings` passed.
- `rustfmt --edition 2021 --check ...` and `git diff --check`: passed.
- Local `python3` has no `torch` module, so no runtime comparison to upstream PyTorch was
  performed. The source equation and a separate higher-precision formula are the oracle.

R729's private GitLab source pipeline #2565 completed successfully on source commit
`2c1d567b8493dfb3b71d96583a7b891f9f2ae843` (gitleaks, Semgrep, Trivy, and fallback all
success), and its 9-file manifest was re-read from GitLab and matched exactly. R730's
publication status and later exact readback are tracked in its handoff/receipt; this report
remains a source-commit snapshot.

## Limits and next boundary

The indexed checkpoint fixture still supplies synthetic weight payloads; no actual model
weight payload was read. R730's residual copies and HC coefficients in the integration
test are synthetic. Learned HC projection/Sinkhorn, HC pre, attention/GDN, tokenizer,
autoregressive token loop and full generation remain unimplemented or unqualified. No
GPU, inter-PC/RDMA, production R53 or performance test was run. The PyTorch runtime is not
installed, so backend reduction-order parity is not asserted.

Next, load and validate the pinned HC projection/base/scale tensors and implement the
source-derived HC coefficient split/Sinkhorn plus pre-mix with an independent CPU oracle;
then compose the pinned attention/GDN operation and continue the token-generation path.
