# R701 — Independent oracle for the DeepSeek gate route fixture

Date: 2026-11-02  
Branch: `r701-deepseek-gate-decimal-oracle-20261102`  
Parent: R700 receipt commit `d48e24130b94d86b9e692e761ced9ca56b63ebec`

## Delivered

Added a second implementation of the R700 synthetic gate fixture in
`r700_gate_decimal_oracle.py`, using only Python's standard-library `Decimal`
at 60 digits. It independently computes sqrt-softplus scores, applies the text
selection correction only for top-6 membership, normalizes the uncorrected
scores, and applies the explicit 1.5 route scale once. It does not import or call
the Rust gate or donor source.

Added `tests/r701_deepseek_gate_decimal_oracle.rs` with the oracle's expected six
expert IDs and coefficients. The focused Rust test passes and checks each
coefficient within `2e-7`. This narrows the R700 caveat: the scalar Rust output
matches an independent high-precision mathematical fixture. It still does not
establish bitwise parity with PyTorch's reduction/top-k kernels or validate the
model's loader mapping for `route_scale`.

## Verification

- Python Decimal oracle: output IDs `[0, 383, 382, 381, 380, 379]`; coefficients
  sum to `1.49999999999999999999984760510354542957937574122110057758166` after
  the pinned `1e-20` normalization epsilon.
- `cargo test --offline --locked --test r701_deepseek_gate_decimal_oracle`:
  1 passed.
- Scoped Clippy with `-D warnings`, rustfmt, and `git diff --check`: passed.
- The Rust check ran in the retained older/grafted local worktree with R700
  source overlaid, not in a full checkout of the GitLab branch.

## Reuse and remaining boundary

R701 copies no donor code. R700 adapts the pinned MIT DeepSeek V4.1 Flash gate
operation order; the pinned model source is at
`314113f8a7dadb388215396ba5fbee2c610a04d6`.

This is a synthetic numerical oracle only. It has no checkpoint, gate projection
weights, six real expert payloads, Session dispatch, GPU, inter-PC transport, or
full generation. R698's qualified live fixture contains real MXFP4 K32 payloads
for only experts 0 and 383 and uses synthetic route coefficients. The top-6
planner correctly refuses the four missing profiles. Keep that refusal; do not
substitute experts or replay a prior UNKNOWN.

## Next useful boundary

Only use artifacts already present in the private source branch or the retained
two-expert CPU fixture. Connect a model-derived route to Sessions only after the
matching projection and all six selected expert payloads are available. If they
are not present, retain fail-closed admission; no full-model download, GPU,
production network, or automatic relaunch is part of this proof.

## Source

- [Pinned DeepSeek V4.1 Flash gate implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py)
