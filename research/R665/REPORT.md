# R665 — Qwen2MoE bridge conforms to the FP16 Collector contract

## Change

R663 validated complete unweighted FP32 down-projection rows, then converted them directly
to f64 before calling VoltaFabric's ordinary routed Collector. That skipped the required
ordinary-Collector representation boundary. R665 now preflights every finite FP32 row by
encoding it as binary16 and decoding that exact value to f64 before constructing or
mutating the Collector. The ordinary Collector remains responsible for applying each
routing coefficient once in its canonical token/slot order. FP16 overflow rejects the
whole result before Collector creation.

## Reused upstream component

This remains a VoltaFabric adapter around pinned ggml-org/llama.cpp commit
3cf03257f219afbe7334045ff7c6a06ac68c627d, licensed MIT. The existing bridge reuses
the donor's GGUF tensor layout and resident CPU Q8_0 grouped-matrix execution for the
separate Qwen2MoE gate, up, and down projections. The donor's Q8_0 reference quantizer
stores the activation scale in FP16; the test's independent scalar oracle now models
that scale rounding in each activation block. No donor engine is installed or used as
VoltaFabric's runtime.

## Verification

An isolated runs harness used the exact GitLab R664 Rust source, Cargo lockfile and
the already-built pinned CPU-only GGML libraries. The only native code changed for this
round is Rust code; no model, GPU, production network, or service was used.

- Locked/offline debug run of the entire native_matrix_session test target: **9 passed**.
- Locked/offline release run of the Qwen2MoE compound FFN test: **1 passed**.
- The Qwen2MoE test exercises a synthetic two-token Q8_0 GGUF fixture, scalar activation
  quantization oracle, FP16-decoded unweighted down rows, f64 routed accumulation,
  stale-Stamp rejection, local replay refusal, and finite-FP32-to-FP16 overflow refusal.

Two pre-existing warnings remain in unrelated code: RawProjectionInputMeta visibility
and unused CapturedQsa fields. An earlier full-target run also found an old rectangular
fixture with a mismatched model_revision and an idealized Q8_0 assertion; the fixture
identity and its oracle were corrected before the passing 9/9 run.

A full-file rustfmt check remains non-clean because the existing source and test files
contain many unrelated unformatted sections. No broad formatting-only rewrite was
included; the changed collector block and new oracle were kept scoped to this tranche.

## Limits and next step

The evidence is synthetic CPU execution with local attempt state. It does not qualify a
real checkpoint, tokenizer, attention, normalization, residual path, autoregressive token
loop, complete generation, external R53 admission, GPU, inter-PC transport, or
performance. The next useful boundary is to derive this Qwen2MoE layer's expert choices
from the pinned donor router semantics and feed those choices into the same identified
FFN path, still on synthetic CPU inputs.
