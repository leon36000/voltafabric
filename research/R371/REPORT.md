# R371 — Atomic FP32 slot buffer for a Strata-style caller

Parent: 815651dac45d8bfc0d137bbbaaf8e7811b71b425. Strata donor reference: fb58e0dbc8399662c0e47c76578c6e878b14f6cf, `include/strata/core/verify.hpp`: pool output is FP32, laid out `(n_tok * k, n_embd)`; the subsequent combine applies weights.

`write_unweighted_f32` validates routing and full output shape, bounds total slot values, converts unweighted FP64 rows to FP32 with nearest rounding, rejects nonfinite values or FP32 overflow, and copies to the caller buffer only after all rows pass. Underflow and ordinary rounding are accepted and can lose precision. Weights are not applied by this routine. This is an explicit output representation conversion, not evidence that QPN2 experts numerically match Strata IQ3 experts or the donor's GPU combine order.

Offline CPU validation: all eight `strata_ingress` tests passed, including tie-to-even examples, signed zero, overflow, nonfinite input, malformed shapes and unchanged destination on rejection. `unweighted_batch_orchestration_is_atomic_and_never_replays` passed with two local synthetic QPN2 sessions, comparing both unweighted rows and flattened FP32 values to dense expert results. Strict all-target Clippy, formatting check and `git diff --check` passed. No GPU or inter-PC operation and no UNKNOWN replay.

Next: handle Strata's GPU hit sentinel without dispatching an already-resident expert or violating slot order, and solve the void callback failure propagation and outstanding GPU wait draining in an isolated donor integration. Real gate provenance, full model execution, production networking and complete generation remain unqualified. No active Strata service was changed.
