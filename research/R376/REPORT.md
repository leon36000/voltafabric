# R376 — Unweighted Strata-style callback admission

Parent: 5f7ed9c0f9b464c8f54d028a60218074a0a73c02. Pinned donor Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf defines `PoolMultiFn` without gate coefficients (`include/strata/core/verify.hpp`) and applies the actual routing weights in the following combine (`src/core/verify.cpp`). Earlier weighted admission therefore required information absent from this callback.

`admit_strata_unweighted_f64` accepts IDs, input rows, final per-slot kind and the complete model expert set without gate weights. It uses unit coefficients only as private routing placeholders, encapsulated by `StrataUnweightedBatch`; its consuming `execute` uses the existing unweighted Session path, so it returns unweighted rows and does not apply those placeholders. The caller must ensure the actual weights are applied once later. This path does not modify the weighted APIs.

All 14 ingress tests passed on CPU in debug. `external_expert_without_registry_copy_can_coexist_with_real_sessions` passed in debug and release without supplying gate coefficients, using a synthetic kind array, one external-only expert and two local synthetic QPN2 Session processes; every miss was checked against a separately decoded dense expert. `strata_callback_admission_requires_no_gate_coefficients` passed in release. Strict offline all-target Clippy, formatting and `git diff --check` passed.

No actual donor callback is wired. The kind array remains supplied by the caller and must be captured after all peer/helper decisions and bound to the same real batch and model. The donor's remote optimized path may pre-reduce helper results and requires its own combine contract; this proof does not cover that path. External completion, error propagation and GPU wait draining remain open. The weights used here are synthetic resident QPN2, not Strata IQ3. No GPU, inter-PC, UNKNOWN replay, full model or generation claim.

Next: define and test an isolated donor handoff at the actual final-kind site, including callback failure handling and wait draining, without altering the live service; preserve the original goal of real generation and distribution.
