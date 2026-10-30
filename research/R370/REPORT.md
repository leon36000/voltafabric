# R370 — Session orchestration for unweighted expert slots

Parent: f57ce5785037986ea676a901f97b19d9acc15b45 (R369).

`run_compressed_batch_unweighted` returns complete unweighted expert rows in token/slot order through the existing compressed Session admission and executor. The ordinary reduced output and the new slot output share preflight, per-rank encoding, attempt allocation, identity claim, send-before-read, draining, and terminal failure behavior. The new path sets recovery false and requests FP64 completions; it never automatically replays. An error exposes no output. Callers must apply routing coefficients exactly once.

CPU test `unweighted_batch_orchestration_is_atomic_and_never_replays` sends a Strata-style FP32 admitted batch through two local QPN2 resident expert sessions, compares every unweighted row to an independently decoded dense expert, verifies both receipts, rejects duplicate identity, and confirms an injected missing completion gives no output and cannot be replayed. Inputs, coefficients and weights are synthetic. This is not a live Strata callback or a GPU, inter-PC, production transport or full-model generation proof. The process-local identity guard is not a durable ledger or exactly-once guarantee.

Existing CPU checks passed: `faults_over_compressed_sessions_fail_closed_without_partial_output`, `fp16_activation_input_wire_matches_f64_and_loads_weights_once`, and `strata_f32_exact_f64_reaches_local_sessions_and_oracle`. Strict offline all-target Clippy, formatting and `git diff --check` passed. No consumed UNKNOWN was replayed.

Next: specify an atomic FP64-to-FP32 slot output policy for Strata, handle its GPU-hit sentinels and its callback's void/error plus GPU wait draining, and test an isolated integration. Actual weights and model formats must remain distinct. Hardware or inter-PC work requires fresh applicable admissions and gates.
