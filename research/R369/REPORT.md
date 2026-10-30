# R369 — Unweighted expert-slot collection for Strata boundary

Parent: 027c9297ecc96668467c106f4f4109b1cdc120a7 (R368).

At Strata donor fb58e0dbc8399662c0e47c76578c6e878b14f6cf, `include/strata/core/verify.hpp` defines PoolMultiFn output as `(n_tok * k, n_embd)` expert rows; `src/core/verify.cpp` subsequently combines those rows with `w_`. Therefore the reduced output in the R367/R368 demonstration is not a drop-in callback result: applying Strata's weights to it would be wrong. GPU-hit rows and failure/wait handling remain unresolved; no active Strata service was modified.

`Collector::finish_unweighted_slots` consumes the collector and returns complete unweighted values in original token/slot order, preserving FP64 values or decoding accepted FP16 values. It refuses incomplete or failed collectors and preweighted FP32 values. Caller owns any subsequent weighting, exactly once. Existing `finish` keeps its previous ordered weighted reduction contract.

CPU validation, offline, single job: seven `strata_ingress` tests passed, including shuffled arrivals, missing and duplicate completions; `unweighted_slot_collector_receives_local_experts_in_strata_order` passed with two local QPN2 session processes and synthetic resident weights, comparing each slot bit-for-bit against a dense expert oracle. The integration test directly drives low-level Session methods with a fresh test attempt: it is not a production batch orchestration or proof of shared identity reservation. `cargo clippy --offline --locked --all-targets -- -D warnings`, format check and `git diff --check` passed.

Next: expose this output via the existing preflight/identity-claim/all-rank-drain Session orchestration, preserving no-replay and atomicity; specify and validate FP64-to-FP32 output conversion, actual gate/hit semantics, and fail-closed handling with GPU wait draining before any isolated donor integration. This proves neither GPU, network production, real Strata callback, nor complete generation or numerical equivalence to Strata's IQ3 experts. No UNKNOWN replayed.
