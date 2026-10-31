# R637 — ownership and shutdown for the Strata callback contexts

## Finding at the pinned Strata boundary

On the Strata source revision pinned by R636 (`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`), `generate.cpp` creates one `Drive`, fills its ordinary `ExpertDispatch` fields, and passes either `&drive` or `&split_drive` to the verifier. It never assigns `vf_cpu_pool` or `vf_cpu_context`. Those candidate fields therefore remain null, so the extracted callback branch is not selected by the actual generation entrypoint. R636's syntax and shape checks did not establish runtime installation.

This is a concrete integration gap. The current bridge also made callers manually keep 48 `PoolContext` objects and their `Client` objects alive in the exact layer order. R637 adds `PoolFleetOwner` to make that lifetime and one-shot session shutdown explicit; it does not wire the owner into Strata's CLI or claim a generation.

## Reuse and adaptation

The adapter continues to target Strata's pinned `ExpertDispatch` / `PoolMultiFn` boundary and VoltaFabric's existing `Client` transport. `PoolFleetOwner` takes ownership of the 48 C++ client objects, builds one context for each exact layer, exposes a stable callback user pointer, and rejects missing, unusable, misordered clients and a non-advanceable initial batch. Socket descriptors remain borrowed and must outlive the owner. The destructor performs no I/O. `shutdown_all()` first blocks new callbacks, waits out an in-flight callback, then attempts one terminal shutdown per session; subsequent calls are rejected instead of replayed.

No Strata or llama.cpp source was copied into the engine. This is a VoltaFabric-side lifecycle adapter over the pinned callback contract.

## Verification

Built and ran `pool_fleet_owner_test.cpp` on CPU with C++20, `-Wall -Wextra -Wpedantic`. The fixture used 48 local AF_UNIX socket pairs. It confirmed that a swapped layer mapping is refused, a correctly ordered set creates a stable callback context, an out-of-range layer leaves the destination unchanged, and a second shutdown attempt is rejected. No IP socket, model, Strata generation, GPU, worker process, or R53 ledger was used.

## Limits and next step

The callback owner is not installed in `generate.cpp`; no real Strata runtime or complete generation is qualified. `Client` still borrows its file descriptors, and a supervisor must keep them open and complete the existing INIT handshake. Next, define and implement an explicit launcher-to-Strata context handoff with the run/model/plan/layer/batch identities preserved. Keep live Strata generation, GPU, inter-PC transport, and real R53 admission gated separately.
