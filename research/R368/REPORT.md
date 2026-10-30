# R368 — Exact FP32 admission for local FP64 sessions

Parent: R367, 27c53c532abfd1a20d8bcf97de04bfc703889367. CPU-only continuation of the Strata-style admission boundary; no modifications to the active Strata service.

The Strata donor examined is pinned at fb58e0dbc8399662c0e47c76578c6e878b14f6cf. Its multi-token pool callback does not supply actual gate coefficients: a future integration must independently provide the coefficients corresponding to the same batch and layer. The callback is not connected by this change.

`admit_pool_batch_f64` validates the same shape, identity, routing, expert IDs and coefficients as the FP16 admission path, rejects non-finite activations, and converts finite FP32 inputs exactly to FP64 (including signed zero). The separate FP16 admission retains its exactness requirement. Admission starts no worker and assigns no attempt. This uses the existing ordinary FP64 Session/Collector contract; it does not establish a GPU or FP16 performance path.

Validation on CPU, offline, one build/test job:

- `cargo test --offline --locked --test strata_ingress`: 6 passed, including all finite FP16 bit patterns, exact FP32 widening and negative cases.
- `cargo test --offline --locked --test a53_compressed_session strata_f32_exact_f64_reaches_local_sessions_and_oracle`: passed. A batch with non-FP16-representable FP32 values reached two actual local QPN2 session processes with synthetic resident expert weights and matched the independent dense oracle bit-for-bit. Gate coefficients in this test are synthetic.
- Both tests passed in `--release` as well.
- `cargo test --offline --locked --lib`: 106 passed, 3 ignored; `cargo clippy --offline --locked --all-targets -- -D warnings`, `cargo fmt --all -- --check`, and `git diff --check` passed.

Use the single-batch Session API for this proof: the series test path scales inputs, including batch zero. This is not a live Strata callback, production networking, GPU execution, a full model, text generation or a performance claim. No consumed UNKNOWN was replayed.

Next: design a fail-closed integration that supplies the real gate coefficients and handles the donor's void callback/error and outstanding GPU waits correctly, without changing the live service; then test in isolation. Live hardware and inter-PC runs remain subject to fresh resource admission and all hardware/lease/ledger gates.
