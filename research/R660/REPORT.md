# R660 — per-slot GGML projection over identified Sessions

## Result

R660 adds a persistent CPU Session request for a resident GGML matrix projection where activation width, matrix output width, and routing dispatch hidden width are represented independently. Inputs are bound to each dispatched `(token, slot, expert, local)` identity. Raw FP32 activations use a bounded little-endian request body. The worker returns unweighted FP32 slot rows; route coefficients and Collector reduction remain the caller's responsibility.

The implementation reuses the pinned llama.cpp/GGML `NativeRoutedMatrix::forward` CPU operator and the existing GGUF-resident grouped-matrix loader. The donor revision is `3cf03257f219afbe7334045ff7c6a06ac68c627d` under MIT, inherited from the R659 adapter. This is a VoltaFabric Session adaptation; it does not add llama.cpp's generation loop, scheduler, router, or checkpoint runtime.

The protocol advertises a separate `cpu_ggml_grouped_projection_session_f32_v1` capability. Existing grouped-matrix requests retain their prior behavior. The new batch path validates every slot and all rank bindings before reserving the local attempt, invalidates all participants if a started rank fails, and returns no partial batch. It does not apply route coefficients and does not pretend the local test ledger is production R53.

## Focused fixture added

`tests/native_matrix_session.rs` adds a synthetic Q8_0 rectangular fixture with dispatch hidden/input width 32 and resident matrix shape 64 output rows by 32 columns. It checks per-slot identities and unweighted values against a scalar oracle, including a route coefficient that must not be applied at this intermediate stage. It also checks that missing slot coverage is rejected before a worker starts.

The fixture is synthetic GGUF only. It is not a real checkpoint, a complete FFN or token-generation path, or a GPU/inter-PC qualification.

## Verification state

Pipeline 2251 passed the configured repository scans. Gitleaks passed with the 53 exact, redacted digest false-positive fingerprints classified in [GITLEAKS_TRIAGE.md](GITLEAKS_TRIAGE.md); Semgrep and Trivy also reported success. Semgrep recorded 155 repository findings, so a successful job does not mean the repository has zero findings. The fallback job only reported that scans completed; it did not compile Rust.

Pipeline 2252 reran the security jobs successfully, but its branch-scoped Rustfmt job could not start: the runner could not resolve the external Docker image host while pulling `rust:1.90-bookworm`.

Pipeline 2253 passed the configured Gitleaks, Semgrep, Trivy and fallback jobs. It had no Rust compile/test job. The complete CI result therefore remains scan-only.

Local Rustfmt 1.95.0 formatted the R660 additions in `src/native_ggml_cpu.rs` and `tests/native_matrix_session.rs`. Then `rustfmt --edition 2021 --config skip_children=true --check` succeeded for those files and `src/session.rs`; the command parsed all three changed Rust files but skipped resolving child modules. It is a syntax/style check, not a type-check or build.

Eight entries in R660's `SHA256SUMS` were recalculated with Python's standard `hashlib.sha256` over the exact files read back from GitLab; all eight matched. No value from the Gitleaks report was copied.

No Rust test pass, type-check, model generation, GPU execution, inter-PC transport, performance result, or production R53 reservation is claimed.

## Next action

Run the focused CPU test only in an admitted isolated environment that already has the verified pinned GGML static libraries. If that build is unavailable, make it reproducible from the pinned donor before composing gate/up activation and down projection stages. Do not add route weighting here; later FFN composition must apply it once at the existing Collector boundary. Real-checkpoint compatibility remains a separate gate.
