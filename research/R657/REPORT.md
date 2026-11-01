# R657 — GGML grouped projection through persistent VoltaFabric Sessions

## Result

R657 connects the R656 resident GGML grouped-matrix provider to VoltaFabric's persistent CPU `Session`/`vf-rank-session` request path. Each rank loads one digest-bound matrix payload and retains its resident provider across requests. Requests carry the routed `Stamp` and attempt; the existing Rust FP16-decoding, f64-accumulating Collector remains the only place that applies route coefficients.

The Session batch coordinator validates every participant and resident order before dispatch, claims the full identity, and durably reserves its local `NativeBatchLedger` before sending work. It shares one deadline across the whole rank wave. A lost acknowledgement, timeout, malformed result, or worker failure invalidates all participating sessions and yields no partial output. A repeated Stamp is refused before another request is sent.

## Reuse and adaptation

This adapts the pinned llama.cpp/GGML `GGML_OP_MUL_MAT_ID` CPU component already linked through VoltaFabric's native bridge. Donor revision `3cf03257f219afbe7334045ff7c6a06ac68c627d` is under MIT; R657 copies no donor source. The adapter keeps GGML's grouped physical expert order explicit and returns raw, unweighted rows to the Rust Collector.

This advances the llama.cpp/GGML component into a persistent VoltaFabric Session boundary. It does not integrate llama.cpp as a complete runtime or checkpoint loader. This wave adds no runtime component from Strata, Maya, vLLM, TensorRT-LLM, SGLang, or DeepInfer.

## Verification

On the PC6 AArch64 CPU build, the focused command passed 4/4 tests in both debug and release: two `vf-rank-session` unit tests and two new subprocess Session integration tests. `rustfmt --check` passed on the changed Rust files. Scoped Clippy completed successfully; it reported two pre-existing warnings in SC117/Strata files, with no warning from the new grouped-matrix coordinator.

The passing integration cases verify two fresh batches through the same live Session and resident weights, expected Collector outputs `[3.5, 2.0]` and `[7.0, 1.0]`, refusal of a repeated full Stamp, and atomic rejection when a second rank produces FP16 overflow after the first rank has completed. Both sessions are terminal after the uncertain/failed batch.

## Limits

- Fixtures use synthetic Q8_0 32×32 matrices; no checkpoint or real model weights were read.
- Session raw loads are one bounded frame (`MAX_RAW_ARTIFACT_BYTES`, below 16 MiB); a chunked checkpoint-to-resident loader is still needed for real projections.
- The test exercises a single grouped projection, not a complete expert MLP, transformer layer, or token generation loop.
- The local `NativeBatchLedger` is not the production R53 ledger; tests use fresh local fixture directories.
- No GPU, V100 compatibility, inter-PC transport, production network, shared R53 admission, performance comparison, or complete generation is qualified.

## Next step

Build a bounded checkpoint-span loader that fills the existing GGML resident without copying weights per batch, then attach this grouped projection to a real expert FFN stage behind an identified CPU Session. Keep model/plan/layer/batch/epoch/attempt identity and atomic no-partial-output semantics. Use a small compatible checkpoint fixture and an independent projection oracle before extending the generation chain. Do not resume hardware or automatic work.
