# R656 — GGML grouped projection through VoltaFabric routing and Collector

## Result

R655 exposed a resident `ggml_mul_mat_id` boundary and tested its raw outputs by manually building a `RoutedRawOutput`. R656 adds `GroupedMatrixBinding`, `GroupedMatrixResident`, and `GroupedMatrixFleet` in `src/native_ggml_cpu.rs`. The reusable API now checks the digest and physical expert/local order, obtains routing from `routing::dispatch`, calls the resident GGML stack, and passes unweighted FP16 rows through `completion::accept_routed_output` to the ordinary Rust Collector. The Collector decodes FP16, accumulates in f64, and applies each route coefficient once.

The fleet stages every rank result before returning a receipt. An error after native work starts poisons the process-local fleet, so an uncertain batch cannot be retried there. Resident matrix payloads are capped at 1,250,000,000 bytes per fleet; input and output staging are each bounded to 16,777,216 values. This is a CPU matrix-projection component; it is not a full MoE expert MLP.

## Source reuse

The native operation remains `GGML_OP_MUL_MAT_ID` from llama.cpp/GGML commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` under MIT. VoltaFabric uses the pinned API through its existing native bridge; it does not copy GGML source files. The API intentionally keeps Q8_0, NVFP4, MXFP4, IQ3, and other formats distinct. This wave only verifies Q8_0.

Primary source references:

- [GGML operation declaration](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/include/ggml.h)
- [GGML graph operation](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml.c)
- [GGML CPU backend](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/ggml/src/ggml-cpu/ggml-cpu.c)
- [Pinned MIT license](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/LICENSE)

## Verification

On PC6 AArch64 CPU, the focused native-matrix integration suite passed 4/4 in debug and 4/4 in release. `rustfmt --check` passed for the two changed Rust files. Clippy passed with existing repository lints allowed (`dead_code`, `too_many_arguments`, `manual_checked_ops`, and an existing `needless_range_loop` in the earlier integration test); all other warnings remained denied.

The tests cover two fresh batches with nonidentity physical resident order, one-time Collector weighting, rejection of a repeated full Stamp, and a second-rank FP16 overflow after the first rank has completed. The failed batch returns no receipt and makes the fleet terminal.

## Limits and next step

All evidence is synthetic Q8_0 CPU work. This wave does not exercise the subprocess `Session` or `RankWorker` protocol, a checkpoint, a complete generation, other matrix formats, GPU, inter-PC transport, production R53, or performance. The replay memory is process-local and is not a durable admission ledger. Next: wire the matrix provider through an identified VoltaFabric Session/owner request path without weakening the existing UNKNOWN/no-replay rules.
