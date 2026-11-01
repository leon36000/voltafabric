# R672 — GGML VFR3 transaction with the real local coordinator and worker

## Result

R672 replaces R671's fake AF_UNIX peer for this test with the actual VoltaFabric `vf-strata-native-bridge` coordinator and its spawned `vf-rank-session` child. The C++ `vf::strata_bridge::Client` still speaks VFB3/VFR3 to the coordinator; the worker's child protocol remains behind that boundary.

The new opt-in CTest starts the coordinator as a local process over an AF_UNIX socketpair, performs its JSON ready and binary-mode handshake, and passes an identified one-slot route through the existing GGML `RouteGraphTransaction` and VFR3 collector. A fresh temporary directory holds the synthetic indexed SC117-format pack and local single-use ledger. It checks the returned row after FP16 encode/decode and a single 0.25 route coefficient, requires the downstream graph node to run once, rejects graph replay, observes clean coordinator exit, and checks one `RESERVED_UNKNOWN_NO_REPLAY` claim.

The logical pack is 50,292,326,400 bytes, created sparsely; the only expert payload written is 2,176,000 bytes. This fixture tests the process and graph boundary, not a real SC117 model or checkpoint.

## Reuse and adaptation

- Donor: `ggml-org/llama.cpp`, pinned revision `3cf03257f219afbe7334045ff7c6a06ac68c627d`, MIT. The existing build checks this revision and keeps CUDA, Vulkan, Metal, native tuning and OpenMP disabled.
- Reuses VoltaFabric's existing C++ VFB3/VFR3 client, Rust coordinator, `vf-rank-session` worker, GGML graph transaction and ordinary FP16-decoded/f64 Collector. No new transport protocol was introduced.
- Adds only a process-level test target in CMake and its synthetic fixture input; the production graph/collector implementation is unchanged by R672.

## Pinned donor route seam

Read-only inspection of the official pinned source identifies a concrete integration point:

- The [Qwen2MoE model graph](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/models/qwen2moe.cpp#L131-L142) calls `build_moe_ffn` with softmax gating and `hparams.expert_weights_scale`.
- In [llm_graph_context::build_moe_ffn](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/llama-graph.cpp#L2107-L2155), the graph produces selected IDs with `ggml_argsort_top_k`, gathers their probabilities, applies any configured normalization and `w_scale`, then expands the resulting weights before expert execution.
- The same function later multiplies the down-projection rows by those weights before summing experts. A VoltaFabric callback must consume IDs and final weights after scaling, replace the routed-expert aggregate, and avoid the donor's subsequent weight multiply; the Qwen2MoE shared-expert and residual paths remain in the graph.

This is source tracing only. The R672 test still supplies a synthetic route and does not execute that llama.cpp runtime callback.

## Verification

On the AArch64 CPU, the four-target CMake build completed with C++17 `-Wall -Wextra -Werror`. CTest passed 4/4: `native_provider`, `route_graph_transaction`, `route_graph_vfr3`, and `route_graph_vfr3_bridge_process`. A direct invocation of the process test also passed with a fresh run/model/plan/batch/epoch/attempt and ledger claim.

The process-test CMake target returns 77 (CTest skip) unless the isolated run explicitly supplies the coordinator binary, worker binary, fixture index and Naos temporary root. The successful run supplied all four paths explicitly.

The Rust executables were built offline from the isolated source snapshot documented in `SNAPSHOT.txt` (R663 source baseline with selected later Session sources). The coordinator and worker entrypoint files, Cargo manifests, and several shared modules were byte-identical to the current GitLab readback. The full Rust source tree was not an exact current-tip checkout: `src/moe.rs`, `src/session.rs`, `src/native_ggml_cpu.rs` and `src/strata_native_bridge.rs` differed. Thus this qualifies an actual coordinator/worker process integration against that local Rust build, but it does not claim a reproducible full Rust build of the current GitLab tip. The exact branch source remains committed on GitLab and this caveat is retained for the next step.

## Limits

- The route ID and final weight in this fixture are synthetic caller inputs. No pinned llama.cpp inference runtime callback supplies them yet.
- The expected row uses the pinned GGML native provider on the synthetic payload; this is an integration consistency check, not an independent mathematical oracle.
- No real checkpoint, tokenizer, attention/residual/token loop, complete generation, production R53 reservation, GPU, inter-PC/RDMA, or performance comparison was exercised.
- The local ledger is an isolated test fixture; no production or shared reservation ledger was used. A reserved UNKNOWN claim is never replayed.
- This does not qualify QPN2/NVFP4 K16, MXFP4 K32, or IQ3 execution.

## Next step

R673: add a dynamic GGML custom op at the pinned `build_moe_ffn` seam that consumes actual selected IDs and post-scale weights, submits the identified route through the existing VoltaFabric client, and replaces only the routed-expert aggregate. Keep coefficient application single-use and preserve the Qwen2MoE shared-expert and residual branches. First reproduce the coordinator build from exact GitLab source readback; then test only with a fresh synthetic CPU fixture. Do not claim complete generation until a compatible checkpoint runs through the complete token loop.

