# R681 — live llama context attachment and coordinator callback path

## Scope

R681 closes the R680 attachment-ownership gap with an AArch64 CPU fixture. It also enables the existing VFR3 process fixture to exercise the Qwen2MoE callback-table bridge while a real VoltaFabric coordinator and `vf-rank-session` worker process the local fixture. This is a bounded connector test before checkpoint loading.

## Reused and adapted components

- Reuses llama.cpp revision `3cf03257f219afbe7334045ff7c6a06ac68c627d` (MIT), the R679 internal Qwen2MoE hook seam, the R680 bridge, the R678 invocation/collector boundary, and the R672 VFB3/VFR3 local client and route-graph path.
- Tightens `Qwen2MoELlamaRouteBridge::attach`: it rejects null factory/collector, non-Qwen2MoE contexts, an already attached bridge, and a context whose hook slot is occupied. Detach clears only this bridge's owned hook; if ownership changed, it leaves the foreign hook intact.
- Adds a live `llama_context` ownership fixture using a small vocab-only test model. This validates architecture/configuration rejection, successful attach, duplicate/competing attach rejection, lost-ownership behavior, and safe destruction-time detach. It does not load checkpoint tensors.
- Adds an opt-in process-test mode that passes the adapted callback table and a fresh route batch through the actual Rust coordinator and `vf-rank-session` worker, then through the GGML VFR3 graph/collector fixture.

## Verification

- Pinned llama.cpp CPU bridge build succeeded with the CUDA, Vulkan, Metal, native-tuning and OpenMP options disabled.
- Focused CTest group: 7/7 passed, including the new live-context ownership test.
- Offline Rust build succeeded for `vf-strata-native-bridge` and `vf-rank-session` with `native-ggml-cpu`.
- One fresh process-level test passed using newly generated run/model/plan/layer/batch/epoch/attempt identities. It invoked the callback bridge once, reached the coordinator and rank worker, and checked the GGML-collected result.
- Receipt: one ledger claim with status `RESERVED_UNKNOWN_NO_REPLAY`. The test is complete and must not be replayed.
- Local `src/session.rs` was verified byte-for-byte against its GitLab blob (`f73e0b48bdc15689acb3640a96fb581c6b31b133`). The Rust snapshot is reconstructed for this validation; this report does not claim a byte-exact checkout of the entire branch.
- The GitLab pipeline on the previous R680 commit covered security scans/fallback only, not this local C++/Rust build evidence. R681's build/test evidence is produced locally.

## Limits

The process fixture calls the bridge callback table with synthetic GGML route data; it does not invoke llama.cpp `llama_context::process_ubatch` or a real Qwen2MoE model graph. It uses no checkpoint, tokenizer, token loop, or full generation. It does not qualify GPU/CUDA graphs, inter-PC/RDMA, production R53, or performance. The sparse local pack has 50,292,326,400 logical bytes; the worker wrote 2,176,000 payload bytes. This is a synthetic transport fixture, not a model artifact.

## Next

R682 should connect the hook through donor `process_ubatch` on a bounded CPU-only Qwen2MoE fixture, or first establish a small compatible model fixture if the pinned donor cannot construct one without checkpoint tensors. Keep the one-shot R681 receipt terminal; do not replay it. Automatic relaunch remains stopped by owner.
