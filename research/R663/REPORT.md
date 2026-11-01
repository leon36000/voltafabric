# R663 — Qwen2MoE FFN to VoltaFabric Collector bridge

## Reused component and adaptation

R663 builds on the private R662 coordinator and the pinned llama.cpp/GGML Qwen2MoE reference at `3cf03257f219afbe7334045ff7c6a06ac68c627d` (MIT). It adds a narrow adapter from the coordinator's complete, ordered, unweighted FP32 down rows to VoltaFabric's existing `moe::Collector` FP64 completion contract.

Before creating the collector, the adapter checks the full layer Stamp, nonzero local attempt, exact down tensor name, rectangular dimensions, complete dispatch slot count/order, selected expert/rank/local bindings, and finite values. It then submits those unweighted rows as ordinary completions; Collector applies the saved route coefficient once and reduces in its established token/slot order. This is a CPU composition boundary, not proof of who produced a caller-constructed public result struct.

## Evidence added

The R662 synthetic Q8_0 integration test now also rejects a stale batch Stamp and compares Collector output against a separate scalar oracle with the route coefficients applied once. This test remains unexecuted. No Rust compile or test is claimed.

## Readback correction carried forward

R662's exact committed `src/session.rs`, test, and report readbacks match their SHA256SUMS entries. The literal `/dev/stdin:` markers in the R661 source/test are absent from the R662 readbacks. R662 scan-only security jobs are still distinct from compilation evidence.

## Scope limits

The code adapts one Qwen2MoE FFN boundary and reuses resident grouped GGML CPU Sessions plus the existing Collector. It does not integrate tokenizer, attention, normalization, residuals, the remaining layer graph, autoregressive scheduling, or a real checkpoint. It does not qualify QPN2/NVFP4 K16, MXFP4/R13 K32, or IQ3 interchange; external R53 reservation, GPU, inter-PC/RDMA, production networking, or performance. The local ledger is not an external lease. No UNKNOWN operation was retried.
