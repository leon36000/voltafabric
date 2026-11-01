# R688 — connect the pinned Qwen2MoE graph to resident VoltaFabric Sessions

## Result

R688 joins the pinned llama.cpp Qwen2MoE graph to VoltaFabric's existing Rust rank-local GGUF loader and high-level Qwen2MoE Session path in a CPU-only host integration. On the synthetic one-token, one-layer Q8_0 fixture, the donor and VoltaFabric outputs differ by at most **8.71369e-8** (test threshold `1e-6`). The test also rejects an invalid output shape without changing the caller's sentinel buffer and retains R687's check that a partial provider failure publishes no result.

This is a useful tested integration boundary, not complete model generation or production readiness.

## What is reused

- **Donor:** llama.cpp / GGML at `3cf03257f219afbe7334045ff7c6a06ac68c627d`, whose checked-out `LICENSE` is MIT. The test uses the existing Qwen2MoE `build_moe_ffn` graph construction and the pinned GGML `MUL_MAT_ID` / SwiGLU execution semantics. The donor checkout is external, pinned, and unmodified; its source is not vendored by this change.
- **VoltaFabric:** the existing `Session::load_grouped_matrix_gguf_expert_slices` boundary loads selected expert slices from GGUF into rank-local Sessions. The high-level `Session::exec_qwen2moe_ffn_batch` composes gate, up, SwiGLU and down under one layer identity and attempt. `collect_qwen2moe_ffn_output` uses the ordinary FP16-decoded Collector with f64 accumulation and applies each route coefficient once.
- **Rank-local mapping:** resident order `[40,3]` maps to source checkpoint experts `[1,0]`. Three Q8_0 tensors each contain two selected expert slices of 2,176 bytes: 6,528 of 13,056 tensor bytes are read (50%). The donor graph routes weights `0.25` and `0.75` to those experts.

The C++ test harness and Rust coordinator run on the same host. Rust Session rank workers are child processes; this test does not claim that all work runs in one process.

## Numerical boundary learned

The first R688 prototype used the lower-level grouped-matrix FP16 Collector after each projection and differed by `0.00195304`. That API rounded gate/up intermediates to FP16 before SwiGLU. The donor graph and VoltaFabric's existing high-level Qwen2MoE path keep those intermediates in FP32 and collect only the final down outputs. R688 now uses that existing higher-level boundary. No new numeric mode was added.

## Verification

- Pinned CPU-only llama.cpp / GGML build and the verified R655 static GGML CPU archives; no GPU or production network.
- Rust loader built offline; the C++ integration compiled with `-std=c++17 -Wall -Wextra -Werror`.
- Donor-vs-VoltaFabric maximum absolute error: `8.71369e-08`, below `1e-6`.
- Invalid output-shape request: refused before output publication; sentinel unchanged.
- Partial provider failure: no partial output returned.

Reproduction entry point: `adapters/native_ggml/run_qwen2moe_r659_rank_local_r688_test.sh`. It requires explicit paths for a clean pinned llama.cpp checkout, the already built CPU library and a test output directory. The runner verifies the donor commit and builds offline.

## Limits and remaining work

The fixture is synthetic, one token, one layer, two Q8_0 experts and one CPU host. It does not qualify a real checkpoint, tokenizer, prompt/decode loop, multiple layers, larger top-k, NVFP4/QPN2 K16, MXFP4 K32, IQ3, GPU/V100/CUDA graphs, inter-PC transport/RDMA, R53 production lease, performance, or complete generation. The bridge is an integration test and loader crate, not yet a complete backend for llama.cpp or another inference engine.

R688 adapts a donor graph boundary and invokes VoltaFabric's existing Session APIs; it does not inherit the donor's correctness guarantees automatically. Follow-up should connect the same resident Session boundary to the model's real layer execution contract, then qualify checkpoint formats and the full token loop without changing route weighting or Collector semantics.
