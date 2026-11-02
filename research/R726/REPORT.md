# R726 — bind ffn RMSNorm activation to the indexed routed-expert plan

## Change

R726 adds a CPU reference for the pinned DeepSeek-V4.1-Flash `ffn_norm.weight` boundary. The loader resolves `layers.{layer}.ffn_norm.weight` through the pinned Safetensors index, verifies the admitted shard/header identity and byte bounds, authenticates the selected BF16 `[5120]` payload by SHA-256, rejects nonfinite weights, then retains decoded FP32 weights in memory. The scalar apply path accepts only finite, exactly BF16-representable FP32 values; it follows the pinned FP32 square-sum/mean/epsilon/sqrt-reciprocal/multiply sequence and rounds the result back to BF16.

`DeepSeekV41IndexedMoeLayer::prepare_ffn_token` now applies that norm once, derives the gate route and indexed expert payload plan from its result, and retains that same normalized activation inside the private token plan. The existing grouped-session integration now feeds `plan.activation()` to the expert owners. This prevents the execution path from routing a normalized vector while handing the owners an independently normalized or raw vector. A direct post-normalization route in the test checks that expert order and coefficient bits match exactly.

## Reuse and source review

- The model configuration and source are pinned to DeepSeek-V4.1-Flash revision `314113f8a7dadb388215396ba5fbee2c610a04d6` (MIT). The math contract is reimplemented in Rust; no model-source code is copied and no dependency is added.
- The code reuses VoltaFabric's R721 indexed named-tensor/Safetensors reader and R725 gate, route, payload reader, resident owners, grouped CPU Sessions and FP64 MXFP4 oracle.
- A focused arXiv/source screen is recorded in `OPEN_SOURCE_SCREEN.md`. MonoMoE's weight-major persistent scheduling and readiness handoffs are worth a later scheduling experiment, but its reported kernels target H200 and are not a V100 port. Flash Linear Attention is a MIT-licensed algorithm/oracle reference for linear-attention work; it is not imported in this change. Gated DeltaNet-2's implementation is under NVIDIA Source Code License-NC, so its code is excluded. FlashKDA's documented SM90+/CUDA 12.9+ floor does not fit V100. The V100 kernel path must be qualified separately; upstream project maturity is not evidence of SM70 compatibility.
- This round does not import components from Strata, Maya, TensorRT-LLM or llama.cpp. R721's earlier vLLM/SGLang review remains limited to checkpoint index conventions; it did not connect their engines to this model.

## Verification

- `cargo test --locked --offline --test r726_deepseek_ffn_norm`: debug 2/2 and release 2/2.
- `cargo test --locked --offline --test r712_deepseek_session_owner gated_top6_is_preflighted_grouped_loaded_and_executed_without_restating_route`: debug 1/1 and release 1/1. The integration uses a synthetic norm/gate/expert shard, validates the exact normalized activation and coefficient bits, loads six indexed expert payloads into five CPU Sessions, and matches the independent FP64 MXFP4 oracle.
- Library and both targeted test Clippy runs pass with `-D warnings`; targeted rustfmt and `cargo fmt --all -- --check` pass.

## Limits and next boundary

The test shard has the real pinned index/header and synthetic norm, gate and routed-expert payload bytes. It does not read the checkpoint's real `ffn_norm.weight` or routed expert payloads, run a full checkpoint, or establish GPU/inter-PC performance.

The pinned `model.py` also shows that this is not yet the model's full FFN: `MoE.forward` adds one shared expert after the routed experts, and `Block.forward` computes Hyper-Connection `hc_pre`/`hc_post` mixes around attention and FFN. R726 currently starts at the FFN activation boundary and implements only the routed branch plus its input norm. Next, pin and implement the FFN Hyper-Connection pre-mix and the shared-expert contribution, then compare the combined FFN sublayer with an independent source-derived oracle. Attention, the HC residual/post-mix, logits and complete generation remain separate work.
