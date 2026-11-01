# R662 — Atomic Qwen2MoE FFN across resident VoltaFabric Sessions

## Donor and operation

This tranche adapts pinned llama.cpp/GGML revision `3cf03257f219afbe7334045ff7c6a06ac68c627d` (MIT), whose Qwen2MoE implementation keeps `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps` separate. Its gate/up path is SiLU(gate) multiplied by up; the selected route coefficient is applied after down.

R661 added the model-specific unweighted gate/up composition boundary. R662 adds one VoltaFabric Session coordinator for all three projections. It preflights tensor names, rectangular shapes, routed resident bindings, slot coverage, and input bounds before claiming the layer Stamp or reserving one local attempt. Gate, up, SwiGLU, and down use that same attempt. The coordinator exposes only complete unweighted down rows. On any post-reservation error it invalidates all participating stage Sessions and exposes no partial result.

## Readback correction

R661 source and test readbacks contained a literal `/dev/stdin:` tool marker copied between Rust blocks. R662 removes those two stray lines. Scan-only CI had not compiled the Rust code and did not detect them. R662 records the correction and does not claim compilation.

## Reuse and adaptation

The implementation reuses VoltaFabric's persistent resident GGML CPU rank Sessions and per-slot projection request/response format from R660. It adapts the donor's Qwen2MoE FFN operation without copying its graph scheduler or runtime. Three tensors remain separate residents. The existing Collector remains the only place to apply selected expert weights once; returned down rows are unweighted FP32.

The local NativeBatchLedger is not external R53. The caller still has to reserve the verified external identity and lease before entering a production path. This change does not bridge the external R53 ticket into Session, change quantization formats, or run on GPU/network.

## Test added

`qwen2moe_compound_ffn_uses_one_identity_and_returns_unweighted_down_rows_and_preflights_all_stages` builds synthetic Q8_0 gate/up/down residents, checks results against an independent scalar SwiGLU/Q8_0 oracle, proves wrong tensor-role preflight leaves Sessions usable and the Stamp available, then verifies replay refusal after success. Route coefficients are asserted absent from returned rows.

The test has **not been executed**; no Rust compile/type-check is claimed. Scan-only CI is not compile evidence.

## Limits

Synthetic CPU fixture only. No real Qwen2MoE checkpoint, tokenizer, complete generation, external R53 reservation, GPU, inter-PC/RDMA, production transport, or performance comparison is qualified. QPN2/NVFP4 K16, MXFP4/R13 K32, and IQ3 remain distinct; no format conversion is introduced.
