# R661 — Qwen2MoE gate/up boundary on the pinned GGML path

## Targeted donor finding

The pinned donor is llama.cpp/GGML revision `3cf03257f219afbe7334045ff7c6a06ac68c627d` (MIT). Its [Qwen2MoE tensor loader](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/models/qwen2moe.cpp) stores separate `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps` matrices. Gate/up use input width `n_embd` and expert width `n_ff_exp`; down maps `n_ff_exp` back to `n_embd`. The pinned [graph builder](https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/llama-graph.cpp) calls `build_moe_ffn` with `LLM_FFN_SILU`, softmax gating, and `expert_weights_scale`.

For Qwen2MoE, the donor multiplies selected expert weights after the down projection, then aggregates expert outputs. R660's unweighted per-slot projection contract therefore matches the right boundary: do not multiply the route coefficient in gate/up or down; apply the selected weight once at the VoltaFabric Collector. This finding is specific to this pinned Qwen2MoE path; llama.cpp has architectures such as Llama 4 that weight before the FFN.

## VoltaFabric change

R661 keeps the resident tensor name in `GroupedProjectionBatchOutput` and adds `qwen2moe_gate_up_to_down_inputs`. It accepts only the pinned donor's separate gate/up tensor roles, identical full stamp and attempt, matching dimensions, and slot identity/order. It applies the ordinary SiLU(gate) × up transform, including the donor's generic optional clamp branch, and returns unweighted per-slot FP32 inputs suitable for the down projection. It rejects malformed tensor roles, mismatched batch identity, shape, non-finite values, and overflow.

This reuses the already integrated GGML CPU grouped-matrix projection at R660 and adapts the donor's separate gate/up + SwiGLU boundary. It does not copy or use llama.cpp's graph scheduler, router, checkpoint loader, or model runtime. It also does not yet execute the down projection or Collector inside one compound Session call.

## Verified blocker and next code seam

R660's public `exec_grouped_projection_batch` claims the complete `moe::Stamp` once and reserves one local attempt. [Its identity guard](https://pc4.tail0c55ff.ts.net/root/voltafabric/-/blob/r660-ggml-per-slot-projection-20261101/src/process.rs#L1013) rejects a second execution of that same stamp unless explicit recovery is used. Therefore, chaining three independent public calls for gate, up, and down with one real model-layer identity is invalid; inventing three batch stamps would break VoltaFabric's batch contract. R661 deliberately does not use recovery or synthesize stamps.

The next implementation must be one compound Session coordinator: preflight all three resident matrices and their per-rank bindings, claim/reserve the layer identity once, execute gate/up/down with the same attempt across their separate resident workers, invalidate every participant on any error, and expose only complete unweighted down rows. The existing Collector then applies the model's selected expert weights once. This is the direct boundary toward a complete Qwen2MoE FFN stage.

## Verification and limits

Two focused CPU integration tests were added: an independent FP64 scalar SiLU oracle over synthetic routed slots, clamp coverage, and refusal for tensor-role or batch mismatch. They have not been executed. The exact pinned GGML CPU static libraries are present in the earlier local build directory `/home/pc6/NAOS/60_travaux/voltafabric-gitlab-r365/.runs/vf-r592-build`, but the available VoltaFabric source checkout is a dirty R367 tree without R660. No R661 Cargo compile/test, real checkpoint, generation, GPU, inter-PC transport, production R53, or performance qualification is claimed.

The change is limited to the CPU projection composition seam. QPN2/NVFP4 K16, MXFP4/R13 K32, and IQ3 remain distinct. No weight-format conversion is introduced.
