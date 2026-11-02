# R727 — pinned DeepSeek FFN Hyper-Connections and shared FP8 operator

## Result

Added two bounded CPU references for the pinned DeepSeek-V4.1-Flash FFN boundary:

1. `deepseek_hc_pre_reference` collapses the four residual copies using the supplied
   FP32 `pre_mix`, checks finite exact-BF16 residual values, and rounds once to BF16.
2. `Fp8E4m3Matrix` and `DeepSeekV41SharedExpert` preserve the shared expert's separate
   E4M3FN/E8M0 representation and execute its three FP8 SwiGLU projections as a CPU
   format oracle. The result is raw FP32 and unweighted; the caller must add the shared
   result once after the routed experts complete.

The pinned model index and layer-0 header confirm that each layer's six shared expert
weight/scale tensors are in the same shard as its gate and `ffn_norm.weight` across
40/40 backbone layers. The exercised matrix payloads are synthetic; no checkpoint
weights were opened.

## Adapted boundary and provenance

The reference follows the pinned MIT implementation at
[`model.py`](https://github.com/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py)
and [`kernel.py`](https://github.com/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/kernel.py).
No upstream implementation code was copied and no dependency was added. FP8 E4M3FN
codes with E8M0 scales remain a distinct type; they are not routed MXFP4 K32 experts
and have no route coefficient. The code remains a scalar CPU reference, not a production
fallback or GPU kernel.

The separate arXiv/repository screen in `OPEN_SOURCE_SCREEN.md` found a promising
write-time KV-residency design for later study and an optional expert-prefetch idea.
Neither is part of this change: routing remains exact, and only completed routed output
may enter the result path.

## Verification

The focused CPU suite checks pinned metadata and 40-layer shard colocation, the HC-pre
equation against an independent FP64 calculation, all three shared FP8 projections
against an independent blockwise oracle, and refusal of malformed FP8/HC inputs. The
exact final command outputs and hashes are in `BUILD_TEST.txt` and `SHA256SUMS`.

## Limits and next boundary

This change does not load the shared expert's actual selected checkpoint payload, join
the shared result to the routed Session/Collector output, compute learned
`hc_mixes`/Sinkhorn coefficients, apply `hc_post`/residual mixing, execute attention,
or generate tokens. It does not qualify GPU, inter-PC transport, production R53, or
performance. R728 should add the indexed six-tensor FP8 shared-expert reader and bind
its unweighted output to the completed routed contribution with full identity and
atomic-failure behavior; `hc_mixes` and `hc_post` stay explicit follow-on boundaries.
