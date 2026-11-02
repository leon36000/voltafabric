# R725 — indexed DeepSeek MoE layer entry point

## Change

R724 loaded the pinned gate projection and then imported the gate-selected experts through separate public calls. R725 adds `DeepSeekV41IndexedMoeLayer`, a resident MoE-sub-layer boundary that binds those operations to one pinned index, one expected header digest, one shard name and one source-size ceiling.

Construction verifies that the gate and expert requests contain the same pinned index bytes and header identity. It resolves the gate tensor names from the pinned index and refuses a gate/expert shard mismatch before opening the source file. It loads only the gate tensors and checks the resulting file/header sizes against the expert source metadata.

For each token, `prepare_token` accepts the post-attention/post-normalization activation plus the full VoltaFabric stamp, attempt, registry and resident ordering. It returns a private route-and-payload plan, preserving the gate's selected order and already-scaled coefficient bits. `read_selected_payloads` feeds that plan to the existing atomic route reader; no partial owner set escapes on failure.

This is an indexed MoE sub-layer entry point. It does not implement attention, RMSNorm, residual updates, the rest of the transformer layer, or text generation.

## Reuse and provenance

- Reuses R721's documented Hugging Face index-to-shard convention and existing bounded Rust Safetensors reader, then R722/R723/R724's pinned DeepSeek expert, gate and route paths.
- Keeps VoltaFabric's gate projection, selector, route identity, coefficient calculation, owner construction, grouped CPU Sessions and FP64 oracle. No donor implementation code was copied and no dependency was added.
- R721 records the inspected vLLM/SGLang source versions and Apache-2.0 licenses. This boundary adapts their checkpoint metadata convention; donor license and maturity do not certify VoltaFabric's model compatibility or runtime behavior.
- The R725 pinned-index regression establishes that, for each of 40 layers, the gate tensors, `attn_norm.weight`, `ffn_norm.weight` and all 384 × 6 expert payload tensors are mapped to one shard (40/40). For layer 0, both norm tensors are BF16 `[5120]` in `model-00003-of-00048.safetensors`. This is pinned index/header metadata; the norm payload values have not been read.
- The pinned DeepSeek config records `rms_norm_eps=1e-20`; the exact model source at revision `314113f8a7dadb388215396ba5fbee2c610a04d6` casts input to FP32, computes mean-square over the last dimension, applies `rsqrt(mean_square + eps)`, multiplies the learned weight and casts back to the input dtype. This is a reference contract for the next CPU normalization boundary, not code copied into VoltaFabric.

## Verification

The synthetic gate → route-and-index-plan → six indexed payloads → resident owners → grouped CPU Sessions → independent FP64 MXFP4 oracle integration passes in debug and release. A negative test gives the gate and expert sides different shard identities and confirms refusal before any file is opened in both profiles. The updated 40-layer pinned-index test confirms co-location of the gate, both norm tensors and expert payload names in debug and release. Scoped library/integration Clippy with `-D warnings`, rustfmt and `git diff --check` pass.

## Limits and next step

All gate and expert payload bytes remain synthetic; the committed index and shard-header metadata are the real pinned inputs. No model generation, real checkpoint payload, GPU, inter-PC execution, production R53 or performance result is established. The new type begins after attention and normalization and covers the MoE sub-layer only.

Next: implement a separate CPU RMSNorm primitive for the pinned contract (`eps=1e-20`, FP32 square/mean/rsqrt/weight multiply, cast back to the input dtype) with an independent numerical oracle, then load the actual selected norm payload and connect its normalized activation to this MoE entry point. The static norm metadata is pinned, but real norm values remain unread. Keep attention-state integration as a later, separately qualified boundary.
