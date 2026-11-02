# R724 — load the pinned DeepSeek gate from the indexed shard

## Change

R723 connected the gate-selected top-6 route to R722's indexed expert reader. R724 adds the parallel indexed checkpoint path for the gate projection itself. `DeepSeekV41Gate::load_indexed` authenticates the pinned index, requires all three gate tensors to map to one source shard, checks the source header digest and size before and after import, and delegates selected-tensor payload validation to R721's bounded Safetensors reader. It then calls the same resident BF16/F32 decoder used by the existing stream loader.

The gate spec is checked before opening a source: model revision/configuration, total gate payload ceiling, and all three payload-hash formats. The index/header and each tensor digest remain separate checks. No source weights are copied into another format during checkpoint selection.

The integration fixture now stores synthetic gate tensors and the six synthetic expert payloads in one sparse source file with the committed pinned DeepSeek index and header. It loads the gate through the new indexed path, compares expert order and coefficient bits against the existing stream gate loader, derives the route, reads all six indexed experts, converts all owners atomically, and executes the existing CPU Sessions and independent FP64 MXFP4 oracle.

## Reuse and provenance

- Reuses R721's pinned-index and root-confined indexed Safetensors reader, R722's expert import path, and R723's route-level preflight/read helper.
- Keeps the existing VoltaFabric gate projection decoder, selector, route identity and coefficient calculation. No upstream implementation code was copied; no dependency was added.
- The pinned index/header metadata and prior vLLM/SGLang license screening are recorded in [R721](../R721/REPORT.md); DeepSeek model metadata is pinned under MIT as noted in [R722](../R722/REPORT.md). These sources provide formats and interfaces, not automatic correctness or checkpoint compatibility.

## Verification

Debug and release pass for the same-file indexed gate → gate route → six indexed experts → resident owners → grouped CPU Sessions → FP64 MXFP4 oracle integration. The non-pinned-index and malformed gate-digest cases both refuse before opening a shard in debug and release. Scoped library/integration Clippy with `-D warnings` and rustfmt pass.

## Limits and next step

All checkpoint payload bytes in this fixture are synthetic; only the pinned index and shard header metadata are actual committed evidence. This proves neither a real model generation nor GPU/inter-PC execution, production R53, or performance. Next: connect indexed gate and expert imports to the model-layer entry point and build a single-token layer/forward path while preserving the existing owner, attention/norm and Collector contracts; actual checkpoint admission remains a separate prerequisite.
