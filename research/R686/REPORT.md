# R686 — R659 rank-local Q8_0 slices at the pinned donor graph boundary

## Change

A CPU-only adapter now models the R659 rank-local slice order `[resident expert 40, local 0, source checkpoint expert 1]`, `[resident expert 3, local 1, source checkpoint expert 0]`. To make the selective read observable, the synthetic source GGUF has four Q8_0 experts while the rank-local GGUF passed to the pinned donor contains only those two resident slices. The source and compact files are both GGUF v3 with separate Qwen2MoE gate/up/down tensors.

The slice reader uses the pinned donor GGUF parser in metadata-only mode, validates type, shape, tensor spans and source expert count, then seeks and reads only each declared Q8_0 expert slice. It reads 6,528 of 13,056 source tensor bytes across the three matrices (50%) and frees the generated source fixture buffer before slice loading. The selected bytes are assembled into a compact rank-local GGUF, re-parsed by the donor, and loaded into the live Qwen2MoE `build_moe_ffn` CPU graph. The route adapter maps compact donor indices `[0,1]` to VoltaFabric experts `[40,3]` once.

## Verification and finding

- Donor: llama.cpp/GGML commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`, MIT; pinned donor sources were not changed.
- The direct runner compiled the bridge and fixture with C++17, `-Wall -Wextra -Werror`, then ran eight donor `process_ubatch` graph cases on CPU.
- Both mapped routes reached the collector with the expected IDs and final sigmoid weights (0.25 and 0.75). Raw donor outputs stayed within `1e-6 + 1e-3 * abs(scalar)` of the separate scalar Q8_0 FFN oracle.
- The bridge callback decoded the final slot as FP16, multiplied the route coefficient once, and the layer wrapper accumulated route results in f64. For resident expert 3 / source checkpoint expert 0, the scalar Collector matched within `8.72e-8`. For resident expert 40 / source checkpoint expert 1, output element 0 differed by `0.00195304` (donor `3.73706`, scalar `3.73901`; per-route FP16 Collector error `0.00195312`). Thus the two-expert combined result does not match the separate FP16 Collector oracle.
- This mismatch is consistent with differing intermediate precision/SiLU behavior between the donor's fused FFN graph and R659's separate projections followed by FP16-decoded Collector stages. This experiment does not isolate which internal operation causes the boundary crossing; it explicitly rejects full numerical equivalence.
- A two-route wrapper staged the first successful expert and then received a provider failure from the second. The graph failed and the wrapper published no partial layer output. A duplicate resident-to-registry map was rejected before collector dispatch.

## Reuse and limits

The adapter reuses the pinned llama.cpp/GGML metadata parser, Q8_0 representation and Qwen2MoE `build_moe_ffn` graph plus the existing VoltaFabric route bridge. It adapts the R659 resident/checkpoint slice mapping and tests bounded slice reads without staging each full source tensor. This test ports the R659 descriptor and extraction behavior in C++; it does **not** invoke the Rust `Session::load_grouped_matrix_gguf_expert_slices` method in the same process as the donor graph.

The result is a synthetic one-layer CPU bridge. It does not use a production checkpoint, prove whole-model generation, qualify GPU/V100, inter-PC/RDMA, production R53, other quantization formats, or performance. GitLab's configured scan pipeline, if present, is not evidence for this C++ runner; the local direct-run receipt is the test evidence. The full adapter CMake/CTest graph is not registered by this standalone slice test.

## Next

R687: isolate the donor-versus-VoltaFabric intermediate rounding/SiLU difference with a scalar FP32 control and a stage-by-stage FP16 reference, then implement or reject an explicit compatibility mode before attempting full-layer equivalence. Keep exact route weighting and atomic refusal, CPU-only, with no automatic relaunch.
