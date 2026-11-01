# R687 — explain the R686 Q8_0 collector delta

## Question and method

R686 found a `0.00195304` maximum difference between the pinned donor layer and a separate scalar Q8_0 / FP16-decoded / f64 collector. R687 captures the actual donor CPU graph's gate matmul, up matmul, SwiGLU node, and down matmul through GGML's evaluation callback. It compares six scalar controls that independently add FP16 rounding at gate, up, and activation stages, then adds an independent scalar implementation of GGML's Q8_0 activation quantization and integer dot semantics.

The fixture remains the synthetic R659-format Qwen2MoE case: Q8_0 gate/up/down, width 32, two rank-local experts `[40,3]`, and checkpoint map `[1,0]`. The donor is pinned to llama.cpp/GGML `3cf03257f219afbe7334045ff7c6a06ac68c627d` (MIT); its source tree was not modified.

## Finding

The six intermediate-rounding variants all reproduce the expert-40 collector delta (`0.00195312`); the captured SwiGLU stage itself agrees with the explicit `expf` formula at zero measured error. The earlier scalar oracle multiplied Q8_0 weights by the original F32 activations directly.

That omitted operation is present in the pinned GGML CPU path: Q8_0 uses Q8_0 vectors for its dot product (`ggml/src/ggml-cpu/ggml-cpu.c:272-275`); `mul_mat_id` converts a non-Q8_0 RHS with the selected `from_float` function (`ggml-cpu.c:1567-1568, 1611-1641`); the reference Q8_0 conversion uses `amax/127`, an FP16 scale, and rounded signed codes (`ggml/src/ggml-quants.c:276-297`); and the dot accumulates integer code products with the decoded FP16 scales (`ggml/src/ggml-cpu/quants.c:451-478`).

The independent scalar oracle now performs that Q8_0 activation conversion before each linear dot. Its gate, up, and down stages match the captured donor nodes with maximum absolute error `0` for both fixture experts. After route weights `0.25` and `0.75` are applied once and the ordinary FP16-decoded collector accumulates in f64, combined output error is `8.71369e-8` max. The previous unquantized-input scalar oracle still diverges `0.00195304` at output 0 (`3.73706` donor, `3.73901` scalar). This identifies the mismatch in this fixture as an oracle omission, not an FP16 gate/up/SwiGLU rounding requirement. No special FP16 compatibility mode is justified by this case.

## Verification

- CPU direct runner: C++17, `-Wall -Wextra -Werror`; eight graph cases, captured four donor stages per valid route, checked all seven scalar controls, duplicate-map refusal and atomic provider-failure withholding; exit code 0.
- The rank-local adapter reads `6528/13056` source tensor bytes (50%) for the two selected experts.
- R659 resident-to-checkpoint mapping and VoltaFabric registry mapping are each applied once; route coefficients are applied once at the collector boundary.
- Source manifest: `SHA256SUMS`; `RESULT.json` records the measurements and limits.

## Limits and next step

This is a synthetic single-layer CPU result. The C++ rank-local adapter still does not call the Rust `Session::load_grouped_matrix_gguf_expert_slices` loader in the same process. It does not qualify a production checkpoint, complete generation/token loop, top-k greater than one, other quantization formats, GPU/V100/CUDA graphs, inter-PC/RDMA, production R53, or performance.

R688 should compose the actual Rust rank-local loader with the pinned donor graph in one CPU process, preserving this Q8_0 activation conversion, the single route-weight application, and atomic refusal. No automatic relaunch, model download, GPU, or production network.
