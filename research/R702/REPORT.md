# R702 — Bounded DeepSeek gate projection from Safetensors

Date: 2026-11-02  
Branch: `r702-deepseek-gate-safetensors-20261102`  
Parent: R701 receipt commit `f897902386f2290a294eb1af7a581b8a87d84f99`

## Delivered

VoltaFabric now has a selective named-tensor reader in `src/checkpoint.rs`. It
preflights a bounded Safetensors header, exact names/dtypes/shapes/payload sizes,
the exact shard length, and caller-pinned SHA-256 values before returning any
selected payload. Its hard payload limit is 64 MiB; a failed hash or read returns
no partial import. Unselected tensor ranges are skipped by seeking.

`src/deepseek_v41_gate.rs` now adapts the pinned DeepSeek V4.1 Flash layer gate
projection boundary. It loads only `layers.{n}.ffn.gate.weight` BF16
`[384,5120]`, `bias` F32 `[384]`, and `bias_vl` F32 `[384]`, for a fixed total
payload of 3,935,232 bytes. Model revision, layer, temperature, route scale and
three per-tensor SHA-256 values are explicit inputs. The immutable BF16 matrix is
kept resident in CPU memory; one hidden vector is projected with a scalar FP32
reference loop, then passed to the existing R700 selector. The selector applies
the supplied route scale once and uses text/image correction biases only for
membership.

The load receipt reports the pinned model revision, layer, exact file/header and
selected payload/I/O byte counts, and the selected tensor hashes. R702's fixture
adds an unrelated 65,536-byte tensor and proves the loader reads only the header
and three requested gate tensors.

## Reuse and provenance

- Reused the already-pinned Rust dependency `safetensors = 0.8.0` (Apache-2.0)
  and VoltaFabric's existing in-crate SHA-256 implementation. No third-party
  source was copied.
- Adapted the gate operation order from the MIT-licensed DeepSeek V4.1 Flash
  source pinned at `314113f8a7dadb388215396ba5fbee2c610a04d6`; no upstream code
  was copied. The loader mapping for `route_scale` is still caller-owned and is
  deliberately not inferred.
- Existing retained metadata identifies the layer-0 gate tensors in
  `model-00003-of-00048.safetensors`: header SHA-256
  `71213f444f9c15a4af6dbcbddc36f7611542884ebb3aadef54de650071357449`; index
  SHA-256
  `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`.
  These are metadata artifacts, not the tensor shard or its payload hashes.
- The donor source and license are retained under
  `evidence/continuation_a97_model_metadata/upstream_314113f8/`.

## Verification

- `CARGO_BUILD_JOBS=1 cargo test --offline --locked --test r702_deepseek_gate_projection`: 2 passed.
- `CARGO_BUILD_JOBS=1 cargo clippy --offline --locked --test r702_deepseek_gate_projection -- -D warnings`: passed.
- `rustfmt --edition 2021 --check src/checkpoint.rs src/deepseek_v41_gate.rs tests/r702_deepseek_gate_projection.rs`: passed.
- `git diff --check`: passed.
- The fixture's BF16 rows encode logits `0..383` exactly. Text and image routes were checked; text coefficients match the independent R701 Decimal oracle within `2e-7`. Revision, per-tensor hash, wrong dtype/shape, missing layer, and under-budget imports all fail closed.

## Limits and next boundary

This is a synthetic CPU fixture, not a load of the real 7.39-GB shard. The
projector's scalar FP32 accumulation is a reference order, not bitwise parity
with PyTorch's linear/reduction kernels. It does not dispatch to Sessions. The
retained R698 Session fixture only qualifies real experts 0 and 383; R700's
synthetic top-6 also selects 382, 381, 380, and 379, whose matching live payloads
were not found in the inspected R698 fixture tree. Keep Session admission closed
until the gate payload and all six selected expert profiles are available.
No full checkpoint download, GPU, inter-PC transport, production service, or
full generation was performed.

## Sources

- [Pinned DeepSeek V4.1 Flash gate source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py)
- [Safetensors 0.8.0 source and Apache-2.0 license](https://github.com/huggingface/safetensors/tree/v0.8.0)
