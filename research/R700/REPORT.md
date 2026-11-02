# R700 — DeepSeek V4.1 Flash gate route boundary

Date: 2026-11-02
Branch: `r700-deepseek-v41-gate-cpu-20261102`
Gate source correction commit: `4483db95eca7b46e7e0ecc7bd6f81ee093e6c3b5`
Gate-to-admission test commit: `3982dff1f1bfd2ad4aaa5faaf2d5253555face9e`
Parent source commit: `7a6cfeb0aa1a85c77d6783ac4dc8322e91c54bf7`

## Work delivered

Added a Rust CPU reference for the primary text/image gate scoring boundary in
the MIT-licensed DeepSeek V4.1 Flash source pinned at
`314113f8a7dadb388215396ba5fbee2c610a04d6`. The implementation is an adaptation
of the published operation order, not copied donor code. It consumes already
projected FP32 router logits and caller-owned text/image correction-bias vectors,
then returns VoltaFabric `moe::Choice` values with normalized coefficients.
Selection correction affects membership only; the returned coefficients use the
unbiased sqrt-softplus scores. Route scale is applied once.

The adapter makes the route scale an explicit constructor argument and rejects
nonfinite or nonpositive values. This avoids hiding a source/config ambiguity:
the pinned model config names `routed_scaling_factor=1.5`, but the inspected
`ModelArgs` demo default is `route_scale=1.0`; no mapping between these fields was
found in the inspected loader code. `ROUTE_SCALE=1.5` is only a fixture reference,
not proof of loader behavior. A real caller must resolve and pass the correct
value.

The implementation rejects invalid vector sizes, NaN/infinity, and ties among
the selected six or across the top-6 cutoff. The stable tie behavior is a
fail-closed adaptation; it is not claimed to reproduce PyTorch `topk` tie order.
The local FP32 scalar reduction is likewise not a bitwise PyTorch oracle.

The new integration test feeds the six-choice gate result into the existing
`plan_routing_advisory_with_memory` boundary with a self-consistent six-expert
route registry but only the two real expert memory profiles retained by R699.
The planner refuses before any Session is created. This verifies that the
model top-6 is not truncated to the two available fixture experts. It is a
negative admission proof, not a positive six-expert dispatch.

## Reuse and limits

| Source | Reuse status in this work | Boundary |
|---|---|---|
| DeepSeek V4.1 Flash, pinned MIT source | Gate equations adapted in Rust; no donor code copied | No gate projection weights/loader, model checkpoint, shared expert, DSpark 128/top-3 gate, GPU kernel, or generation |
| llama.cpp/GGML, pinned MIT source at `3cf03257f219afbe7334045ff7c6a06ac68c627d` | Existing VoltaFabric CPU GGUF/MoE/Session path is prior work (R654–R659, R691); R700 does not alter it | The gate boundary is not wired into the llama.cpp generation callback in this commit |
| Strata, pinned MIT donor previously studied | R652 launcher environment sanitization is an existing adaptation; R653 native build stopped before callback code on the available AArch64 headers and the inspected CUDA path requires SM80+ | No Strata model or production generation used here |
| vLLM, TensorRT-LLM, SGLang, Maya, DeepInfer | No runtime code copied or invoked by R700 | Earlier investigations remain source references; no V100 or complete-generation qualification is implied |

No upstream donor quality is inherited automatically. The R699 live Session test
still uses a two-expert fixture and synthetic route weights. The new R700
route-to-admission test proves refusal when four selected payloads are missing;
it does not dispatch a valid six-expert batch. A positive
model-route-to-session result still requires the projection, all selected
experts, identity, memory admission, and atomic Collector to be composed without
dropping any of the six choices.

## Verification

`cargo test --offline --locked --test r700_deepseek_gate_route` passes 5/5 tests.
The focused
`cargo test --offline --locked --test r699_deepseek_memory_admission full_gate_top6_route_refuses_when_only_two_deepseek_experts_have_weight_profiles`
passes 1/1. `rustfmt --check --edition 2021` for changed Rust files, scoped
Clippy with `-D warnings`, and `git diff --check` pass. The tests were compiled in the
retained local Rust worktree at base `604764703f0a12af4fe264041ebecc0ed9c21a90`
with the R700 source overlaid; that worktree is an older/grafted snapshot, not a
full checkout of the current GitLab branch. GitLab readback is byte-identical
for the gate/provenance at `4483db95...` and the integration test at
`3982dff1...`.

Pipelines 2398, 2399, and 2400 passed the configured Gitleaks, Semgrep, Trivy,
and fallback jobs. Pipeline 2400 validates the route-to-admission test commit.
These GitLab jobs are security scans and fallback only and do not compile Rust;
the six CPU test passes and lint checks above are the separate local evidence.
The five R700 receipt files were read back byte-for-byte from private GitLab
commit `6e3e5f1492ce7ea2de312efb77bd993e96f18bf9`.

## arXiv scout shortlist

One bounded GPT-6 Luna scout produced the actionable shortlist in
`ARXIV_SHORTLIST.md`. This is not a claim that every arXiv paper was exhaustively
reviewed. DAOP and the prefetch/locality papers are candidates for ideas, not
benchmarked evidence for VoltaFabric's V100/PC4/PC6 topology. No code from these
papers was copied in R700, and no performance comparison was run.

## Next engineering step

Bind the gate-produced six-choice route to a fresh VoltaFabric identity and the
existing per-rank memory admission path. The currently retained DeepSeek layer
fixture contains only experts 0 and 383, so a positive top-6 Session dispatch
must wait for all selected pinned expert slices and the matching gate projection
to be available. Keep the path atomic and refuse missing experts; do not invent
four substitute routes or reuse R699's synthetic 0.375 coefficients.

## Sources

- [Pinned DeepSeek V4.1 Flash config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/config.json)
- [Pinned DeepSeek gate implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/inference/model.py)
- [Pinned DeepSeek model license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/LICENSE)
