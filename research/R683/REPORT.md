# R683 — donor `build_moe_ffn` CPU hook path

## Scope

R683 moves the Qwen2MoE route-hook fixture from R682's synthetic graph callback to the pinned donor's actual `llm_graph_context::build_moe_ffn` implementation. The live pinned `llama_context::process_ubatch` still drives the CPU graph.

## Reused and adapted

- Donor: `ggml-org/llama.cpp` revision `3cf03257f219afbe7334045ff7c6a06ac68c627d`, MIT, with the previously published R679/R680 hook and bridge changes.
- Reuses R681's live context attachment and R682's bounded pinned-library runner, adding a second opt-in CTest for `build_moe_ffn`.
- The synthetic internal Qwen2MoE graph supplies activation `[3,4]`, logits `[0,0]`, selected expert `1`, expert-1 identity up-projection, and expert-1 down-projection `diag(2,5)`. The donor's `build_moe_ffn` computes raw routed output `[6,20]` and sigmoid route coefficient `0.5` before dispatching the hook.
- The VoltaFabric collector fixture receives expert `1`, owner kind `0`, original activation and the final coefficient, applies it once, and returns `[3,10]`. Complete run/model/plan/layer/epoch/batch/attempt identity and exactly one callback are checked.

## Verification

- Direct bounded runner: C++17 `-Wall -Wextra -Werror`, pinned donor/shared-library checks, exit 0.
- CMake configure using the exact updated branch CMakeLists mirror and CPU-only pinned GGML: passed.
- Focused CTest `qwen2moe_llama_build_moe_ffn`: 1/1 passed in 1.69 s.
- Output checked: donor raw `[6,20]`; route weight `0.5` applied once; final `[3,10]`.
- Source/manifest readback and GitLab pipeline status will be recorded after publication.

## Limits

This reaches the donor's actual `build_moe_ffn` implementation inside a live `llama_context::process_ubatch`, but the test graph directly calls that method. The supplied selected expert is a synthetic input; this does not test Qwen2MoE's architecture-level top-k routing, the full model layer graph, or shared-expert/residual paths. No checkpoint, tokenizer, token loop, Rust coordinator/worker, complete generation, GPU/CUDA graph, inter-PC/RDMA, production R53 or performance was tested. R681's terminal process receipt was not replayed.

## Next

R684: connect this qualified routed-aggregate boundary to a compatible Qwen2MoE checkpoint/token loop only after identifying an already available small fixture and validating its format, license and loading requirements. Keep shared/residual in the donor, preserve complete identities, single coefficient application and atomic refusal. Do not download a checkpoint or start hardware work without fresh resource and safety admission.
