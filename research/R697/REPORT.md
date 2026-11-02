# R697 — advisory memory envelope at the public llama.cpp decode boundary

## Change

R697 connects the per-rank memory planner from R694–R696 to the existing Qwen2MoE route callback used by the pinned llama.cpp `llama_decode` path. On each callback, the adapter builds an advisory from the route selected by the donor gate and tags it with the complete VoltaFabric run/model/plan/layer/batch/epoch identity and attempt. The advice is read-only: it does not alter selected experts, coefficient bits, the owner registry, or the decode decision.

The Rust loader reads selected Q8_0 expert slices from the synthetic GGUF fixture and compares the observed byte count with the expected slice: 2,176 selected bytes per projection tensor (1,088 per selected expert), 13,056 bytes across six projection tensors. A capacity refusal produces “no advice” and leaves the donor route intact. A deliberately stale attempt is rejected by the FFI getter. The test separately exercises a late-layer execution refusal and confirms that no logits or sampled token escape from the incomplete decode.

## Reuse and adaptation

- Runtime boundary reused: the public `llama_decode` API at `ggml-org/llama.cpp` commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`, whose checkout `LICENSE` is MIT. R697 calls that API; it does not copy or patch llama.cpp source.
- VoltaFabric logic reused: R693’s route callback and GGUF expert-slice reader, plus the R694/R695/R696 route-bound placement and per-rank memory planner. The new Rust FFI object only returns advisory memory information.
- No code is imported in this wave from vLLM, SGLang, TensorRT-LLM, DeepInfer, Maya, Strata, or other surveyed engines. Their code and licenses remain separate review items.
- Q8_0 is the only fixture format here. This work does not convert or equate QPN2/NVFP4 K16, MXFP4 K32, or IQ3.

## Verification

On the local CPU build, the VoltaFabric `vf-rank-session` debug binary built offline with the lockfile and `native-ggml-cpu`; the R697 FFI loader built offline in release mode; the C++17 adapter linked to the pinned llama.cpp libraries with `-Wall -Wextra -Werror`; shell syntax and Rust formatting checks passed. The synthetic fixture was executed once.

The single CPU run completed three two-layer decode steps. Donor and VoltaFabric sampled tokens were `3,3,3`; maximum absolute logit error was `0`. Layer 0 reported the synthetic resident/candidate memory envelope (`13,056` / `6,528` bytes). Layer 1 deliberately refused memory advice under a one-byte test capacity while decode still followed the donor route. The late-layer execution refusal returned `-3` and `sampled_after_late_layer_reject=false`.

Seven local ledger claims for the run are consumed and marked `RESERVED_UNKNOWN_NO_REPLAY`; this run must not be repeated. GitLab pipeline results, if any, are separate security/fallback evidence and do not replace these CPU results.

## Limits

The GGUF and all budgets are synthetic. The donor reported zero model tensors loaded by llama.cpp; expert payloads were supplied to the VoltaFabric adapter fixture, not a trained checkpoint. There is no compatible real checkpoint admission, complete model generation, V100 execution, multi-PC worker, production-network test, or measured performance result. No memory capacity reported here describes the actual GPU fleet. The advisory is not yet evidence that a complete production allocator can use it safely.

## Next action

R698 should bind the advice to a verified checkpoint-derived inventory or reject the candidate if no admitted checkpoint exists. Preserve the route gate and registry as authorities; do not download a model, launch a GPU, or replay this fixture without a fresh, separate admission.
