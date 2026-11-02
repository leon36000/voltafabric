# R697 — upstream reuse and arXiv triage

## Reused in code

| Source | Pinned identity / license | Use in R697 | Boundary |
|---|---|---|---|
| `ggml-org/llama.cpp` | `3cf03257f219afbe7334045ff7c6a06ac68c627d`; MIT | Call the public `llama_decode` route callback already established by R693; R697 adds a VoltaFabric advisory FFI on that boundary | No llama.cpp source copied or modified. The fixture is synthetic and does not establish trained-model compatibility. |
| VoltaFabric R694–R696 | Private GitLab source commits documented in handoff | Reuse the route-bound placement and conservative memory-peak planner | Advisory only; routes, coefficients, identity, registry and atomic output contract remain authoritative. |

## Candidate code reviewed, not imported

- **MoE-Infinity**, arXiv [2401.14361](https://arxiv.org/abs/2401.14361), GitHub release tag `v0.2.2`, Apache-2.0 ([repository](https://github.com/EfficientMoE/MoE-Infinity)). The current open repository states that it differs from the implementation used for the paper and does not yet support multi-node inference. Its documented build targets SM80/SM90 by default, with a separate SM120 option, so it is not a drop-in V100 component. No code copied.
- **Efficient CPU-GPU Collaborative Inference for MoE**, arXiv [2512.16473](https://arxiv.org/abs/2512.16473), repository `elsa-lab/MoE-CPU-GPU-Collaborative-Inference`, MIT ([license](https://github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference/blob/main/LICENSE)). The checked repository view is a moving `main` without a release/tag or SHA pinned in this pass. Its exact-route cache/miss-execution idea is a possible design reference, not imported code or V100 evidence.
- **HybriMoE**, arXiv [2504.05897](https://arxiv.org/abs/2504.05897), repository `PKU-SEC-Lab/HybriMoE`, Apache-2.0 ([license](https://github.com/PKU-SEC-Lab/HybriMoE/blob/main/LICENSE)). No release/tag/source SHA was pinned in this pass; the project depends on a newer CUDA/kTransformers/GGUF stack. Its CPU placement and prefetch ideas remain a source review, not an imported dependency.
- **Fiddler**, arXiv [2402.07033](https://arxiv.org/abs/2402.07033), repository `efeslab/fiddler`, Apache-2.0 ([license](https://github.com/efeslab/fiddler/blob/main/LICENSE)). The repository presents a proof of concept still under construction and no exact release was pinned here. No code copied.

The 24-paper survey in the R690 handoff already covers UCCL-EP, ThunderEP, VarioPath, cache/offload, quantization, attention and KV scheduling. A single GPT-6 Luna scout added a bounded set of further candidates; its search was not exhaustive of all arXiv. We checked these additional primary pages and retained the following as **ideas only** because no compatible, licensed donor code was established:

- [Kernel-Managed Expert Cache, arXiv:2608.12103v2](https://arxiv.org/abs/2608.12103): OS page-cache admission and readahead advice; paper reports GH200 traces and does not provide a verified donor implementation/license in this pass.
- [PagedWeight, arXiv:2607.16184v1](https://arxiv.org/abs/2607.16184): runtime quality-aware weight quantization coupled to KV pressure. Do not merge its format into QPN2/NVFP4, MXFP4, or IQ3.
- [SliceMoE, arXiv:2512.12990v5](https://arxiv.org/abs/2512.12990): bit-sliced expert cache/precision adaptation; no code repository or code license was verified. Its quantization is a distinct format.
- [Occult, arXiv:2505.13345](https://arxiv.org/abs/2505.13345), with [official repository](https://github.com/UNITES-Lab/Occult): duplicate-token communication grouping could be evaluated only after confirming exact route/weight identity and repository license at a pinned commit. The repository license was not verified for R697; no code copied. Route pruning, expert replacement, and its training-oriented results are excluded.

## Decision

R697 reuses the mature engine at an explicit public decoder seam instead of vendoring a full inference engine. The implementation borrows only the stable decode boundary and keeps VoltaFabric’s routing, memory planner, identities, expert registry and collector semantics in charge. Licensing, version pinning, the synthetic test and the unmodified donor tree are recorded separately; correctness of the donor alone is not inherited by this adapter.
