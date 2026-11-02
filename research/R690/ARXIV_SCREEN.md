# R690 — targeted arXiv screen for VoltaFabric

## Method and coverage

One GPT-6 Luna subagent searched recent arXiv work across MoE expert parallelism,
placement, CPU/GPU collaboration, PCIe communication, RDMA control, quantization, and
KV tiers; I independently opened the listed arXiv primary pages for the candidates
below. This is a broad, targeted survey, **not a literal review of every arXiv paper**.
The paper abstracts and dates are primary-source metadata. Code availability and code
licenses were not audited except where separately pinned in the R689 donor record.

## Prioritized candidates

| Paper | Finding in the paper | VoltaFabric adaptation to evaluate | Boundary / reason not to copy directly |
|---|---|---|---|
| [ThunderEP, arXiv:2609.40093](https://arxiv.org/abs/2609.40093) | Designs PCIe expert-parallel transfer around DMA engines and fewer relay hops / completion polls; reports RTX 4090/5090 results. | After a fresh V100 admission, compare transfer count, host staging, DMA overlap and CPU synchronization using exact VoltaFabric slot buffers; first create a CPU/analytical schedule model. | RTX consumer results do not qualify V100. Do not add rsocket handles to POSIX poll/epoll/ppoll or infer throughput from the paper. |
| [UCCL-EP, arXiv:2512.19849](https://arxiv.org/abs/2512.19849) | Uses compact routing commands and CPU proxies to initiate RDMA; discusses ordering semantics. | Strengthen the existing CPU bridge with explicit sequence, identity, completion and ordering checks before any separately admitted transport work. | Its GPU/NIC systems differ; no production RDMA qualification here. Preserve atomic completion, and do not replay UNKNOWN. |
| [Director, arXiv:2607.08782](https://arxiv.org/abs/2607.08782) | Predicts expert activations and optimizes placement under capacity/migration cost. | Start only with an offline cost estimator over actually observed route/session traces; preserve observed expert choices and compare estimated memory/transfer cost. | Prediction-driven online migration could alter route semantics and relies on model/system-specific results. No prediction may replace the gate's exact route. |
| [CPU–GPU collaborative MoE inference, arXiv:2512.16473](https://arxiv.org/abs/2512.16473) | Describes GPU expert caching with CPU fallback on misses. | A deterministic hit/miss fixture could count exact resident bytes and exercise CPU-owner fallback without changing expert identity. | Linked repository licence and compatibility were not checked. No speed claim transfers to V100 or this topology. |
| [When the KV Cache Is Slower Than the GPU, arXiv:2610.21500](https://arxiv.org/abs/2610.21500) | Reports a GLM-5.3/B300 deployment where remote KV hits can be slower than recomputation for some request lengths. | Once real decode/KV exists, derive a local break-even model from measured transfer and recompute costs; retain recomputation as an option. | B300 numbers do not transfer; R690 has no attention, KV cache, or remote tier. |

## Rejected or deferred transfer

- Do not apply speculative routing as an output route: only the gate's admitted choices
  determine the experts. At most, evaluate harmless post-route prefetch in a later
  identity-bound experiment.
- Do not treat papers' hardware speedups, quantization kernels, or linked repositories
  as evidence of VoltaFabric accuracy, licensing, or V100 support. QPN2/NVFP4 K16,
  MXFP4 K32, and IQ3 remain separate formats.
- Paged KV and prefix-cache ideas remain downstream of a real decoder/KV implementation;
  adding a remote cache before measuring its break-even point is not justified.

## Verified source pages

- [ThunderEP](https://arxiv.org/abs/2609.40093), submitted 2026-09-30.
- [UCCL-EP](https://arxiv.org/abs/2512.19849), v2 page currently dated 2026-01-22.
- [Director](https://arxiv.org/abs/2607.08782), submitted 2026-06-13.
- [CPU–GPU collaborative MoE](https://arxiv.org/abs/2512.16473), submitted 2025-12-18.
- [Remote KV tier study](https://arxiv.org/abs/2610.21500), submitted 2026-10-15.
