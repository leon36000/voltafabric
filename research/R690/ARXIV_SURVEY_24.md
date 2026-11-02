# R690 — 24-paper arXiv survey for VoltaFabric

## Scope and method

This is a bounded survey produced by one GPT-6 Luna research agent, followed by
primary-page checks for the shortlisted records. Searches covered arXiv categories
cs.DC, cs.LG, cs.AI, and cs.NI (2023–2026) with terms for MoE/expert parallelism,
expert placement and load balance, CPU–GPU cooperation, PCIe/RDMA/overlap,
quantization, and KV/serving. The 24 records below are selected for a plausible
engineering transfer to VoltaFabric; this is not an exhaustive review of all arXiv.
Paper metadata and claims are not a license audit of linked implementations.

## Communication, topology, and expert ownership

1. [ThunderEP, arXiv:2609.40093](https://arxiv.org/abs/2609.40093) — PCIe expert
   parallelism using DMA engines and fewer relay hops/polls. Candidate: model exact
   copies, staging, overlap, and completion costs for VoltaFabric buffers. Its RTX
   4090/5090 measurements do not qualify V100; no rsocket handles in POSIX
   poll/epoll/ppoll.
2. [VarioPath, arXiv:2609.34340](https://arxiv.org/abs/2609.34340) — offline,
   topology-aware channel catalog with online demand-aware all-to-all selection.
   Candidate: CPU-only simulator fed by verified VoltaFabric topology and routes;
   external platform results do not establish our topology.
3. [UCCL-EP, arXiv:2512.19849](https://arxiv.org/abs/2512.19849) — compact routing
   commands, CPU proxies, and ordering/completion concerns for expert communication.
   Candidate: tighten CPU bridge sequence, identity, completion, and ordering checks.
   It does not qualify VoltaFabric production RDMA.
4. [Director, arXiv:2607.08782](https://arxiv.org/abs/2607.08782) — predictive expert
   placement with capacity and migration costs. Candidate: offline cost estimate on
   admitted traces only; predictions cannot replace gate-selected expert routes.
5. [Expert Coupling, arXiv:2610.09372](https://arxiv.org/abs/2610.09372) — exploits
   correlated expert activations for placement. Candidate: measure co-activation and
   placement cost without token shuffling. Its training/MI300X setting differs.
6. [Semantic Parallelism, arXiv:2503.04398](https://arxiv.org/abs/2503.04398) —
   coordinates semantic placement and data parallelism. Candidate: CPU placement
   solver constrained by the VoltaFabric owner registry; do not change route meaning.

## Heterogeneous expert execution and transfer

7. [Achieving Cloud-Grade SLOs for Local MoE, arXiv:2606.10493](https://arxiv.org/abs/2606.10493)
   — streaming model loading and CPU/GPU overlap. Candidate: reason about buffer
   lifetime and dependencies in CPU code before making hardware claims.
8. [Adaptive Expert Assignment, arXiv:2610.23006](https://arxiv.org/abs/2610.23006)
   — chooses CPU execution versus transfer-plus-GPU for expert misses. Candidate:
   compare measured local costs only; do not infer V100 crossover from paper results.
9. [ATSInfer, arXiv:2607.10183](https://arxiv.org/abs/2607.10183) — tensor-granularity
   placement and dynamic transfer. Candidate: revisit only if measurements justify
   the added transfer/state complexity.
10. [Efficient CPU-GPU Collaborative MoE Inference, arXiv:2512.16473](https://arxiv.org/abs/2512.16473)
    — GPU expert caching and CPU fallback. Candidate: deterministic cache hit/miss
    fixture with exact resident-byte and owner accounting; linked-code license and
    compatibility remain unchecked.
11. [HeteGen, arXiv:2403.01164](https://arxiv.org/abs/2403.01164) — CPU/GPU
    collaboration and overlap. Candidate: model dependencies and CPU work; do not
    transfer speed claims to this hardware.
12. [OD-MoE, arXiv:2512.03927](https://arxiv.org/abs/2512.03927) — predicts future
    layer expert misses. Candidate: offline miss/byte analysis only; prediction is not
    route truth and cannot authorize output from an unselected expert.
13. [SSD Offloading for LLM Inference Is Harmful in Energy Efficiency, arXiv:2508.06978](https://arxiv.org/abs/2508.06978)
    — reports energy costs of SSD-backed inference. Candidate: reject SSD as a hot
    expert pool absent local evidence; preserve storage for cold recovery, not an
    assumed low-latency tier.

## Numeric formats and decoder boundaries

14. [Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization,
    arXiv:2509.23202](https://arxiv.org/abs/2509.23202) — introduces MR-GPTQ and
    studies microscaling FP4 quality/performance. Candidate: use as a quantization
    methodology reference only. It does not validate V100 kernels or VoltaFabric's
    formats; keep QPN2/NVFP4 K16, R13/MXFP4 K32, and IQ3 separate.
15. [PagedAttention, arXiv:2309.06180](https://arxiv.org/abs/2309.06180) — KV block
    management and sharing. Candidate: borrow block/refcount invariants only after a
    real decoder and KV implementation exist.
16. [SGLang, arXiv:2312.07104](https://arxiv.org/abs/2312.07104) — RadixAttention and
    structured generation. Candidate: prefix/cache scheduling later, after exact
    tokenizer and KV identity contracts exist.
17. [FlashAttention-2, arXiv:2307.08691](https://arxiv.org/abs/2307.08691) — attention
    IO-aware tiling and work partitioning. Candidate: implementation reference for a
    future admitted attention kernel; no GPU measurement is authorized by this CPU
    tranche.

## KV capacity, admission, and scheduling

18. [The KV Cache Is the New Memory Wall, arXiv:2609.30854](https://arxiv.org/abs/2609.30854)
    — analyzes memory demand from KV state. Candidate: add context length and KV bytes
    to later capacity models rather than considering weights alone.
19. [When the KV Cache Is Slower Than the GPU, arXiv:2610.21500](https://arxiv.org/abs/2610.21500)
    — studies remote KV versus recomputation in a GLM-5.3/B300 scenario. Candidate:
    compute a local break-even model only after decode/KV and transfer costs can be
    measured; reported hardware numbers do not transfer.
20. [KVLearn, arXiv:2610.28637](https://arxiv.org/abs/2610.28637) — cache retention
    and admission decisions. Candidate: CPU admission model after a coherent local KV
    contract exists.
21. [SmartGen, arXiv:2607.28150](https://arxiv.org/abs/2607.28150) — proactive,
    partial KV strategies. Candidate: defer until the complete decode/cache path has
    stable ownership and correctness evidence.
22. [Online Scheduling for LLM Inference with KV Cache Constraints,
    arXiv:2502.07115](https://arxiv.org/abs/2502.07115) — scheduling subject to KV
    capacity. Candidate: later CPU batch-admission oracle with explicit capacity
    limits.
23. [Preble, arXiv:2407.00023](https://arxiv.org/abs/2407.00023) — prefix-aware
    request scheduling. Candidate: defer until tokenizer, prefix identity, and KV
    semantics are implemented.
24. [DeepSeek-V4.1-Flash, arXiv:2609.19969](https://arxiv.org/abs/2609.19969) —
    model/architecture paper relevant to a target model. Candidate: inspect exact
    checkpoint revision, architecture, tensor formats, license, and memory admission
    before considering it. The paper alone does not admit a compatible checkpoint or
    prove it fits the available devices.

## Immediate engineering order

The most directly actionable items are the UCCL-EP ordering/identity ideas for the
CPU bridge and topology/copy-count modeling from VarioPath and ThunderEP. They can
first be translated into CPU/analytic checks without GPU or production-network
admission. Expert placement papers are secondary: estimate cost from actual routes,
but never let prediction rewrite a gate-selected route or the Rust owner registry.
KV and attention papers stay downstream of a real decoder/KV path. For every donor,
inspect the implementation revision and software license before copying code; a
paper citation or arXiv license does not settle repository code rights.
