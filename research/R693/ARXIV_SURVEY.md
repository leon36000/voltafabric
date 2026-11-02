# ArXiv scan for VoltaFabric — focused shortlist

This is a broad, targeted scan of arXiv papers in distributed/parallel systems,
networking, AI and machine learning (roughly 2023–2026), not an exhaustive read
of all arXiv. Abstracts and version/date metadata were checked on arXiv pages;
papers are sources of ideas, not proof for our hardware. A paper's publication
license does not establish the license of associated software. Any code reuse
requires a separate repository/version/license audit.

## First five engineering actions

1. Build a deterministic CPU simulator for candidate expert placements and
   communication schedules, taking routes, tensor bytes, measured link topology,
   compute estimates and capacity as explicit inputs. Start from the existing
   owner registry; the simulator must not change router choices or publish a
   partial owner set. VarioPath and Expert Coupling motivate this work.
2. Add route-trace summaries (per layer, owner, bytes and batch) to the existing
   CPU fixture, then replay them through the simulator. Reject any placement that
   violates `(expert, rank)` uniqueness, owner capacity, or the chosen reduction
   contract. This is observability and planning only.
3. Extend the bridge protocol's CPU oracle for ordering, exact identity, duplicate,
   missing-owner, stale-attempt and late-failure cases. UCCL-EP motivates an
   explicit control/data boundary; its RDMA implementation is not admitted here.
4. Specify and unit-test a KV block-table allocator with reference counts,
   copy-on-write and fail-atomic OOM. This can adapt PagedAttention's memory
   management concept after the current public decode boundary is stable.
5. Simulate cache/offload break-even on explicit bytes and latency inputs before
   any placement policy is adopted. CPU/GPU collaboration papers do not qualify
   our local transfer speeds or a V100 code path.

## Prioritized papers

For each paper, the test is deliberately CPU/offline first; reported speedups are
not projected onto VoltaFabric.

| Paper | Useful transferable idea | VoltaFabric check / limit |
|---|---|---|
| [ThunderEP, arXiv:2609.40093 (2026-09-30 v1)](https://arxiv.org/abs/2609.40093) | Avoid relay/contention on PCIe consumer-GPU expert-parallel paths. | Simulate bytes and hops; audit low-level polling/ordering. Never use rsocket handles with POSIX poll/epoll/ppoll. Tested on other GPU systems. |
| [VarioPath, arXiv:2609.34340 (2026-09-28 v1)](https://arxiv.org/abs/2609.34340) | Offline topology catalog plus online demand-aware decomposition of AlltoAllv. | A CPU planner over verified topology/routes is the strongest next planning artifact; no speed result transfers to our cluster. |
| [UCCL-EP, arXiv:2512.19849 (2025-12-22 v1; 2026-01-22 v2)](https://arxiv.org/abs/2512.19849) | Compact CPU proxy control channel and explicit communication ordering. | Adapt only message format/order invariants first; no RDMA worker or platform qualification. |
| [Director, arXiv:2607.08782 (2026-06-13 v1)](https://arxiv.org/abs/2607.08782) | Trace-based expected expert placement before requests. | Offline placement suggestion only; never change gate routing or model function. Stability is model/workload-specific. |
| [Expert Coupling in MoE Pretraining, arXiv:2610.09372 (2026-10-07 v1)](https://arxiv.org/abs/2610.09372) | Co-activation statistics can inform physical grouping. | Count co-routed expert pairs from actual traces; keep logical IDs, route weights and Collector order unchanged. Training result only motivates placement. |
| [Semantic Parallelism, arXiv:2503.04398 (2025-03-06 v1; 2026-02-27 v5)](https://arxiv.org/abs/2503.04398) | Co-schedule model work and data to reduce communication. | Prototype an exact CPU schedule planner around immutable owner sets; page title/version changed, deduplicate older indexed title. |
| [Achieving Cloud-Grade SLOs for Local MoE, arXiv:2606.10493 (2026-06-09 v1)](https://arxiv.org/abs/2606.10493) | CPU/GPU overlap and dual-batch execution graph. | Model dependencies/buffer lifetimes first; evaluated on different CPUs/GPUs, not our hosts. |
| [Adaptive Expert Assignment, arXiv:2610.23006 (2026-10-16 v1)](https://arxiv.org/abs/2610.23006) | Assignment based on CPU/GPU miss costs. | Exact CPU oracle with explicit costs before policy; moving work to a different logical expert is not allowed. |
| [ATSInfer, arXiv:2607.10183 (2026-07-11 v1; 2026-07-14 v2)](https://arxiv.org/abs/2607.10183) | Tensor-level placement and load-aware async movement. | Simulate residency/transfer and refuse until every required output is present; async adds identity complexity. |
| [Efficient CPU-GPU Collaborative Inference for MoE, arXiv:2512.16473 (2025-12-18 v1)](https://arxiv.org/abs/2512.16473) | GPU expert cache with CPU execution for misses. | The paper links the [official source repository](https://github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference/tree/1687abe25d79b27fe26367620a24d79c72eed491); pinned `main` HEAD `1687abe25d79b27fe26367620a24d79c72eed491` has an MIT root LICENSE. The implementation is not yet audited/reused. First model hit/miss on CPU. |
| [HeteGen, arXiv:2403.01164 (2024-03-02 v1)](https://arxiv.org/abs/2403.01164) | Heterogeneous CPU/GPU partition and overlap model. | Build a dependency-aware simulator; reported OPT-30B/A10 results do not qualify V100. |
| [OD-MoE, arXiv:2512.03927 (2025-12-03 v1)](https://arxiv.org/abs/2512.03927) | Predictive just-in-time expert loads. | Offline replay only; prediction misses cannot yield partial output or alter exact routing. |
| [SSD Offloading for MoE Weights Considered Harmful in Energy Efficiency, arXiv:2508.06978 (2025-08-09 v1)](https://arxiv.org/abs/2508.06978) | Treat storage tier movement as a measurable cost. | Use byte/latency/energy formulas with local inputs; no SSD offload policy without measurements. |
| [MR-GPTQ / microscaling FP4, arXiv:2509.23202 (2025-09-27 v1; 2026-03-03 v3)](https://arxiv.org/abs/2509.23202) | Quantization calibration interacts with microscale FP4 behavior. | Per-layer CPU error histogram only after format provenance; keep QPN2/NVFP4 K16, R13/MXFP4 K32 and IQ3 separate. |
| [PagedAttention, arXiv:2309.06180 (2023-09-12 v1)](https://arxiv.org/abs/2309.06180) | Block-based KV allocation and shared-prefix memory management. | Add allocator invariants (refcount, COW, atomic OOM) after decoder integration. The paper points to vLLM source; audit its exact software license separately. |
| [SGLang, arXiv:2312.07104 (2023-12-12 v1; 2024-06-06 v2)](https://arxiv.org/abs/2312.07104) | Radix/prefix cache and programmatic serving. | CPU cache-key tests must bind model, plan and tokenizer; serving runtime is not qualified here. |
| [FlashAttention-2, arXiv:2307.08691 (2023-07-17 v1)](https://arxiv.org/abs/2307.08691) | Tiling reduces attention intermediate memory traffic. | Verify mathematical oracle on CPU before a kernel; A100 results do not establish SM70 support. |
| [The KV Cache Is the New Memory Wall, arXiv:2609.30854 (2026-09-25 v1)](https://arxiv.org/abs/2609.30854) | Roofline/crossover framework for weights versus KV traffic. | Derive crossover from admitted checkpoint and measured link/compute inputs; do not reuse H100/B200/MI300X thresholds. |
| [When KV Cache Is Slower Than GPU, arXiv:2610.21500 (2026-10-15 v1)](https://arxiv.org/abs/2610.21500) | A remote KV hit may cost more than recomputation. | CPU break-even calculator with local latency assumptions; do not infer local values from B300/Valkey. |
| [KVLearn, arXiv:2610.28637 (2026-10-08 v1)](https://arxiv.org/abs/2610.28637) | KV retention/admission can use compute, transfer and storage costs. | Replay cache traces before learning; paper links [KVLearn source](https://github.com/FastLM/KVLearn), whose commit and license need separate audit. |
| [SmartGen, arXiv:2607.28150 (2026-07-30 v1)](https://arxiv.org/abs/2607.28150) | Selective/parallel KV transfer to lower time-to-first-token. | Simulate transfer ranges but assert complete KV availability before token publication. |
| [Online Scheduling for LLM Inference with KV Cache Constraints, arXiv:2502.07115 (2025-02-10 v1; 2026-01-15 v5)](https://arxiv.org/abs/2502.07115) | Admission and batching under KV capacity constraints. | Exact toy simulator for capacity/fairness; local request distributions are unknown. |
| [Preble, arXiv:2407.00023 (2024-05-08 v1; 2024-10-03 v2)](https://arxiv.org/abs/2407.00023) | Prefix-aware global scheduling. | Offline trace replay after real tokenizer/KV exists; repository/license not verified. |
| [DeepSeek V4.1 Flash, arXiv:2609.19969 (2026-09-17 v1)](https://arxiv.org/abs/2609.19969) | A target model candidate with MoE, compressed KV and substantial active parameter budget. | Metadata, tensor shapes, quant formats and authoritative license first. No current proof it fits V100 or the admitted stack. |

## Excluded or deferred

- Dynamic balancing that moves a token to a different logical expert is rejected
  unless it is a mathematically identical replica with an identity/weight proof;
  a registry violation is not repaired by optimizer policy.
- Predictors, cache hits, async transfers and remote KV do not authorize stale or
  partial outputs. `UNKNOWN` remains non-replayable.
- No reported speedup from these papers is a VoltaFabric performance claim.
- The scan was query- and category-based, not exhaustive over all arXiv records.
  Code links, source versions and software licenses remain to be checked at their
  primary repositories before any reuse.
