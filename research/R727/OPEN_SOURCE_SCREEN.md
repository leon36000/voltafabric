# R727 focused source screen

## Pinned donor consulted

- DeepSeek-V4.1-Flash commit `314113f8a7dadb388215396ba5fbee2c610a04d6`, MIT;
  provenance and license snapshot are in `third_party/deepseek-v41-flash/PROVENANCE.md`.
- `inference/model.py`: one shared `Expert` is required and invoked for every token;
  E4M3 weights use the FP8 activation quantizer with block size 32; the shared Expert
  runs `w1`, `w3`, clamp, SiLU-times-up, casts to the source activation dtype, then `w2`.
- `Block.hc_pre`: weighted sum of the four residual copies followed by a cast back to
  the input dtype. The returned coefficients come from `hc_split_sinkhorn`; this wave
  accepts them as input and does not reimplement the learned projection or Sinkhorn.
- The committed index/header metadata shows the six FP8 shared weight/scale tensors
  colocated with the gate and FFN norm in all 40 backbone layers. This is metadata only;
  the model payloads were not read.

## ArXiv and repository survey (2026-11-02; one Luna survey)

- **InplaceKVCache / WriteScope — highest-priority design lead.** The [paper,
  arXiv:2609.14507](https://arxiv.org/abs/2609.14507) fixes physical KV byte residency
  at write time, divides storage into four device-affinity/access-pattern regions, and
  schedules CPU versus GPU attention work as sequence length changes. The authors
  report experiments on V100 and a DeepSeek-V4-Flash composition case. This is close
  to our memory/topology constraints, but the survey found no source repository or
  reuse license to pin. Carry forward the residency/scheduling design, not donor code;
  verify our actual V100 memory, KV layout, GDN requirements, and transfer path before
  implementation. The paper's reported results are not VoltaFabric measurements.
- **Speculating Experts / YALIS — optional prefetch idea.** [arXiv:2603.19289](https://arxiv.org/abs/2603.19289)
  predicts future experts to overlap CPU-to-GPU transfers with computation. The linked
  [YALIS `offload_prefetch` branch](https://github.com/axonn-ai/yalis/tree/offload_prefetch)
  has an [Apache-2.0 license](https://github.com/axonn-ai/yalis/blob/offload_prefetch/LICENSE).
  No code or commit was pinned or copied. For VoltaFabric, prediction may only issue
  cancellable prefetches: the actual gate remains authoritative, and mispredicted data
  cannot enter accepted results or alter coefficients/identity. Consider only after
  the exact routed Collector path is complete.
- **FinDEP — scheduling model, not yet an integration.** [arXiv:2512.21487](https://arxiv.org/abs/2512.21487)
  models fine-grained compute/communication task overlap for disaggregated expert
  parallelism. This survey did not establish a public implementation artifact or a
  V100/inter-PC result. Its task graph could inform a deterministic CPU simulator;
  it does not qualify transport, reduction, or the R53 network worker.
- **Fiddler — CPU placement precedent, prototype rejected for direct reuse.**
  [arXiv:2402.07033](https://arxiv.org/abs/2402.07033) and the
  [Apache-2.0 repository](https://github.com/efeslab/fiddler) place CPU compute on
  missing experts to avoid moving large expert weights. Its README calls the repository
  a proof of concept under heavy construction and describes a latency-oriented Mixtral
  path. Keep only the data-movement cost model as a lead; no source was copied.

## Previously screened candidates

- [MonoMoE, arXiv:2609.04244](https://arxiv.org/abs/2609.04244) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer): scheduler/weight-major ideas are research leads; their reported kernels target H200/modern GPUs and were not copied or qualified for this V100-focused code path.
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention): useful operator/oracle reference for the separate GDN boundary; no dependency or code copied here.
- vLLM's Apache-2.0 checkpoint-index convention was adapted earlier in R721 into VoltaFabric's bounded reader; no vLLM runtime or kernel was imported in R727.
- The survey did not establish an admissible V100 component from Strata, Maya, TensorRT-LLM, llama.cpp, or SGLang for this shared DeepSeek FP8 boundary. None was imported in this wave.

This is a focused, non-exhaustive screen, not a claim that all arXiv has been exhaustively
searched. Source licenses and hardware claims do not transfer automatically to
VoltaFabric, and none of these upstream kernels is assumed to work on the target
equipment. R727 imports no third-party implementation code: the only source boundary
implemented here is an independently written CPU reference for the pinned MIT
DeepSeek checkpoint semantics.
