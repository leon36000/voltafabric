# R694 — route-bound advisory for topology planning

## Change

The R693 branch already contains a bounded planner that models directed links,
wire variants, device capacity, and a scheduled demand graph. It lacked a direct
adapter from the validated Qwen2MoE route batch into those planning demands.
R694 adds planner::plan_qwen2moe_advisory.

The adapter accepts the donor route batch produced by VoltaFabric's existing
Qwen2MoE boundary. It verifies the batch through the current routing validator
and dispatch, requires the planner's expert profiles to match the logical
expert set exactly, and emits one planner demand per token. It retains the full
run/model/plan/layer/batch/epoch Stamp plus the separate nonzero attempt. It also
records selected expert IDs in token/slot order and the exact f64 coefficient
bit patterns as hex. Coefficients are evidence, not costs: the adapter never
renormalizes, rescales, or uses them to alter dispatch.

The returned placement is explicitly advisory. It does not mutate the
RoutingProblem, active registry, or route, and this helper offers no path to
commit a new owner set into a Session. Existing capacity, work-budget, directed
path, and all-or-error checks remain in force.

## Open-source research and reuse record

- VarioPath, arXiv:2609.34340 v1 (submitted 2026-09-28), describes an offline
  topology-aware channel catalog and an online demand-aware AlltoAllv schedule.
  R694 adapts the separation between fixed topology input and per-batch demand
  at the planner boundary. No VarioPath code was found or copied; the reported
  platform speedups are not applied to our equipment.
- Efficient CPU-GPU Collaborative Inference for MoE, arXiv:2512.16473, links
  elsa-lab/MoE-CPU-GPU-Collaborative-Inference. Its main was pinned to
  1687abe25d79b27fe26367620a24d79c72eed491; the root LICENSE at that commit
  is MIT. We inspected its Python cache path but copied no code. The inspected
  implementation stores cache tags/valid flags as int8, allocates a BF16 GPU
  cache, and hardcodes two selected experts in that forward path. Those choices
  do not establish safe behavior for our model formats or V100; format metadata,
  expert-ID range, and production transfer behavior are not qualified.
- R693 separately records the prior use of the public llama.cpp model/decode
  API at its pinned MIT commit. R694 adds no llama.cpp files.

## Evidence

Three CPU tests on an exact source extraction of the R693 planner, topology,
traffic, schedule, routing, MoE, collector, and utility modules passed. The test
checks ordered route preservation, exact coefficient bits, identity/attempt
binding, unchanged dispatch and registry, profile mismatch refusal, and missing
path refusal. rustfmt and Clippy on the extracted test target pass. The local
test harness used cached serde_json 1.0.154 because the workspace's 1.0.145
was not present in Cargo's offline cache; this is not a full workspace build.

All topology and compute costs in the fixture are marked synthetic. The result
is not a hardware measurement, a performance claim, a live registry update, or
a complete generation. No GPU, network worker, or model download was used.

