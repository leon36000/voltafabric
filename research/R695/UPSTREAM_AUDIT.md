# R695 upstream audit

## InplaceKVCache

- Source: arXiv:2609.14507v1, submitted 2026-09-13. The abstract describes a KV layout that fixes physical CPU/GPU residency at write time and a CPU/GPU scheduling policy.
- The paper authors report tests involving V100, Qwen3-30B-A3B, DeepSeek-V2-Lite, and Mixtral; those are author claims and do not establish performance on our cards, model checkpoint, code, or inter-PC links.
- No implementation repository or code license was verified on the arXiv page in this pass. No source was copied.
- Adapted idea: account for KV bytes as a separate per-rank budget, distinct from expert weights and recurrent state, and include transient loading memory in the placement admission.

## FluxMoE

- Source: arXiv:2604.02715v3, revised 2026-09-10. The abstract describes expert paging and a residency planner implemented on vLLM.
- The reported target evaluations name H20 and L40S; they do not qualify V100 or our transport. No code/license provenance was verified in this pass, so no code was reused.
- Rejected as a direct runtime drop-in; kept only as a design reference for later residency and transfer work.

## Existing llama.cpp donor

- `ggml-org/llama.cpp` commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`, MIT, used through the existing R693 public API seam. R695 adds no upstream code and does not change the donor integration.
