# R729 reuse note

R729 reuses the source boundary already recorded in R727/R728:

- DeepSeek-V4.1-Flash pinned revision `314113f8a7dadb388215396ba5fbee2c610a04d6`,
  MIT semantics; no source copied.
- VoltaFabric's R712 grouped CPU Sessions and FP32-preweighted Collector.
- VoltaFabric's R728 indexed FP8 E4M3FN/E8M0 shared expert and identity-bound join.
- The one-agent Luna arXiv screen remains in R727's `OPEN_SOURCE_SCREEN.md`; YALIS
  prefetch and InplaceKVCache residency remain unimplemented leads, not dependencies.

This wave imports no external runtime or kernel. Other engine projects were not
requalified for this CPU receipt-composition target.
