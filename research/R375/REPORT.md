# R375 — Admit the pinned Strata ownership kinds

Parent: a252127f9e87fa319390657c679d477df2f83708. The source reference is Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf, `src/core/expert_source.cpp`, approximately lines 3140–3264: `kind` is -1 for CPU, 0 for VRAM, 1 for PCIe and 2 for peer/helper. Remote helpers may change kinds after the initial GPU plan, so the final array must be captured only after all peer and helper decisions. Its existing array is local to the donor routine; this change does not wire it into VoltaFabric.

`admit_strata_kind_f64` validates the bounded kind array, rejects every code outside -1/0/1/2 and maps it to the R374 mixed admission. The caller must establish genuine origin, model, layer, batch and synchronization of the ownership decision; the adapter cannot authenticate them. No claim of GPU, PCIe, peer execution or numerical parity follows from the labels.

CPU-only validation: all 13 ingress tests passed in debug, including every kind, invalid codes and retained slot order. A two-local-Session QPN2 integration test passed in debug and release using a synthetic kind array and an external-only expert. The kind mapping test passed in release. Strict offline all-target Clippy, formatting and `git diff --check` passed. No consumed UNKNOWN replay and no hardware or inter-PC workload.

Fresh read-only observation: PC6 had 4 KiB free swap, below the inter-PC admission threshold. Three historical PC1 fixture/checkpoint paths referenced by R360 were absent at those exact paths; this does not establish global absence or rule out reconstruction elsewhere. No fixture was downloaded and no service changed.

Next: produce a safe, isolated donor interface that passes the actual final kind decisions, real model identity and gate provenance, and handles the donor void callback failure plus GPU wait draining. CPU-only synthetic tests do not prove this connection or complete generation.
