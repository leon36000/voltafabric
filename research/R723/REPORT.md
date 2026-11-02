# R723 — gate-selected route to indexed checkpoint reads and CPU Sessions

## Change

R722 added a pinned DeepSeek-V4.1 expert reader over the R721 indexed Safetensors loader. R723 adds the missing route-level bridge in `deepseek_v41_gate`: a gate-selected top-6 route now reuses one shared preflight/read path and can invoke the pinned indexed reader for each selected expert directly from a checkpoint directory.

The common route operation validates the route stamp and attempt, recomputes and compares the metadata plan, verifies the complete 36-name digest manifest before any reader callback runs, checks each dispatch slot and coefficient bit pattern, enforces the aggregate and per-expert byte ceilings, and compares each expert receipt against its exact slice of the route-level index/header plan. It returns the six-read receipt only after every read passes. The existing generic stream-reader entry point uses the same helper and retained its atomicity checks.

The integration fixture starts with a gate-produced route, creates a temporary sparse 7.39-GB Safetensors shard with 112,803,840 bytes of synthetic selected payloads, then calls the new directory-backed route API. The resulting six residents pass existing all-owner preflight/grouping, CPU Session LOAD/EXEC, and the independent FP64 MXFP4 oracle. A separate negative test provides no shard file and an incomplete digest map; the route rejects the map before attempting to open a file.

## Reuse and provenance

- Reuses VoltaFabric's R721 bounded index-to-shard loader and R722 pinned per-expert indexed read contract; no donor code was copied and no crate was added.
- Reuses the existing DeepSeek gate route identity/dispatch, atomic top-6 owner preflight, grouped rank scheduling, CPU Session LOAD/EXEC and Collector path.
- The index/header conventions and Apache-2.0 donor-source screening are documented in [R721](../R721/REPORT.md); the pinned DeepSeek checkpoint metadata is MIT-licensed. This tranche adapts interfaces inside VoltaFabric and does not inherit donor correctness or model compatibility automatically.

## Verification

Focused debug and release tests pass for the integrated gate → indexed shard → six residents → CPU Sessions → independent oracle path. The incomplete-hash preflight test passes in both profiles. The existing in-memory route payload reader test also passes after extraction into the shared helper. Scoped library/integration Clippy with `-D warnings` and rustfmt pass.

## Limits and next step

All tensor payload bytes are synthetic; only the pinned index and header metadata are real. This does not demonstrate generation with actual checkpoint weights, a full model forward, GPU, inter-PC transport, production R53, or performance. Automatic relaunch stays stopped by owner request. Next: connect the route-level indexed import to a model-layer loader entry point and add a manifest/admission boundary for real checkpoint payloads before any full-generation claim.
