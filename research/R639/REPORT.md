# R639 — Mixed Strata ownership at the native CPU-miss boundary

## Change

Added `pool_cpu_misses_by_layer` to the VoltaFabric-side Strata bridge. It accepts the final per-route ownership kinds (`-1` CPU, `0` VRAM, `1` PCIe and `2` peer/helper), sends that mask over the existing identity-bound VFB3 client, and copies back only the CPU-owned rows. The VFB3 client validates the full response before the hook writes any CPU row; externally owned destination rows are left unchanged. If every route is externally owned, the hook returns success without dispatch and without consuming a batch or attempt.

This uses the existing Rust unweighted FP32 ingress contract. It does not pass gate coefficients to VoltaFabric, so the pinned Strata path remains responsible for applying each coefficient once. The existing all-CPU callback stays available.

The source seam was checked against pinned Strata revision `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`: its dispatcher assigns `kind` before CPU job construction and the CPU fallback collects only rows with `kind < 0`; the VFB3 client already accepts the same `-1..2` ownership values and returns only `-1` rows. That makes the new helper a compatible boundary for a later donor patch without moving GPU placement into VoltaFabric.

## Verification and limits

This wave received source review only. No compiler, CTest, Rust test, model, Strata generation, GPU, network worker, or R53 admission was run for R639. The new helper is not called by the pinned Strata donor yet; `strata-fb58-cpu-hook.patch` remains the earlier CPU-only candidate and still refuses GPU/peer/PCIe ownership. The generator does not yet construct the 48 sessions or install this callback.

No complete generation or V100 distribution is claimed. This local bridge uses AF_UNIX and a local ledger; it is not a shared R53 registry or production network.

## Next step

R640 should extend the patch for the exact pinned donor to call the mixed hook only after its final ownership map is complete, skip local CPU computation only for successfully handled `-1` rows, and propagate failures through the verifier's existing drain/refusal path. Then compile and exercise the patched donor CPU path against an independent row oracle. Keep GPU, inter-PC transport, production networking and real R53 suspended absent fresh admission.
