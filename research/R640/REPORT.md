# R640 — Mixed-owner callback in the pinned Strata dispatch

## Adaptation

Prepared an incremental, opt-in patch for Strata revision
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`. The reproducible generator first applies the already-reviewed VoltaFabric CPU-hook patch to a temporary copy of five pinned source files, then emits only the R640 delta. It verifies the donor revision, clean donor files, base-patch SHA and unique edit anchors. The donor checkout remains unchanged.

The delta adds a mixed-owner callback to `ExpertDispatch`. It calls the existing VoltaFabric bridge only after Strata has assigned its final `-1..2` owner map, including peer and remote claims. It keeps Strata's owner kinds unchanged and tracks VoltaFabric-completed CPU rows in a separate fixed mask. Only those rows bypass Strata's activation quantization, prefetch and CPU expert job; other CPU misses still use Strata. The bridge continues to apply no gate coefficient, leaving the pinned combine stage responsible for exactly one weighting.

If the bridge refuses or fails, the patch drains already-started peer/remote owners and returns through the existing verifier failure/drain path. No partial model result is accepted. The no-CPU-miss path consumes no VFB3 ticket.

## Reuse boundary and remaining work

This is an adapter patch artifact; it does not copy or modify the Strata donor and it does not change the default runtime. The new callback fields remain unset until a later generator/launcher integration supplies the 48-session owner and matching identity. Thus the callback is not active in a generation yet.

The generator produced the incremental patch against the pinned source and the patch was reviewed as text. No compiler, CTest, Rust test, model generation, GPU, network worker or R53 admission was run for R640. R641 should connect the owner lifetime and launcher identity to the Strata generation context, then exercise the CPU path against an independent row oracle before claiming generation integration.
