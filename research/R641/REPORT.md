# R641 — Connect Strata single-token generation to the VoltaFabric pool

## Result

Prepared an incremental patch for pinned Strata revision
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf` (MIT; see the pinned donor's
`LICENSE`, reproduced in `STRATA-LICENSE-MIT.txt`). R641 applies after the
published R639 CPU-miss callback patch and R640 mixed-owner dispatch patch. It
is an opt-in source adaptation; the donor checkout is unchanged.

With `STRATA_VOLTAFABRIC_POOL=ON`, the patch adds a Linux/native-CUDA-only
build option, requires the matching VoltaFabric bridge sources, installs the
R640 callback into Strata's generation `Drive`, and keeps a `PoolFleetOwner`
alive until the generator returns. The CLI accepts 48 unique inherited
AF_UNIX stream descriptors and explicit run, model-revision, plan, epoch,
initial-batch, initial-attempt, and timeout fields. The existing VFB3 bridge
checks these values on every request. The owner then sends one-token CPU-miss
rows from the normal decode dispatch; GPU and remote rows remain owned by
Strata. Gate coefficients remain applied once in Strata's existing combine.

The generator rejects `--no_pool`, `--no_capture`, and non-transactional
sidecar outputs in VoltaFabric mode. It calls `shutdown_all` on the owner after
the generator returns; a shutdown failure turns an otherwise successful
process into failure. On a VoltaFabric callback failure the dispatch attempts
to finish started remote owners, zeroes its output rows, and latches the
existing generation failure path before token/final output is accepted.

## Provenance and reuse

- Strata donor revision: `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`.
- Effective donor license: MIT, from that pinned checkout's `LICENSE`.
- Prior incremental dependencies: R639
  `2e86bbcb65ad6763cd9c47ec02a85da7ce3147ee50633e974ec5815f2f6bd10a` and
  R640 `c8e2848813ebd079a71cfa81a8d31c62214a413e0be50803e0b9619b8481da9b`.
- The patch adapts Strata's own dispatch/generator boundary and calls the
  existing VoltaFabric bridge. It does not copy a complete engine or claim
  inherited quality from Strata.

## Remaining boundary and evidence

The launcher that creates/initializes these 48 sessions and passes their file
descriptors is not implemented by this patch. The CLI identity is supplied by
the caller; this tranche does not prove that it matches the loaded checkpoint
or a production plan. No checkpoint hash binding, launcher-to-model admission,
compiler build, test, independent output oracle, complete generation, GPU,
inter-PC worker, production network or shared R53 admission was performed.
The donor failure path also does not cancel already submitted CUDA work; it
prevents accepting the generation output and is not a qualified hardware drain
protocol. The adapter is not enabled by default and must be independently
built and qualified before use.

No runtime integration of vLLM, SGLang, TensorRT-LLM, Maya or DeepInfer is
established by R641. Earlier llama.cpp GGML and Strata-head work remains as
recorded in R626/R627; IQ3, MXFP4 and NVFP4 formats are not conflated here.

The patch generator checks the pinned donor revision and clean target files,
checks the two dependency patch hashes, applies the cumulative patches to a
temporary copy, applies the R641 delta to a fresh post-R640 copy, and compares
the resulting source bytes with the generated expected files. This is patch
application/readback verification only, not a compiler or runtime test.
