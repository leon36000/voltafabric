# R719 — route-bound payloads through Session LOAD/EXEC

## Change

R718 stopped after reading the six route-selected owners and grouping them by rank. R719 extends that same CPU fixture through Session LOAD, one routed EXEC, and output collection, so the checkpoint-read boundary and the existing rank Session path are exercised together.

The fixture uses the pinned DeepSeek index/header and a sparse in-memory shard. Each selected MXFP4 weight plane contains the synthetic E2M1 byte `0x22`; each E8M0 scale plane contains `0x6f`. A nonzero BF16-exact input of `0.5` is used. The independent resident MXFP4 reference and the Session result are both asserted nonzero, and the maximum absolute difference must be at most `5e-4`. This closes a vacuity found during development: the previous all-zero payload fixture returned zero from both sides and did not check numerical work. The initial all-zero attempt was rejected by the new nonzero guard; it did not pass as evidence.

The test first confirms that a 35/36 tensor digest map is rejected before source I/O. With the complete synthetic digest map, it reads the six experts in gate order, converts them to owners, preflights the route, and groups them into five rank artifacts. The first two selected experts remain colocated on rank 4. It then loads each group into one local CPU Session, executes one token using the original gate-bound route, verifies six expected owner slots across five rank receipts, and compares the collected result with the per-expert MXFP4 CPU reference.

## Reuse and provenance

R719 reuses the VoltaFabric pinned-range planner, route-level SHA-256 reader, resident conversion, route identity and coefficient preflight, rank grouping, Session protocol, and MXFP4 CPU reference. The pinned DeepSeek-V4.1-Flash revision `314113f8a7dadb388215396ba5fbee2c610a04d6` remains an MIT semantic reference; no donor code is copied. Synthetic payload bytes are generated in the fixture and are not model weights.

The source file also receives a one-line rustfmt import-order correction so the repository-wide `cargo fmt --check` passes with this fixture.

## Verification

- Focused Linux test, debug: 1/1 passed in 71.12 s.
- Focused Linux test, release: 1/1 passed in 4.46 s.
- Strict targeted Clippy (`-D warnings`): passed.
- Repository `cargo fmt --check` and `git diff --check`: passed.
- The numerical nonzero assertions passed; the payload remains synthetic and sparse.

## Limits

This is a one-token CPU integration fixture, not a model generation. It does not read actual checkpoint payloads or establish that the complete selected tensor inventory is present. It does not qualify GPU execution, production R53, inter-PC transport, production registry state, service behavior, or performance. No UNKNOWN operation was replayed. The route coefficient is taken from the original gate-bound route; this test does not replace the earlier nonzero route/oracle proofs in R716.

## Source hashes

Recorded in `SHA256SUMS` after the final local verification.
