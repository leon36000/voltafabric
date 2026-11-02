# R711 — build one route-bound resident MXFP4 expert

## Integrated boundary

`PinnedMxfp4RankExpertRead::into_session_resident` consumes the exact R710 read receipt and builds the existing `session::mxfp4::Artifact` for one expert and one rank. Before allocation it rechecks the full stamp, nonzero attempt, unique top-6 route containing the selected expert, pinned model/layer/namespace, the exact six names, dtypes, shapes, offset lengths, and each payload SHA-256. It rejects missing or changed payloads and rejects an insufficient caller memory budget before serialization.

The packing step uses the pinned Safetensors serializer to assemble a compact one-expert artifact from only the six selected tensors. The large source shard is neither materialized nor retained. The receipt carries the original run/model/plan/layer/batch/epoch, attempt, route order, rank/local binding, shard header digest, and payload hashes beside the Session artifact's rank/layer/epoch binding. A conservative peak bound is `2 * selected_payload_bytes + MAX_HEADER_BYTES + 8`; the packed owner must also fit the existing 64 MiB MXFP4 Session artifact cap. Temporary payload vectors are dropped before the Session owner is parsed.

## CPU evidence

The synthetic R710 sparse-shard fixture now verifies the selected one-expert tensors become a valid existing Session artifact, with the expected expert identity, rank 4, local 0, epoch and MXFP4 matrix shapes. The 18,800,640-byte synthetic expert produces an owner below the 64 MiB artifact cap with a conservative 45,989,896-byte packing peak bound. Separate cases reject a one-byte-too-small memory budget, a changed payload digest and a missing tensor before returning any owner.

- Focused debug suite: 5 passed, 0 failed.
- Same focused release suite: 5 passed, 0 failed.
- `cargo clippy --offline --locked --lib -- -D warnings`: passed.
- `rustfmt --edition 2021 --check src/deepseek_v41.rs`: passed.
- `git diff --check -- src/deepseek_v41.rs`: passed.

## Reuse and limits

Reused the R710 bounded `checkpoint::read_named_tensors` result, Safetensors serialization, and the existing `session::mxfp4::Artifact::new` validation path. No donor engine code was copied. The source fixture uses official pinned DeepSeek metadata but synthetic payload bytes; no real checkpoint weight payload was read. One in-memory Session artifact was constructed, but no Session worker was started or LOAD sent. This proves neither the six-expert route set nor generation, inter-PC distribution, production R53, GPU execution, or performance. Missing route payloads and digests still require atomic refusal.

Next: use this owner in a fresh CPU Session path with one identity, and prove the complete multi-rank preflight refuses before worker start while any of the other route experts are absent. No full model download, GPU, production network, UNKNOWN replay, deletion, or automatic relaunch.

R712 is the successor source snapshot: it adds a consuming adapter into the existing CPU Session LOAD path and changes `src/deepseek_v41.rs`'s checksum. The R711 checksum in `RESULT.json` is the pre-R712 snapshot; R712 records and verifies the successor checksum and re-runs the selective-read/packing suite.
