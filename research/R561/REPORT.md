# R561 — provisional GDN state batch

Adds a bounded, provisional CPU GDN batch: convolution history and recurrence state are cloned together, each position advances the private candidate, and only a complete successful batch commits both. Invalid, incomplete, overfilled or dropped batches preserve the live state. Preview outputs are tentative; the caller remains responsible for enclosing transaction admission and publication.

Verification: `cargo test --offline --locked --test gdn_cpu` and the release equivalent both passed eight tests; `cargo clippy --offline --locked --lib --test gdn_cpu -- -D warnings` passed. The batch test compares three positions with serial GDN execution and both final states. Failure, incomplete and dropped batches are tested. Existing independent scalar recurrence oracle also passes.

Scope: CPU state primitive only. It is not yet integrated into the reserved SC117 GDN stage or the full model prefill; no real-checkpoint batch, network, GPU, generation or performance qualification is claimed. Next: integrate guarded reserved GDN stage with real checkpoint positions and an independent layer oracle, preserving all-or-none state and collector semantics. Do not replay UNKNOWN claims.
