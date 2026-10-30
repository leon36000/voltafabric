# R408 — joint CPU GDN token state

`GdnState` combines R406 convolution history and R407 recurrent state. Each call preprocesses against a private candidate history; only after the bounded recurrent step succeeds does it publish that history. The recurrence already computes into a private candidate before its own commit. A test forces rejection at the recurrence after successful preprocessing and verifies both states remain unchanged; it also verifies identity rejection and successful successive advancement. This atomicity covers the two in-memory GDN states of this call only, not an enclosing model layer, delivery, durable checkpoint or process failure. The caller remains responsible for run/plan/batch/attempt admission, ordering and never replaying UNKNOWN.

The implementation builds upon the earlier traced Strata fb58 MIT-licensed algorithm adaptation. The previous independent scalar recurrence oracle remains in the targeted test suite; the transaction test uses synthetic data. No claim is made for GPU parity, real-weight recurrence, complete generation, distributed execution or performance.

Next: compute real SC117 alpha/beta projections and gate from R404's mixed token-42 input, feed them with the R406 QKV into this combined state, and independently verify the complete CPU result. GPU and inter-PC execution retain their separate admission gates.

Verification on PC6: `cargo test --offline --test gdn_cpu` and `cargo test --offline --release --test gdn_cpu` each passed all four tests; `cargo clippy --offline --test gdn_cpu -- -D warnings` passed. This is targeted validation, not the full historical CPU gate.
