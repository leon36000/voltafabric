# R555 — provisional QSA cache transactions

`qsa_cpu::State::begin_batch` prepares up to 128 consecutive positions against an independent provisional cache. The committed cache is only replaced when the caller commits exactly the expected count; an invalid input, foreign identity, extra position, incomplete commit or dropped transaction leaves it unchanged. A distinct state identity prevents provisional candidates crossing into the live cache. Preview outputs are tentative and the enclosing layer must not publish them before its own complete validation. Durable reservation and model-layer atomicity remain caller-owned.

The focused `qsa_cpu` suite passes 9 tests in debug and release; clippy with `-D warnings` and rustfmt checks pass. Tests compare provisional and sequential causal outputs and cover abort, incomplete commit, drop, foreign identity, extra rows, stale proposals, and bounds. This is a CPU cache building block, not a full prefill implementation, GPU qualification, performance claim, or assurance that all other layer state rolls back.

The transaction currently clones the already committed K/V cells. Under the existing geometry limit, their FP16 payload adds up to 64 MiB for one fully populated provisional copy, plus vector overhead and new cells. It is bounded but has not been optimized or measured; account for it before admission.

The previously published R554 branch remains the parent. Next connect the transaction to the QSA stage and complete the enclosing multi-position expert transaction, with corresponding GDN and PLE state semantics, a fresh ledger, and whole-layer rollback tests before any full prefill claim.
