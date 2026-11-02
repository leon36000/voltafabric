# R729 — completed CPU Session receipt through shared-expert join

## Result

Extended the existing R712 integration test so one gate-produced token is evaluated on
the existing grouped CPU Sessions and its real `Completed` route receipt is passed to
`combine_pinned_moe_route_with_shared`. The same token plan executes the resident shared
FP8 expert; both outputs carry the identical Stamp and attempt. The fixture exercises
five CPU Sessions delivering all six routed slots, the FP32-preweighted routed
Collector, the unweighted shared output, and the final single FP32 add followed by one
BF16 round-to-nearest-even cast.

The test compares the routed session output with the existing independent FP64 MoE
oracle, then checks every final BF16 element using a separate test-side bit-level RNE
formula. This closes the R728 synthetic-receipt integration gap for the CPU route/session
boundary. Route choices and coefficients are unchanged; the shared result receives no
route coefficient. R728 still refuses Aborted, Unknown, mismatched identity, partial
rows and overflow atomically.

## Reuse and provenance

This is an adaptation over VoltaFabric's existing R712 CPU Session/Collector path and
R728's indexed FP8 shared-expert loader/join. It uses the pinned DeepSeek-V4.1-Flash
revision `314113f8a7dadb388215396ba5fbee2c610a04d6` (MIT semantics); no donor code was
copied and no dependency was added. The focused one-agent Luna arXiv and repository
survey is retained in R727, with the selected/declined leads summarized in R728. No
Strata, Maya, vLLM runtime, llama.cpp, TensorRT-LLM or SGLang component was imported by
this test seam.

## Verification and limits

R712 debug suite: 12/12 passed after the change (123.62 s). Release suite: 12/12 passed
(7.64 s). The new integration case also passed alone in debug and release; Clippy,
formatting and diff checks passed. The indexed fixture uses the pinned index and header
with synthetic tensor planes in a sparse shard. It does not read actual DeepSeek weight
payloads.

This verifies a CPU Session-produced receipt entering the shared join, but not a full
model block, attention/GDN, HC post/residual mixing, tokenizer, autoregressive
multi-token generation, GPU, inter-PC transport, production R53, or performance. These
remain separate admissions and implementation boundaries.

## Next boundary

R730 should compose HC post/residual over this verified routed-plus-shared sublayer and
compare it to a separate oracle. Then bind the pinned attention/GDN operation and token
loop; preserve the full identity, resident weights, no-partial-output behavior and
UNKNOWN-no-retry contract at each boundary.
