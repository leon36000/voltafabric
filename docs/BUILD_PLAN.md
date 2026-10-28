# Offline core implementation plan

Status: historical work plan. Actual implementation names and completed gates are in
README.md and evidence/DELIVERY.json. Registry/collector live in moe.rs. Read-only
hardware probes are documented, not an implemented or executed collector.

Goal: deliver a tested CPU-only vertical slice and a bounded Codex continuation.
Spec: docs/BUILD_SPEC.md. Method: tests, failing evidence, implementation, rerun.

1. memory.rs: checked layouts and capacity incl. load peak; overflow, alignment, padding.
2. moe.rs + registry.rs: independent sequential/partitioned paths and exact routing.
3. collector.rs: out-of-order slots, duplicate/stale/missing failure, epoch isolation.
4. topology.rs + traffic.rs: directed explicit links, grouping counterexamples, metadata.
5. planner.rs + pipeline.rs: feasible placement and contiguous partitions; exhaustive small-case oracle.
6. schedule.rs: dependency validation, exclusive shared resources, concurrency/cycle cases.
7. CLI: bounded JSON input, strict schema, no silent unsupported GPU execution.
8. Python differential/property tests, LiteLLM runner with local fake-server tests,
   future read-only probe procedure, reproducible verification and delivery.
9. Review, mutation sensitivity tests, release build, integrity manifest and handoff.

Review focus: integer overflow; unintended weight renormalization; duplicate/late replies;
shared PCIe resources; fake GPU evidence; HTTP redirects leaking credentials; retries;
unbounded queues; ambiguous model aliases; input paths escaping the project.

Independent implementations are deliberately limited: small CPU oracle and tests, not
another GPU kernel library. The plan is finished by evidence, not by document volume.
