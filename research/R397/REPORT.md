# R397 — stream resident experts into local sessions

After validating the complete routing/binding map and payload-byte budget, the coordinator now reads, digest-checks and sends one expert at a time to each local rank-session rather than retaining every expert payload in its own `loads` map. Each worker is sealed after its assigned blobs, and the pack descriptor's metadata is checked again after loading and before reporting Ready. Sessions remain owned until admission completes; a load, digest, worker or metadata failure drops and reaps all started sessions without reporting Ready. The selected payload, each worker copy and protocol buffers still consume memory; this is not a fleet-wide RSS budget or a measured performance claim.

Seven native-provider and three durable-ledger tests pass with the complete patched donor two-layer CPU path. The five-resident synthetic test still executes correctly. A deliberately incorrect digest on the fifth expert caused refusal after an actual worker had started; a PID-recording local test confirmed that worker was reaped before the failure reply. Rust clippy and formatting pass. No real checkpoint, GPU, distributed transport or generation was exercised.

Next: establish trusted real-checkpoint bindings, aggregate memory admission and owned per-layer initialization; the local worker limits and hardware/UNKNOWN protections continue to apply.
