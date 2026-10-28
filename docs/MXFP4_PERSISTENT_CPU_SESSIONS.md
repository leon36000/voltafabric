# Persistent MXFP4 CPU sessions (R12)

R14 adds an explicitly selected native GPU worker using the same session and
preweighted collector. Its producer includes the qualified library SHA-256;
CPU remains the default. See `../gpu-worker/README.md` for the actual native
qualification and bounds. The CPU reference described below remains available.

`vf session-run-mxfp4 INPUT.json` exercises the existing authenticated process/SSH
session with one immutable compressed shard per rank. This is a bounded reference
producer and integration test, not a GPU fallback or a complete model service.
The existing dense/QPN2 producer and unweighted collector retain their contracts.

## Arithmetic and ownership

The capability `cpu_mxfp4_r9_preweighted_fp32_v1` admits only the explicit
`r9_mxfp4_preweighted_fp32_slot_sum_v1` arithmetic contract. Input F32 bits must
already represent finite BF16 values exactly. W1/W3 use 32 F32 lane partials,
ascending K32 blocks, separate rounded multiplies/adds and a delta 16,8,4,2,1
shuffle reduction. The existing FP64 format oracle remains separate. SiLU/clamp,
the original coefficient before BF16, second activation packing and W2 follow the
R9 laboratory contract. No claim is made about an upstream Tensor Core GEMM's
reduction order or every host/GPU exponential boundary.

The collector receives already weighted finite F32 rows and sums slots once in
fixed order. Coefficients must round-trip exactly through F32; no reweighting,
renormalization or silent dtype conversion occurs. A zero coefficient requires a
zero row. The tiny independent Torch CPU corpus and the preserved full-expert R9
corpus are qualification inputs, not model quality or GPU throughput evidence.

`mxfp4::Artifact` owns an immutable Arc-backed SafeTensors byte vector. Its digest,
projection metadata, layer, hidden dimension and exact expert set are validated
before LOAD. The worker retains compressed codes/scales; it does not retain dense
weight planes. The client and worker each own their shard; one copy does not imply
shared memory across processes. Rebind verifies the current session epoch and
unchanged mapping/revision before transmitting metadata only. Eviction releases
the resident slot; there is no hidden CPU/GPU migration.

## Protocol and failure

LOAD sends metadata and at most 64 MiB of compressed artifact. This uses the
existing shard owner's admission cap, separately from the 16 MiB normal frame cap.
The series CLI admits at most 128 MiB of aggregate client shard owners, 1–64
workers, 1–64 batches and a 1–60000 ms timeout. Each expert reference is bounded by
48M MACs and each rank EXEC by 100M MACs. These CPU reference limits do not specify
the capacity of a future GPU service or a fixed number of GPUs.

All workers and batches are validated before spawn. The series CLI holds one
artifact epoch fixed across its batches; the library API separately supports an
explicit strictly increasing epoch rebind. Only activations, selected slots and
contract metadata follow the initial LOAD. Physical replica/local IDs remain
distinct from logical expert IDs.

Completion admission checks full Stamp, rank, attempt, artifact SHA, arithmetic
contract, hidden width and slot count before allocating/reading a raw reply body.
Actual received header length plus body is bounded. The reader consumes a cumulative
slot budget before each body read; repeated bounded frames cannot enqueue
unbounded raw FP32 payload for one EXEC. The existing session reader now uses a
lossless two-frame queue for all reply kinds, with stream backpressure; one further
frame may be under decoding/sending. Closing disconnects that queue before joining
the reader. This bounds backlog by frame count, not total coordinator memory:
per-frame allocations, staged rows, artifacts and multiple sessions remain separate
costs. Every slot identity and
coefficient bit pattern is checked, and rows remain staged until Finished and
exact coverage. A duplicate, missing/late result, stale identity, truncation or
nonfinite value expires the whole current collector; no partial batch is returned.
A correctly drained peer may report Completed with BATCH_ABORTED if another peer
already failed: completion of its execution is distinct from publication.

A worker's explicit computation failure after Started is represented by a Started
receipt plus its error, with no batch output. A connection loss, malformed stream,
or timeout after transmission may be UNKNOWN; no retry is implied. A replacement
worker may process a different batch identity, not silently replay that batch.
The coordinator's process-local claim guard is not durable distributed exactly-once.

Unix sessions connect child stdin/stdout/stderr through private standard-library socket pairs.
This preserves the byte stream while allowing the new MXFP4 writer to recompute
one absolute write deadline for every partial write, without a detached thread
retaining the shard. Custom endpoints that inspect whether their standard descriptors are pipes may
observe this difference. No listening socket or network service is added. R12
bounded MXFP4 writes require Unix; existing generic session writes have not gained
a general deadline guarantee. An expired deadline before transmission sends no
bytes; a failure after partial transmission shuts down the stream and kills only
the owned child. `close`/drop reaps it. No descendant or remote process is killed
implicitly. On Unix, close preserves SESSION_PIPES diagnostics for inherited output
descriptors, then shuts down the local endpoints and joins readers; implicit Drop
also disconnects the frame queue, cancels those endpoints and reaps its own child.
Thus a descendant retaining stdout/stderr cannot leave a permanently blocked
reader after the session is dropped. Non-Unix legacy readers retain their earlier
best-effort cleanup behavior. System timeout granularity is not a real-time scheduling guarantee.

Traffic counters count completed logical writes and received application bytes.
A failed partial write can have transmitted bytes absent from those successful
write counters; they are not network packet accounting or proof of nonexecution.
SSH framing/encryption bytes, link utilization and CUDA synchronization are not
measured by these counters. Worker RSS and cold/persistent wall times must be
reported separately from compressed payload and avoided LOAD bytes.

## Reproduction

Use Rust 1.90.0, the locked existing crates and the project's isolated CPU Python
for Torch-dependent oracle regeneration. No new dependency is required.

```
cargo +1.90.0 test --offline --locked --test r12_r9_fp32 --test r12_mxfp4_session
cargo +1.90.0 test --offline --locked --release --test r12_r9_fp32 --test r12_mxfp4_session
python tests/reference/mxfp4_tiny_oracle.py
python3 scripts/mxfp4_session_crosscheck.py --binary /absolute/path/to/vf --out /new/result.json
```

`--worker-map FILE.json` accepts existing Endpoint argv arrays keyed by rank;
for example rank 9 may use `ssh -T -o BatchMode=yes pc1 /path/to/vf-rank-session`.
Both binaries must implement the same capability and identified source snapshot.
The helper compares three persistent batches to three distinct cold runs, verifies
exact independent output bits and repeats reversed result arrival. Paths in a
series input are deployment inputs, not an embedded dependency on PC1's mount.
The complete `scripts/verify.py` gate includes these independent tiny oracles.

H1/H2/H3, actual RDMA/RoCE/GPUDirect, long sessions, service throughput, energy and
complete DeepSeek model quality require their own hardware executions. This CPU
producer does not close them or replace the preserved H1 GPU package.
