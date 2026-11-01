# R642 — Local 48-session launcher for Strata and VoltaFabric

## Result

R642 adds a standard-library Python launcher that closes the session-startup
gap left by R641. It reads a bounded launch manifest, a checkpoint manifest,
and a plan; derives the model revision and plan identity from their exact bytes;
checks the 48 SC117 layer templates and bindings; and prepares one local
AF_UNIX stream for each native layer coordinator. It initializes every
coordinator through its JSON/VFB3 boundary, checks the admitted rank set, enters
binary mode, then starts the pinned Strata executable with all 48 descriptors
and the derived identity fields. An incomplete startup is refused before
generation. It never retries a submitted attempt.

The launcher checks that the SC117 expert pack is a regular file of the pinned
v3 size without reading the 50,292,326,400-byte file into memory. The Rust
coordinator remains responsible for loading and hashing each resident expert
payload. The manifest digest therefore identifies the manifest and its
per-expert expectations; it is not a whole-checkpoint digest.

## Reuse and provenance

- Base branch: private `r641-strata-token-decode-voltafabric-20261101`, tip
  `70beeca4bd1d67644ad623042a3227affb083c45` at pre-publication inspection.
- Reuses the R641 integration boundary: pinned Strata revision
  `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` and the existing VoltaFabric
  `vf-strata-native-bridge` VFB3 protocol. R641 records the donor's effective
  MIT license. R642 is new VoltaFabric launcher code and does not copy donor
  engine code.
- It keeps the owner work local and bounded: 48 coordinator children, local
  AF_UNIX streams, caller-provided executables and working directory, and
  bounded manifest, plan, index, payload, and timeout fields.

## Verification

`python3 -m py_compile` succeeded for the launcher and test file. The targeted
CPU suite passed 2/2 tests:

1. 48 fake coordinators accept the identity-bound initialization, enter binary
   mode, and receive the inherited descriptors in a fake Strata process.
2. A mismatched plan template is rejected before any child process starts.

The fixture replaces the pinned 50 GB pack size with a tiny file and uses fake
bridge and Strata executables. No model weights, real Rust bridge, or real
Strata binary were loaded or built.

## Limits and next step

This qualifies launcher control flow against local protocol fakes only. It does
not qualify the real Rust coordinator's startup/shutdown exchange, the Strata
generator's CLI integration, checkpoint bytes, expert outputs, or complete
generation. The launcher checks pack type and size; it does not hash all pack
bytes. The coordinator checks resident expert payload hashes when it loads
them. The resident-payload caps cover declared payload bytes, not process RSS,
allocator overhead, or other model state. No GPU, RDMA, inter-PC worker,
production network, shared R53 reservation, performance, or service claim is
made.

Next: compile and exercise the real Rust bridge and the pinned donor adapter in
a bounded CPU-only fixture, preserving no-replay behavior. Complete checkpoint
generation remains dependent on a separately admitted model/resource plan.
