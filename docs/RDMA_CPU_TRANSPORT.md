# Explicit bounded host-RDMA CPU transport

`scripts/rdma_cpu_bridge.c` is the exact 100 µs candidate exercised in R29/R30.
It adapts the existing Rust worker stdin/stdout byte stream with the installed
rdma-core public rsockets ABI. Rust still owns Stamp, dispatch, immutable compressed
artifacts and Collector semantics. No application protocol or GPU fallback is added.
SSH remains the default: R30 did not establish a performance advantage against it.

Build natively on each architecture, in a **new** output directory:

```
python3 scripts/build_rdma_cpu.py --out /absolute/new/build-directory
python3 scripts/build_rdma_cpu.py --out /absolute/new/sanitized-directory --sanitized
```

The parent must exist. Existing directories and symlinks are refused, including
failed/uncertain builds; inspect their receipts instead of silently rebuilding.
`build.json` identifies source bytes, binary, machine/libc, compiler hash, exact
argv, exit and status. The source snapshot and binary use relative names in that
directory. Moving the directory does not require the original source checkout.
Native compilation is COMPILED_NATIVE_ONLY, not an RDMA/GPU execution result.
The x86-64 executable cannot be used on ARM64. Rebuild for that host; preserve
its libc, loader and rdma-core compatibility requirements.

For the controlled PC1/PC6 link, endpoint argv is:

```
vf-rsocket-cpu server 10.40.13.1 18900 /private/session-cookie /absolute/owned/vf-rank-session
vf-rsocket-cpu client 10.40.13.2 18900 /private/session-cookie -
```

Only private IPs 10.40.13.1/.2 and ports 18900–18919 are admitted by this bounded
implementation. These are explicit deployment constraints, not topology discovery.
Each process accepts/connects once and has a 12 s alarm. The existing external
guard uses 15 s, hash admission, a shared exclusive NIC lease, parent-death cleanup
and terminal receipts. Provision cookies using the existing authenticated SSH
bootstrap, run only under those controls, and stop/reconcile the owned group after
each bounded session. The executable is not a daemon or public RPC listener.

The cookie must be an owned non-symlink regular file, mode 0600, exactly 64 bytes:
32 random client-auth bytes and 32 random server-auth bytes. Authentication is
checked before worker fork. It does not encrypt the link or establish security
against active on-link interception. No remote command is accepted from the stream;
the owner supplies the local worker executable at launch. Never put keys/cookies in
arguments, manifests or logs. Only their hashes and session identities are receipts.

The two fixed C queues are 65 536 bytes each. `rpoll(...,0)` advances RDMA;
`ppoll` on local descriptors requests at most 100 µs of sleep when idle. This
is not a real-time wakeup guarantee, and its wakeup/CPU cost must count at scale.
Directional EOF drains the other direction; the server ends worker stdin while
still draining worker stdout. A transport disconnect/timeout after EXEC may be
UNKNOWN even when the worker computed. No automatic reconnect/retry/exactly-once
claim is provided. Rust stages rows until full identity and coverage validate;
the adapter must never publish a partial batch or reinstall weights per EXEC.

The existing compressed CPU session is bounded by its own frame, artifact, MAC,
token/slot and aggregate-owner limits; this transport does not raise any of them.
The 12 s transport/15 s guard bounds are deliberately retained. Long sessions,
GPU inter-PC, GPUDirect, model quality/energy and full H1/H2/H3 remain separate.

The C code is project-owned laboratory code; no third-party implementation was
copied. It dynamically uses system `librdmacm.so.1`; that library and the libc/compiler
are not bundled or relicensed. Existing project licensing remains unchanged; no
new open-source license or publication right is inferred from this integration.

The canonical CPU-only supervisor is `scripts/rdma_cpu_guard.py` (packaged as
`scripts/rping_guard.py` for the existing group launcher). It fails before spawn
when Python optimization is enabled, rather than allowing `-O` to elide its
assertion-based hash checks. It also refuses GPU mode before any inventory call.
This CPU package does not replace the separate GPU guardian or hardware gate.
