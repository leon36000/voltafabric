# R718 — route-wide authenticated MXFP4 payload read

## Change

R717 joined the DeepSeek top-6 gate result to a pinned Safetensors range plan, but the existing authenticated read API operated one expert at a time. A caller could therefore begin reading selected bytes for early experts before learning that a later expert's digest was absent from its retained source manifest.

R718 adds a route-level read boundary. It rechecks the immutable route plan against the pinned index/header, validates exact coverage and SHA-256 syntax for all 36 selected tensor entries before touching the payload reader, then invokes the existing bounded per-expert reader in gate order. The returned value is available only if all six selected experts pass source-header stability and tensor-digest checks. Conversion to six Session resident owners and grouping by the exact rank/local dispatch is also all-or-error.

The integration test proves both refusal and success on a sparse CPU fixture using the tracked pinned DeepSeek index/header and synthetic payload bytes. A 35/36 digest map fails with the reader position still at zero. A complete synthetic map reads six experts, creates six immutable owners, passes the existing top-6 route preflight, and groups them into five ranks, with the first two colocated on rank 4. Existing route identity, attempt, route order, coefficient bits, and rank/local bindings remain authoritative.

## Provenance and reuse

The implementation reuses VoltaFabric's pinned-range planner, `routing::dispatch`, one-expert authenticated Safetensors reader, resident-owner conversion, route preflight, and rank-group builder. `safetensors` 0.8.0 remains the already-pinned Apache-2.0 parser. The pinned DeepSeek-V4.1-Flash revision `314113f8a7dadb388215396ba5fbee2c610a04d6` is an MIT-licensed semantic reference; no donor code is copied.

## Verification

- Focused Linux integration test: 1/1 debug and 1/1 release passed.
- Strict targeted Clippy (`-D warnings`), rustfmt check, and `git diff --check` passed.
- The synthetic route exercised 36 selected ranges (112,803,840 bytes), not real model payloads.

## Limits

This establishes a CPU file-to-owner boundary only. The selected expert payload bytes are not all in the tracked inventory (experts 379–382 are missing there; that is not a claim of global absence). No real checkpoint payload was loaded. No Session worker was started from this returned fixture, no generation ran, and no GPU, production R53, inter-PC transport, or production registry was used. No performance claim is made. The donor shard must remain quiescent during the route read. Model-wide checkpoint admission and full generation remain unfinished.

## Source hashes

- `src/deepseek_v41_gate.rs`: `c443d26993ac12ecbeb340641b613dee667c009d47df35c9ec730c9602bac312`
- `tests/r712_deepseek_session_owner.rs`: `044c44a4dd4cc6ea612e83a362269f20391f2450ee331bdbf43af43c704df07f`

