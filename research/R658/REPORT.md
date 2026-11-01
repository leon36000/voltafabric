# R658 — GGUF tensor span to persistent grouped-matrix Session

## Change

VoltaFabric can now resolve one expert-stack tensor from an already-open GGUF checkpoint using the pinned GGML/GGUF metadata API, then stream only that tensor's byte span into one identified CPU `Session`. The owner worker stages bounded chunks, validates the declared byte count, planned digest and GGML shape before creating the resident matrix. Any short source, protocol failure, refusal, timeout or mismatched acknowledgement poisons that Session; no partial resident or output is reported.

The transfer uses 1 MiB chunks under one deadline and an existing raw-frame-independent request header. The per-stack ceiling is 1,250,000,000 bytes; a grouped batch also rejects participating resident stacks whose combined transferred bytes exceed that ceiling. The native bridge retains its bounded payload checks. The model's tensor bytes are never parsed or copied by GGUF metadata resolution; the bytes are read once from the validated offset and staged by the receiving worker before digest validation.

## Reuse and provenance

This is a narrow reuse of the GGUF metadata reader already shipped in the repository's pinned llama.cpp/GGML source at `3cf03257f219afbe7334045ff7c6a06ac68c627d` (MIT, copyright ggml authors). VoltaFabric calls its metadata-only `gguf_*` API to resolve tensor type, dimensions, data offset and span. The runtime's existing VoltaFabric Session, routing, resident matrix, worker lifecycle and Collector remain in control. No llama.cpp generation loop, scheduler, executor or model runtime was copied or invoked, and no quality claim is inherited from the donor.

## Verification

The isolated Linux CPU harness ran the targeted Session integration tests in debug and release: 2 worker unit tests and 4 integration tests passed in each profile. The integration fixture uses a hand-serialized GGUF v3 Q8_0 expert tensor and compares resident Session results with an independent scalar one-hot projection oracle. A second fixture streams 16,816,128 bytes, exceeding the existing 16 MiB raw Session frame limit, and reaches resident state. A short source test confirms the Session is poisoned and closed before it can be reused. CMake's `native_provider` CTest also passed. Targeted rustfmt and Clippy succeeded; Clippy reports two pre-existing warnings in untouched `src/sc117_layer_chain.rs` and `src/strata_native_bridge.rs`. Whole-workspace rustfmt remains noisy because numerous unrelated older binaries are not formatted.

## Limits

This proves one synthetic Q8_0 tensor stack through local CPU Sessions only. It does not qualify a production checkpoint, a full model or generation, V100/CUDA, multi-machine transfer, RDMA, or performance. `gguf_init_from_file(no_alloc=true)` avoids tensor allocation but the upstream parser still parses metadata before VoltaFabric checks the resulting data offset against 128 MiB. The current cap is per Session at load time, with an aggregate participating-stack cap at execution time; callers still need an admission/placement decision before loading many Sessions concurrently. The resident path also stages bytes before constructing the native matrix, so peak host memory can exceed the packed tensor size.

## Next

R659 should map one verified, compatible checkpoint architecture's expert FFN tensors and formats to these span bindings, load only a small complete set into identified CPU Sessions, and compare the gate/up/down projection sequence with an independent oracle. Do not load a full checkpoint or invoke hardware/network routes until a fresh resource and hardware gate permits it.
