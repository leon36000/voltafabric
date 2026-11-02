# R722 — pinned DeepSeek index reader reaches the CPU Session path

## Change

R721 added a bounded generic Hugging Face Safetensors index-to-shard reader. R722 connects that reader to VoltaFabric's pinned DeepSeek-V4.1 selected-expert contract and returns the existing `PinnedMxfp4RankExpertRead` receipt, which already feeds resident-owner construction and CPU Session LOAD/EXEC.

The new `read_pinned_mxfp4_rank_expert_indexed` entry point validates the complete route identity, nonzero attempt, selected expert, rank/local registry binding, exact pinned index SHA-256, index tensor count and total size, exact six expert tensor names, pinned header SHA-256, and all six caller-supplied payload SHA-256 values. It uses the existing pinned planner to verify the six MXFP4 tensor dtypes, shapes, byte ranges, and index/header mapping, then uses R721's safe indexed-file reader for selected payload reads. It rechecks the shard header and file length after the read and returns no receipt if any validation fails.

The pinned DeepSeek metadata currently maps all six tensors for a selected backbone expert to one shard. R722 explicitly refuses if that pinned mapping changes to split the six tensors across shards; the generic R721 reader itself remains capable of reading a selected set spread across multiple shards. The shared header-only helper uses the same root confinement, non-symlink regular-file check, `O_NOFOLLOW` open on Unix, file bounds, and stable-size/mtime check as payload reads.

## Reuse and provenance

- Reuses R721's existing Rust Safetensors index planner, per-tensor digest reader, and no-follow shard opening; no new crate was added.
- Reuses VoltaFabric's pinned DeepSeek metadata plan (R709), selected expert receipt and resident owner (R710), route preflight, CPU Session, and Collector path. No upstream source code was copied.
- The index-to-shard mapping idea follows Hugging Face/vLLM checkpoint conventions documented in [R721](../R721/REPORT.md). The vLLM and SGLang sources screened there are Apache-2.0; DeepSeek-V4.1-Flash checkpoint metadata is pinned to revision `314113f8a7dadb388215396ba5fbee2c610a04d6` (MIT). Those licenses do not qualify model compatibility or donor quality.

## Verification

The targeted CPU fixture uses the committed pinned DeepSeek index and one committed shard header, but writes synthetic bytes only for the 36 selected tensors. The 7.39-GB shard file is sparse and temporary under the repository's `.work`; only the selected payload ranges are written. The test reads six expert owners through the new indexed entry point, sends them through the existing CPU Sessions, executes one top-6 route, and compares against the independent FP64 MXFP4 expert oracle. Route order, coefficient bits, expert identity, quantization format, and collector contract remain unchanged.

Debug and release both pass the focused indexed-reader unit test and the six-session route integration test. Scoped `-D warnings` Clippy passes for the library and integration target; rustfmt and `git diff --check` pass.

## Limits and next step

This verifies a pinned metadata plus synthetic-payload CPU path, not actual checkpoint weights, a complete model generation, GPU execution, inter-PC transport, production R53, or performance. The indexed path is now connected through one selected-expert owner to the existing six-expert CPU Session and oracle test. Next, connect the actual gate-selected token route to this file-backed indexed loader and Session execution while preserving atomic preflight; real checkpoint admission remains a separate prerequisite. Automatic relaunch remains stopped by the owner.
