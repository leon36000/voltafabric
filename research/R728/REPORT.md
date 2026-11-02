# R728 — indexed shared FP8 load and terminal routed-result join

## Result

R728 extends the pinned DeepSeek-V4.1-Flash CPU reference with an indexed loader for
its always-active shared expert and an identity-bound join to a completed routed
Collector result. The loader selects exactly six weight/scale tensors for one layer,
checks the pinned index, shard header, caller-admitted tensor hashes, shapes and dtypes,
confirms that the shard header did not change during the read, and returns a resident
FP8 expert plus a load receipt. It reads only the selected tensor ranges through the
existing bounded VoltaFabric Safetensors reader.

The checkpoint uses E4M3FN weight bytes and E8M0 block scales. Scales are stored as
`[M/32, K/32]`; the existing CPU matrix view expects `[M, K/32]`. R728 adds a bounded
32-row expansion and unit coverage for its block mapping, invalid shape/length and
E8M0 NaN. The FP8 planes remain distinct from the routed experts' MXFP4 K32 codes.
The selected shared payload is 35,424,000 bytes across the three projections.

The layer evaluates the resident shared path on the activation already attached to the
gate-produced token plan. Its result remains raw and unweighted, with the full Stamp and
attempt. The final join accepts only a `Completed` routed receipt with matching
run/model/plan/layer/batch/epoch and attempt, six unique selected experts, complete
rank receipts, and one finite hidden row. It performs one FP32 routed-plus-shared sum
per element and one BF16 round-to-nearest-even cast. Aborted, Unknown, mismatched,
partial, nonfinite and overflowing cases return an error and expose no combined row.
No Session request or retry is issued by the join.

## Reuse and provenance

- DeepSeek-V4.1-Flash revision `314113f8a7dadb388215396ba5fbee2c610a04d6`, MIT. The
  semantic boundary and checkpoint layout were adapted independently; no donor source
  code was copied and no dependency was added.
- R728 reuses VoltaFabric's existing indexed Safetensors planner/reader and the R727
  resident FP8 matrix/reference operator. The narrow adaptation here is the six-plane
  indexed load, checkpoint-scale layout conversion, identity-preserving raw output,
  and terminal route/shared composition.
- The one GPT-6 Luna arXiv survey is recorded in R727's
  [`OPEN_SOURCE_SCREEN.md`](../R727/OPEN_SOURCE_SCREEN.md). Its V100 KV-residency and
  cancellable-prefetch ideas remain research leads; no arXiv implementation was copied
  into R728. R721's vLLM index convention was already adapted into VoltaFabric's
  bounded reader; this tranche imports no vLLM runtime or kernel.
- No component from Strata, Maya, llama.cpp, TensorRT-LLM or SGLang was selected for
  this specific shared-FP8 boundary. Their code, versions and licenses are not
  requalified by R728.

## Verification and limits

Local CPU tests use pinned index/header metadata with synthetic selected tensor
payloads in a sparse shard fixture. They verify selected ranges, hashes, exact
identity, one addition/cast, overflow refusal and withheld output for Unknown,
Aborted, mismatched-attempt/plan and partial rows. Existing R712 CPU Session tests also
pass in debug and release; however, the R728 join test constructs a synthetic completed
route receipt. It therefore does **not yet** prove that an actual R712 Session-produced
receipt passes through the new join.

This is not an actual model-weight load, whole transformer block, token generation,
GPU, inter-PC, production R53 or performance qualification. HC post/residual mixing,
attention/GDN integration and a full generation path remain to be built and tested.
No UNKNOWN attempt was replayed.

## Next boundary

R729 should feed a completed route receipt produced by the existing CPU Session path
into the R728 shared join under one exact token identity, with an independent numerical
oracle. Then continue composing HC post/residual and attention/GDN into a complete
generation path without changing route coefficients or accepting partial results.
