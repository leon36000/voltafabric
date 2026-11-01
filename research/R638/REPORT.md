# R638 — Identity-bound Strata native pipe attempts

## Change

R638 adapts the existing VoltaFabric AF_UNIX bridge so a caller can bind every VFB3 batch to the trusted launcher identity and to an explicit attempt. The 52-byte fixed request header carries layer, epoch, the run/model-revision/plan field lengths, and the u64 attempt; the three identity strings follow the fixed header. Batch and attempt remain separate coordinates. VFR3 echoes the identity and attempt.

The C++ Client validates the echoed coordinates before accepting staged rows. PoolFleetOwner::create_from_fds accepts already initialized inherited AF_UNIX descriptors, validates the same run/model/plan/epoch across all 48 layers, and maintains independent starting batch and attempt counters. The Rust coordinator compares VFB3 identity against its bound Init before dispatch or local ledger reservation, reserves the exact batch/attempt ticket, and passes that ticket through the unweighted FP32 ingress and native session boundary. A failure consumes the attempt; no retry or partial output is published.

VFB1/VFB2 remain supported by the coordinator for older explicitly managed callers. This change copies no Strata or llama.cpp implementation. The donor remains Strata revision fb58e0dbc8399662c0e47c76578c6e878b14f6cf (MIT); the implementation is a VoltaFabric-side adaptation of its callback boundary.

## Verification

CPU-only checks previously passed against the source set published in this branch; GitLab readback then matched all 13 full-file source updates byte-for-byte. The one reserved Session method was also read back; a final formatting-only blank line is aligned in the metadata commit.

- CMake configure/build and CTest: 3/3 pass (strata_pipe_client, strata_pipe_client_identity, strata_pool_fleet_owner).
- Rust identity unit: 1/1 pass; the comparison covers run, model revision, plan, layer, epoch, batch, and attempt.
- Rust synthetic provider integration: 1/1 pass. A mismatched identity is refused before a ledger claim; the same batch under the correct identity then succeeds. The fixture records attempts 23 and 24 separately with batch IDs 89 and 1234 in the local NativeBatchLedger.
- The verification used local CPU AF_UNIX fixtures and synthetic SC117 data; it did not invoke Strata generation or read a real checkpoint.

The first code commit c771edf45c3dc8aecadbb9b4e90c236bc076101a exposed a connector encoding mistake: 13 files were stored as base64 text. The following commit db2fbfc45e9953b6e14fe5842789d44ea08de90f replaced them with plain source text. History was preserved without a force update. The readback checks below apply to the corrected branch contents.

## Limits and next step

This is still a protocol and ownership boundary. The pinned Strata generator leaves vf_cpu_pool and vf_cpu_context unset; no launcher or actual generate.cpp callback installation is included. The pinned callback owner expects already initialized inherited descriptors; it does not create the coordinator or perform the Init/binary handshakes.

No complete generation, real-model load, GPU, V100, production network, inter-PC worker, or real R53 admission was run. The ledger in this test is local and cannot authorize or replace R53. The callback remains disconnected from generation. No runtime integration of vLLM, SGLang, TensorRT-LLM, Maya, or DeepInfer is qualified by R638; DeepInfer remains a source of ideas rather than the target engine.

Next: R639 should wire the launcher-supplied identity and session set at the pinned Strata callback lifetime boundary, first with the real pinned generator on CPU and an independent oracle. Keep GPU, inter-PC transport, production networking, and real R53 suspended absent fresh admission.
