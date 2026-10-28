# Lossless BF16 activation wire format (R32)

MXFP4 `TokenInput.value_bits` continues to identify finite, exactly BF16 values
using binary32 bits. The negotiated encoding changes only their wire bytes.
Weights stay compressed and resident after LOAD. Output slots, routing
coefficients, accumulation order and the FP32 arithmetic contract are unchanged.

## Negotiation and compatibility

A worker implementing `mxfp4_exec_bf16_exact_raw_v1` advertises that capability in
HELLO, separately from its CPU or native GPU producer. Raw32 remains the default.
For an explicit experiment, a series WorkerSource can set
`"prefer_bf16_activations": true`, or a Rust caller can invoke
`Session::prefer_mxfp4_bf16_activations()`. When requested and supported, the client sends
`exec_preweighted_bf16_raw`; otherwise it sends the original
`exec_preweighted_fp32_raw`. Both operations carry the same artifact digest,
RankDispatch, contract, attempt, token metadata and fault controls. The raw BF16
body is exactly two bytes per value, little-endian, containing binary32 bits
31..16. Decoding appends sixteen zero bits. There is no FP16 cast, rounding,
clamping or lossy quantization. Signed zero and finite subnormals survive exactly.

Old clients continue to use raw32 with new workers. New clients use raw32 with
old workers, even when BF16 was preferred. This compatibility choice is made before transmission; a failure
after transmission never causes a format fallback or execution retry.

## Admission and failures

Before reading/allocating the raw body, the worker bounds metadata to 4096 rows,
65536 values per row, and a checked aggregate of at most
`MAX_RAW_ACTIVATION_BYTES / 4` values. The old decoded-memory limit remains in
force even though the wire body is shorter. Declared bytes must equal twice the
aggregate value count. The existing combined header/body frame cap also applies.
Nonfinite BF16 encodings are rejected before rows are published. Ordinary
dispatch, artifact, coefficient, identity and epoch checks remain mandatory
before Started or any provider call. Partial/corrupt input closes the session.

The existing UNKNOWN semantics apply after ambiguous transmission. Missing
Started is not evidence of non-execution. Neither raw32 fallback nor automatic
replay is allowed after stream loss. All completion frames remain FP32.

## Evidence and limits

R32 contains an exhaustive finite-pattern codec check, an independent byte
oracle, five compilable mutation checks, real subprocess tests of negotiation,
legacy raw32 and loss during BF16 input, plus the normal CPU qualification suite.
`tests/reference/bf16_legacy_hello_proxy.py` masks only a real worker's capability;
it does not simulate numeric outputs. Interoperability against the frozen R31
binary is a separate experiment under the R32 phase directory.

The full verifier builds the current CLI and workers before the Python
integration tests. This prevents testing an obsolete JSON schema or silently
skipping native tests when release binaries are absent. `--python-only` remains
an explicitly partial check and never yields a full CPU qualification.

Halving the activation body does not halve total traffic: request metadata,
FP32 completions and startup still use bytes. Local-pipe measurements cannot
establish inter-PC latency or GPU performance. The frozen R31 RDMA package is
unchanged; a successor requires fresh cross-architecture transport qualification.
No CUDA/GPUDirect/full-model result follows from this wire-format change.
