# R692 — GGUF checkpoint metadata inventory (CPU-only)

## Result

Added a small metadata-only inventory executable around the pinned llama.cpp/GGML
GGUF parser. It reports the GGUF version, file size, tensor count and quantization
type histogram; architecture dimensions; selected tokenizer metadata; and the
checkpoint's optional `general.license` metadata. It opens GGUF with
`no_alloc=true`, so it does not allocate or load tensor weights. Compatibility
is deliberately reported as `NOT_ATTEMPTED`.

The local test used the retained R691 synthetic Qwen2MoE GGUF (13,536 bytes):
GGUF v3, three Q8_0 tensors, hidden length 32, one block, and four checkpoint
experts. The fixture has no tokenizer or license metadata, which the inventory
reports as absent rather than inferring either from the architecture or its
synthetic data. A malformed GGUF was rejected with status 3 and emitted no JSON.

## Reuse and adaptation

- Donor: `ggml-org/llama.cpp` commit
  `3cf03257f219afbe7334045ff7c6a06ac68c627d`, MIT.
- Reused the donor's public GGUF metadata reader (`gguf_init_from_file`,
  `gguf_find_key`, `gguf_get_*`) and GGML tensor type names. No upstream source
  was copied into VoltaFabric.
- The output is a screening record, not a model loader, tokenizer validator,
  license determination, or compatibility claim. The model card/source terms
  still need independent verification.

## Verification

`BUILD_TEST.txt` records the direct compile and run. C++17
`-Wall -Wextra -Werror` passed; the shell runner passed `bash -n`. The
runner verified the known synthetic fixture's metadata/type histogram and
rejected malformed input without publishing partial JSON. No model weights
were read.

## Scope and next step

This pass inventories the retained synthetic fixture only. Noninteractive SSH
probes to the configured PC4 aliases returned exit 255 with diagnostics
suppressed; the cause was not established, and this is not evidence that PC4
contains no candidate. R689's earlier path inventory remains limited evidence.
No checkpoint was downloaded, no GPU or production network was used, and no
model compatibility or performance claim is made.

R693 should use this inventory on an already-present, owner-admitted real GGUF
candidate once its source path is available, record its actual architecture,
formats, tokenizer metadata, and authoritative license, then select one CPU
layer for checkpoint-backed comparison. Do not infer absence from the blocked
inventory or introduce downloads/GPU execution.
