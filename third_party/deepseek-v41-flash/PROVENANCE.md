# DeepSeek-V4.1-Flash provenance (A58)

- Source: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- Revision: 314113f8a7dadb388215396ba5fbee2c610a04d6
- License: MIT (copy in `LICENSE`, Copyright (c) 2023 DeepSeek)
- Pinned metadata fixture: `fixtures/deepseek_v41_config_314113f8.json`, fetched
  from the official `config.json` at this exact revision; 3,311 bytes,
  SHA-256 `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`.
  This is architecture/quantization metadata only; no checkpoint shard was downloaded.
- Real expert fixture: `fixtures/deepseek_v41_layer0_expert0_314113f8.safetensors`
  contains only the six `w1/w2/w3` I8/F8_E8M0 tensors for layer 0, expert 0,
  extracted by exact HTTP byte ranges from the pinned shard after checking the
  official Safetensors index and header. It has 18,800,640 payload bytes and a
  separate extraction receipt under `evidence/continuation_a97_model_metadata/`;
  the 7.39 GB shard and 510 GB checkpoint were not downloaded.
- Exact pinned source snapshots for `inference/convert.py`, `inference/kernel.py`
  and `inference/model.py` are retained under
  `evidence/continuation_a97_model_metadata/upstream_314113f8/` with their
  SHA-256 hashes and the repository MIT license. They are reference evidence,
  not vendored runtime code. The converter maps the low nibble before the high
  nibble, and the official FP4 GEMM treats the packed values as consecutive K
  elements with one E8M0 scale per 32 K values. The A97 format cross-check
  verifies that pinned convention against all 35,389,440 codes in the real
  expert fixture.
- The official `Expert.forward` does gate/up FP4 GEMMs, clamps up to ±10 and
  gate from above at 10, computes SiLU(gate) × up, applies the routing weight,
  casts back to the input dtype, then performs the down projection. The CPU
  reference covers compressed weight decoding and a single w1 matvec only; it
  does not reproduce activation quantization, the official GEMM accumulation,
  expert forward, routing, or a GPU kernel.
- Files consulted: `config.json` (`swiglu_limit` 10.0, `expert_dtype` fp4,
  `scale_fmt` ue8m0), `inference/model.py` (`Linear` FP4 layout: weight
  `[out, in//2]` float4_e2m1fn_x2, scale `[out, in//32]` float8_e8m0fnu;
  `Expert.forward` clamps and routing weight), one shard safetensors header
  (`layers.0.ffn.experts.0.w{1,2,3}` I8/F8_E8M0 shapes).
- Adaptation: semantics re-implemented in Rust (`src/deepseek_v41.rs`); no source
  code is copied or vendored. Only the pre-down activation and the MXFP4 format
  decode are implemented; no BF16 cast, activation quant, GEMM, routing, GPU kernel
  or full model.
