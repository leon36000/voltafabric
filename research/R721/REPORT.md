# R721 — bounded indexed Safetensors shard import

## Result

R720 identified a practical checkpoint boundary from mature loaders: Hugging Face's `model.safetensors.index.json` maps tensor names to shard files. R721 adds that mapping and the bounded file-read step to VoltaFabric's existing Rust checkpoint module. The implementation reuses the already pinned `safetensors` 0.8.0 crate and `read_named_tensors`; it adds no package and copies no donor code.

The caller supplies the expected SHA-256 for the index and for each selected tensor, along with its expected dtype, shape, and byte count. The reader validates the bounded index, resolves only the requested names, groups them by shard, then reads those payloads from safe files directly under the supplied root. It rejects duplicate JSON keys, unsafe shard names, path escapes, symlink shards, non-regular files, changed file size/mtime, absent tensors, wrong metadata or payload digest, and aggregate requests over 64 tensors or 64 MiB. It stages the map locally and returns it only after every requested shard and tensor succeeds.

This is a real checkpoint-loading boundary, but it is not yet connected to a model-generation entry point. The index is a location map, not a trust root: its expected digest and the per-tensor digests must come from a pinned manifest. The generic loader does not convert formats or establish that a checkpoint is compatible with QPN2/NVFP4 K16, MXFP4 K32, IQ3, or any target model.

## Donor/version and compatibility screen

- vLLM v0.17.0's official installation page describes its CUDA 12.8 binary path and a generic compute-capability 7.0+ floor, which includes V100. That generic floor does not qualify DeepSeek V4.1 Flash on V100: the current official V4.1 recipe asks for vLLM 0.31.0 or newer and at least 614 GB of VRAM, and specifies MXFP4 MoE with MXFP8 dense weights. This excludes the known V100 PC1 configuration as an immediate full-engine target; it does not establish that every earlier vLLM build fails for every V100 workload.
- The SGLang v0.5.21 release page lists DeepSeek-V4.1 Flash and V4.1-specific changes. The releases page identifies v0.5.19 as the last release with CUDA 12.x wheels/images, so the V4.1 release point is not an established CUDA-12/V100 deployment path. No installation or model download was attempted.
- The cited vLLM and SGLang projects use Apache-2.0 licensing at the inspected source points. R721 imports no source code from either engine; it adapts the documented index-to-shard loading boundary using VoltaFabric's existing Safetensors reader. Donor license does not certify this adaptation or any complete-model behavior.

Sources:

- [vLLM v0.17.0 GPU installation](https://docs.vllm.ai/en/v0.17.0/getting_started/installation/gpu/)
- [vLLM v0.17.0 `weight_utils.py`](https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/model_executor/model_loader/weight_utils.py)
- [DeepSeek V4.1 Flash vLLM recipe](https://github.com/vllm-project/vllm/blob/main/recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml)
- [vLLM v0.31.0 DeepSeek V4 implementation](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/models/deepseek_v4/nvidia/model.py)
- [vLLM v0.31.0 license](https://github.com/vllm-project/vllm/blob/v0.31.0/LICENSE)
- [SGLang v0.5.21 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.21)
- [SGLang releases and CUDA 12.x cutoff](https://github.com/sgl-project/sglang/releases)
- [SGLang license](https://github.com/sgl-project/sglang/blob/v0.5.21/LICENSE)

## Verification and limits

Five synthetic CPU tests cover two-shard extraction while skipping an unrelated tensor, bad second-shard payload digest with no returned partial import, absent shard, wrong index digest, symlink refusal, and total-byte-cap rejection. Debug and release targeted tests, scoped Clippy with `-D warnings`, `cargo fmt --check`, and `git diff --check` pass. No model, GPU, production R53, service, or inter-PC operation was used. No performance comparison was made.

The next slice is to connect this reader to a pinned VoltaFabric model/checkpoint manifest and a selected-expert load plan, then exercise one synthetic sharded layer through the existing CPU Session path and an independent tensor oracle. Keep coefficients, route, expert identities and quantization formats unchanged.
