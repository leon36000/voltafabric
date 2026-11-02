# R728 focused reuse record

The bounded upstream/arXiv screen is in [R727](../R727/OPEN_SOURCE_SCREEN.md); the
single GPT-6 Luna survey was reused rather than repeated. R728's implementation uses
only already-published VoltaFabric boundaries plus independently written adaptation
code.

| Source | Version/license evidence | R728 decision |
| --- | --- | --- |
| DeepSeek-V4.1-Flash | Pinned revision `314113f8a7dadb388215396ba5fbee2c610a04d6`, MIT; official revision tree: <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/314113f8a7dadb388215396ba5fbee2c610a04d6> | Adapt the shared FP8 layout and execution semantics; copy no source code. |
| VoltaFabric indexed Safetensors reader (R721) | Existing project code and tests; no external dependency | Reused for bounded selected-tensor I/O and source-header revalidation. |
| VoltaFabric R727 FP8 operator | Existing project CPU reference; pinned by R727's same DeepSeek revision | Reused for resident E4M3FN/E8M0 projections. |
| vLLM | Apache-2.0 index-convention adaptation was already recorded in R721; runtime version is outside this tranche | No runtime/kernel reused; only the existing VoltaFabric reader is called. |
| InplaceKVCache / WriteScope, arXiv:2609.14507 | R727 survey found no source repository/license to pin | Residency-at-write-time remains a design lead; no code copied. |
| Speculating Experts / YALIS, arXiv:2603.19289 | Survey linked an Apache-2.0 branch but did not pin a commit | Cancellable prefetch only is a later option; no code copied and routing stays authoritative. |
| Strata, Maya, llama.cpp, TensorRT-LLM, SGLang | Versions/licenses not requalified for this exact boundary | Not selected or imported in R728. |

The arXiv screen is focused, not exhaustive. A donor's license, code quality, numerical
contract or hardware support is not inherited by VoltaFabric.
