# R694 publication correction and readback

Private branch `r694-qwen2moe-route-bound-placement-20261102` first published source commit `b2ec530cec030218da9e011e95c65009a52a190c`. A fresh GitLab API readback found one duplicate `#[derive(Debug, Clone, Serialize, Deserialize)]` immediately before `VariantCost`; the local exact-source test had not included that duplicate. The issue was corrected without rewriting history in commit `ff97fed0ad9374c07194fb56a7972fdd67e309e4`.

The corrected `src/planner.rs` readback is byte-for-byte equal to the local corrected source (26,923 bytes). The seven R694 source/evidence files were read back exactly. The targeted local module test passed (3/3) with rustfmt and Clippy. GitLab pipelines 2367 (b2ec) and 2368 (ff97) ended success; configured checks were Gitleaks, Semgrep, Trivy, and fallback. Those pipelines are not a Rust compile proof; local exact-module tests provide that scoped evidence.
