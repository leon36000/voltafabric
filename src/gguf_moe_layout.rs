//! Metadata-only resolver for the llama.cpp GGUF expert-FFN tensor convention.
//!
//! This identifies checkpoint tensor names and physical dimensions. It does not
//! interpret router semantics, activation functions, shared experts, biases, or
//! execute an MoE layer; those remain model-specific runtime work.

#![cfg(all(feature = "native-ggml-cpu", target_os = "linux"))]

use std::collections::BTreeMap;

use vf_ggml_native_ffi::{GgufIndex, GgufTensorSpan};

const MAX_LAYERS: u32 = 4096;
const MAX_MODEL_DIMENSION: u32 = 1 << 20;
const MAX_ARCHITECTURE_BYTES: usize = 64;

/// Small interface keeps the name/shape policy testable without loading model data.
pub trait ExpertFfnIndex {
    fn string(&self, name: &str) -> Result<String, &'static str>;
    fn u32(&self, name: &str) -> Result<u32, &'static str>;
    fn has_tensor(&self, name: &str) -> Result<bool, &'static str>;
    fn tensor_span(&self, name: &str) -> Result<GgufTensorSpan, &'static str>;
}

impl ExpertFfnIndex for GgufIndex {
    fn string(&self, name: &str) -> Result<String, &'static str> {
        GgufIndex::string(self, name)
    }

    fn u32(&self, name: &str) -> Result<u32, &'static str> {
        GgufIndex::u32(self, name)
    }

    fn has_tensor(&self, name: &str) -> Result<bool, &'static str> {
        GgufIndex::has_tensor(self, name)
    }

    fn tensor_span(&self, name: &str) -> Result<GgufTensorSpan, &'static str> {
        GgufIndex::tensor_span(self, name)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertFfnLayer {
    pub layer: u32,
    pub gate_name: String,
    pub up_name: String,
    pub down_name: String,
    pub gate: GgufTensorSpan,
    pub up: GgufTensorSpan,
    pub down: GgufTensorSpan,
    /// True when gate and up share the `ffn_gate_up_exps` checkpoint tensor.
    pub fused_gate_up: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertFfnLayout {
    pub architecture: String,
    pub hidden_size: u32,
    pub expert_count: u32,
    pub expert_ffn_size: u32,
    pub layers: Vec<ExpertFfnLayer>,
}

/// Resolve exact `blk.N.ffn_{gate,up,down}_exps.weight` GGUF names and their
/// llama.cpp physical dimensions. Fused gate/up is accepted only when both halves
/// and their order are represented by one `[hidden, 2*ffn, experts, 1]` tensor.
pub fn resolve_expert_ffn_layout<I: ExpertFfnIndex>(
    index: &I,
    file_bytes: u64,
) -> Result<ExpertFfnLayout, &'static str> {
    let architecture = index.string("general.architecture")?;
    if architecture.is_empty()
        || architecture.len() > MAX_ARCHITECTURE_BYTES
        || !architecture
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err("unsupported GGUF architecture identifier");
    }
    let metadata_key = |suffix: &str| format!("{architecture}.{suffix}");
    let layers = index.u32(&metadata_key("block_count"))?;
    let hidden_size = index.u32(&metadata_key("embedding_length"))?;
    let expert_count = index.u32(&metadata_key("expert_count"))?;
    let expert_ffn_size = index.u32(&metadata_key("expert_feed_forward_length"))?;
    if layers == 0
        || layers > MAX_LAYERS
        || hidden_size == 0
        || hidden_size > MAX_MODEL_DIMENSION
        || expert_count == 0
        || expert_count > MAX_MODEL_DIMENSION
        || expert_ffn_size == 0
        || expert_ffn_size > MAX_MODEL_DIMENSION / 2
    {
        return Err("GGUF expert-FFN metadata is outside supported bounds");
    }

    let mut resolved = Vec::with_capacity(layers as usize);
    for layer in 0..layers {
        let stem = format!("blk.{layer}.ffn_");
        let fused_name = format!("{stem}gate_up_exps.weight");
        let gate_name = format!("{stem}gate_exps.weight");
        let up_name = format!("{stem}up_exps.weight");
        let down_name = format!("{stem}down_exps.weight");
        let fused = index.has_tensor(&fused_name)?;
        let separate_gate = index.has_tensor(&gate_name)?;
        let separate_up = index.has_tensor(&up_name)?;
        let has_down = index.has_tensor(&down_name)?;
        if !has_down
            || (fused && (separate_gate || separate_up))
            || (!fused && !(separate_gate && separate_up))
        {
            return Err("GGUF expert FFN tensors are incomplete or ambiguous");
        }

        let down = checked_span(
            index,
            &down_name,
            [
                expert_ffn_size as i64,
                hidden_size as i64,
                expert_count as i64,
                1,
            ],
            file_bytes,
        )?;
        let (gate_name, gate, up_name, up, fused_gate_up) = if fused {
            let fused = checked_span(
                index,
                &fused_name,
                [
                    hidden_size as i64,
                    i64::from(expert_ffn_size) * 2,
                    expert_count as i64,
                    1,
                ],
                file_bytes,
            )?;
            (fused_name.clone(), fused, fused_name, fused, true)
        } else {
            let expected = [
                hidden_size as i64,
                expert_ffn_size as i64,
                expert_count as i64,
                1,
            ];
            (
                gate_name.clone(),
                checked_span(index, &gate_name, expected, file_bytes)?,
                up_name.clone(),
                checked_span(index, &up_name, expected, file_bytes)?,
                false,
            )
        };
        resolved.push(ExpertFfnLayer {
            layer,
            gate_name,
            up_name,
            down_name,
            gate,
            up,
            down,
            fused_gate_up,
        });
    }

    Ok(ExpertFfnLayout {
        architecture,
        hidden_size,
        expert_count,
        expert_ffn_size,
        layers: resolved,
    })
}

fn checked_span<I: ExpertFfnIndex>(
    index: &I,
    name: &str,
    dimensions: [i64; 4],
    file_bytes: u64,
) -> Result<GgufTensorSpan, &'static str> {
    let span = index.tensor_span(name)?;
    if span.dimensions != dimensions
        || span.bytes == 0
        || span
            .offset
            .checked_add(span.bytes)
            .is_none_or(|end| end > file_bytes)
    {
        return Err("GGUF expert FFN tensor shape or file span is invalid");
    }
    Ok(span)
}

/// Build the unambiguous tensor->span view for loaders. Fused gate/up contributes one
/// source tensor; separate gate/up contributes two. The result contains no tensor bytes.
pub fn layer_spans(layer: &ExpertFfnLayer) -> BTreeMap<&str, GgufTensorSpan> {
    let mut spans = BTreeMap::from([(layer.down_name.as_str(), layer.down)]);
    spans.insert(layer.gate_name.as_str(), layer.gate);
    if layer.up_name != layer.gate_name {
        spans.insert(layer.up_name.as_str(), layer.up);
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Default)]
    struct FixtureIndex {
        strings: HashMap<String, String>,
        integers: HashMap<String, u32>,
        tensors: HashMap<String, GgufTensorSpan>,
    }

    impl ExpertFfnIndex for FixtureIndex {
        fn string(&self, name: &str) -> Result<String, &'static str> {
            self.strings.get(name).cloned().ok_or("missing string")
        }
        fn u32(&self, name: &str) -> Result<u32, &'static str> {
            self.integers.get(name).copied().ok_or("missing integer")
        }
        fn has_tensor(&self, name: &str) -> Result<bool, &'static str> {
            Ok(self.tensors.contains_key(name))
        }
        fn tensor_span(&self, name: &str) -> Result<GgufTensorSpan, &'static str> {
            self.tensors.get(name).copied().ok_or("missing tensor")
        }
    }

    fn span(dimensions: [i64; 4], offset: u64) -> GgufTensorSpan {
        GgufTensorSpan {
            offset,
            bytes: 32,
            ggml_type: 8,
            dimensions,
        }
    }

    fn metadata() -> FixtureIndex {
        FixtureIndex {
            strings: HashMap::from([("general.architecture".into(), "qwen2moe".into())]),
            integers: HashMap::from([
                ("qwen2moe.block_count".into(), 1),
                ("qwen2moe.embedding_length".into(), 32),
                ("qwen2moe.expert_count".into(), 2),
                ("qwen2moe.expert_feed_forward_length".into(), 16),
            ]),
            ..FixtureIndex::default()
        }
    }

    #[test]
    fn resolves_separate_gate_up_and_down_names_with_exact_shapes() {
        let mut index = metadata();
        let matrix = [32, 16, 2, 1];
        let down = [16, 32, 2, 1];
        index
            .tensors
            .insert("blk.0.ffn_gate_exps.weight".into(), span(matrix, 32));
        index
            .tensors
            .insert("blk.0.ffn_up_exps.weight".into(), span(matrix, 64));
        index
            .tensors
            .insert("blk.0.ffn_down_exps.weight".into(), span(down, 96));
        let layout = resolve_expert_ffn_layout(&index, 256).unwrap();
        assert_eq!(layout.architecture, "qwen2moe");
        assert_eq!(layout.hidden_size, 32);
        assert_eq!(layout.expert_count, 2);
        assert_eq!(layout.expert_ffn_size, 16);
        assert_eq!(layout.layers.len(), 1);
        assert!(!layout.layers[0].fused_gate_up);
        assert_eq!(layer_spans(&layout.layers[0]).len(), 3);
    }

    #[test]
    fn accepts_fused_gate_up_only_at_its_declared_doubled_extent() {
        let mut index = metadata();
        index.tensors.insert(
            "blk.0.ffn_gate_up_exps.weight".into(),
            span([32, 32, 2, 1], 32),
        );
        index.tensors.insert(
            "blk.0.ffn_down_exps.weight".into(),
            span([16, 32, 2, 1], 96),
        );
        let layout = resolve_expert_ffn_layout(&index, 256).unwrap();
        assert!(layout.layers[0].fused_gate_up);
        assert_eq!(layer_spans(&layout.layers[0]).len(), 2);
    }

    #[test]
    fn rejects_incomplete_ambiguous_and_out_of_file_tensor_sets() {
        let mut index = metadata();
        index.tensors.insert(
            "blk.0.ffn_gate_up_exps.weight".into(),
            span([32, 32, 2, 1], 32),
        );
        index.tensors.insert(
            "blk.0.ffn_gate_exps.weight".into(),
            span([32, 16, 2, 1], 64),
        );
        index.tensors.insert(
            "blk.0.ffn_down_exps.weight".into(),
            span([16, 32, 2, 1], 96),
        );
        assert_eq!(
            resolve_expert_ffn_layout(&index, 256).unwrap_err(),
            "GGUF expert FFN tensors are incomplete or ambiguous"
        );

        index.tensors.remove("blk.0.ffn_gate_exps.weight");
        index.tensors.remove("blk.0.ffn_gate_up_exps.weight");
        index.tensors.insert(
            "blk.0.ffn_gate_exps.weight".into(),
            span([32, 16, 2, 1], 240),
        );
        index
            .tensors
            .insert("blk.0.ffn_up_exps.weight".into(), span([32, 16, 2, 1], 64));
        assert_eq!(
            resolve_expert_ffn_layout(&index, 256).unwrap_err(),
            "GGUF expert FFN tensor shape or file span is invalid"
        );
    }
}
