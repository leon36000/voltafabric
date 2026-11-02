//! Identity-bound CPU reference for DeepSeek-V4.1's layer-specific window RoPE.
//!
//! This adapts the pinned MIT donor's adjacent-complex-pair rotation. It does
//! not use the local Qwen split-half helper: that convention is incompatible.
//! Only the pinned uncompressed sliding-window profile is admitted here.

use crate::deepseek_v41::cast_bf16_rne_reference;
use crate::deepseek_v41_attention::{
    DeepSeekV41WindowAttentionOutput, DeepSeekV41WindowAttentionRequest, DeepSeekV41WindowCache,
};
use crate::deepseek_v41_gate::PINNED_MODEL_REVISION;
use crate::deepseek_v41_projection::DeepSeekV41ProjectedQkv;
use crate::moe::Stamp;
use crate::{invalid, Error, Result};

pub const PINNED_CONFIG_SHA256: &str =
    "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879";
pub const PINNED_WINDOW_HEAD_DIM: usize = 512;
pub const PINNED_WINDOW_ROPE_DIM: usize = 64;
pub const PINNED_WINDOW_HEADS: usize = 64;
pub const PINNED_WINDOW_SIZE: usize = 128;
pub const PINNED_WINDOW_ROPE_THETA: f32 = 10_000.0;
pub const PINNED_WINDOW_MAX_POSITION: u64 = 1_048_576;

/// Parsed only from the pinned Hugging Face config and a layer whose compressed
/// attention ratio is zero in that config. The source's pure-window path
/// explicitly disables YaRN and uses `rope_theta` directly.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV41WindowRopeConfig {
    layer: u32,
    head_dim: usize,
    rope_dim: usize,
    query_heads: usize,
    window_size: usize,
    theta: f32,
    max_position: u64,
    config_sha256: String,
}

impl DeepSeekV41WindowRopeConfig {
    pub fn from_pinned_config(config_bytes: &[u8], layer: u32) -> Result<Self> {
        if crate::sha256::hex(config_bytes) != PINNED_CONFIG_SHA256 || layer >= 40 {
            return Err(invalid(
                "DeepSeek window RoPE config digest or backbone layer is not pinned",
            ));
        }
        let config: serde_json::Value = serde_json::from_slice(config_bytes)
            .map_err(|_| invalid("DeepSeek pinned RoPE config is not valid JSON"))?;
        let text = config
            .get("text_config")
            .ok_or_else(|| invalid("DeepSeek pinned config lacks text_config"))?;
        let ratio = text
            .get("compress_ratios")
            .and_then(serde_json::Value::as_array)
            .and_then(|ratios| ratios.get(layer as usize))
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| invalid("DeepSeek layer compression ratio is absent"))?;
        let hidden_layers = uint(text, "num_hidden_layers")?;
        let heads = uint(text, "num_attention_heads")?;
        let head_dim = uint(text, "head_dim")?;
        let rope_dim = uint(text, "qk_rope_head_dim")?;
        let window = uint(text, "sliding_window")?;
        let theta = text
            .get("rope_theta")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| invalid("DeepSeek window RoPE theta is absent"))?;
        let max_position = uint(text, "max_position_embeddings")?;
        if layer >= hidden_layers as u32
            || ratio != 0
            || heads != PINNED_WINDOW_HEADS as u64
            || head_dim != PINNED_WINDOW_HEAD_DIM as u64
            || rope_dim != PINNED_WINDOW_ROPE_DIM as u64
            || window != PINNED_WINDOW_SIZE as u64
            || theta != f64::from(PINNED_WINDOW_ROPE_THETA)
            || max_position != PINNED_WINDOW_MAX_POSITION
        {
            return Err(invalid(
                "DeepSeek config does not match the pinned uncompressed window RoPE profile",
            ));
        }
        Ok(Self {
            layer,
            head_dim: head_dim as usize,
            rope_dim: rope_dim as usize,
            query_heads: heads as usize,
            window_size: window as usize,
            theta: theta as f32,
            max_position,
            config_sha256: PINNED_CONFIG_SHA256.to_owned(),
        })
    }

    pub fn layer(&self) -> u32 {
        self.layer
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn rope_dim(&self) -> usize {
        self.rope_dim
    }

    pub fn query_heads(&self) -> usize {
        self.query_heads
    }

    pub fn window_size(&self) -> usize {
        self.window_size
    }

    pub fn config_sha256(&self) -> &str {
        &self.config_sha256
    }
}

fn uint(value: &serde_json::Value, key: &str) -> Result<u64> {
    value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| invalid(format!("DeepSeek pinned config field {key} is absent")))
}

/// Post-RoPE Q/KV values ready for the R733 window cache; identity and absolute
/// position travel with the values so callers cannot silently mismatch layers.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV41PostRopeQkv {
    stamp: Stamp,
    attempt: u64,
    position: u64,
    query_heads: usize,
    head_dim: usize,
    query_bf16_f32: Vec<f32>,
    key_value_bf16_f32: Vec<f32>,
}

impl DeepSeekV41PostRopeQkv {
    pub fn stamp(&self) -> &Stamp {
        &self.stamp
    }

    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    pub fn position(&self) -> u64 {
        self.position
    }

    pub fn query_heads(&self) -> usize {
        self.query_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn query_bf16_f32(&self) -> &[f32] {
        &self.query_bf16_f32
    }

    pub fn key_value_bf16_f32(&self) -> &[f32] {
        &self.key_value_bf16_f32
    }

    /// Compose the post-RoPE projection with the existing atomic R733 cache.
    /// Cache identity/position checks happen before any cache mutation.
    pub fn append_and_attend(
        &self,
        cache: &mut DeepSeekV41WindowCache,
        attention_sink_f32: &[f32],
        softmax_scale: f32,
    ) -> Result<DeepSeekV41WindowAttentionOutput> {
        cache.append_and_attend(DeepSeekV41WindowAttentionRequest {
            stamp: &self.stamp,
            attempt: self.attempt,
            position: self.position,
            query_bf16_f32: &self.query_bf16_f32,
            key_value_bf16_f32: &self.key_value_bf16_f32,
            attention_sink_f32,
            softmax_scale,
        })
    }
}

impl DeepSeekV41ProjectedQkv {
    /// Apply the pinned model's interleaved complex-pair RoPE to the final
    /// `rope_dim` values of every query head and the single shared KV row.
    /// Input and output boundaries are exact BF16 values represented as FP32.
    pub fn apply_window_rope(
        &self,
        config: &DeepSeekV41WindowRopeConfig,
        position: u64,
    ) -> Result<DeepSeekV41PostRopeQkv> {
        if self.stamp().model_revision != PINNED_MODEL_REVISION
            || self.stamp().layer != config.layer
            || self.query_heads() != config.query_heads
            || self.head_dim() != config.head_dim
            || self.attempt() == 0
            || position >= config.max_position
            || self.query_pre_rope_bf16_f32().len() != config.query_heads * config.head_dim
            || self.key_value_pre_rope_bf16_f32().len() != config.head_dim
        {
            return Err(invalid(
                "DeepSeek pre-RoPE Q/KV identity, shape or position differs from pinned config",
            ));
        }
        let mut query = self.query_pre_rope_bf16_f32().to_vec();
        let mut key_value = self.key_value_pre_rope_bf16_f32().to_vec();
        validate_bf16(&query)?;
        validate_bf16(&key_value)?;
        rotate_query_tail(
            &mut query,
            config.query_heads,
            config.head_dim,
            config.rope_dim,
            config.theta,
            position,
        )?;
        rotate_vector_tail(
            &mut key_value,
            config.head_dim,
            config.rope_dim,
            config.theta,
            position,
        )?;
        Ok(DeepSeekV41PostRopeQkv {
            stamp: self.stamp().clone(),
            attempt: self.attempt(),
            position,
            query_heads: config.query_heads,
            head_dim: config.head_dim,
            query_bf16_f32: query,
            key_value_bf16_f32: key_value,
        })
    }
}

fn validate_bf16(values: &[f32]) -> Result<()> {
    if values.iter().any(|value| {
        !value.is_finite()
            || cast_bf16_rne_reference(*value)
                .map_or(true, |rounded| rounded.to_bits() != value.to_bits())
    }) {
        return Err(invalid(
            "DeepSeek RoPE requires finite exact-BF16 input values",
        ));
    }
    Ok(())
}

fn rotate_query_tail(
    query: &mut [f32],
    heads: usize,
    head_dim: usize,
    rope_dim: usize,
    theta: f32,
    position: u64,
) -> Result<()> {
    let expected = heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid("DeepSeek RoPE query size overflow"))?;
    if query.len() != expected {
        return Err(invalid("DeepSeek RoPE query layout is invalid"));
    }
    for head in 0..heads {
        let start = head
            .checked_mul(head_dim)
            .ok_or_else(|| invalid("DeepSeek RoPE head offset overflow"))?;
        rotate_slice_tail(
            &mut query[start..start + head_dim],
            head_dim,
            rope_dim,
            theta,
            position,
        )?;
    }
    Ok(())
}

fn rotate_vector_tail(
    vector: &mut [f32],
    head_dim: usize,
    rope_dim: usize,
    theta: f32,
    position: u64,
) -> Result<()> {
    if vector.len() != head_dim {
        return Err(invalid("DeepSeek RoPE KV layout is invalid"));
    }
    rotate_slice_tail(vector, head_dim, rope_dim, theta, position)
}

fn rotate_slice_tail(
    vector: &mut [f32],
    head_dim: usize,
    rope_dim: usize,
    theta: f32,
    position: u64,
) -> Result<()> {
    if vector.len() != head_dim
        || rope_dim == 0
        || rope_dim > head_dim
        || !rope_dim.is_multiple_of(2)
        || !theta.is_finite()
        || theta <= 1.0
        || position > PINNED_WINDOW_MAX_POSITION
    {
        return Err(invalid("DeepSeek RoPE rotation bounds are invalid"));
    }
    let start = head_dim - rope_dim;
    let position_f32 = position as f32;
    for pair in 0..rope_dim / 2 {
        let exponent = (2 * pair) as f32 / rope_dim as f32;
        let inverse_frequency = theta.powf(-exponent);
        let angle = position_f32 * inverse_frequency;
        if !angle.is_finite() {
            return Err(Error::new(
                "NUMERICAL_OVERFLOW",
                "DeepSeek RoPE angle is nonfinite",
            ));
        }
        let (sin, cos) = angle.sin_cos();
        let index = start + 2 * pair;
        let real = vector[index];
        let imaginary = vector[index + 1];
        vector[index] = cast_bf16_rne_reference(real * cos - imaginary * sin)?;
        vector[index + 1] = cast_bf16_rne_reference(real * sin + imaginary * cos)?;
    }
    Ok(())
}
