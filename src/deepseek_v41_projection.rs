//! Resident, bounded CPU reference for the pinned DeepSeek-V4.1 attention Q/KV
//! projection boundary. It stops before RoPE; the window-attention API consumes
//! post-RoPE inputs and must not be given this output directly.
//!
//! This adapts the MIT donor's `Attention.forward` ordering without copying its
//! implementation: FP8 `wq_a`, BF16 `q_norm`, FP8 `wq_b`, then FP8 `wkv` and BF16
//! `kv_norm`. The CPU matvec is an independent-format reference, not a bitwise
//! replica of the donor's CUDA/Torch GEMM reduction.

use crate::checkpoint::{self, NamedTensorSpec};
use crate::deepseek_v41::{
    cast_bf16_rne_reference, fp8_activation_quantize_reference, Fp8E4m3Matrix,
    MAX_FP8_REFERENCE_MACS, PINNED_EXPERT_INDEX_SHA256,
};
use crate::deepseek_v41_gate::PINNED_MODEL_REVISION;
use crate::moe::Stamp;
use crate::{invalid, Error, Result};
use safetensors::tensor::Dtype;
use std::collections::BTreeMap;
use std::path::Path;

pub const PINNED_ATTENTION_HIDDEN: usize = 5120;
pub const PINNED_ATTENTION_Q_RANK: usize = 1280;
pub const PINNED_ATTENTION_HEADS: usize = 64;
pub const PINNED_ATTENTION_HEAD_DIM: usize = 512;
pub const PINNED_ATTENTION_NORM_EPSILON: f32 = 1.0e-20;
const MAX_ATTENTION_PROJECTION_PAYLOAD_BYTES: u64 = 56 * 1024 * 1024;
const MAX_ATTENTION_PROJECTION_MACS: u64 = 56 * 1024 * 1024;
const MATRIX_CHUNK_ROWS: usize = 8192;

#[derive(Debug, Clone)]
pub struct AttentionProjectionTensorHashes {
    pub wq_a_weight: String,
    pub wq_a_scale: String,
    pub wq_b_weight: String,
    pub wq_b_scale: String,
    pub q_norm_weight: String,
    pub wkv_weight: String,
    pub wkv_scale: String,
    pub kv_norm_weight: String,
}

#[derive(Debug, Clone)]
pub struct AttentionProjectionTensorSpec {
    pub layer: u32,
    pub tensor_sha256: AttentionProjectionTensorHashes,
}

#[derive(Debug, Clone, Copy)]
pub struct PinnedAttentionProjectionSource<'a> {
    pub index_bytes: &'a [u8],
    pub header_bytes: &'a [u8],
    pub expected_header_sha256: &'a str,
    pub header_shard_name: &'a str,
    pub source_shard_bytes: u64,
    pub max_source_shard_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionProjectionLoadReceipt {
    pub model_revision: String,
    pub layer: u32,
    pub source_shard_name: String,
    pub index_sha256: String,
    pub header_sha256: String,
    pub shard_bytes: u64,
    pub header_bytes: u64,
    pub payload_bytes: u64,
    pub io_bytes: u64,
    pub tensor_sha256: AttentionProjectionTensorHashesReceipt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionProjectionTensorHashesReceipt {
    pub wq_a_weight: String,
    pub wq_a_scale: String,
    pub wq_b_weight: String,
    pub wq_b_scale: String,
    pub q_norm_weight: String,
    pub wkv_weight: String,
    pub wkv_scale: String,
    pub kv_norm_weight: String,
}

/// Owned resident projection planes and BF16 normalization vectors.
#[derive(Debug)]
pub struct DeepSeekV41AttentionProjectionWeights {
    pub wq_a: Fp8E4m3Matrix,
    pub q_norm_bf16_f32: Vec<f32>,
    pub wq_b: Fp8E4m3Matrix,
    pub wkv: Fp8E4m3Matrix,
    pub kv_norm_bf16_f32: Vec<f32>,
}

/// Identity-bearing Q/KV projections before the donor's RoPE application.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV41ProjectedQkv {
    stamp: Stamp,
    attempt: u64,
    query_heads: usize,
    head_dim: usize,
    query_pre_rope_bf16_f32: Vec<f32>,
    key_value_pre_rope_bf16_f32: Vec<f32>,
}

impl DeepSeekV41ProjectedQkv {
    pub fn stamp(&self) -> &Stamp {
        &self.stamp
    }

    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    pub fn query_heads(&self) -> usize {
        self.query_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Head-major flattened query in the model's pre-RoPE projection space.
    pub fn query_pre_rope_bf16_f32(&self) -> &[f32] {
        &self.query_pre_rope_bf16_f32
    }

    /// Single KV row in the model's pre-RoPE projection space.
    pub fn key_value_pre_rope_bf16_f32(&self) -> &[f32] {
        &self.key_value_pre_rope_bf16_f32
    }
}

/// Immutable resident weights for the pinned low-rank query and KV projections.
#[derive(Debug)]
pub struct DeepSeekV41AttentionProjection {
    layer: u32,
    query_heads: usize,
    head_dim: usize,
    hidden: usize,
    wq_a: Fp8E4m3Matrix,
    wq_b: Fp8E4m3Matrix,
    wkv: Fp8E4m3Matrix,
    q_norm_bf16_f32: Box<[f32]>,
    kv_norm_bf16_f32: Box<[f32]>,
    receipt: Option<AttentionProjectionLoadReceipt>,
}

impl DeepSeekV41AttentionProjection {
    /// Construct a bounded synthetic/reference layer. Production tensor loading
    /// should use `load_indexed`, which pins names, dimensions, index, shard and
    /// each payload digest before this immutable owner is returned.
    pub fn new(
        layer: u32,
        query_heads: usize,
        head_dim: usize,
        weights: DeepSeekV41AttentionProjectionWeights,
    ) -> Result<Self> {
        let hidden = usize::try_from(weights.wq_a.cols())
            .map_err(|_| invalid("attention hidden size overflow"))?;
        validate_projection_shapes(layer, query_heads, head_dim, &weights)?;
        Ok(Self {
            layer,
            query_heads,
            head_dim,
            hidden,
            wq_a: weights.wq_a,
            wq_b: weights.wq_b,
            wkv: weights.wkv,
            q_norm_bf16_f32: weights.q_norm_bf16_f32.into_boxed_slice(),
            kv_norm_bf16_f32: weights.kv_norm_bf16_f32.into_boxed_slice(),
            receipt: None,
        })
    }

    /// Authenticate and import only this layer's attention projection tensors.
    /// All eight tensors must be in the same admitted shard; no model-wide
    /// payload is copied or downloaded by this function.
    pub fn load_indexed(
        root: &Path,
        source: PinnedAttentionProjectionSource<'_>,
        spec: &AttentionProjectionTensorSpec,
    ) -> Result<Self> {
        if spec.layer >= 40
            || !crate::sha256::is_hex_digest(source.expected_header_sha256)
            || crate::sha256::hex(source.index_bytes) != PINNED_EXPERT_INDEX_SHA256
            || crate::sha256::hex(source.header_bytes) != source.expected_header_sha256
            || source.source_shard_bytes == 0
            || source.source_shard_bytes > source.max_source_shard_bytes
            || source.header_shard_name.is_empty()
        {
            return Err(invalid(
                "pinned DeepSeek attention source identity or bounds are invalid",
            ));
        }
        let selected = named_specs(spec)?;
        let names = selected
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect::<Vec<_>>();
        let plan = checkpoint::plan_indexed_safetensors_shards(
            source.index_bytes,
            PINNED_EXPERT_INDEX_SHA256,
            &names,
        )?;
        if plan.tensor_to_shard.len() != selected.len()
            || plan
                .tensor_to_shard
                .values()
                .any(|shard| shard != source.header_shard_name)
        {
            return Err(invalid(
                "pinned attention tensors must share the admitted layer shard",
            ));
        }
        let header = checkpoint::read_indexed_safetensors_shard_header(
            root,
            source.header_shard_name,
            source.max_source_shard_bytes,
        )?;
        if header.sha256 != source.expected_header_sha256
            || header.file_bytes != source.source_shard_bytes
            || header.header_bytes != source.header_bytes
        {
            return Err(invalid("pinned attention shard header or size differs"));
        }
        let imported = checkpoint::read_indexed_named_tensors(
            root,
            source.index_bytes,
            PINNED_EXPERT_INDEX_SHA256,
            &selected,
            MAX_ATTENTION_PROJECTION_PAYLOAD_BYTES,
        )?;
        if imported.plan != plan
            || imported.file_bytes != source.source_shard_bytes
            || imported.header_bytes != header.header_bytes.len() as u64
            || imported.tensors.len() != selected.len()
        {
            return Err(invalid(
                "indexed attention payload import differs from its preflight",
            ));
        }
        let payload_bytes = selected.iter().try_fold(0u64, |sum, tensor| {
            sum.checked_add(tensor.bytes)
                .ok_or_else(|| invalid("attention payload size overflow"))
        })?;
        let mut tensors = imported.tensors;
        let layer = spec.layer;
        let hidden = PINNED_ATTENTION_HIDDEN;
        let q_rank = PINNED_ATTENTION_Q_RANK;
        let query_width = PINNED_ATTENTION_HEADS * PINNED_ATTENTION_HEAD_DIM;
        let wq_a = matrix_from_import(
            &mut tensors,
            &format!("layers.{layer}.attn.wq_a"),
            q_rank,
            hidden,
        )?;
        let q_norm = bf16_vector_from_import(
            &mut tensors,
            &format!("layers.{layer}.attn.q_norm.weight"),
            q_rank,
        )?;
        let wq_b = matrix_from_import(
            &mut tensors,
            &format!("layers.{layer}.attn.wq_b"),
            query_width,
            q_rank,
        )?;
        let wkv = matrix_from_import(
            &mut tensors,
            &format!("layers.{layer}.attn.wkv"),
            PINNED_ATTENTION_HEAD_DIM,
            hidden,
        )?;
        let kv_norm = bf16_vector_from_import(
            &mut tensors,
            &format!("layers.{layer}.attn.kv_norm.weight"),
            PINNED_ATTENTION_HEAD_DIM,
        )?;
        if !tensors.is_empty() {
            return Err(invalid("attention import returned an unrequested tensor"));
        }
        let mut layer_owner = Self::new(
            layer,
            PINNED_ATTENTION_HEADS,
            PINNED_ATTENTION_HEAD_DIM,
            DeepSeekV41AttentionProjectionWeights {
                wq_a,
                q_norm_bf16_f32: q_norm,
                wq_b,
                wkv,
                kv_norm_bf16_f32: kv_norm,
            },
        )?;
        let final_header = checkpoint::read_indexed_safetensors_shard_header(
            root,
            source.header_shard_name,
            source.max_source_shard_bytes,
        )?;
        if final_header.file_bytes != header.file_bytes || final_header.sha256 != header.sha256 {
            return Err(Error::new(
                "CHECKPOINT_CHANGED",
                "DeepSeek attention shard changed during indexed payload import",
            ));
        }
        layer_owner.receipt = Some(AttentionProjectionLoadReceipt {
            model_revision: PINNED_MODEL_REVISION.to_owned(),
            layer,
            source_shard_name: source.header_shard_name.to_owned(),
            index_sha256: crate::sha256::hex(source.index_bytes),
            header_sha256: header.sha256,
            shard_bytes: imported.file_bytes,
            header_bytes: imported.header_bytes,
            payload_bytes,
            io_bytes: imported.io_bytes,
            tensor_sha256: hashes_receipt(&spec.tensor_sha256),
        });
        Ok(layer_owner)
    }

    pub fn receipt(&self) -> Option<&AttentionProjectionLoadReceipt> {
        self.receipt.as_ref()
    }

    /// Execute one token through the resident Q/KV projections. Input and outputs
    /// use exact finite BF16 values represented as FP32. RoPE, sink attention,
    /// compressed KV, output projection and residual combination remain separate.
    pub fn project(
        &self,
        stamp: &Stamp,
        attempt: u64,
        hidden_bf16_f32: &[f32],
    ) -> Result<DeepSeekV41ProjectedQkv> {
        if stamp.model_revision != PINNED_MODEL_REVISION
            || stamp.run_id.is_empty()
            || stamp.plan_id.is_empty()
            || stamp.layer != self.layer
            || stamp.layer >= 40
            || attempt == 0
            || hidden_bf16_f32.len() != self.hidden
        {
            return Err(invalid(
                "DeepSeek Q/KV projection identity or input shape differs",
            ));
        }
        validate_exact_bf16("attention hidden", hidden_bf16_f32)?;
        let hidden_quantized = fp8_activation_quantize_reference(hidden_bf16_f32)?;
        let q_low = bounded_matvec(&self.wq_a, &hidden_quantized)?
            .into_iter()
            .map(cast_bf16_rne_reference)
            .collect::<Result<Vec<_>>>()?;
        let q_low = rms_norm_bf16_reference(&q_low, &self.q_norm_bf16_f32)?;
        let q_low_quantized = fp8_activation_quantize_reference(&q_low)?;
        let query = bounded_matvec(&self.wq_b, &q_low_quantized)?
            .into_iter()
            .map(cast_bf16_rne_reference)
            .collect::<Result<Vec<_>>>()?;
        let kv = bounded_matvec(&self.wkv, &hidden_quantized)?
            .into_iter()
            .map(cast_bf16_rne_reference)
            .collect::<Result<Vec<_>>>()?;
        let kv = rms_norm_bf16_reference(&kv, &self.kv_norm_bf16_f32)?;
        Ok(DeepSeekV41ProjectedQkv {
            stamp: stamp.clone(),
            attempt,
            query_heads: self.query_heads,
            head_dim: self.head_dim,
            query_pre_rope_bf16_f32: query,
            key_value_pre_rope_bf16_f32: kv,
        })
    }
}

fn validate_projection_shapes(
    layer: u32,
    query_heads: usize,
    head_dim: usize,
    weights: &DeepSeekV41AttentionProjectionWeights,
) -> Result<()> {
    let wq_a = &weights.wq_a;
    let wq_b = &weights.wq_b;
    let wkv = &weights.wkv;
    let q_norm = &weights.q_norm_bf16_f32;
    let kv_norm = &weights.kv_norm_bf16_f32;
    let hidden =
        usize::try_from(wq_a.cols()).map_err(|_| invalid("attention hidden size overflow"))?;
    let q_rank = usize::try_from(wq_a.rows()).map_err(|_| invalid("attention Q rank overflow"))?;
    let query_width = query_heads.checked_mul(head_dim);
    let work = wq_a
        .rows()
        .checked_mul(wq_a.cols())
        .and_then(|value| value.checked_add(wq_b.rows().checked_mul(wq_b.cols())?))
        .and_then(|value| value.checked_add(wkv.rows().checked_mul(wkv.cols())?));
    if layer >= 40
        || query_heads == 0
        || query_heads > PINNED_ATTENTION_HEADS
        || head_dim == 0
        || head_dim > PINNED_ATTENTION_HEAD_DIM
        || hidden == 0
        || hidden > PINNED_ATTENTION_HIDDEN
        || q_rank == 0
        || q_rank > PINNED_ATTENTION_Q_RANK
        || [hidden, q_rank, head_dim]
            .iter()
            .any(|value| !value.is_multiple_of(32))
        || wq_a.rows() != q_rank as u64
        || wq_a.cols() != hidden as u64
        || wq_b.rows() != query_width.unwrap_or(0) as u64
        || wq_b.cols() != q_rank as u64
        || wkv.rows() != head_dim as u64
        || wkv.cols() != hidden as u64
        || q_norm.len() != q_rank
        || kv_norm.len() != head_dim
        || q_norm.iter().chain(kv_norm).any(|value| {
            !value.is_finite()
                || cast_bf16_rne_reference(*value)
                    .map_or(true, |rounded| rounded.to_bits() != value.to_bits())
        })
        || work.is_none_or(|value| value > MAX_ATTENTION_PROJECTION_MACS)
    {
        return Err(invalid(
            "DeepSeek attention projection dimensions, norms or CPU work exceed the pinned contract",
        ));
    }
    Ok(())
}

fn bounded_matvec(
    matrix: &Fp8E4m3Matrix,
    input: &crate::deepseek_v41::Fp8ActivationQuantized,
) -> Result<Vec<f32>> {
    bounded_matvec_with_limit(matrix, input, MAX_FP8_REFERENCE_MACS)
}

fn bounded_matvec_with_limit(
    matrix: &Fp8E4m3Matrix,
    input: &crate::deepseek_v41::Fp8ActivationQuantized,
    max_macs: u64,
) -> Result<Vec<f32>> {
    let rows =
        usize::try_from(matrix.rows()).map_err(|_| invalid("FP8 matrix row count overflow"))?;
    let cols =
        usize::try_from(matrix.cols()).map_err(|_| invalid("FP8 matrix column count overflow"))?;
    let blocks = cols / 32;
    let (codes, scales) = matrix.compressed_planes();
    let rows_per_chunk = MATRIX_CHUNK_ROWS.min((max_macs as usize / cols) / 32 * 32);
    if rows_per_chunk == 0 {
        return Err(invalid("FP8 projection cannot fit one bounded row chunk"));
    }
    let mut output = Vec::new();
    output.try_reserve_exact(rows).map_err(|_| {
        Error::new(
            "ALLOCATION_FAILED",
            "DeepSeek projection output allocation failed",
        )
    })?;
    for row_start in (0..rows).step_by(rows_per_chunk) {
        let row_end = (row_start + rows_per_chunk).min(rows);
        let code_start = row_start
            .checked_mul(cols)
            .ok_or_else(|| invalid("FP8 projection code offset overflow"))?;
        let code_end = row_end
            .checked_mul(cols)
            .ok_or_else(|| invalid("FP8 projection code end overflow"))?;
        let scale_start = row_start
            .checked_mul(blocks)
            .ok_or_else(|| invalid("FP8 projection scale offset overflow"))?;
        let scale_end = row_end
            .checked_mul(blocks)
            .ok_or_else(|| invalid("FP8 projection scale end overflow"))?;
        let chunk = Fp8E4m3Matrix::new(
            (row_end - row_start) as u64,
            cols as u64,
            codes[code_start..code_end].to_vec(),
            scales[scale_start..scale_end].to_vec(),
        )?;
        output.extend(chunk.matvec_reference(input)?);
    }
    if output.len() != rows || output.iter().any(|value| !value.is_finite()) {
        return Err(Error::new(
            "NUMERICAL_OVERFLOW",
            "DeepSeek projection output is incomplete or nonfinite",
        ));
    }
    Ok(output)
}

fn rms_norm_bf16_reference(input: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
    if input.is_empty()
        || input.len() != weight.len()
        || input.len() > PINNED_ATTENTION_Q_RANK
        || !input.len().is_multiple_of(32)
        || weight.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(
            "DeepSeek attention RMSNorm shape or weight is invalid",
        ));
    }
    let mut sum_squares = 0.0f32;
    for &value in input {
        if !value.is_finite() || cast_bf16_rne_reference(value)?.to_bits() != value.to_bits() {
            return Err(invalid(
                "DeepSeek attention RMSNorm input must be exact finite BF16",
            ));
        }
        sum_squares += value * value;
    }
    let variance = sum_squares / input.len() as f32 + PINNED_ATTENTION_NORM_EPSILON;
    if !variance.is_finite() || variance <= 0.0 {
        return Err(Error::new(
            "NUMERICAL_OVERFLOW",
            "DeepSeek attention RMSNorm variance is invalid",
        ));
    }
    let inverse_rms = variance.sqrt().recip();
    if !inverse_rms.is_finite() {
        return Err(Error::new(
            "NUMERICAL_OVERFLOW",
            "DeepSeek attention RMSNorm reciprocal RMS is invalid",
        ));
    }
    input
        .iter()
        .zip(weight)
        .map(|(&value, &weight)| cast_bf16_rne_reference((value * inverse_rms) * weight))
        .collect()
}

fn validate_exact_bf16(label: &str, values: &[f32]) -> Result<()> {
    if values.iter().any(|value| {
        !value.is_finite()
            || cast_bf16_rne_reference(*value)
                .map_or(true, |rounded| rounded.to_bits() != value.to_bits())
    }) {
        return Err(invalid(format!(
            "DeepSeek {label} must contain finite exact-BF16 values"
        )));
    }
    Ok(())
}

fn named_specs(spec: &AttentionProjectionTensorSpec) -> Result<Vec<NamedTensorSpec>> {
    let layer = spec.layer;
    if layer >= 40 {
        return Err(invalid(
            "DeepSeek attention layer index is outside the backbone",
        ));
    }
    let hashes = &spec.tensor_sha256;
    let entries = [
        (
            format!("layers.{layer}.attn.wq_a.weight"),
            Dtype::F8_E4M3,
            vec![1280, 5120],
            1280u64 * 5120,
            &hashes.wq_a_weight,
        ),
        (
            format!("layers.{layer}.attn.wq_a.scale"),
            Dtype::F8_E8M0,
            vec![40, 160],
            40u64 * 160,
            &hashes.wq_a_scale,
        ),
        (
            format!("layers.{layer}.attn.wq_b.weight"),
            Dtype::F8_E4M3,
            vec![32768, 1280],
            32768u64 * 1280,
            &hashes.wq_b_weight,
        ),
        (
            format!("layers.{layer}.attn.wq_b.scale"),
            Dtype::F8_E8M0,
            vec![1024, 40],
            1024u64 * 40,
            &hashes.wq_b_scale,
        ),
        (
            format!("layers.{layer}.attn.q_norm.weight"),
            Dtype::BF16,
            vec![1280],
            1280u64 * 2,
            &hashes.q_norm_weight,
        ),
        (
            format!("layers.{layer}.attn.wkv.weight"),
            Dtype::F8_E4M3,
            vec![512, 5120],
            512u64 * 5120,
            &hashes.wkv_weight,
        ),
        (
            format!("layers.{layer}.attn.wkv.scale"),
            Dtype::F8_E8M0,
            vec![16, 160],
            16u64 * 160,
            &hashes.wkv_scale,
        ),
        (
            format!("layers.{layer}.attn.kv_norm.weight"),
            Dtype::BF16,
            vec![512],
            512u64 * 2,
            &hashes.kv_norm_weight,
        ),
    ];
    if entries
        .iter()
        .any(|(_, _, _, _, digest)| !crate::sha256::is_hex_digest(digest))
    {
        return Err(invalid("DeepSeek attention tensor SHA-256 is malformed"));
    }
    Ok(entries
        .into_iter()
        .map(|(name, dtype, shape, bytes, sha256)| NamedTensorSpec {
            name,
            dtype,
            shape,
            bytes,
            sha256: sha256.clone(),
        })
        .collect())
}

fn matrix_from_import(
    tensors: &mut BTreeMap<String, checkpoint::NamedTensor>,
    prefix: &str,
    rows: usize,
    cols: usize,
) -> Result<Fp8E4m3Matrix> {
    let weight_name = format!("{prefix}.weight");
    let scale_name = format!("{prefix}.scale");
    let weight = tensors
        .remove(&weight_name)
        .ok_or_else(|| invalid("DeepSeek attention weight payload is missing"))?;
    let scale = tensors
        .remove(&scale_name)
        .ok_or_else(|| invalid("DeepSeek attention scale payload is missing"))?;
    if weight.dtype != Dtype::F8_E4M3
        || weight.shape != [rows, cols]
        || weight.bytes.len() != rows * cols
        || scale.dtype != Dtype::F8_E8M0
        || scale.shape != [rows / 32, cols / 32]
    {
        return Err(invalid(
            "DeepSeek attention FP8 tensor metadata differs from the pinned shape",
        ));
    }
    Fp8E4m3Matrix::new(
        rows as u64,
        cols as u64,
        weight.bytes,
        expand_scales(rows, cols, &scale.bytes)?,
    )
}

fn bf16_vector_from_import(
    tensors: &mut BTreeMap<String, checkpoint::NamedTensor>,
    name: &str,
    len: usize,
) -> Result<Vec<f32>> {
    let tensor = tensors
        .remove(name)
        .ok_or_else(|| invalid("DeepSeek attention norm payload is missing"))?;
    if tensor.dtype != Dtype::BF16 || tensor.shape != [len] || tensor.bytes.len() != len * 2 {
        return Err(invalid(
            "DeepSeek attention RMSNorm metadata differs from the pinned shape",
        ));
    }
    tensor
        .bytes
        .chunks_exact(2)
        .map(|pair| {
            let bits = u16::from_le_bytes([pair[0], pair[1]]);
            let value = f32::from_bits(u32::from(bits) << 16);
            if value.is_finite() {
                Ok(value)
            } else {
                Err(Error::new(
                    "NONFINITE_INPUT",
                    "DeepSeek attention norm contains NaN or infinity",
                ))
            }
        })
        .collect()
}

fn expand_scales(rows: usize, cols: usize, block_scales: &[u8]) -> Result<Vec<u8>> {
    if rows == 0
        || cols == 0
        || !rows.is_multiple_of(32)
        || !cols.is_multiple_of(32)
        || block_scales.len() != (rows / 32) * (cols / 32)
        || block_scales.contains(&255)
    {
        return Err(invalid(
            "DeepSeek attention E8M0 block-scale plane is invalid",
        ));
    }
    let blocks_per_row = cols / 32;
    let mut expanded = Vec::new();
    expanded
        .try_reserve_exact(rows * blocks_per_row)
        .map_err(|_| {
            Error::new(
                "ALLOCATION_FAILED",
                "DeepSeek scale expansion allocation failed",
            )
        })?;
    for block_row in 0..rows / 32 {
        let row = &block_scales[block_row * blocks_per_row..(block_row + 1) * blocks_per_row];
        for _ in 0..32 {
            expanded.extend_from_slice(row);
        }
    }
    Ok(expanded)
}

fn hashes_receipt(
    hashes: &AttentionProjectionTensorHashes,
) -> AttentionProjectionTensorHashesReceipt {
    AttentionProjectionTensorHashesReceipt {
        wq_a_weight: hashes.wq_a_weight.clone(),
        wq_a_scale: hashes.wq_a_scale.clone(),
        wq_b_weight: hashes.wq_b_weight.clone(),
        wq_b_scale: hashes.wq_b_scale.clone(),
        q_norm_weight: hashes.q_norm_weight.clone(),
        wkv_weight: hashes.wkv_weight.clone(),
        wkv_scale: hashes.wkv_scale.clone(),
        kv_norm_weight: hashes.kv_norm_weight.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_matrix(width: usize) -> Fp8E4m3Matrix {
        let mut codes = vec![0u8; width * width];
        for index in 0..width {
            codes[index * width + index] = 0x38; // E4M3FN 1.0
        }
        Fp8E4m3Matrix::new(
            width as u64,
            width as u64,
            codes,
            vec![127; width * width / 32],
        )
        .unwrap()
    }

    fn stamp(layer: u32) -> Stamp {
        Stamp {
            run_id: "projection-test".into(),
            model_revision: PINNED_MODEL_REVISION.into(),
            plan_id: "plan-a".into(),
            layer,
            batch: 4,
            epoch: 2,
        }
    }

    #[test]
    fn resident_identity_projection_returns_exact_bf16_pre_rope_qkv() {
        let width = 32;
        let projection = DeepSeekV41AttentionProjection::new(
            3,
            1,
            width,
            DeepSeekV41AttentionProjectionWeights {
                wq_a: identity_matrix(width),
                q_norm_bf16_f32: vec![1.0; width],
                wq_b: identity_matrix(width),
                wkv: identity_matrix(width),
                kv_norm_bf16_f32: vec![1.0; width],
            },
        )
        .unwrap();
        let mut hidden = vec![0.0; width];
        hidden[0] = 1.0;
        let output = projection.project(&stamp(3), 7, &hidden).unwrap();
        assert_eq!(output.stamp(), &stamp(3));
        assert_eq!(output.attempt(), 7);
        assert_eq!(output.query_heads(), 1);
        assert_eq!(output.head_dim(), width);
        assert_eq!(output.query_pre_rope_bf16_f32().len(), width);
        assert_eq!(output.key_value_pre_rope_bf16_f32().len(), width);
        assert!(output
            .query_pre_rope_bf16_f32()
            .iter()
            .all(|value| value.is_finite()
                && cast_bf16_rne_reference(*value).unwrap().to_bits() == value.to_bits()));
        assert!(output
            .key_value_pre_rope_bf16_f32()
            .iter()
            .all(|value| value.is_finite()
                && cast_bf16_rne_reference(*value).unwrap().to_bits() == value.to_bits()));
        assert_eq!(output.key_value_pre_rope_bf16_f32()[1..], [0.0; 31]);
    }

    #[test]
    fn batch_identity_is_preserved_and_wrong_layer_attempt_and_non_bf16_fail_closed() {
        let width = 32;
        let projection = DeepSeekV41AttentionProjection::new(
            3,
            1,
            width,
            DeepSeekV41AttentionProjectionWeights {
                wq_a: identity_matrix(width),
                q_norm_bf16_f32: vec![1.0; width],
                wq_b: identity_matrix(width),
                wkv: identity_matrix(width),
                kv_norm_bf16_f32: vec![1.0; width],
            },
        )
        .unwrap();
        let hidden = vec![1.0; width];
        let mut other = stamp(3);
        other.batch += 1;
        assert_eq!(
            projection.project(&other, 7, &hidden).unwrap().stamp(),
            &other
        );
        assert!(projection.project(&stamp(4), 7, &hidden).is_err());
        assert!(projection.project(&stamp(3), 0, &hidden).is_err());
        let mut not_bf16 = hidden;
        not_bf16[0] = 1.001;
        assert!(projection.project(&stamp(3), 7, &not_bf16).is_err());
    }

    #[test]
    fn identity_matrix_matches_independent_scalar_activation_oracle() {
        let width = 32;
        let matrix = identity_matrix(width);
        let mut input = vec![0.0; width];
        input[0] = 1.0;
        let quantized = fp8_activation_quantize_reference(&input).unwrap();
        let got = bounded_matvec(&matrix, &quantized).unwrap();
        let activation_scale = f32::from_bits(u32::from(quantized.scale_codes()[0]) << 23);
        let code = quantized.values()[0] & 0x7f;
        let exponent = (code >> 3) & 15;
        let mantissa = code & 7;
        let fp8 = if exponent == 0 {
            f64::from(mantissa) * 2.0f64.powi(-9)
        } else {
            f64::from(8 + mantissa) * 2.0f64.powi(i32::from(exponent) - 10)
        };
        let expected = (fp8 * f64::from(activation_scale)) as f32;
        assert_eq!(got[0].to_bits(), expected.to_bits());
        assert_eq!(&got[1..], &[0.0; 31]);
    }

    #[test]
    fn bounded_row_chunks_match_one_shot_reference() {
        let width = 32;
        let mut codes = vec![0u8; 64 * width];
        for row in 0..64 {
            codes[row * width + row % width] = 0x38;
        }
        let matrix = Fp8E4m3Matrix::new(64, width as u64, codes, vec![127; 64]).unwrap();
        let mut input = vec![0.0; width];
        input[0] = 1.0;
        input[7] = -0.5;
        let quantized = fp8_activation_quantize_reference(&input).unwrap();
        let one_shot = matrix.matvec_reference(&quantized).unwrap();
        let chunked = bounded_matvec_with_limit(&matrix, &quantized, 32 * width as u64).unwrap();
        assert_eq!(chunked, one_shot);
    }

    #[test]
    fn scale_expansion_matches_checkpoint_block_row_layout() {
        let expanded = expand_scales(64, 64, &[10, 11, 12, 13]).unwrap();
        assert_eq!(&expanded[..2], &[10, 11]);
        assert_eq!(&expanded[62..66], &[10, 11, 12, 13]);
        assert_eq!(&expanded[126..], &[12, 13]);
        assert!(expand_scales(63, 64, &[10, 11]).is_err());
        assert!(expand_scales(64, 64, &[10, 11, 12, 255]).is_err());
    }

    #[test]
    fn pinned_loader_rejects_untrusted_tensor_hashes_before_io() {
        let malformed = "0".repeat(63);
        let spec = AttentionProjectionTensorSpec {
            layer: 0,
            tensor_sha256: AttentionProjectionTensorHashes {
                wq_a_weight: malformed.clone(),
                wq_a_scale: malformed.clone(),
                wq_b_weight: malformed.clone(),
                wq_b_scale: malformed.clone(),
                q_norm_weight: malformed.clone(),
                wkv_weight: malformed.clone(),
                wkv_scale: malformed.clone(),
                kv_norm_weight: malformed,
            },
        };
        assert!(named_specs(&spec).is_err());
    }

    #[test]
    fn pinned_layer_zero_attention_names_and_shapes_match_local_index_and_header() {
        let index = include_bytes!(
            "../evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
        );
        let header = include_bytes!(
            "../evidence/continuation_a97_model_metadata/model-00003.header.314113f8.json"
        );
        let digest = "a".repeat(64);
        let spec = AttentionProjectionTensorSpec {
            layer: 0,
            tensor_sha256: AttentionProjectionTensorHashes {
                wq_a_weight: digest.clone(),
                wq_a_scale: digest.clone(),
                wq_b_weight: digest.clone(),
                wq_b_scale: digest.clone(),
                q_norm_weight: digest.clone(),
                wkv_weight: digest.clone(),
                wkv_scale: digest.clone(),
                kv_norm_weight: digest,
            },
        };
        let selected = named_specs(&spec).unwrap();
        let plan = checkpoint::plan_indexed_safetensors_shards(
            index,
            PINNED_EXPERT_INDEX_SHA256,
            &selected
                .iter()
                .map(|tensor| tensor.name.clone())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(plan.tensor_to_shard.len(), 8);
        assert!(plan
            .tensor_to_shard
            .values()
            .all(|shard| shard == "model-00003-of-00048.safetensors"));
        let parsed: serde_json::Value = serde_json::from_slice(header).unwrap();
        for tensor in selected {
            let metadata = &parsed[&tensor.name];
            assert_eq!(metadata["shape"], serde_json::json!(tensor.shape));
            assert_eq!(
                metadata["dtype"],
                serde_json::json!(match tensor.dtype {
                    Dtype::F8_E4M3 => "F8_E4M3",
                    Dtype::F8_E8M0 => "F8_E8M0",
                    Dtype::BF16 => "BF16",
                    _ => panic!("unexpected pinned attention tensor dtype"),
                })
            );
        }
    }
}
