//! Bounded CPU reference for the pinned DeepSeek-V4.1 pure-window attention
//! sublayer. This composes resident indexed output weights, inverse RoPE, and
//! the matching HC attention residual; it is not donor CUDA/Torch code.

use crate::checkpoint::{self, NamedTensorSpec};
use crate::deepseek_v41::{
    cast_bf16_rne_reference, fp8_activation_quantize_reference, Fp8E4m3Matrix,
    MAX_FP8_REFERENCE_MACS, PINNED_EXPERT_INDEX_SHA256,
};
use crate::deepseek_v41_attention::DeepSeekV41WindowCache;
use crate::deepseek_v41_gate::{
    hc::{DeepSeekV41HcMixes, HcProjectionBranch, HcProjectionLoadReceipt},
    DeepSeekV41ResidualStream, PinnedGateIndexSource, PINNED_MODEL_REVISION,
};
use crate::deepseek_v41_projection::{
    AttentionProjectionLoadReceipt, DeepSeekV41AttentionProjection,
};
use crate::deepseek_v41_rope::{DeepSeekV41UnrotatedAttentionOutput, DeepSeekV41WindowRopeConfig};
use crate::moe::Stamp;
use crate::{invalid, Error, Result};
use safetensors::tensor::Dtype;
use std::collections::BTreeMap;
use std::path::Path;

pub const PINNED_ATTENTION_OUTPUT_HIDDEN: usize = 5120;
pub const PINNED_ATTENTION_OUTPUT_HEADS: usize = 64;
pub const PINNED_ATTENTION_OUTPUT_HEAD_DIM: usize = 512;
pub const PINNED_ATTENTION_OUTPUT_GROUPS: usize = 8;
pub const PINNED_ATTENTION_OUTPUT_GROUP_RANK: usize = 1024;
pub const PINNED_ATTENTION_OUTPUT_GROUP_INPUT: usize = 4096;
const WO_A_ROWS: usize = PINNED_ATTENTION_OUTPUT_GROUPS * PINNED_ATTENTION_OUTPUT_GROUP_RANK;
const WO_B_INPUT: usize = PINNED_ATTENTION_OUTPUT_GROUPS * PINNED_ATTENTION_OUTPUT_GROUP_RANK;
const MAX_OUTPUT_PROJECTION_PAYLOAD_BYTES: u64 = 112 * 1024 * 1024;
const MAX_OUTPUT_PROJECTION_MACS: u64 = 80_000_000;
const OUTPUT_PROJECTION_MACS: u64 = (WO_A_ROWS as u64 * PINNED_ATTENTION_OUTPUT_GROUP_INPUT as u64)
    + (PINNED_ATTENTION_OUTPUT_HIDDEN as u64 * WO_B_INPUT as u64);
const MATRIX_CHUNK_ROWS: usize = 2048;

#[derive(Debug, Clone)]
pub struct AttentionOutputTensorHashes {
    pub attention_norm: String,
    pub sink: String,
    pub wo_a: String,
    pub wo_b: String,
    pub wo_b_scale: String,
}

#[derive(Debug, Clone)]
pub struct AttentionOutputTensorSpec {
    pub layer: u32,
    pub tensor_sha256: AttentionOutputTensorHashes,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionOutputLoadReceipt {
    pub model_revision: String,
    pub layer: u32,
    pub source_shard_name: String,
    pub index_sha256: String,
    pub header_sha256: String,
    pub shard_bytes: u64,
    pub header_bytes: u64,
    pub payload_bytes: u64,
    pub io_bytes: u64,
    pub tensor_sha256: AttentionOutputTensorHashesReceipt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionOutputTensorHashesReceipt {
    pub attention_norm: String,
    pub sink: String,
    pub wo_a: String,
    pub wo_b: String,
    pub wo_b_scale: String,
}

/// Identity-bearing BF16 result of the complete attention output projection.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV41AttentionSublayerOutput {
    stamp: Stamp,
    attempt: u64,
    output_bf16_f32: Vec<f32>,
}

impl DeepSeekV41AttentionSublayerOutput {
    fn new(stamp: Stamp, attempt: u64, output_bf16_f32: Vec<f32>) -> Result<Self> {
        validate_identity(&stamp, attempt)?;
        validate_exact_bf16("attention output", &output_bf16_f32)?;
        if output_bf16_f32.len() != PINNED_ATTENTION_OUTPUT_HIDDEN {
            return Err(invalid("DeepSeek attention output row width is not pinned"));
        }
        Ok(Self {
            stamp,
            attempt,
            output_bf16_f32,
        })
    }

    pub fn stamp(&self) -> &Stamp {
        &self.stamp
    }

    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    pub fn output_bf16_f32(&self) -> &[f32] {
        &self.output_bf16_f32
    }
}

/// One completed single-token attention sublayer and its newly composed HC
/// residual. The cache advances only when every operation in the sublayer passes.
#[derive(Debug, Clone)]
pub struct DeepSeekV41AttentionLayerOutput {
    attention: DeepSeekV41AttentionSublayerOutput,
    residual: DeepSeekV41ResidualStream,
    next_position: u64,
    source_shard_name: String,
    header_sha256: String,
}

impl DeepSeekV41AttentionLayerOutput {
    pub fn attention(&self) -> &DeepSeekV41AttentionSublayerOutput {
        &self.attention
    }

    pub fn residual(&self) -> &DeepSeekV41ResidualStream {
        &self.residual
    }

    pub fn next_position(&self) -> u64 {
        self.next_position
    }

    pub fn source_shard_name(&self) -> &str {
        &self.source_shard_name
    }

    pub fn header_sha256(&self) -> &str {
        &self.header_sha256
    }
}

/// Immutable resident attention output planes for the pinned pure-window path.
/// `wo_a` retains its original BF16 bytes; `wo_b` remains FP8 E4M3/E8M0.
#[derive(Debug)]
pub struct DeepSeekV41AttentionOutputProjection {
    layer: u32,
    attention_norm_bf16_f32: Box<[f32]>,
    sink_f32: Box<[f32]>,
    wo_a_bf16_le: Box<[u8]>,
    wo_b: Fp8E4m3Matrix,
    receipt: AttentionOutputLoadReceipt,
}

impl DeepSeekV41AttentionOutputProjection {
    /// Import only the pinned layer's attention norm, sink and `wo_a`/`wo_b`
    /// tensors from one indexed shard. Payload reads have a fixed resident cap;
    /// no full source shard is copied into memory.
    pub fn load_indexed(
        root: &Path,
        source: PinnedGateIndexSource<'_>,
        spec: &AttentionOutputTensorSpec,
    ) -> Result<Self> {
        validate_spec(spec)?;
        if source.max_source_shard_bytes == 0
            || !crate::sha256::is_hex_digest(source.expected_header_sha256)
            || crate::sha256::hex(source.index_bytes) != PINNED_EXPERT_INDEX_SHA256
        {
            return Err(invalid(
                "DeepSeek attention output source identity or shard ceiling is invalid",
            ));
        }
        let selected = tensor_specs(spec)?;
        let selected_names = selected
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect::<Vec<_>>();
        let plan = checkpoint::plan_indexed_safetensors_shards(
            source.index_bytes,
            PINNED_EXPERT_INDEX_SHA256,
            &selected_names,
        )?;
        if plan.tensor_to_shard.len() != selected.len() || plan.tensors_by_shard.len() != 1 {
            return Err(invalid(
                "DeepSeek attention output tensors must all resolve to one indexed shard",
            ));
        }
        let shard_name = plan
            .tensors_by_shard
            .keys()
            .next()
            .ok_or_else(|| invalid("DeepSeek attention output index selected no shard"))?;
        let shard_names = plan
            .tensors_by_shard
            .get(shard_name)
            .ok_or_else(|| invalid("DeepSeek attention output shard plan disappeared"))?;
        if shard_names.len() != selected.len()
            || shard_names
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                != selected_names.iter().collect()
        {
            return Err(invalid(
                "DeepSeek attention output index does not select exactly the requested tensors",
            ));
        }
        let header = checkpoint::read_indexed_safetensors_shard_header(
            root,
            shard_name,
            source.max_source_shard_bytes,
        )?;
        if header.sha256 != source.expected_header_sha256 {
            return Err(invalid(
                "DeepSeek attention output shard header digest differs from admission",
            ));
        }
        let imported = checkpoint::read_indexed_named_tensors(
            root,
            source.index_bytes,
            PINNED_EXPERT_INDEX_SHA256,
            &selected,
            MAX_OUTPUT_PROJECTION_PAYLOAD_BYTES,
        )?;
        if imported.plan != plan
            || imported.file_bytes != header.file_bytes
            || imported.header_bytes
                != u64::try_from(header.header_bytes.len())
                    .map_err(|_| invalid("DeepSeek attention output header length overflow"))?
            || imported.tensors.len() != selected.len()
        {
            return Err(invalid(
                "DeepSeek attention output import differs from its complete indexed plan",
            ));
        }
        let payload_bytes = selected.iter().try_fold(0u64, |sum, tensor| {
            sum.checked_add(tensor.bytes)
                .ok_or_else(|| invalid("DeepSeek attention output payload size overflow"))
        })?;
        if payload_bytes > MAX_OUTPUT_PROJECTION_PAYLOAD_BYTES {
            return Err(Error::new(
                "ARTIFACT_BUDGET",
                "DeepSeek attention output payload exceeds its resident bound",
            ));
        }

        let mut tensors = imported.tensors;
        let layer = spec.layer;
        let norm = take_bf16_vector(
            &mut tensors,
            &format!("layers.{layer}.attn_norm.weight"),
            PINNED_ATTENTION_OUTPUT_HIDDEN,
        )?;
        let sink = take_f32_vector(
            &mut tensors,
            &format!("layers.{layer}.attn.attn_sink"),
            PINNED_ATTENTION_OUTPUT_HEADS,
        )?;
        let wo_a_name = format!("layers.{layer}.attn.wo_a.weight");
        let wo_a = tensors
            .remove(&wo_a_name)
            .ok_or_else(|| invalid("DeepSeek wo_a payload is missing"))?;
        if wo_a.dtype != Dtype::BF16
            || wo_a.shape != [WO_A_ROWS, PINNED_ATTENTION_OUTPUT_GROUP_INPUT]
            || wo_a.bytes.len() != WO_A_ROWS * PINNED_ATTENTION_OUTPUT_GROUP_INPUT * 2
        {
            return Err(invalid(
                "DeepSeek wo_a BF16 tensor metadata differs from pin",
            ));
        }
        let wo_b = take_wo_b(&mut tensors, layer)?;
        if !tensors.is_empty() {
            return Err(invalid(
                "DeepSeek attention output import has extra tensors",
            ));
        }
        let final_header = checkpoint::read_indexed_safetensors_shard_header(
            root,
            shard_name,
            source.max_source_shard_bytes,
        )?;
        if final_header.file_bytes != header.file_bytes || final_header.sha256 != header.sha256 {
            return Err(Error::new(
                "CHECKPOINT_CHANGED",
                "DeepSeek attention output shard changed during indexed import",
            ));
        }
        let receipt = AttentionOutputLoadReceipt {
            model_revision: PINNED_MODEL_REVISION.to_owned(),
            layer,
            source_shard_name: shard_name.clone(),
            index_sha256: crate::sha256::hex(source.index_bytes),
            header_sha256: header.sha256,
            shard_bytes: imported.file_bytes,
            header_bytes: imported.header_bytes,
            payload_bytes,
            io_bytes: imported.io_bytes,
            tensor_sha256: AttentionOutputTensorHashesReceipt {
                attention_norm: spec.tensor_sha256.attention_norm.clone(),
                sink: spec.tensor_sha256.sink.clone(),
                wo_a: spec.tensor_sha256.wo_a.clone(),
                wo_b: spec.tensor_sha256.wo_b.clone(),
                wo_b_scale: spec.tensor_sha256.wo_b_scale.clone(),
            },
        };
        Ok(Self {
            layer,
            attention_norm_bf16_f32: norm.into_boxed_slice(),
            sink_f32: sink.into_boxed_slice(),
            wo_a_bf16_le: wo_a.bytes.into_boxed_slice(),
            wo_b,
            receipt,
        })
    }

    pub fn receipt(&self) -> &AttentionOutputLoadReceipt {
        &self.receipt
    }

    /// Execute one full pure-window attention sublayer from the HC residual.
    /// Cache changes are staged on a 256-KiB clone and committed only after
    /// output projection and the identity-matched HC post both succeed.
    pub fn forward_token(
        &self,
        residual: &DeepSeekV41ResidualStream,
        mixes: &DeepSeekV41HcMixes,
        qkv_projection: &DeepSeekV41AttentionProjection,
        rope: &DeepSeekV41WindowRopeConfig,
        cache: &mut DeepSeekV41WindowCache,
        position: u64,
    ) -> Result<DeepSeekV41AttentionLayerOutput> {
        validate_identity(residual.stamp(), residual.attempt())?;
        let stamp = residual.stamp();
        if mixes.branch() != HcProjectionBranch::Attention
            || mixes.stamp() != stamp
            || mixes.attempt() != residual.attempt()
            || self.layer != stamp.layer
            || rope.layer() != stamp.layer
            || rope.query_heads() != PINNED_ATTENTION_OUTPUT_HEADS
            || rope.head_dim() != PINNED_ATTENTION_OUTPUT_HEAD_DIM
            || position != cache.next_position()
        {
            return Err(invalid(
                "DeepSeek attention sublayer inputs do not share one pinned layer/token position",
            ));
        }
        let qkv_receipt = qkv_projection.receipt().ok_or_else(|| {
            invalid("DeepSeek attention QKV owner must come from indexed checkpoint loading")
        })?;
        require_same_shard(&self.receipt, qkv_receipt, mixes.receipt())?;

        let hidden = mixes.apply_pre(residual)?;
        let normalized = rms_norm_bf16(&hidden, &self.attention_norm_bf16_f32)?;
        let qkv = qkv_projection.project(stamp, residual.attempt(), &normalized)?;
        let post_rope = qkv.apply_window_rope(rope, position)?;
        let mut candidate_cache = cache.clone();
        let attention = post_rope.append_and_attend(&mut candidate_cache, &self.sink_f32)?;
        let unrotated = attention.apply_inverse_window_rope(rope)?;
        let output_row = self.project_output(&unrotated)?;
        let attention_sublayer =
            DeepSeekV41AttentionSublayerOutput::new(stamp.clone(), residual.attempt(), output_row)?;
        let next_residual = mixes.apply_attention_post(&attention_sublayer, residual)?;
        let next_position = candidate_cache.next_position();
        *cache = candidate_cache;
        Ok(DeepSeekV41AttentionLayerOutput {
            attention: attention_sublayer,
            residual: next_residual,
            next_position,
            source_shard_name: self.receipt.source_shard_name.clone(),
            header_sha256: self.receipt.header_sha256.clone(),
        })
    }

    fn project_output(&self, output: &DeepSeekV41UnrotatedAttentionOutput) -> Result<Vec<f32>> {
        if output.stamp().model_revision != PINNED_MODEL_REVISION
            || output.stamp().layer != self.layer
            || output.attempt() == 0
            || output.values_bf16_f32().len()
                != PINNED_ATTENTION_OUTPUT_HEADS * PINNED_ATTENTION_OUTPUT_HEAD_DIM
        {
            return Err(invalid(
                "DeepSeek inverse-RoPE output identity or shape differs from indexed weights",
            ));
        }
        if OUTPUT_PROJECTION_MACS > MAX_OUTPUT_PROJECTION_MACS {
            return Err(Error::new(
                "ARTIFACT_BUDGET",
                "DeepSeek attention output projections exceed the CPU reference work bound",
            ));
        }
        let quantized = fp8_activation_quantize_reference(output.values_bf16_f32())?;
        let wo_a_input = quantized.dequantized_bf16_f32()?;
        let mut wo_a_output = Vec::new();
        wo_a_output.try_reserve_exact(WO_B_INPUT).map_err(|_| {
            Error::new(
                "ALLOCATION_FAILED",
                "DeepSeek wo_a output allocation failed",
            )
        })?;
        for group in 0..PINNED_ATTENTION_OUTPUT_GROUPS {
            let input_start = group * PINNED_ATTENTION_OUTPUT_GROUP_INPUT;
            let input = &wo_a_input[input_start..input_start + PINNED_ATTENTION_OUTPUT_GROUP_INPUT];
            let first_row = group * PINNED_ATTENTION_OUTPUT_GROUP_RANK;
            for row_offset in 0..PINNED_ATTENTION_OUTPUT_GROUP_RANK {
                let row = first_row + row_offset;
                let weight_start = row * PINNED_ATTENTION_OUTPUT_GROUP_INPUT * 2;
                let mut sum = 0.0f32;
                for (column, activation) in input.iter().enumerate() {
                    let offset = weight_start + column * 2;
                    let bits = u16::from_le_bytes([
                        self.wo_a_bf16_le[offset],
                        self.wo_a_bf16_le[offset + 1],
                    ]);
                    let weight = f32::from_bits(u32::from(bits) << 16);
                    sum += weight * activation;
                    if !sum.is_finite() {
                        return Err(Error::new(
                            "NUMERICAL_OVERFLOW",
                            "DeepSeek wo_a FP32 accumulation became nonfinite",
                        ));
                    }
                }
                wo_a_output.push(cast_bf16_rne_reference(sum)?);
            }
        }
        if wo_a_output.len() != WO_B_INPUT {
            return Err(invalid("DeepSeek wo_a projection returned a partial row"));
        }
        let wo_b_input = fp8_activation_quantize_reference(&wo_a_output)?;
        matvec_chunked(&self.wo_b, &wo_b_input)?
            .into_iter()
            .map(cast_bf16_rne_reference)
            .collect()
    }
}

fn require_same_shard(
    output: &AttentionOutputLoadReceipt,
    qkv: &AttentionProjectionLoadReceipt,
    hc: &HcProjectionLoadReceipt,
) -> Result<()> {
    if output.model_revision != PINNED_MODEL_REVISION
        || output.index_sha256 != PINNED_EXPERT_INDEX_SHA256
        || output.layer != qkv.layer
        || output.layer != hc.layer
        || qkv.model_revision != PINNED_MODEL_REVISION
        || hc.model_revision != PINNED_MODEL_REVISION
        || output.source_shard_name != qkv.source_shard_name
        || output.source_shard_name != hc.source_shard_name
        || output.header_sha256 != qkv.header_sha256
        || output.header_sha256 != hc.header_sha256
        || output.index_sha256 != qkv.index_sha256
        || output.index_sha256 != hc.index_sha256
    {
        return Err(invalid(
            "DeepSeek attention, QKV and HC weights do not share the same indexed layer shard",
        ));
    }
    Ok(())
}

fn rms_norm_bf16(input: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
    if input.len() != PINNED_ATTENTION_OUTPUT_HIDDEN
        || weight.len() != PINNED_ATTENTION_OUTPUT_HIDDEN
        || weight.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(
            "DeepSeek attention RMSNorm vectors differ from pin",
        ));
    }
    validate_exact_bf16("attention HC pre activation", input)?;
    let mut sum_squares = 0.0f32;
    for value in input {
        sum_squares += value * value;
        if !sum_squares.is_finite() {
            return Err(Error::new(
                "NUMERICAL_OVERFLOW",
                "DeepSeek attention RMSNorm square sum became nonfinite",
            ));
        }
    }
    let variance = sum_squares / input.len() as f32 + 1.0e-20f32;
    if !variance.is_finite() || variance <= 0.0 {
        return Err(Error::new(
            "NUMERICAL_OVERFLOW",
            "DeepSeek attention RMSNorm variance is invalid",
        ));
    }
    let inverse_rms = variance.sqrt().recip();
    input
        .iter()
        .zip(weight)
        .map(|(&value, &weight)| cast_bf16_rne_reference((value * inverse_rms) * weight))
        .collect()
}

fn matvec_chunked(
    matrix: &Fp8E4m3Matrix,
    input: &crate::deepseek_v41::Fp8ActivationQuantized,
) -> Result<Vec<f32>> {
    let rows = usize::try_from(matrix.rows()).map_err(|_| invalid("wo_b row count overflow"))?;
    let cols = usize::try_from(matrix.cols()).map_err(|_| invalid("wo_b column count overflow"))?;
    let blocks = cols / 32;
    let (codes, scales) = matrix.compressed_planes();
    if cols != WO_B_INPUT || rows != PINNED_ATTENTION_OUTPUT_HIDDEN {
        return Err(invalid("DeepSeek wo_b resident dimensions differ from pin"));
    }
    let mut output = Vec::new();
    output.try_reserve_exact(rows).map_err(|_| {
        Error::new(
            "ALLOCATION_FAILED",
            "DeepSeek wo_b output allocation failed",
        )
    })?;
    for row_start in (0..rows).step_by(MATRIX_CHUNK_ROWS) {
        let row_end = (row_start + MATRIX_CHUNK_ROWS).min(rows);
        let code_start = row_start * cols;
        let code_end = row_end * cols;
        let scale_start = row_start * blocks;
        let scale_end = row_end * blocks;
        let chunk = Fp8E4m3Matrix::new(
            (row_end - row_start) as u64,
            cols as u64,
            codes[code_start..code_end].to_vec(),
            scales[scale_start..scale_end].to_vec(),
        )?;
        if chunk.rows() * chunk.cols() > MAX_FP8_REFERENCE_MACS {
            return Err(invalid(
                "DeepSeek wo_b chunk exceeds CPU reference work bound",
            ));
        }
        output.extend(chunk.matvec_reference(input)?);
    }
    if output.len() != rows || output.iter().any(|value| !value.is_finite()) {
        return Err(Error::new(
            "NUMERICAL_OVERFLOW",
            "DeepSeek wo_b projection returned an incomplete or nonfinite row",
        ));
    }
    Ok(output)
}

fn take_wo_b(
    tensors: &mut BTreeMap<String, checkpoint::NamedTensor>,
    layer: u32,
) -> Result<Fp8E4m3Matrix> {
    let prefix = format!("layers.{layer}.attn.wo_b");
    let weight = tensors
        .remove(&format!("{prefix}.weight"))
        .ok_or_else(|| invalid("DeepSeek wo_b FP8 weight payload is missing"))?;
    let scale = tensors
        .remove(&format!("{prefix}.scale"))
        .ok_or_else(|| invalid("DeepSeek wo_b E8M0 scale payload is missing"))?;
    if weight.dtype != Dtype::F8_E4M3
        || weight.shape != [PINNED_ATTENTION_OUTPUT_HIDDEN, WO_B_INPUT]
        || weight.bytes.len() != PINNED_ATTENTION_OUTPUT_HIDDEN * WO_B_INPUT
        || scale.dtype != Dtype::F8_E8M0
        || scale.shape != [PINNED_ATTENTION_OUTPUT_HIDDEN / 32, WO_B_INPUT / 32]
        || scale.bytes.len() != (PINNED_ATTENTION_OUTPUT_HIDDEN / 32) * (WO_B_INPUT / 32)
        || scale.bytes.contains(&255)
    {
        return Err(invalid(
            "DeepSeek wo_b FP8/E8M0 metadata differs from pinned shape",
        ));
    }
    let expanded = expand_block_scales(PINNED_ATTENTION_OUTPUT_HIDDEN, WO_B_INPUT, &scale.bytes)?;
    Fp8E4m3Matrix::new(
        PINNED_ATTENTION_OUTPUT_HIDDEN as u64,
        WO_B_INPUT as u64,
        weight.bytes,
        expanded,
    )
}

fn expand_block_scales(rows: usize, cols: usize, scales: &[u8]) -> Result<Vec<u8>> {
    if !rows.is_multiple_of(32)
        || !cols.is_multiple_of(32)
        || scales.len() != (rows / 32) * (cols / 32)
    {
        return Err(invalid(
            "DeepSeek wo_b block-scale plane dimensions are invalid",
        ));
    }
    let blocks_per_row = cols / 32;
    let mut expanded = Vec::new();
    expanded
        .try_reserve_exact(rows * blocks_per_row)
        .map_err(|_| Error::new("ALLOCATION_FAILED", "wo_b scales allocation failed"))?;
    for block_row in 0..rows / 32 {
        let row = &scales[block_row * blocks_per_row..(block_row + 1) * blocks_per_row];
        for _ in 0..32 {
            expanded.extend_from_slice(row);
        }
    }
    Ok(expanded)
}

fn take_bf16_vector(
    tensors: &mut BTreeMap<String, checkpoint::NamedTensor>,
    name: &str,
    len: usize,
) -> Result<Vec<f32>> {
    let tensor = tensors
        .remove(name)
        .ok_or_else(|| invalid(format!("DeepSeek tensor {name} is missing")))?;
    if tensor.dtype != Dtype::BF16 || tensor.shape != [len] || tensor.bytes.len() != len * 2 {
        return Err(invalid(format!(
            "DeepSeek BF16 tensor {name} differs from pin"
        )));
    }
    tensor
        .bytes
        .chunks_exact(2)
        .map(|pair| {
            let value = f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16);
            if value.is_finite() {
                Ok(value)
            } else {
                Err(Error::new(
                    "NONFINITE_INPUT",
                    format!("DeepSeek BF16 tensor {name} contains NaN or infinity"),
                ))
            }
        })
        .collect()
}

fn take_f32_vector(
    tensors: &mut BTreeMap<String, checkpoint::NamedTensor>,
    name: &str,
    len: usize,
) -> Result<Vec<f32>> {
    let tensor = tensors
        .remove(name)
        .ok_or_else(|| invalid(format!("DeepSeek tensor {name} is missing")))?;
    if tensor.dtype != Dtype::F32 || tensor.shape != [len] || tensor.bytes.len() != len * 4 {
        return Err(invalid(format!(
            "DeepSeek F32 tensor {name} differs from pin"
        )));
    }
    tensor
        .bytes
        .chunks_exact(4)
        .map(|bytes| {
            let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            if value.is_finite() {
                Ok(value)
            } else {
                Err(Error::new(
                    "NONFINITE_INPUT",
                    format!("DeepSeek F32 tensor {name} contains NaN or infinity"),
                ))
            }
        })
        .collect()
}

fn tensor_specs(spec: &AttentionOutputTensorSpec) -> Result<Vec<NamedTensorSpec>> {
    let layer = spec.layer;
    if layer >= 40 {
        return Err(invalid(
            "DeepSeek attention output layer is outside backbone",
        ));
    }
    let hashes = &spec.tensor_sha256;
    let entries = [
        (
            format!("layers.{layer}.attn_norm.weight"),
            Dtype::BF16,
            vec![PINNED_ATTENTION_OUTPUT_HIDDEN],
            (PINNED_ATTENTION_OUTPUT_HIDDEN * 2) as u64,
            &hashes.attention_norm,
        ),
        (
            format!("layers.{layer}.attn.attn_sink"),
            Dtype::F32,
            vec![PINNED_ATTENTION_OUTPUT_HEADS],
            (PINNED_ATTENTION_OUTPUT_HEADS * 4) as u64,
            &hashes.sink,
        ),
        (
            format!("layers.{layer}.attn.wo_a.weight"),
            Dtype::BF16,
            vec![WO_A_ROWS, PINNED_ATTENTION_OUTPUT_GROUP_INPUT],
            (WO_A_ROWS * PINNED_ATTENTION_OUTPUT_GROUP_INPUT * 2) as u64,
            &hashes.wo_a,
        ),
        (
            format!("layers.{layer}.attn.wo_b.weight"),
            Dtype::F8_E4M3,
            vec![PINNED_ATTENTION_OUTPUT_HIDDEN, WO_B_INPUT],
            (PINNED_ATTENTION_OUTPUT_HIDDEN * WO_B_INPUT) as u64,
            &hashes.wo_b,
        ),
        (
            format!("layers.{layer}.attn.wo_b.scale"),
            Dtype::F8_E8M0,
            vec![PINNED_ATTENTION_OUTPUT_HIDDEN / 32, WO_B_INPUT / 32],
            ((PINNED_ATTENTION_OUTPUT_HIDDEN / 32) * (WO_B_INPUT / 32)) as u64,
            &hashes.wo_b_scale,
        ),
    ];
    if entries
        .iter()
        .any(|(_, _, _, _, digest)| !crate::sha256::is_hex_digest(digest))
    {
        return Err(invalid("DeepSeek attention output hashes are malformed"));
    }
    let selected = entries
        .into_iter()
        .map(|(name, dtype, shape, bytes, sha256)| NamedTensorSpec {
            name,
            dtype,
            shape,
            bytes,
            sha256: sha256.clone(),
        })
        .collect::<Vec<_>>();
    let payload = selected.iter().try_fold(0u64, |sum, tensor| {
        sum.checked_add(tensor.bytes)
            .ok_or_else(|| invalid("DeepSeek attention output payload length overflow"))
    })?;
    if payload > MAX_OUTPUT_PROJECTION_PAYLOAD_BYTES {
        return Err(Error::new(
            "ARTIFACT_BUDGET",
            "DeepSeek attention output payload exceeds its resident bound",
        ));
    }
    Ok(selected)
}

fn validate_spec(spec: &AttentionOutputTensorSpec) -> Result<()> {
    let _ = tensor_specs(spec)?;
    Ok(())
}

fn validate_identity(stamp: &Stamp, attempt: u64) -> Result<()> {
    if stamp.model_revision != PINNED_MODEL_REVISION
        || stamp.run_id.trim().is_empty()
        || stamp.plan_id.trim().is_empty()
        || stamp.layer >= 40
        || attempt == 0
    {
        return Err(invalid(
            "DeepSeek attention output requires pinned run/model/plan/layer identity and attempt",
        ));
    }
    Ok(())
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
