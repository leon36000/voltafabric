//! Bounded CPU adaptation of the pinned DeepSeek-V4.1 Hyper-Connections path.
//!
//! The source formula and tensor names are pinned to the MIT DeepSeek-V4.1-Flash
//! revision used by the parent module. This implementation is a scalar CPU
//! reference; it is not copied donor code and does not claim TileLang/CUDA parity.

use super::{
    checkpoint, invalid, validate_hc_identity, DeepSeekV41HcPostCoefficients,
    DeepSeekV41MoeTokenOutput, DeepSeekV41ResidualStream, PinnedGateIndexSource,
    PINNED_MODEL_REVISION,
};
use crate::{deepseek_v41, Error, Result};
use safetensors::tensor::Dtype;
use std::{collections::BTreeSet, path::Path};

pub const HC_MULTIPLICITY: usize = deepseek_v41::PINNED_HC_MULT;
pub const HC_MIX_COUNT: usize = (2 + HC_MULTIPLICITY) * HC_MULTIPLICITY;
pub const HC_INPUT_WIDTH: usize = HC_MULTIPLICITY * super::HIDDEN_SIZE;
pub const HC_PROJECTION_WEIGHT_BYTES: u64 =
    (HC_MIX_COUNT * HC_INPUT_WIDTH * std::mem::size_of::<f32>()) as u64;
pub const HC_PROJECTION_BASE_BYTES: u64 = (HC_MIX_COUNT * std::mem::size_of::<f32>()) as u64;
pub const HC_PROJECTION_SCALE_BYTES: u64 = (3 * std::mem::size_of::<f32>()) as u64;
pub const HC_PROJECTION_PAYLOAD_BYTES: u64 =
    HC_PROJECTION_WEIGHT_BYTES + HC_PROJECTION_BASE_BYTES + HC_PROJECTION_SCALE_BYTES;
pub const PINNED_HC_NORMALIZATION_EPSILON: f32 = 1.0e-20;
pub const PINNED_HC_SINKHORN_EPSILON: f32 = 1.0e-6;
pub const PINNED_HC_SINKHORN_ITERATIONS: usize = 20;
const MAX_SINKHORN_ITERATIONS: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HcProjectionBranch {
    Attention,
    Ffn,
}

impl HcProjectionBranch {
    fn tensor_prefix(self) -> &'static str {
        match self {
            Self::Attention => "hc_attn",
            Self::Ffn => "hc_ffn",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HcProjectionTensorHashes {
    pub weight: String,
    pub base: String,
    pub scale: String,
}

/// Runtime values that are explicit because the checkpoint header does not
/// authenticate model configuration. The pinned donor defaults are exported
/// above; production callers must pass values admitted from their config.
#[derive(Debug, Clone)]
pub struct HcProjectionSpec {
    pub model_revision: String,
    pub layer: u32,
    pub branch: HcProjectionBranch,
    pub normalization_epsilon: f32,
    pub sinkhorn_epsilon: f32,
    pub sinkhorn_iterations: usize,
    pub tensor_sha256: HcProjectionTensorHashes,
    pub max_payload_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HcProjectionLoadReceipt {
    pub model_revision: String,
    pub layer: u32,
    pub branch: HcProjectionBranch,
    pub source_shard_name: String,
    pub index_sha256: String,
    pub header_sha256: String,
    pub shard_bytes: u64,
    pub header_bytes: u64,
    pub tensor_bytes: u64,
    pub io_bytes: u64,
    pub tensor_sha256: HcProjectionTensorHashes,
}

/// Resident F32 projection for one single-token attention or FFN HC operation.
/// Its bounded payload is about 1.88 MiB; the source shard itself is never
/// loaded into memory.
#[derive(Debug)]
pub struct DeepSeekV41HcProjection {
    layer: u32,
    branch: HcProjectionBranch,
    weight: Box<[f32]>,
    base: Box<[f32]>,
    scale: Box<[f32]>,
    normalization_epsilon: f32,
    sinkhorn_epsilon: f32,
    sinkhorn_iterations: usize,
    receipt: HcProjectionLoadReceipt,
}

/// The identity-bound outputs of one HC projection and split/Sinkhorn stage.
#[derive(Debug, Clone)]
pub struct DeepSeekV41HcMixes {
    stamp: crate::moe::Stamp,
    attempt: u64,
    branch: HcProjectionBranch,
    pre_mix: Box<[f32]>,
    post_coefficients: DeepSeekV41HcPostCoefficients,
}

impl DeepSeekV41HcMixes {
    pub fn stamp(&self) -> &crate::moe::Stamp {
        &self.stamp
    }

    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    pub fn branch(&self) -> HcProjectionBranch {
        self.branch
    }

    pub fn pre_mix(&self) -> &[f32] {
        &self.pre_mix
    }

    pub fn post_coefficients(&self) -> &DeepSeekV41HcPostCoefficients {
        &self.post_coefficients
    }

    pub fn post(&self) -> &[f32] {
        self.post_coefficients.post()
    }

    pub fn comb(&self) -> &[Vec<f32>] {
        self.post_coefficients.comb()
    }

    /// Apply this HC pre-vector to the exact residual stream that produced it.
    /// A stale layer, token, epoch, or attempt is rejected before any output is
    /// constructed.
    pub fn apply_pre(&self, residual: &DeepSeekV41ResidualStream) -> Result<Vec<f32>> {
        if residual.stamp() != &self.stamp || residual.attempt() != self.attempt {
            return Err(invalid(
                "DeepSeek HC pre mix and residual stream have different token identity",
            ));
        }
        let flattened = residual
            .copies_bf16_f32()
            .iter()
            .flat_map(|copy| copy.iter().copied())
            .collect::<Vec<_>>();
        deepseek_v41::deepseek_hc_pre_reference(&flattened, &self.pre_mix, super::HIDDEN_SIZE)
    }

    /// Apply this exact FFN branch's post/comb mixes to the completed routed-plus-
    /// shared FFN result. Keeping this operation on the projection receipt prevents
    /// callers from pairing post coefficients from another HC pass or sublayer.
    pub fn apply_ffn_post(
        &self,
        sublayer: &DeepSeekV41MoeTokenOutput,
        residual: &DeepSeekV41ResidualStream,
    ) -> Result<DeepSeekV41ResidualStream> {
        if self.branch != HcProjectionBranch::Ffn
            || self.post_coefficients.branch() != HcProjectionBranch::Ffn
        {
            return Err(invalid(
                "DeepSeek HC projection is not the FFN branch required by the routed MoE post",
            ));
        }
        super::deepseek_hc_post_reference(sublayer, residual, &self.post_coefficients)
    }
}

impl DeepSeekV41HcProjection {
    /// Resolve exactly the requested projection/base/scale tensors from one
    /// indexed Safetensors shard. The index and selected tensor digests are
    /// checked, the full source shard remains off-heap, and no result is exposed
    /// until the second header check succeeds.
    pub fn load_indexed(
        root: &Path,
        source: PinnedGateIndexSource<'_>,
        spec: &HcProjectionSpec,
    ) -> Result<Self> {
        validate_spec(spec)?;
        if source.max_source_shard_bytes == 0
            || !crate::sha256::is_hex_digest(source.expected_header_sha256)
        {
            return Err(invalid(
                "DeepSeek HC source needs a positive shard ceiling and valid header digest",
            ));
        }

        let selected = tensor_specs(spec);
        let selected_names = selected
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect::<Vec<_>>();
        let planned = checkpoint::plan_indexed_safetensors_shards(
            source.index_bytes,
            deepseek_v41::PINNED_EXPERT_INDEX_SHA256,
            &selected_names,
        )?;
        if planned.tensors_by_shard.len() != 1 || planned.tensor_to_shard.len() != selected.len() {
            return Err(invalid(
                "DeepSeek HC projection tensors must resolve completely to one source shard",
            ));
        }
        let (shard_name, shard_tensors) = planned
            .tensors_by_shard
            .iter()
            .next()
            .ok_or_else(|| invalid("pinned DeepSeek HC index selected no shard"))?;
        let expected_names = selected_names.iter().collect::<BTreeSet<_>>();
        let actual_names = shard_tensors.iter().collect::<BTreeSet<_>>();
        if shard_tensors.len() != selected.len() || actual_names != expected_names {
            return Err(invalid(
                "pinned DeepSeek HC index differs from the exact projection/base/scale tensors",
            ));
        }

        let header = checkpoint::read_indexed_safetensors_shard_header(
            root,
            shard_name,
            source.max_source_shard_bytes,
        )?;
        if header.sha256 != source.expected_header_sha256 {
            return Err(invalid(
                "pinned DeepSeek HC source shard header digest differs",
            ));
        }
        let imported = checkpoint::read_indexed_named_tensors(
            root,
            source.index_bytes,
            deepseek_v41::PINNED_EXPERT_INDEX_SHA256,
            &selected,
            spec.max_payload_bytes,
        )?;
        if imported.plan != planned
            || imported.file_bytes != header.file_bytes
            || imported.header_bytes
                != u64::try_from(header.header_bytes.len())
                    .map_err(|_| invalid("DeepSeek HC header length does not fit u64"))?
            || imported.tensors.len() != selected.len()
        {
            return Err(invalid(
                "indexed DeepSeek HC import differs from its complete preflight plan",
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
                "DeepSeek HC shard changed during indexed payload import",
            ));
        }

        Self::from_imported(imported, header.file_bytes, header.sha256, shard_name, spec)
    }

    fn from_imported(
        imported: checkpoint::IndexedNamedTensorImport,
        shard_bytes: u64,
        header_sha256: String,
        shard_name: &str,
        spec: &HcProjectionSpec,
    ) -> Result<Self> {
        validate_spec(spec)?;
        if imported.tensors.len() != 3 {
            return Err(invalid(
                "DeepSeek HC import does not contain exactly three selected tensors",
            ));
        }
        let tensor_bytes = imported
            .tensors
            .values()
            .map(|tensor| tensor.bytes.len() as u64)
            .sum();
        let prefix = format!("layers.{}.{}", spec.layer, spec.branch.tensor_prefix());
        let weight_name = format!("{prefix}_fn");
        let base_name = format!("{prefix}_base");
        let scale_name = format!("{prefix}_scale");
        let mut selected = imported.tensors;
        let weight = selected
            .remove(&weight_name)
            .ok_or_else(|| invalid("HC projection weight absent after complete import"))?;
        let base = selected
            .remove(&base_name)
            .ok_or_else(|| invalid("HC projection base absent after complete import"))?;
        let scale = selected
            .remove(&scale_name)
            .ok_or_else(|| invalid("HC projection scale absent after complete import"))?;
        let weights = decode_f32_exact(&weight.bytes, HC_MIX_COUNT * HC_INPUT_WIDTH)?;
        let bases = decode_f32_exact(&base.bytes, HC_MIX_COUNT)?;
        let scales = decode_f32_exact(&scale.bytes, 3)?;
        if weights.len() != HC_MIX_COUNT * HC_INPUT_WIDTH
            || bases.len() != HC_MIX_COUNT
            || scales.len() != 3
            || weights
                .iter()
                .chain(&bases)
                .chain(&scales)
                .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "DeepSeek HC projection payload has invalid shape or nonfinite values",
            ));
        }

        Ok(Self {
            layer: spec.layer,
            branch: spec.branch,
            weight: weights.into_boxed_slice(),
            base: bases.into_boxed_slice(),
            scale: scales.into_boxed_slice(),
            normalization_epsilon: spec.normalization_epsilon,
            sinkhorn_epsilon: spec.sinkhorn_epsilon,
            sinkhorn_iterations: spec.sinkhorn_iterations,
            receipt: HcProjectionLoadReceipt {
                model_revision: spec.model_revision.clone(),
                layer: spec.layer,
                branch: spec.branch,
                source_shard_name: shard_name.to_owned(),
                index_sha256: deepseek_v41::PINNED_EXPERT_INDEX_SHA256.to_owned(),
                header_sha256,
                shard_bytes,
                header_bytes: imported.header_bytes,
                tensor_bytes,
                io_bytes: imported.io_bytes,
                tensor_sha256: spec.tensor_sha256.clone(),
            },
        })
    }

    pub fn receipt(&self) -> &HcProjectionLoadReceipt {
        &self.receipt
    }

    /// Compute source-pinned HC mixes for one token. The operations are explicit
    /// FP32 scalar reductions. Reduction order may differ from the donor's
    /// PyTorch/TileLang path, so this is a bounded CPU reference, not a backend
    /// parity claim.
    pub fn project(&self, residual: &DeepSeekV41ResidualStream) -> Result<DeepSeekV41HcMixes> {
        let stamp = residual.stamp();
        validate_hc_identity(stamp, residual.attempt())?;
        if stamp.layer != self.layer || stamp.model_revision != self.receipt.model_revision {
            return Err(invalid(
                "DeepSeek HC projection and residual token identify different model layer",
            ));
        }
        let copies = residual.copies_bf16_f32();
        let mut square_sum = 0.0f32;
        for &value in copies.iter().flatten() {
            square_sum += value * value;
            if !square_sum.is_finite() {
                return Err(Error::new(
                    "NUMERICAL_OVERFLOW",
                    "DeepSeek HC normalization square sum became nonfinite",
                ));
            }
        }
        let mean_square = square_sum / HC_INPUT_WIDTH as f32;
        let mean_square = mean_square + self.normalization_epsilon;
        if !mean_square.is_finite() || mean_square <= 0.0 {
            return Err(Error::new(
                "NUMERICAL_OVERFLOW",
                "DeepSeek HC normalization denominator is invalid",
            ));
        }
        let inverse_rms = mean_square.sqrt().recip();
        if !inverse_rms.is_finite() {
            return Err(Error::new(
                "NUMERICAL_OVERFLOW",
                "DeepSeek HC inverse RMS became nonfinite",
            ));
        }

        let mut mixes = Vec::with_capacity(HC_MIX_COUNT);
        for row in 0..HC_MIX_COUNT {
            let start = row * HC_INPUT_WIDTH;
            let mut projection = 0.0f32;
            for (copy_index, copy) in copies.iter().enumerate() {
                for (hidden, &value) in copy.iter().enumerate() {
                    let input_index = copy_index * super::HIDDEN_SIZE + hidden;
                    projection += self.weight[start + input_index] * value;
                }
            }
            let mix = projection * inverse_rms;
            if !mix.is_finite() {
                return Err(Error::new(
                    "NUMERICAL_OVERFLOW",
                    "DeepSeek HC projection output became nonfinite",
                ));
            }
            mixes.push(mix);
        }

        split_sinkhorn(
            stamp,
            residual.attempt(),
            &mixes,
            &HcSplitParameters {
                branch: self.branch,
                base: &self.base,
                scale: &self.scale,
                sinkhorn_iterations: self.sinkhorn_iterations,
                epsilon: self.sinkhorn_epsilon,
            },
        )
    }
}

fn validate_spec(spec: &HcProjectionSpec) -> Result<()> {
    if spec.model_revision != PINNED_MODEL_REVISION || spec.layer >= 40 {
        return Err(invalid("unsupported DeepSeek HC model revision or layer"));
    }
    if !spec.normalization_epsilon.is_finite()
        || spec.normalization_epsilon <= 0.0
        || !spec.sinkhorn_epsilon.is_finite()
        || spec.sinkhorn_epsilon <= 0.0
        || spec.sinkhorn_iterations == 0
        || spec.sinkhorn_iterations > MAX_SINKHORN_ITERATIONS
    {
        return Err(invalid(
            "DeepSeek HC epsilon or Sinkhorn iteration count is invalid",
        ));
    }
    if spec.max_payload_bytes < HC_PROJECTION_PAYLOAD_BYTES
        || spec.max_payload_bytes > checkpoint::MAX_NAMED_TENSOR_PAYLOAD_BYTES
    {
        return Err(Error::new(
            "ARTIFACT_BUDGET",
            "DeepSeek HC projection payload ceiling is too small or exceeds the importer cap",
        ));
    }
    if [
        spec.tensor_sha256.weight.as_str(),
        spec.tensor_sha256.base.as_str(),
        spec.tensor_sha256.scale.as_str(),
    ]
    .iter()
    .any(|digest| !crate::sha256::is_hex_digest(digest))
    {
        return Err(invalid("DeepSeek HC tensor hashes must be SHA-256 digests"));
    }
    Ok(())
}

fn tensor_specs(spec: &HcProjectionSpec) -> [checkpoint::NamedTensorSpec; 3] {
    let prefix = format!("layers.{}.{}", spec.layer, spec.branch.tensor_prefix());
    [
        checkpoint::NamedTensorSpec {
            name: format!("{prefix}_fn"),
            dtype: Dtype::F32,
            shape: vec![HC_MIX_COUNT, HC_INPUT_WIDTH],
            bytes: HC_PROJECTION_WEIGHT_BYTES,
            sha256: spec.tensor_sha256.weight.clone(),
        },
        checkpoint::NamedTensorSpec {
            name: format!("{prefix}_base"),
            dtype: Dtype::F32,
            shape: vec![HC_MIX_COUNT],
            bytes: HC_PROJECTION_BASE_BYTES,
            sha256: spec.tensor_sha256.base.clone(),
        },
        checkpoint::NamedTensorSpec {
            name: format!("{prefix}_scale"),
            dtype: Dtype::F32,
            shape: vec![3],
            bytes: HC_PROJECTION_SCALE_BYTES,
            sha256: spec.tensor_sha256.scale.clone(),
        },
    ]
}

struct HcSplitParameters<'a> {
    branch: HcProjectionBranch,
    base: &'a [f32],
    scale: &'a [f32],
    sinkhorn_iterations: usize,
    epsilon: f32,
}

fn split_sinkhorn(
    stamp: &crate::moe::Stamp,
    attempt: u64,
    mixes: &[f32],
    parameters: &HcSplitParameters<'_>,
) -> Result<DeepSeekV41HcMixes> {
    validate_hc_identity(stamp, attempt)?;
    if mixes.len() != HC_MIX_COUNT
        || parameters.base.len() != HC_MIX_COUNT
        || parameters.scale.len() != 3
        || mixes
            .iter()
            .chain(parameters.base)
            .chain(parameters.scale)
            .any(|value| !value.is_finite())
        || parameters.sinkhorn_iterations == 0
        || parameters.sinkhorn_iterations > MAX_SINKHORN_ITERATIONS
        || !parameters.epsilon.is_finite()
        || parameters.epsilon <= 0.0
    {
        return Err(invalid("DeepSeek HC split/Sinkhorn inputs are invalid"));
    }

    let mut pre = Vec::with_capacity(HC_MULTIPLICITY);
    let mut post = Vec::with_capacity(HC_MULTIPLICITY);
    for copy in 0..HC_MULTIPLICITY {
        pre.push(
            sigmoid_f32(mixes[copy] * parameters.scale[0] + parameters.base[copy])
                + parameters.epsilon,
        );
        post.push(
            2.0 * sigmoid_f32(
                mixes[copy + HC_MULTIPLICITY] * parameters.scale[1]
                    + parameters.base[copy + HC_MULTIPLICITY],
            ),
        );
    }

    let mut comb = vec![0.0f32; HC_MULTIPLICITY * HC_MULTIPLICITY];
    for row in 0..HC_MULTIPLICITY {
        let row_start = row * HC_MULTIPLICITY;
        let mut row_max = f32::NEG_INFINITY;
        for column in 0..HC_MULTIPLICITY {
            let mix_index = 2 * HC_MULTIPLICITY + row_start + column;
            let value = mixes[mix_index] * parameters.scale[2] + parameters.base[mix_index];
            comb[row_start + column] = value;
            row_max = row_max.max(value);
        }
        let mut row_sum = 0.0f32;
        for column in 0..HC_MULTIPLICITY {
            let index = row_start + column;
            comb[index] = (comb[index] - row_max).exp();
            row_sum += comb[index];
        }
        if !row_sum.is_finite() || row_sum <= 0.0 {
            return Err(Error::new(
                "NUMERICAL_OVERFLOW",
                "DeepSeek HC row softmax denominator is invalid",
            ));
        }
        for column in 0..HC_MULTIPLICITY {
            let index = row_start + column;
            comb[index] = comb[index] / row_sum + parameters.epsilon;
        }
    }

    normalize_columns(&mut comb, parameters.epsilon)?;
    for _ in 1..parameters.sinkhorn_iterations {
        normalize_rows(&mut comb, parameters.epsilon)?;
        normalize_columns(&mut comb, parameters.epsilon)?;
    }
    if pre
        .iter()
        .chain(&post)
        .chain(&comb)
        .any(|value| !value.is_finite())
    {
        return Err(Error::new(
            "NUMERICAL_OVERFLOW",
            "DeepSeek HC split/Sinkhorn result became nonfinite",
        ));
    }
    let comb_rows = comb
        .chunks_exact(HC_MULTIPLICITY)
        .map(<[f32]>::to_vec)
        .collect::<Vec<_>>();
    let post_coefficients = DeepSeekV41HcPostCoefficients::new(
        stamp.clone(),
        attempt,
        parameters.branch,
        post,
        comb_rows,
    )?;
    Ok(DeepSeekV41HcMixes {
        stamp: stamp.clone(),
        attempt,
        branch: parameters.branch,
        pre_mix: pre.into_boxed_slice(),
        post_coefficients,
    })
}

fn normalize_rows(matrix: &mut [f32], epsilon: f32) -> Result<()> {
    for row in 0..HC_MULTIPLICITY {
        let start = row * HC_MULTIPLICITY;
        let sum = matrix[start..start + HC_MULTIPLICITY]
            .iter()
            .copied()
            .sum::<f32>();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(Error::new(
                "NUMERICAL_OVERFLOW",
                "DeepSeek HC row normalization denominator is invalid",
            ));
        }
        for value in &mut matrix[start..start + HC_MULTIPLICITY] {
            *value /= sum + epsilon;
        }
    }
    Ok(())
}

fn normalize_columns(matrix: &mut [f32], epsilon: f32) -> Result<()> {
    for column in 0..HC_MULTIPLICITY {
        let mut sum = 0.0f32;
        for row in 0..HC_MULTIPLICITY {
            sum += matrix[row * HC_MULTIPLICITY + column];
        }
        if !sum.is_finite() || sum <= 0.0 {
            return Err(Error::new(
                "NUMERICAL_OVERFLOW",
                "DeepSeek HC column normalization denominator is invalid",
            ));
        }
        for row in 0..HC_MULTIPLICITY {
            matrix[row * HC_MULTIPLICITY + column] /= sum + epsilon;
        }
    }
    Ok(())
}

fn sigmoid_f32(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn decode_f32_exact(bytes: &[u8], expected_values: usize) -> Result<Vec<f32>> {
    let expected_bytes = expected_values
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| invalid("DeepSeek HC F32 tensor byte count overflow"))?;
    if bytes.len() != expected_bytes {
        return Err(invalid(
            "DeepSeek HC F32 tensor has an unexpected byte length",
        ));
    }
    let mut values = Vec::with_capacity(expected_values);
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if !value.is_finite() {
            return Err(Error::new(
                "NONFINITE_INPUT",
                "DeepSeek HC F32 tensor contains NaN or infinity",
            ));
        }
        values.push(value);
    }
    Ok(values)
}
