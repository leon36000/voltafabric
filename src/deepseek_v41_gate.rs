//! CPU reference for the pinned DeepSeek-V4.1-Flash expert-selection boundary.
//!
//! This adapts the official MIT-licensed implementation at revision
//! `314113f8a7dadb388215396ba5fbee2c610a04d6`; it does not copy upstream code.
//! Inputs are the 384 FP32 outputs of the router linear projection, before the
//! model's gate temperature. The projection itself and loading of its checkpoint
//! tensors remain outside this module. The pinned defaults are temperature 1.0,
//! sqrt-softplus scoring, top-6, and normalized selected weights. The route scale
//! is an explicit constructor input. Upstream config names `routed_scaling_factor`
//! (1.5), while the reference `ModelArgs` demo default says `route_scale=1.0`; the
//! loader mapping must be established by the caller rather than assumed here.
//! Selection bias affects expert choice only. Image tokens use the separate
//! `bias_vl` vector. The shared expert is not represented by these routed choices
//! and must be executed by the layer separately. This covers the primary
//! 40-layer text backbone (384 routed experts, top-6), not DSpark's separate
//! auxiliary 128-expert/top-3 gate.
//!
//! This is an FP32 CPU reference boundary, not a bitwise oracle for PyTorch's
//! reduction kernel or a complete model forward. Ties in the selected top-6
//! ordering or at its cutoff are rejected because Torch's `topk` tie ordering is
//! not a stable cross-backend contract.
use crate::checkpoint::{self, NamedTensorSpec};
use crate::moe::Choice;
use crate::{invalid, Error, Result};
use safetensors::tensor::Dtype;
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Read, Seek},
    path::Path,
};

pub mod hc;

pub const ROUTED_EXPERTS: usize = 384;
pub const TOP_K: usize = 6;
pub const GATE_TEMPERATURE: f32 = 1.0;
pub const ROUTE_SCALE: f32 = 1.5;
const SOFTPLUS_THRESHOLD: f32 = 20.0;
const ROUTE_EPSILON: f32 = 1.0e-20;
pub const HIDDEN_SIZE: usize = 5120;
pub const PINNED_MODEL_REVISION: &str = "314113f8a7dadb388215396ba5fbee2c610a04d6";
pub const GATE_WEIGHT_BYTES: u64 = (ROUTED_EXPERTS * HIDDEN_SIZE * 2) as u64;
pub const GATE_BIAS_BYTES: u64 = (ROUTED_EXPERTS * 4) as u64;
pub const GATE_PAYLOAD_BYTES: u64 = GATE_WEIGHT_BYTES + 2 * GATE_BIAS_BYTES;
pub const PINNED_RMS_NORM_EPSILON: f32 = 1.0e-20;
pub const RMS_NORM_WEIGHT_BYTES: u64 = (HIDDEN_SIZE * 2) as u64;
pub const SHARED_EXPERT_INTERMEDIATE_SIZE: usize = 2304;
pub const SHARED_EXPERT_WEIGHT_BYTES: u64 = (SHARED_EXPERT_INTERMEDIATE_SIZE * HIDDEN_SIZE) as u64;
pub const SHARED_EXPERT_SCALE_BYTES: u64 =
    ((SHARED_EXPERT_INTERMEDIATE_SIZE / 32) * (HIDDEN_SIZE / 32)) as u64;
pub const SHARED_EXPERT_PAYLOAD_BYTES: u64 =
    3 * (SHARED_EXPERT_WEIGHT_BYTES + SHARED_EXPERT_SCALE_BYTES);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Text,
    Image,
}

/// Immutable, shape-checked copies of the two pinned gate correction-bias vectors.
#[derive(Debug, Clone)]
pub struct RouterBiases {
    text: Box<[f32]>,
    image: Box<[f32]>,
    temperature: f32,
    route_scale: f32,
}

impl RouterBiases {
    pub fn new(text: Vec<f32>, image: Vec<f32>, route_scale: f32) -> Result<Self> {
        Self::new_with_temperature(text, image, GATE_TEMPERATURE, route_scale)
    }

    pub fn new_with_temperature(
        text: Vec<f32>,
        image: Vec<f32>,
        temperature: f32,
        route_scale: f32,
    ) -> Result<Self> {
        validate_vector("text correction bias", &text)?;
        validate_vector("image correction bias", &image)?;
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(invalid(
                "DeepSeek gate temperature must be finite and positive",
            ));
        }
        if !route_scale.is_finite() || route_scale <= 0.0 {
            return Err(invalid("DeepSeek route scale must be finite and positive"));
        }
        Ok(Self {
            text: text.into_boxed_slice(),
            image: image.into_boxed_slice(),
            temperature,
            route_scale,
        })
    }

    /// Route one token from pre-temperature FP32 router logits.
    ///
    /// Returned choices are in descending corrected-score order, matching the
    /// pinned `topk(..., sorted=True)` boundary when no relevant ties exist. Each
    /// coefficient already includes normalization and the supplied route scale; a
    /// downstream executor must apply it exactly once.
    pub fn route(&self, router_logits: &[f32], kind: TokenKind) -> Result<Vec<Choice>> {
        validate_vector("router logits", router_logits)?;

        let correction = match kind {
            TokenKind::Text => &self.text,
            TokenKind::Image => &self.image,
        };
        let mut candidates = Vec::with_capacity(ROUTED_EXPERTS);
        for (expert, (&logit, &bias)) in router_logits.iter().zip(correction.iter()).enumerate() {
            let temperature_scaled = logit / self.temperature;
            let positive_score = softplus_fp32(temperature_scaled).sqrt();
            let corrected_score = positive_score + bias;
            if !positive_score.is_finite() || !corrected_score.is_finite() {
                return Err(Error::new(
                    "NUMERICAL_OVERFLOW",
                    "DeepSeek gate score became nonfinite",
                ));
            }
            candidates.push(Candidate {
                expert: expert as u32,
                score: positive_score,
                corrected_score,
            });
        }

        candidates.sort_by(|left, right| {
            right
                .corrected_score
                .total_cmp(&left.corrected_score)
                .then_with(|| left.expert.cmp(&right.expert))
        });
        if candidates[..=TOP_K]
            .windows(2)
            .any(|pair| pair[0].corrected_score == pair[1].corrected_score)
        {
            return Err(invalid(
                "DeepSeek top-6 or top-6 cutoff has tied corrected scores",
            ));
        }

        // The donor performs this reduction in FP32. A local scalar order is
        // explicit here; exact backend parity still requires a pinned Torch oracle.
        let score_sum = candidates[..TOP_K]
            .iter()
            .fold(0.0f32, |sum, candidate| sum + candidate.score);
        let denominator = score_sum + ROUTE_EPSILON;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(Error::new(
                "NUMERICAL_OVERFLOW",
                "DeepSeek gate normalization denominator is invalid",
            ));
        }

        candidates
            .iter()
            .take(TOP_K)
            .map(|candidate| {
                let normalized = candidate.score / denominator;
                let weighted = normalized * self.route_scale;
                if !weighted.is_finite() || weighted < 0.0 {
                    return Err(Error::new(
                        "NUMERICAL_OVERFLOW",
                        "DeepSeek routed coefficient is invalid",
                    ));
                }
                Ok(Choice {
                    expert: candidate.expert,
                    weight: f64::from(weighted),
                })
            })
            .collect()
    }
}

/// Tensor hashes are supplied from the caller's pinned source manifest. They
/// bind the exact three tensors selected from a larger checkpoint shard.
#[derive(Debug, Clone)]
pub struct GateTensorHashes {
    pub weight: String,
    pub text_bias: String,
    pub image_bias: String,
}

/// Caller-verified model identity and gate configuration. Neither route scale
/// nor temperature is inferred from a similarly named field in another config.
#[derive(Debug, Clone)]
pub struct GateShardSpec {
    pub model_revision: String,
    pub layer: u32,
    pub gate_temperature: f32,
    pub route_scale: f32,
    pub tensor_sha256: GateTensorHashes,
    pub max_payload_bytes: u64,
}

/// Pinned index-backed source for the three gate projection tensors. The index
/// digest is fixed to the pinned DeepSeek checkpoint revision; the selected
/// source shard header digest and source-size ceiling are supplied by admission.
#[derive(Debug, Clone, Copy)]
pub struct PinnedGateIndexSource<'a> {
    pub index_bytes: &'a [u8],
    pub expected_header_sha256: &'a str,
    pub max_source_shard_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateTensorReceipt {
    pub name: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateLoadReceipt {
    pub model_revision: String,
    pub layer: u32,
    pub file_bytes: u64,
    pub header_bytes: u64,
    pub selected_payload_bytes: u64,
    pub io_bytes: u64,
    pub tensors: Vec<GateTensorReceipt>,
}

/// Pinned checkpoint metadata for planning the gate-selected experts. This contains
/// only the Safetensors index/header and explicit source-size bounds; it is not proof
/// that any selected weight payload is present or authenticated.
#[derive(Debug, Clone, Copy)]
pub struct PinnedMxfp4PayloadSource<'a> {
    pub index_bytes: &'a [u8],
    pub header_bytes: &'a [u8],
    pub expected_header_sha256: &'a str,
    pub header_shard_name: &'a str,
    pub source_shard_bytes: u64,
    pub max_source_shard_bytes: u64,
}

/// Caller-provided digest and the exact epsilon for the pinned layer's ffn_norm.
/// The model index/header identify its location and BF16 shape, not its payload.
#[derive(Debug, Clone)]
pub struct FfnNormTensorSpec {
    pub layer: u32,
    pub epsilon: f32,
    pub tensor_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FfnNormLoadReceipt {
    pub layer: u32,
    pub tensor_name: String,
    pub tensor_sha256: String,
    pub tensor_bytes: u64,
    pub shard_bytes: u64,
    pub header_bytes: u64,
    pub io_bytes: u64,
}

/// Caller-pinned hashes for all six planes in the always-active shared FP8 expert.
/// The weight and scale encodings remain distinct: E4M3FN and E8M0 respectively.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedExpertTensorHashes {
    pub w1_weight: String,
    pub w1_scale: String,
    pub w2_weight: String,
    pub w2_scale: String,
    pub w3_weight: String,
    pub w3_scale: String,
}

#[derive(Debug, Clone)]
pub struct SharedExpertTensorSpec {
    pub layer: u32,
    pub tensor_sha256: SharedExpertTensorHashes,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedExpertLoadReceipt {
    pub layer: u32,
    pub source_shard_name: String,
    pub index_sha256: String,
    pub header_sha256: String,
    pub shard_bytes: u64,
    pub header_bytes: u64,
    pub tensor_bytes: u64,
    pub io_bytes: u64,
    pub tensor_sha256: SharedExpertTensorHashes,
}

#[derive(Debug)]
struct ResidentSharedExpert {
    expert: crate::deepseek_v41::DeepSeekV41SharedExpert,
    receipt: SharedExpertLoadReceipt,
}

/// Raw, unweighted FP32 result of the always-active shared expert. It carries the
/// full VoltaFabric token identity and attempt so it can only be joined to the
/// matching completed routed Collector result.
#[derive(Debug, Clone)]
pub struct DeepSeekV41SharedTokenOutput {
    stamp: crate::moe::Stamp,
    attempt: u64,
    output_fp32: Vec<f32>,
}

impl DeepSeekV41SharedTokenOutput {
    pub fn stamp(&self) -> &crate::moe::Stamp {
        &self.stamp
    }

    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    pub fn output_fp32(&self) -> &[f32] {
        &self.output_fp32
    }
}

/// Final single-token MoE activation. Values are exact BF16 values represented as
/// FP32 after one pinned FP32 routed-plus-shared addition and one BF16 RNE cast.
#[derive(Debug, Clone)]
pub struct DeepSeekV41MoeTokenOutput {
    stamp: crate::moe::Stamp,
    attempt: u64,
    output_bf16_f32: Vec<f32>,
}

impl DeepSeekV41MoeTokenOutput {
    pub fn stamp(&self) -> &crate::moe::Stamp {
        &self.stamp
    }

    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    pub fn output_bf16_f32(&self) -> &[f32] {
        &self.output_bf16_f32
    }
}

/// The four pinned residual copies entering one HC post operation. Every row is
/// exact BF16 represented in FP32 and retains the same token identity.
#[derive(Debug, Clone)]
pub struct DeepSeekV41ResidualStream {
    stamp: crate::moe::Stamp,
    attempt: u64,
    copies_bf16_f32: Vec<Vec<f32>>,
}

impl DeepSeekV41ResidualStream {
    pub fn new(
        stamp: crate::moe::Stamp,
        attempt: u64,
        copies_bf16_f32: Vec<Vec<f32>>,
    ) -> Result<Self> {
        validate_hc_identity(&stamp, attempt)?;
        if copies_bf16_f32.len() != crate::deepseek_v41::PINNED_HC_MULT
            || copies_bf16_f32.iter().any(|copy| {
                copy.len() != HIDDEN_SIZE
                    || copy.iter().any(|value| {
                        !value.is_finite()
                            || crate::deepseek_v41::cast_bf16_rne_reference(*value)
                                .map(|rounded| rounded.to_bits() != value.to_bits())
                                .unwrap_or(true)
                    })
            })
        {
            return Err(invalid(
                "DeepSeek HC residual must contain four finite exact-BF16 hidden rows",
            ));
        }
        Ok(Self {
            stamp,
            attempt,
            copies_bf16_f32,
        })
    }

    pub fn stamp(&self) -> &crate::moe::Stamp {
        &self.stamp
    }

    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    pub fn copies_bf16_f32(&self) -> &[Vec<f32>] {
        &self.copies_bf16_f32
    }
}

/// `post` and `comb` outputs of the pinned HC mix operator, bound to one exact token
/// and sublayer. The branch tag prevents attention coefficients from being applied
/// to the routed FFN result accepted by `deepseek_hc_post_reference`.
#[derive(Debug, Clone)]
pub struct DeepSeekV41HcPostCoefficients {
    stamp: crate::moe::Stamp,
    attempt: u64,
    branch: hc::HcProjectionBranch,
    post: Vec<f32>,
    comb: Vec<Vec<f32>>,
}

impl DeepSeekV41HcPostCoefficients {
    pub fn new(
        stamp: crate::moe::Stamp,
        attempt: u64,
        branch: hc::HcProjectionBranch,
        post: Vec<f32>,
        comb: Vec<Vec<f32>>,
    ) -> Result<Self> {
        validate_hc_identity(&stamp, attempt)?;
        let copies = crate::deepseek_v41::PINNED_HC_MULT;
        if post.len() != copies
            || comb.len() != copies
            || comb.iter().any(|row| row.len() != copies)
            || post
                .iter()
                .chain(comb.iter().flatten())
                .any(|value| !value.is_finite())
        {
            return Err(invalid(
                "DeepSeek HC post/comb coefficients have invalid shape or nonfinite values",
            ));
        }
        Ok(Self {
            stamp,
            attempt,
            branch,
            post,
            comb,
        })
    }

    pub fn branch(&self) -> hc::HcProjectionBranch {
        self.branch
    }

    pub fn post(&self) -> &[f32] {
        &self.post
    }

    pub fn comb(&self) -> &[Vec<f32>] {
        &self.comb
    }
}

fn validate_hc_identity(stamp: &crate::moe::Stamp, attempt: u64) -> Result<()> {
    if stamp.run_id.trim().is_empty()
        || stamp.model_revision != PINNED_MODEL_REVISION
        || stamp.plan_id.trim().is_empty()
        || stamp.layer >= 40
        || attempt == 0
    {
        return Err(invalid(
            "DeepSeek HC input requires a pinned model, layer, run, plan and nonzero attempt",
        ));
    }
    Ok(())
}

/// CPU scalar adaptation of the pinned Block.hc_post expression:
/// `post[i] * x + sum_j(comb[i,j] * residual[j])`, rounded once to BF16.
/// It accepts caller-supplied mixes and does not compute `hc_mixes` or Sinkhorn.
pub fn deepseek_hc_post_reference(
    sublayer: &DeepSeekV41MoeTokenOutput,
    residual: &DeepSeekV41ResidualStream,
    coefficients: &DeepSeekV41HcPostCoefficients,
) -> Result<DeepSeekV41ResidualStream> {
    let stamp = sublayer.stamp();
    validate_hc_identity(stamp, sublayer.attempt())?;
    if stamp != residual.stamp()
        || stamp != &coefficients.stamp
        || sublayer.attempt() != residual.attempt()
        || sublayer.attempt() != coefficients.attempt
        || coefficients.branch != hc::HcProjectionBranch::Ffn
        || sublayer.output_bf16_f32.len() != HIDDEN_SIZE
        || sublayer.output_bf16_f32.iter().any(|value| {
            !value.is_finite()
                || crate::deepseek_v41::cast_bf16_rne_reference(*value)
                    .map(|rounded| rounded.to_bits() != value.to_bits())
                    .unwrap_or(true)
        })
    {
        return Err(invalid(
            "DeepSeek HC post inputs do not share one exact token identity and BF16 row",
        ));
    }

    // Keep the result private until all four full rows are finite and rounded.
    let mut outputs = Vec::with_capacity(crate::deepseek_v41::PINNED_HC_MULT);
    for destination in 0..crate::deepseek_v41::PINNED_HC_MULT {
        let mut row = Vec::with_capacity(HIDDEN_SIZE);
        for hidden in 0..HIDDEN_SIZE {
            let mut residual_mix = 0.0f32;
            for source in 0..crate::deepseek_v41::PINNED_HC_MULT {
                residual_mix += coefficients.comb[destination][source]
                    * residual.copies_bf16_f32[source][hidden];
                if !residual_mix.is_finite() {
                    return Err(Error::new(
                        "NUMERICAL_OVERFLOW",
                        "DeepSeek HC residual mix became nonfinite",
                    ));
                }
            }
            let value =
                coefficients.post[destination] * sublayer.output_bf16_f32[hidden] + residual_mix;
            if !value.is_finite() {
                return Err(Error::new(
                    "NUMERICAL_OVERFLOW",
                    "DeepSeek HC post result became nonfinite",
                ));
            }
            row.push(crate::deepseek_v41::cast_bf16_rne_reference(value)?);
        }
        outputs.push(row);
    }
    DeepSeekV41ResidualStream::new(stamp.clone(), sublayer.attempt(), outputs)
}

/// Add the shared expert exactly once after the routed FP32-preweighted Collector
/// completes. Any Aborted/Unknown, identity mismatch, malformed or nonfinite row
/// returns no output. The route receipt is borrowed and remains available to the
/// caller for audit; this function performs no Session requests and never retries.
pub fn combine_pinned_moe_route_with_shared(
    route: &crate::deepseek_v41::PinnedMxfp4RouteExecutionReceipt,
    shared: &DeepSeekV41SharedTokenOutput,
) -> Result<DeepSeekV41MoeTokenOutput> {
    use crate::deepseek_v41::PinnedMxfp4RouteExecutionState;

    match route.state() {
        PinnedMxfp4RouteExecutionState::Completed => {}
        PinnedMxfp4RouteExecutionState::Aborted => {
            return Err(Error::new(
                "ROUTE_ABORTED",
                "DeepSeek shared result withheld because the routed Collector aborted",
            ));
        }
        PinnedMxfp4RouteExecutionState::Unknown => {
            return Err(Error::new(
                "ROUTE_UNKNOWN",
                "DeepSeek shared result withheld because the routed attempt is unknown",
            ));
        }
    }

    let stamp = route.stamp();
    if stamp != shared.stamp()
        || route.attempt() == 0
        || route.attempt() != shared.attempt()
        || stamp.model_revision != PINNED_MODEL_REVISION
        || stamp.layer != shared.stamp().layer
        || route.route_experts().len() != TOP_K
        || !crate::sha256::is_hex_digest(route.route_fingerprint())
        || shared.output_fp32.len() != HIDDEN_SIZE
        || shared.output_fp32.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(
            "DeepSeek routed and shared outputs do not share one complete token identity",
        ));
    }
    let receipts = route.exec_receipts();
    let unique_route_experts = route
        .route_experts()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let unique_ranks = receipts
        .iter()
        .map(|receipt| receipt.rank)
        .collect::<BTreeSet<_>>();
    let complete_slots = receipts.iter().try_fold(0usize, |sum, receipt| {
        if receipt.attempt != route.attempt()
            || receipt.state != crate::process::ReceiptState::Completed
            || !receipt.started
            || receipt.error.is_some()
            || receipt.expected_slots == 0
            || receipt.received_slots != receipt.expected_slots
        {
            return None;
        }
        sum.checked_add(receipt.received_slots)
    });
    if route.error().is_some()
        || receipts.is_empty()
        || unique_ranks.len() != receipts.len()
        || unique_route_experts.len() != TOP_K
        || complete_slots != Some(TOP_K)
    {
        return Err(invalid(
            "DeepSeek completed Collector receipt lacks six completed route slots",
        ));
    }
    let routed_rows = route
        .output_bits()
        .ok_or_else(|| invalid("completed DeepSeek route has no Collector output"))?;
    if routed_rows.len() != 1 || routed_rows[0].len() != HIDDEN_SIZE {
        return Err(invalid(
            "DeepSeek completed route must contain one full hidden row",
        ));
    }

    // Stage locally so no partially combined row can escape if a value overflows.
    let mut output_bf16_f32 = Vec::with_capacity(HIDDEN_SIZE);
    for (&routed_bits, &shared_value) in routed_rows[0].iter().zip(&shared.output_fp32) {
        let routed_value = f32::from_bits(routed_bits);
        let sum = routed_value + shared_value;
        if !routed_value.is_finite() || !sum.is_finite() {
            return Err(Error::new(
                "NUMERICAL_OVERFLOW",
                "DeepSeek routed-plus-shared FP32 sum is nonfinite",
            ));
        }
        output_bf16_f32.push(crate::deepseek_v41::cast_bf16_rne_reference(sum)?);
    }
    Ok(DeepSeekV41MoeTokenOutput {
        stamp: stamp.clone(),
        attempt: route.attempt(),
        output_bf16_f32,
    })
}

/// Resident CPU RMSNorm weight imported from the pinned BF16 ffn_norm tensor.
/// Math follows the pinned source's FP32 accumulation/multiply and BF16 output
/// cast; it is a scalar CPU reference, not a GPU kernel.
#[derive(Debug, Clone)]
pub struct DeepSeekV41FfnRmsNorm {
    layer: u32,
    epsilon: f32,
    weight_f32: Box<[f32]>,
    receipt: FfnNormLoadReceipt,
}

impl DeepSeekV41FfnRmsNorm {
    /// Authenticate and load only `layers.{layer}.ffn_norm.weight` from the
    /// caller-admitted shard. The pinned index, header digest, tensor identity,
    /// byte count and tensor SHA-256 must all agree before the resident weight
    /// is returned.
    pub fn load_indexed(
        root: &Path,
        source: PinnedMxfp4PayloadSource<'_>,
        spec: &FfnNormTensorSpec,
    ) -> Result<Self> {
        if spec.layer >= 40
            || spec.epsilon.to_bits() != PINNED_RMS_NORM_EPSILON.to_bits()
            || !crate::sha256::is_hex_digest(&spec.tensor_sha256)
            || crate::sha256::hex(source.index_bytes)
                != crate::deepseek_v41::PINNED_EXPERT_INDEX_SHA256
            || source.expected_header_sha256.len() != 64
            || !crate::sha256::is_hex_digest(source.expected_header_sha256)
            || crate::sha256::hex(source.header_bytes) != source.expected_header_sha256
            || source.source_shard_bytes == 0
            || source.source_shard_bytes > source.max_source_shard_bytes
        {
            return Err(invalid(
                "pinned DeepSeek ffn_norm identity, epsilon, digest or bounds are invalid",
            ));
        }

        let tensor_name = format!("layers.{}.ffn_norm.weight", spec.layer);
        let expected_plan = checkpoint::plan_indexed_safetensors_shards(
            source.index_bytes,
            crate::deepseek_v41::PINNED_EXPERT_INDEX_SHA256,
            std::slice::from_ref(&tensor_name),
        )?;
        if expected_plan
            .tensor_to_shard
            .get(&tensor_name)
            .map(String::as_str)
            != Some(source.header_shard_name)
        {
            return Err(invalid(
                "pinned DeepSeek ffn_norm must use the admitted layer shard",
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
            return Err(invalid(
                "pinned DeepSeek ffn_norm shard header or size differs",
            ));
        }

        let named = NamedTensorSpec {
            name: tensor_name.clone(),
            dtype: Dtype::BF16,
            shape: vec![HIDDEN_SIZE],
            bytes: RMS_NORM_WEIGHT_BYTES,
            sha256: spec.tensor_sha256.clone(),
        };
        let imported = checkpoint::read_indexed_named_tensors(
            root,
            source.index_bytes,
            crate::deepseek_v41::PINNED_EXPERT_INDEX_SHA256,
            &[named],
            RMS_NORM_WEIGHT_BYTES,
        )?;
        if imported.plan != expected_plan
            || imported.file_bytes != source.source_shard_bytes
            || imported.header_bytes != header.header_bytes.len() as u64
            || imported.tensors.len() != 1
        {
            return Err(invalid(
                "indexed DeepSeek ffn_norm payload import differs from its preflight",
            ));
        }
        let tensor = imported
            .tensors
            .get(&tensor_name)
            .ok_or_else(|| invalid("indexed DeepSeek ffn_norm payload is absent"))?;
        if tensor.dtype != Dtype::BF16
            || tensor.shape != [HIDDEN_SIZE]
            || tensor.bytes.len() as u64 != RMS_NORM_WEIGHT_BYTES
            || tensor.sha256 != spec.tensor_sha256
        {
            return Err(invalid(
                "indexed DeepSeek ffn_norm tensor metadata or digest differs",
            ));
        }
        let final_header = checkpoint::read_indexed_safetensors_shard_header(
            root,
            source.header_shard_name,
            source.max_source_shard_bytes,
        )?;
        if final_header.file_bytes != header.file_bytes || final_header.sha256 != header.sha256 {
            return Err(Error::new(
                "CHECKPOINT_CHANGED",
                "DeepSeek ffn_norm shard changed during indexed payload import",
            ));
        }

        let mut weight_f32 = Vec::with_capacity(HIDDEN_SIZE);
        for pair in tensor.bytes.chunks_exact(2) {
            let bf16 = u16::from_le_bytes([pair[0], pair[1]]);
            let value = f32::from_bits(u32::from(bf16) << 16);
            if !value.is_finite() {
                return Err(Error::new(
                    "NONFINITE_INPUT",
                    "DeepSeek ffn_norm weight contains NaN or infinity",
                ));
            }
            weight_f32.push(value);
        }
        if weight_f32.len() != HIDDEN_SIZE {
            return Err(invalid("DeepSeek ffn_norm weight has wrong hidden width"));
        }

        Ok(Self {
            layer: spec.layer,
            epsilon: spec.epsilon,
            weight_f32: weight_f32.into_boxed_slice(),
            receipt: FfnNormLoadReceipt {
                layer: spec.layer,
                tensor_name,
                tensor_sha256: spec.tensor_sha256.clone(),
                tensor_bytes: RMS_NORM_WEIGHT_BYTES,
                shard_bytes: imported.file_bytes,
                header_bytes: imported.header_bytes,
                io_bytes: imported.io_bytes,
            },
        })
    }

    pub fn layer(&self) -> u32 {
        self.layer
    }

    pub fn receipt(&self) -> &FfnNormLoadReceipt {
        &self.receipt
    }

    /// Apply the pinned per-token RMSNorm to one exact-BF16 activation represented
    /// as FP32. Accumulation and the learned-weight product are FP32, followed by
    /// the model's BF16 round-to-nearest-even output cast.
    pub fn apply(&self, input_bf16_f32: &[f32]) -> Result<Vec<f32>> {
        if input_bf16_f32.len() != HIDDEN_SIZE
            || self.weight_f32.len() != HIDDEN_SIZE
            || !self.epsilon.is_finite()
            || self.epsilon.to_bits() != PINNED_RMS_NORM_EPSILON.to_bits()
        {
            return Err(invalid(
                "DeepSeek ffn_norm input, weight or epsilon is invalid",
            ));
        }
        let mut sum_squares = 0.0f32;
        for &value in input_bf16_f32 {
            if !value.is_finite()
                || crate::deepseek_v41::cast_bf16_rne_reference(value)?.to_bits() != value.to_bits()
            {
                return Err(invalid(
                    "DeepSeek ffn_norm input must contain finite exact BF16 values",
                ));
            }
            sum_squares += value * value;
        }
        if !sum_squares.is_finite() {
            return Err(Error::new(
                "NUMERICAL_OVERFLOW",
                "DeepSeek ffn_norm square sum overflowed",
            ));
        }
        let mean_square = sum_squares / HIDDEN_SIZE as f32;
        let variance = mean_square + self.epsilon;
        if !variance.is_finite() || variance <= 0.0 {
            return Err(Error::new(
                "NUMERICAL_OVERFLOW",
                "DeepSeek ffn_norm variance is invalid",
            ));
        }
        let inverse_rms = variance.sqrt().recip();
        if !inverse_rms.is_finite() {
            return Err(Error::new(
                "NUMERICAL_OVERFLOW",
                "DeepSeek ffn_norm reciprocal RMS is invalid",
            ));
        }
        input_bf16_f32
            .iter()
            .zip(self.weight_f32.iter())
            .map(|(&value, &weight)| {
                crate::deepseek_v41::cast_bf16_rne_reference((value * inverse_rms) * weight)
            })
            .collect()
    }
}

impl SharedExpertTensorSpec {
    fn named_tensors(&self) -> Result<Vec<NamedTensorSpec>> {
        if self.layer >= 40 {
            return Err(invalid(
                "pinned DeepSeek shared expert layer is out of range",
            ));
        }
        let layer = self.layer;
        let intermediate = SHARED_EXPERT_INTERMEDIATE_SIZE;
        let hidden = HIDDEN_SIZE;
        let entries = [
            (
                format!("layers.{layer}.ffn.shared_experts.w1.weight"),
                Dtype::F8_E4M3,
                vec![intermediate, hidden],
                SHARED_EXPERT_WEIGHT_BYTES,
                &self.tensor_sha256.w1_weight,
            ),
            (
                format!("layers.{layer}.ffn.shared_experts.w1.scale"),
                Dtype::F8_E8M0,
                vec![intermediate / 32, hidden / 32],
                SHARED_EXPERT_SCALE_BYTES,
                &self.tensor_sha256.w1_scale,
            ),
            (
                format!("layers.{layer}.ffn.shared_experts.w2.weight"),
                Dtype::F8_E4M3,
                vec![hidden, intermediate],
                SHARED_EXPERT_WEIGHT_BYTES,
                &self.tensor_sha256.w2_weight,
            ),
            (
                format!("layers.{layer}.ffn.shared_experts.w2.scale"),
                Dtype::F8_E8M0,
                vec![hidden / 32, intermediate / 32],
                SHARED_EXPERT_SCALE_BYTES,
                &self.tensor_sha256.w2_scale,
            ),
            (
                format!("layers.{layer}.ffn.shared_experts.w3.weight"),
                Dtype::F8_E4M3,
                vec![intermediate, hidden],
                SHARED_EXPERT_WEIGHT_BYTES,
                &self.tensor_sha256.w3_weight,
            ),
            (
                format!("layers.{layer}.ffn.shared_experts.w3.scale"),
                Dtype::F8_E8M0,
                vec![intermediate / 32, hidden / 32],
                SHARED_EXPERT_SCALE_BYTES,
                &self.tensor_sha256.w3_scale,
            ),
        ];
        if entries
            .iter()
            .any(|(_, _, _, _, digest)| !crate::sha256::is_hex_digest(digest))
        {
            return Err(invalid(
                "pinned DeepSeek shared expert tensor digest is malformed",
            ));
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
}

impl ResidentSharedExpert {
    fn load_indexed(
        root: &Path,
        source: PinnedMxfp4PayloadSource<'_>,
        spec: &SharedExpertTensorSpec,
    ) -> Result<Self> {
        if spec.layer >= 40
            || !crate::sha256::is_hex_digest(source.expected_header_sha256)
            || crate::sha256::hex(source.index_bytes)
                != crate::deepseek_v41::PINNED_EXPERT_INDEX_SHA256
            || crate::sha256::hex(source.header_bytes) != source.expected_header_sha256
            || source.source_shard_bytes == 0
            || source.source_shard_bytes > source.max_source_shard_bytes
            || source.header_shard_name.is_empty()
        {
            return Err(invalid(
                "pinned DeepSeek shared expert source identity or bounds are invalid",
            ));
        }

        let selected = spec.named_tensors()?;
        let names = selected
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect::<Vec<_>>();
        let expected_plan = checkpoint::plan_indexed_safetensors_shards(
            source.index_bytes,
            crate::deepseek_v41::PINNED_EXPERT_INDEX_SHA256,
            &names,
        )?;
        if expected_plan.tensor_to_shard.len() != selected.len()
            || expected_plan
                .tensor_to_shard
                .values()
                .any(|shard| shard != source.header_shard_name)
        {
            return Err(invalid(
                "pinned DeepSeek shared expert planes must share the admitted layer shard",
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
            return Err(invalid(
                "pinned DeepSeek shared expert header or shard size differs",
            ));
        }

        let imported = checkpoint::read_indexed_named_tensors(
            root,
            source.index_bytes,
            crate::deepseek_v41::PINNED_EXPERT_INDEX_SHA256,
            &selected,
            SHARED_EXPERT_PAYLOAD_BYTES,
        )?;
        if imported.plan != expected_plan
            || imported.file_bytes != source.source_shard_bytes
            || imported.header_bytes != header.header_bytes.len() as u64
            || imported.tensors.len() != selected.len()
        {
            return Err(invalid(
                "indexed DeepSeek shared expert import differs from its preflight",
            ));
        }

        let mut tensors = imported.tensors;
        let mut take = |name: &str,
                        dtype: Dtype,
                        shape: &[usize],
                        byte_count: u64,
                        expected_sha256: &str|
         -> Result<Vec<u8>> {
            let tensor = tensors
                .remove(name)
                .ok_or_else(|| invalid(format!("indexed shared tensor is absent: {name}")))?;
            if tensor.dtype != dtype
                || tensor.shape != shape
                || tensor.bytes.len() as u64 != byte_count
                || tensor.sha256 != expected_sha256
            {
                return Err(invalid(format!(
                    "indexed shared tensor metadata or digest differs: {name}"
                )));
            }
            Ok(tensor.bytes)
        };

        let layer = spec.layer;
        let intermediate = SHARED_EXPERT_INTERMEDIATE_SIZE;
        let hidden = HIDDEN_SIZE;
        let w1_weight = take(
            &format!("layers.{layer}.ffn.shared_experts.w1.weight"),
            Dtype::F8_E4M3,
            &[intermediate, hidden],
            SHARED_EXPERT_WEIGHT_BYTES,
            &spec.tensor_sha256.w1_weight,
        )?;
        let w1_scale_blocks = take(
            &format!("layers.{layer}.ffn.shared_experts.w1.scale"),
            Dtype::F8_E8M0,
            &[intermediate / 32, hidden / 32],
            SHARED_EXPERT_SCALE_BYTES,
            &spec.tensor_sha256.w1_scale,
        )?;
        let w1 = crate::deepseek_v41::Fp8E4m3Matrix::new(
            intermediate as u64,
            hidden as u64,
            w1_weight,
            expand_fp8_block_scales(intermediate, hidden, &w1_scale_blocks)?,
        )?;
        let w2_weight = take(
            &format!("layers.{layer}.ffn.shared_experts.w2.weight"),
            Dtype::F8_E4M3,
            &[hidden, intermediate],
            SHARED_EXPERT_WEIGHT_BYTES,
            &spec.tensor_sha256.w2_weight,
        )?;
        let w2_scale_blocks = take(
            &format!("layers.{layer}.ffn.shared_experts.w2.scale"),
            Dtype::F8_E8M0,
            &[hidden / 32, intermediate / 32],
            SHARED_EXPERT_SCALE_BYTES,
            &spec.tensor_sha256.w2_scale,
        )?;
        let w2 = crate::deepseek_v41::Fp8E4m3Matrix::new(
            hidden as u64,
            intermediate as u64,
            w2_weight,
            expand_fp8_block_scales(hidden, intermediate, &w2_scale_blocks)?,
        )?;
        let w3_weight = take(
            &format!("layers.{layer}.ffn.shared_experts.w3.weight"),
            Dtype::F8_E4M3,
            &[intermediate, hidden],
            SHARED_EXPERT_WEIGHT_BYTES,
            &spec.tensor_sha256.w3_weight,
        )?;
        let w3_scale_blocks = take(
            &format!("layers.{layer}.ffn.shared_experts.w3.scale"),
            Dtype::F8_E8M0,
            &[intermediate / 32, hidden / 32],
            SHARED_EXPERT_SCALE_BYTES,
            &spec.tensor_sha256.w3_scale,
        )?;
        let w3 = crate::deepseek_v41::Fp8E4m3Matrix::new(
            intermediate as u64,
            hidden as u64,
            w3_weight,
            expand_fp8_block_scales(intermediate, hidden, &w3_scale_blocks)?,
        )?;
        if !tensors.is_empty() {
            return Err(invalid(
                "indexed DeepSeek shared expert import returned unrequested tensors",
            ));
        }

        let final_header = checkpoint::read_indexed_safetensors_shard_header(
            root,
            source.header_shard_name,
            source.max_source_shard_bytes,
        )?;
        if final_header.file_bytes != header.file_bytes || final_header.sha256 != header.sha256 {
            return Err(Error::new(
                "CHECKPOINT_CHANGED",
                "DeepSeek shared expert shard changed during indexed payload import",
            ));
        }

        let expert = crate::deepseek_v41::DeepSeekV41SharedExpert::new(w1, w2, w3)?;
        Ok(Self {
            expert,
            receipt: SharedExpertLoadReceipt {
                layer,
                source_shard_name: source.header_shard_name.to_owned(),
                index_sha256: crate::sha256::hex(source.index_bytes),
                header_sha256: header.sha256,
                shard_bytes: imported.file_bytes,
                header_bytes: imported.header_bytes,
                tensor_bytes: SHARED_EXPERT_PAYLOAD_BYTES,
                io_bytes: imported.io_bytes,
                tensor_sha256: spec.tensor_sha256.clone(),
            },
        })
    }
}

/// Expand the checkpoint's E8M0 [M/32,K/32] scales into the existing CPU
/// reference's row-major [M,K/32] view. One source scale applies to 32 matrix rows.
fn expand_fp8_block_scales(rows: usize, cols: usize, block_scales: &[u8]) -> Result<Vec<u8>> {
    if rows == 0 || cols == 0 || !rows.is_multiple_of(32) || !cols.is_multiple_of(32) {
        return Err(invalid(
            "DeepSeek FP8 block scales require dimensions divisible by 32",
        ));
    }
    let scale_blocks_per_row = cols / 32;
    let expected = (rows / 32)
        .checked_mul(scale_blocks_per_row)
        .ok_or_else(|| invalid("DeepSeek FP8 block-scale count overflow"))?;
    if block_scales.len() != expected || block_scales.contains(&255) {
        return Err(invalid(
            "DeepSeek FP8 block-scale plane has invalid size or E8M0 NaN",
        ));
    }
    let expanded_len = rows
        .checked_mul(scale_blocks_per_row)
        .ok_or_else(|| invalid("DeepSeek expanded FP8 scale count overflow"))?;
    let mut expanded = Vec::with_capacity(expanded_len);
    for block_row in 0..rows / 32 {
        let start = block_row * scale_blocks_per_row;
        let row = &block_scales[start..start + scale_blocks_per_row];
        for _ in 0..32 {
            expanded.extend_from_slice(row);
        }
    }
    Ok(expanded)
}

#[cfg(test)]
mod r728_fp8_scale_layout_tests {
    use super::expand_fp8_block_scales;

    #[test]
    fn checkpoint_32_by_32_scales_expand_to_the_existing_row_major_view() {
        let source = [10, 11, 12, 13]; // [M/32=2,K/32=2]
        let expanded = expand_fp8_block_scales(64, 64, &source).unwrap();
        assert_eq!(expanded.len(), 128);
        for row in 0..32 {
            assert_eq!(&expanded[row * 2..row * 2 + 2], &[10, 11]);
        }
        for row in 32..64 {
            assert_eq!(&expanded[row * 2..row * 2 + 2], &[12, 13]);
        }
    }

    #[test]
    fn checkpoint_scale_expansion_rejects_wrong_shape_length_and_e8m0_nan() {
        assert!(expand_fp8_block_scales(63, 64, &[10, 11]).is_err());
        assert!(expand_fp8_block_scales(64, 64, &[10, 11, 12]).is_err());
        assert!(expand_fp8_block_scales(64, 64, &[10, 11, 12, 255]).is_err());
    }
}

/// Immutable metadata-only payload plan derived from one gate-produced top-6 route.
/// The stamp, attempt, route order, coefficient/rank dispatch fingerprint and exact
/// selected source ranges stay together for the later authenticated read stage.
#[derive(Debug)]
pub struct DeepSeekV41RoutePayloadPlan {
    stamp: crate::moe::Stamp,
    attempt: u64,
    route_fingerprint: String,
    payload_plan: crate::deepseek_v41::Mxfp4ExpertPayloadPlan,
}

/// All six route-selected payload reads, returned only after every selected tensor
/// passed its digest check. Intermediate per-expert reads remain private until the
/// route-level operation succeeds, so a caller cannot accidentally start Sessions
/// from a partial top-6 payload set.
#[derive(Debug)]
pub struct DeepSeekV41RoutePayloadReads {
    stamp: crate::moe::Stamp,
    attempt: u64,
    route_fingerprint: String,
    payload_plan: crate::deepseek_v41::Mxfp4ExpertPayloadPlan,
    reads: Vec<crate::deepseek_v41::PinnedMxfp4RankExpertRead>,
}

impl DeepSeekV41RoutePayloadReads {
    pub fn stamp(&self) -> &crate::moe::Stamp {
        &self.stamp
    }

    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    pub fn route_fingerprint(&self) -> &str {
        &self.route_fingerprint
    }

    pub fn payload_plan(&self) -> &crate::deepseek_v41::Mxfp4ExpertPayloadPlan {
        &self.payload_plan
    }

    pub fn reads(&self) -> &[crate::deepseek_v41::PinnedMxfp4RankExpertRead] {
        &self.reads
    }

    /// Convert the complete route read into six resident owner inputs. If any one
    /// owner fails its own peak-memory or tensor-identity admission, this method
    /// drops the partial vector and returns no owners.
    pub fn into_session_residents(
        self,
        max_peak_bytes_per_expert: u64,
    ) -> Result<Vec<crate::deepseek_v41::PinnedMxfp4RankResidentExpert>> {
        if self.attempt == 0
            || self.reads.len() != TOP_K
            || self.payload_plan.experts.len() != TOP_K
            || self.route_fingerprint.is_empty()
        {
            return Err(invalid(
                "DeepSeek route payload read receipt is not a complete top-6 set",
            ));
        }
        let mut residents = Vec::with_capacity(TOP_K);
        for (expert, read) in self.payload_plan.experts.iter().zip(self.reads) {
            if read.stamp() != &self.stamp
                || read.attempt() != self.attempt
                || read.route_fingerprint() != self.route_fingerprint
                || read.route_experts() != self.payload_plan.experts
                || read.expert() != *expert
            {
                return Err(invalid(
                    "DeepSeek route payload read identity or selected expert order changed",
                ));
            }
            residents.push(read.into_session_resident(max_peak_bytes_per_expert)?);
        }
        if residents.len() != TOP_K {
            return Err(invalid(
                "DeepSeek route owner conversion did not produce all six experts",
            ));
        }
        Ok(residents)
    }
}

impl DeepSeekV41RoutePayloadPlan {
    pub fn stamp(&self) -> &crate::moe::Stamp {
        &self.stamp
    }

    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    pub fn route_experts(&self) -> &[u32] {
        &self.payload_plan.experts
    }

    pub fn route_fingerprint(&self) -> &str {
        &self.route_fingerprint
    }

    pub fn payload_plan(&self) -> &crate::deepseek_v41::Mxfp4ExpertPayloadPlan {
        &self.payload_plan
    }
}

/// A one-token gate decision bound to VoltaFabric's route identity and attempt.
/// The route problem is private so the validated stamp cannot be replaced after
/// construction; its already-scaled coefficients are carried through unchanged.
#[derive(Debug, Clone)]
pub struct DeepSeekV41RouteBatch {
    problem: crate::routing::RoutingProblem,
    attempt: u64,
}

/// Complete input identity for one token entering the indexed DeepSeek MoE layer.
/// `hidden` is the caller's post-attention/post-normalization activation; this type
/// does not claim to implement the attention or normalization sublayers.
pub struct DeepSeekV41IndexedTokenRequest<'a> {
    pub hidden: &'a [f32],
    pub token_kind: TokenKind,
    pub stamp: crate::moe::Stamp,
    pub attempt: u64,
    pub registry: crate::moe::Registry,
    pub preferred_owners: BTreeMap<u32, u32>,
    pub resident_order: BTreeMap<u32, Vec<u32>>,
    pub max_total_selected_payload_bytes: u64,
}

/// Raw block activation entering the model's ffn_norm immediately before this
/// indexed MoE sublayer. The normalization is applied exactly once by the layer.
pub struct DeepSeekV41IndexedFfnTokenRequest<'a> {
    pub hidden: &'a [f32],
    pub token_kind: TokenKind,
    pub stamp: crate::moe::Stamp,
    pub attempt: u64,
    pub registry: crate::moe::Registry,
    pub preferred_owners: BTreeMap<u32, u32>,
    pub resident_order: BTreeMap<u32, Vec<u32>>,
    pub max_total_selected_payload_bytes: u64,
}

/// Route and selected source ranges produced together by one resident indexed gate.
/// The route remains private so callers cannot swap in expert IDs or coefficient bits.
#[derive(Debug)]
pub struct DeepSeekV41IndexedMoeTokenPlan {
    batch: DeepSeekV41RouteBatch,
    payloads: DeepSeekV41RoutePayloadPlan,
    activation: Vec<f32>,
}

impl DeepSeekV41IndexedMoeTokenPlan {
    pub fn batch(&self) -> &DeepSeekV41RouteBatch {
        &self.batch
    }

    pub fn payloads(&self) -> &DeepSeekV41RoutePayloadPlan {
        &self.payloads
    }

    /// Exact BF16-representable activation used to derive this route. Keeping it
    /// beside the private route prevents a caller from routing one tensor and
    /// executing the selected experts on a separately normalized tensor.
    pub fn activation(&self) -> &[f32] {
        &self.activation
    }
}

impl DeepSeekV41RouteBatch {
    pub fn problem(&self) -> &crate::routing::RoutingProblem {
        &self.problem
    }

    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    /// Resolve all six gate-selected experts against the pinned index and one exact
    /// Safetensors header, without reading any weight range. The checkpoint index
    /// must place those six experts in the supplied shard; incomplete or oversized
    /// plans refuse before a caller opens a payload reader or starts Sessions.
    pub fn plan_selected_mxfp4_payloads(
        &self,
        source: PinnedMxfp4PayloadSource<'_>,
        max_total_selected_payload_bytes: u64,
    ) -> Result<DeepSeekV41RoutePayloadPlan> {
        crate::routing::validate(&self.problem)?;
        if self.attempt == 0
            || self.problem.stamp.model_revision != PINNED_MODEL_REVISION
            || self.problem.hidden != HIDDEN_SIZE
            || self.problem.routes.len() != 1
            || self.problem.routes[0].len() != TOP_K
            || self.problem.experts.len() != TOP_K
            || self.problem.experts
                != self.problem.routes[0]
                    .iter()
                    .map(|choice| choice.expert)
                    .collect::<Vec<_>>()
            || max_total_selected_payload_bytes == 0
        {
            return Err(invalid(
                "DeepSeek payload planning requires one identified top-6 gate route and a positive byte ceiling",
            ));
        }
        let route_fingerprint =
            crate::deepseek_v41::pinned_top6_route_fingerprint(&self.problem, self.attempt)?;
        let payload_plan = crate::deepseek_v41::plan_pinned_mxfp4_expert_payloads(
            crate::deepseek_v41::PinnedExpertPayloadRequest {
                index_bytes: source.index_bytes,
                header_bytes: source.header_bytes,
                expected_header_sha256: source.expected_header_sha256,
                header_shard_name: source.header_shard_name,
                source_shard_bytes: source.source_shard_bytes,
                max_source_shard_bytes: source.max_source_shard_bytes,
                namespace: crate::deepseek_v41::Mxfp4ExpertNamespace::Backbone,
                layer: self.problem.stamp.layer,
                experts: &self.problem.experts,
            },
        )?;
        if payload_plan.experts != self.problem.experts {
            return Err(invalid(
                "pinned payload planner changed gate-selected expert order",
            ));
        }
        if payload_plan.selected_payload_bytes > max_total_selected_payload_bytes {
            return Err(crate::Error::new(
                "ARTIFACT_BUDGET",
                "gate-selected MXFP4 payload plan exceeds its explicit aggregate byte ceiling",
            ));
        }
        Ok(DeepSeekV41RoutePayloadPlan {
            stamp: self.problem.stamp.clone(),
            attempt: self.attempt,
            route_fingerprint,
            payload_plan,
        })
    }

    /// Authenticate one complete gate-selected top-6 payload set from the planned
    /// shard. All 36 tensor digests are checked for exact coverage and valid syntax
    /// before the first source read. Each expert is then read through the existing
    /// bounded Safetensors reader; no partial route receipt escapes if any read,
    /// payload digest, or source-stability check fails. The source must remain
    /// quiescent for the duration of the call.
    pub fn read_selected_mxfp4_payloads<R: Read + Seek>(
        &self,
        reader: &mut R,
        plan: &DeepSeekV41RoutePayloadPlan,
        source: PinnedMxfp4PayloadSource<'_>,
        expected_payload_sha256: &BTreeMap<String, String>,
        max_total_selected_payload_bytes: u64,
    ) -> Result<DeepSeekV41RoutePayloadReads> {
        self.read_selected_mxfp4_payloads_with(
            plan,
            source,
            expected_payload_sha256,
            max_total_selected_payload_bytes,
            |expert, rank, expert_hashes, max_payload_bytes| {
                crate::deepseek_v41::read_pinned_mxfp4_rank_expert(
                    reader,
                    crate::deepseek_v41::PinnedMxfp4RankExpertReadRequest {
                        route: &self.problem,
                        attempt: self.attempt,
                        expert,
                        rank,
                        index_bytes: source.index_bytes,
                        header_shard_name: source.header_shard_name,
                        expected_header_sha256: source.expected_header_sha256,
                        expected_payload_sha256: expert_hashes,
                        max_source_shard_bytes: source.max_source_shard_bytes,
                        max_payload_bytes,
                    },
                )
            },
        )
    }

    /// Authenticate the gate-selected top-6 directly from the pinned checkpoint
    /// directory. All route identities and all 36 payload hashes are validated
    /// before the first file-backed read. The method exposes no partial read set.
    pub fn read_selected_mxfp4_indexed_payloads(
        &self,
        root: &Path,
        plan: &DeepSeekV41RoutePayloadPlan,
        source: PinnedMxfp4PayloadSource<'_>,
        expected_payload_sha256: &BTreeMap<String, String>,
        max_total_selected_payload_bytes: u64,
    ) -> Result<DeepSeekV41RoutePayloadReads> {
        self.read_selected_mxfp4_payloads_with(
            plan,
            source,
            expected_payload_sha256,
            max_total_selected_payload_bytes,
            |expert, rank, expert_hashes, max_payload_bytes| {
                crate::deepseek_v41::read_pinned_mxfp4_rank_expert_indexed(
                    root,
                    crate::deepseek_v41::PinnedMxfp4IndexedRankExpertReadRequest {
                        route: &self.problem,
                        attempt: self.attempt,
                        expert,
                        rank,
                        index_bytes: source.index_bytes,
                        expected_header_sha256: source.expected_header_sha256,
                        expected_payload_sha256: expert_hashes,
                        max_source_shard_bytes: source.max_source_shard_bytes,
                        max_payload_bytes,
                    },
                )
            },
        )
    }

    fn read_selected_mxfp4_payloads_with<F>(
        &self,
        plan: &DeepSeekV41RoutePayloadPlan,
        source: PinnedMxfp4PayloadSource<'_>,
        expected_payload_sha256: &BTreeMap<String, String>,
        max_total_selected_payload_bytes: u64,
        mut read_expert: F,
    ) -> Result<DeepSeekV41RoutePayloadReads>
    where
        F: FnMut(
            u32,
            u32,
            &BTreeMap<String, String>,
            u64,
        ) -> Result<crate::deepseek_v41::PinnedMxfp4RankExpertRead>,
    {
        crate::routing::validate(&self.problem)?;
        if self.attempt == 0
            || plan.stamp != self.problem.stamp
            || plan.attempt != self.attempt
            || plan.route_fingerprint
                != crate::deepseek_v41::pinned_top6_route_fingerprint(&self.problem, self.attempt)?
        {
            return Err(invalid(
                "DeepSeek payload plan differs from the gate-bound route identity",
            ));
        }

        let verified_plan =
            self.plan_selected_mxfp4_payloads(source, max_total_selected_payload_bytes)?;
        if verified_plan.stamp != plan.stamp
            || verified_plan.attempt != plan.attempt
            || verified_plan.route_fingerprint != plan.route_fingerprint
            || verified_plan.payload_plan != plan.payload_plan
        {
            return Err(invalid(
                "DeepSeek payload plan differs from the pinned source metadata",
            ));
        }

        let payload_plan = &verified_plan.payload_plan;
        if payload_plan.experts.as_slice() != self.problem.experts.as_slice()
            || payload_plan.tensors.len() != TOP_K * 6
            || expected_payload_sha256.len() != payload_plan.tensors.len()
        {
            return Err(invalid(
                "DeepSeek route payload digest manifest must cover exactly 36 selected tensors",
            ));
        }
        let expected_names = payload_plan
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        if expected_names.len() != TOP_K * 6
            || expected_payload_sha256
                .keys()
                .any(|name| !expected_names.contains(name.as_str()))
        {
            return Err(invalid(
                "DeepSeek route payload digest manifest contains duplicate or foreign tensor names",
            ));
        }
        for name in &expected_names {
            let digest = expected_payload_sha256.get(*name).ok_or_else(|| {
                invalid(format!("DeepSeek route payload digest is missing: {name}"))
            })?;
            if !crate::sha256::is_hex_digest(digest) {
                return Err(invalid(format!(
                    "DeepSeek route payload digest is malformed: {name}"
                )));
            }
        }

        let dispatch = crate::routing::dispatch(&self.problem)?;
        if dispatch.slots.len() != TOP_K {
            return Err(invalid(
                "DeepSeek route dispatch differs from the selected top-6 payload set",
            ));
        }
        let route_fingerprint = verified_plan.route_fingerprint.clone();
        let mut reads = Vec::with_capacity(TOP_K);
        for (slot, (choice, binding)) in self.problem.routes[0]
            .iter()
            .zip(&dispatch.slots)
            .enumerate()
        {
            if binding.token != 0
                || binding.slot != slot
                || binding.expert != choice.expert
                || binding.weight.to_bits() != choice.weight.to_bits()
            {
                return Err(invalid(
                    "DeepSeek dispatch changed selected route order or coefficient bits",
                ));
            }
            let expert_bytes = payload_plan
                .tensors
                .iter()
                .filter(|tensor| tensor.expert == choice.expert)
                .try_fold(0_u64, |sum, tensor| {
                    sum.checked_add(
                        tensor
                            .file_offset_end
                            .checked_sub(tensor.file_offset_start)?,
                    )
                })
                .ok_or_else(|| invalid("DeepSeek per-expert payload byte count overflow"))?;
            if expert_bytes == 0 || expert_bytes > crate::checkpoint::MAX_NAMED_TENSOR_PAYLOAD_BYTES
            {
                return Err(crate::Error::new(
                    "ARTIFACT_BUDGET",
                    "DeepSeek route expert payload exceeds the selective-read byte ceiling",
                ));
            }
            let expert_hashes = payload_plan
                .tensors
                .iter()
                .filter(|tensor| tensor.expert == choice.expert)
                .map(|tensor| {
                    expected_payload_sha256
                        .get(&tensor.name)
                        .cloned()
                        .map(|digest| (tensor.name.clone(), digest))
                        .ok_or_else(|| {
                            invalid("DeepSeek route payload digest disappeared after preflight")
                        })
                })
                .collect::<Result<BTreeMap<_, _>>>()?;
            let read = read_expert(choice.expert, binding.rank, &expert_hashes, expert_bytes)?;
            let expected_expert_tensors = payload_plan
                .tensors
                .iter()
                .filter(|tensor| tensor.expert == choice.expert)
                .collect::<Vec<_>>();
            let read_plan = read.plan();
            let tensors_match = expected_expert_tensors.len() == read_plan.tensors.len()
                && expected_expert_tensors
                    .iter()
                    .zip(&read_plan.tensors)
                    .all(|(expected, actual)| **expected == *actual);
            if read_plan.model_revision != payload_plan.model_revision
                || read_plan.namespace != payload_plan.namespace
                || read_plan.layer != payload_plan.layer
                || read_plan.experts.as_slice() != [choice.expert]
                || read_plan.source_shard != payload_plan.source_shard
                || read_plan.source_shard_bytes != payload_plan.source_shard_bytes
                || read_plan.header_bytes != payload_plan.header_bytes
                || read_plan.data_start_bytes != payload_plan.data_start_bytes
                || read_plan.selected_payload_bytes != expert_bytes
                || !tensors_match
            {
                return Err(invalid(
                    "DeepSeek selected indexed expert read differs from its route-level source plan",
                ));
            }
            if read.stamp() != &self.problem.stamp
                || read.route_experts() != payload_plan.experts
                || read.route_fingerprint() != route_fingerprint
                || read.attempt() != self.attempt
                || read.expert() != choice.expert
                || read.rank() != binding.rank
                || read.local() != binding.local
            {
                return Err(invalid(
                    "DeepSeek selected payload read differs from its exact dispatch binding",
                ));
            }
            reads.push(read);
        }
        if reads.len() != TOP_K {
            return Err(invalid(
                "DeepSeek route payload read did not complete all six selected experts",
            ));
        }
        Ok(DeepSeekV41RoutePayloadReads {
            stamp: self.problem.stamp.clone(),
            attempt: self.attempt,
            route_fingerprint,
            payload_plan: verified_plan.payload_plan,
            reads,
        })
    }

    /// Build the existing one-token MXFP4 Session input without changing routes,
    /// coefficients, stamp, or attempt. The Session wire accepts exact BF16 values;
    /// callers must supply those bits explicitly rather than relying on a cast here.
    pub fn into_mxfp4_session_batch(
        &self,
        inputs: Vec<crate::session::mxfp4::TokenInput>,
    ) -> Result<crate::session::mxfp4::Batch> {
        if self.attempt == 0
            || self.problem.routes.len() != 1
            || inputs.len() != 1
            || inputs[0].token != 0
            || inputs[0].value_bits.len() != self.problem.hidden
        {
            return Err(invalid(
                "DeepSeek MXFP4 Session requires one routed token and one full-width activation",
            ));
        }
        for &bits in &inputs[0].value_bits {
            let value = f32::from_bits(bits);
            if crate::deepseek_v41::cast_bf16_rne_reference(value)?.to_bits() != bits {
                return Err(invalid(
                    "DeepSeek MXFP4 Session activation must already be exactly BF16",
                ));
            }
        }
        Ok(crate::session::mxfp4::Batch {
            problem: self.problem.clone(),
            attempt: self.attempt,
            inputs,
            faults: BTreeMap::new(),
        })
    }

    /// Bind selectively-read MXFP4 experts to the exact route produced by this gate
    /// batch. Callers do not re-create a `RoutingProblem` or restate the attempt.
    pub fn preflight_mxfp4_residents(
        &self,
        residents: &[crate::deepseek_v41::PinnedMxfp4RankResidentExpert],
        max_total_artifact_bytes: u64,
    ) -> Result<crate::deepseek_v41::PinnedMxfp4RoutePreflight> {
        crate::deepseek_v41::preflight_pinned_mxfp4_route_residents(
            &self.problem,
            self.attempt,
            residents,
            max_total_artifact_bytes,
        )
    }

    /// Group those same route-bound residents according to the gate batch's exact
    /// rank/local dispatch. Extra resident locals and changed route receipts refuse.
    pub fn group_mxfp4_residents(
        &self,
        preflight: crate::deepseek_v41::PinnedMxfp4RoutePreflight,
        residents: Vec<crate::deepseek_v41::PinnedMxfp4RankResidentExpert>,
        max_peak_bytes: u64,
    ) -> Result<(
        crate::deepseek_v41::PinnedMxfp4RoutePreflight,
        Vec<crate::deepseek_v41::PinnedMxfp4RankResidentGroup>,
    )> {
        crate::deepseek_v41::group_pinned_mxfp4_route_residents(
            &self.problem,
            preflight,
            residents,
            max_peak_bytes,
        )
    }

    /// Execute a consumed gate route through grouped rank-local MXFP4 Sessions.
    /// The method derives its route and attempt from this private gate receipt.
    pub fn execute_grouped_mxfp4(
        self,
        preflight: crate::deepseek_v41::PinnedMxfp4RoutePreflight,
        inputs: Vec<crate::session::mxfp4::TokenInput>,
        load_receipts: Vec<crate::deepseek_v41::PinnedMxfp4RouteRankGroupLoadReceipt>,
        sessions: &mut BTreeMap<u32, crate::session::Session>,
        timeout: std::time::Duration,
    ) -> Result<crate::deepseek_v41::PinnedMxfp4RouteExecutionReceipt> {
        crate::deepseek_v41::execute_pinned_mxfp4_route_grouped(
            preflight,
            &self.problem,
            self.attempt,
            inputs,
            load_receipts,
            sessions,
            timeout,
        )
    }

    /// Bind this validated route to VoltaFabric's existing route-aware memory planner.
    /// This returns advisory placement only: it does not load, move, or dispatch weights.
    /// The stored attempt and gate coefficients pass through unchanged.
    pub fn plan_memory_advisory(
        &self,
        base: &crate::planner::PlacementProblem,
        source_rank: u32,
        envelopes: &[crate::planner::PerRankMemoryEnvelope],
    ) -> Result<crate::planner::RoutedPlacementMemoryAdvice> {
        crate::planner::plan_routing_advisory_with_memory(
            base,
            &self.problem,
            self.attempt,
            source_rank,
            envelopes,
        )
    }
}

/// Resident CPU projection weights and the pinned selector. This deliberately
/// loads only layer gate weight and biases, never the whole Safetensors shard.
#[derive(Debug)]
pub struct DeepSeekV41Gate {
    weight_bf16: Box<[u16]>,
    router: RouterBiases,
    receipt: GateLoadReceipt,
}

fn validate_gate_spec(spec: &GateShardSpec) -> Result<()> {
    if spec.model_revision != PINNED_MODEL_REVISION {
        return Err(invalid("unsupported DeepSeek V4.1 Flash model revision"));
    }
    if !spec.gate_temperature.is_finite()
        || spec.gate_temperature <= 0.0
        || !spec.route_scale.is_finite()
        || spec.route_scale <= 0.0
    {
        return Err(invalid(
            "DeepSeek gate configuration is not finite and positive",
        ));
    }
    if spec.max_payload_bytes < GATE_PAYLOAD_BYTES
        || spec.max_payload_bytes > checkpoint::MAX_NAMED_TENSOR_PAYLOAD_BYTES
    {
        return Err(Error::new(
            "ARTIFACT_BUDGET",
            "DeepSeek gate payload ceiling is zero, too small, or exceeds the bounded importer",
        ));
    }
    if [
        spec.tensor_sha256.weight.as_str(),
        spec.tensor_sha256.text_bias.as_str(),
        spec.tensor_sha256.image_bias.as_str(),
    ]
    .iter()
    .any(|digest| !crate::sha256::is_hex_digest(digest))
    {
        return Err(invalid(
            "DeepSeek gate payload hashes must be three complete SHA-256 digests",
        ));
    }
    Ok(())
}

fn gate_tensor_specs(spec: &GateShardSpec) -> [NamedTensorSpec; 3] {
    let prefix = format!("layers.{}.ffn.gate", spec.layer);
    [
        NamedTensorSpec {
            name: format!("{prefix}.weight"),
            dtype: Dtype::BF16,
            shape: vec![ROUTED_EXPERTS, HIDDEN_SIZE],
            bytes: GATE_WEIGHT_BYTES,
            sha256: spec.tensor_sha256.weight.clone(),
        },
        NamedTensorSpec {
            name: format!("{prefix}.bias"),
            dtype: Dtype::F32,
            shape: vec![ROUTED_EXPERTS],
            bytes: GATE_BIAS_BYTES,
            sha256: spec.tensor_sha256.text_bias.clone(),
        },
        NamedTensorSpec {
            name: format!("{prefix}.bias_vl"),
            dtype: Dtype::F32,
            shape: vec![ROUTED_EXPERTS],
            bytes: GATE_BIAS_BYTES,
            sha256: spec.tensor_sha256.image_bias.clone(),
        },
    ]
}

impl DeepSeekV41Gate {
    pub fn load<R: Read + Seek>(reader: &mut R, spec: &GateShardSpec) -> Result<Self> {
        validate_gate_spec(spec)?;
        let imported = checkpoint::read_named_tensors(
            reader,
            &gate_tensor_specs(spec),
            spec.max_payload_bytes,
        )?;
        Self::from_imported(imported, spec)
    }

    /// Load the same resident gate projection through the pinned Safetensors
    /// index. All three tensors must resolve to one shard; its header is checked
    /// before and after payload import, and the index and payloads are hashed.
    pub fn load_indexed(
        root: &Path,
        source: PinnedGateIndexSource<'_>,
        spec: &GateShardSpec,
    ) -> Result<Self> {
        Self::load_indexed_expected_shard(root, source, spec, None)
    }

    fn load_indexed_expected_shard(
        root: &Path,
        source: PinnedGateIndexSource<'_>,
        spec: &GateShardSpec,
        expected_source_shard: Option<&str>,
    ) -> Result<Self> {
        validate_gate_spec(spec)?;
        if source.max_source_shard_bytes == 0
            || !crate::sha256::is_hex_digest(source.expected_header_sha256)
        {
            return Err(invalid(
                "DeepSeek gate source needs a positive byte ceiling and valid header digest",
            ));
        }
        let selected = gate_tensor_specs(spec);
        let selected_names = selected
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect::<Vec<_>>();
        let planned = checkpoint::plan_indexed_safetensors_shards(
            source.index_bytes,
            crate::deepseek_v41::PINNED_EXPERT_INDEX_SHA256,
            &selected_names,
        )?;
        if planned.tensors_by_shard.len() != 1 || planned.tensor_to_shard.len() != selected.len() {
            return Err(invalid(
                "pinned DeepSeek gate tensors must resolve completely to one source shard",
            ));
        }
        let (shard_name, shard_tensors) = planned
            .tensors_by_shard
            .iter()
            .next()
            .ok_or_else(|| invalid("pinned DeepSeek gate index selected no shard"))?;
        if expected_source_shard.is_some_and(|expected| expected != shard_name.as_str()) {
            return Err(invalid(
                "pinned gate and routed expert tensors must share one source shard",
            ));
        }
        let expected_names = selected_names
            .iter()
            .collect::<std::collections::BTreeSet<_>>();
        let actual_names = shard_tensors
            .iter()
            .collect::<std::collections::BTreeSet<_>>();
        if shard_tensors.len() != selected.len() || actual_names != expected_names {
            return Err(invalid(
                "pinned DeepSeek gate index differs from the exact three gate tensors",
            ));
        }
        let header = checkpoint::read_indexed_safetensors_shard_header(
            root,
            shard_name,
            source.max_source_shard_bytes,
        )?;
        if header.sha256 != source.expected_header_sha256 {
            return Err(invalid(
                "pinned DeepSeek gate source shard header digest differs",
            ));
        }
        let indexed = checkpoint::read_indexed_named_tensors(
            root,
            source.index_bytes,
            crate::deepseek_v41::PINNED_EXPERT_INDEX_SHA256,
            &selected,
            spec.max_payload_bytes,
        )?;
        if indexed.plan != planned
            || indexed.file_bytes != header.file_bytes
            || indexed.header_bytes
                != u64::try_from(header.header_bytes.len())
                    .map_err(|_| invalid("DeepSeek gate header length does not fit u64"))?
            || indexed.tensors.len() != selected.len()
        {
            return Err(invalid(
                "indexed DeepSeek gate payload import differs from its preflight plan",
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
                "DeepSeek gate shard changed during indexed payload import",
            ));
        }
        Self::from_imported(
            checkpoint::NamedTensorImport {
                file_bytes: indexed.file_bytes,
                header_bytes: indexed.header_bytes,
                io_bytes: indexed.io_bytes,
                tensors: indexed.tensors,
            },
            spec,
        )
    }

    fn from_imported(
        imported: checkpoint::NamedTensorImport,
        spec: &GateShardSpec,
    ) -> Result<Self> {
        validate_gate_spec(spec)?;
        if imported.tensors.len() != 3 {
            return Err(invalid(
                "DeepSeek gate import does not contain exactly three selected tensors",
            ));
        }
        let file_bytes = imported.file_bytes;
        let header_bytes = imported.header_bytes;
        let io_bytes = imported.io_bytes;
        let tensors = imported
            .tensors
            .iter()
            .map(|(name, tensor)| GateTensorReceipt {
                name: name.clone(),
                sha256: tensor.sha256.clone(),
                bytes: tensor.bytes.len() as u64,
            })
            .collect::<Vec<_>>();

        let mut selected = imported.tensors;
        let weight_name = format!("layers.{}.ffn.gate.weight", spec.layer);
        let text_name = format!("layers.{}.ffn.gate.bias", spec.layer);
        let image_name = format!("layers.{}.ffn.gate.bias_vl", spec.layer);
        let weight = selected
            .remove(&weight_name)
            .ok_or_else(|| invalid("gate weight absent after complete shard preflight"))?;
        let text_bias = decode_f32(
            &selected
                .remove(&text_name)
                .ok_or_else(|| invalid("text gate bias absent after complete shard preflight"))?
                .bytes,
        )?;
        let image_bias = decode_f32(
            &selected
                .remove(&image_name)
                .ok_or_else(|| invalid("image gate bias absent after complete shard preflight"))?
                .bytes,
        )?;
        let mut weight_bf16 = Vec::with_capacity(ROUTED_EXPERTS * HIDDEN_SIZE);
        for bytes in weight.bytes.chunks_exact(2) {
            let value = u16::from_le_bytes([bytes[0], bytes[1]]);
            if !f32::from_bits(u32::from(value) << 16).is_finite() {
                return Err(Error::new(
                    "NONFINITE_INPUT",
                    "gate weight contains NaN or infinity",
                ));
            }
            weight_bf16.push(value);
        }
        if weight_bf16.len() != ROUTED_EXPERTS * HIDDEN_SIZE {
            return Err(invalid("gate weight element count differs after decode"));
        }
        let router = RouterBiases::new_with_temperature(
            text_bias,
            image_bias,
            spec.gate_temperature,
            spec.route_scale,
        )?;
        let receipt = GateLoadReceipt {
            model_revision: spec.model_revision.clone(),
            layer: spec.layer,
            file_bytes,
            header_bytes,
            selected_payload_bytes: GATE_PAYLOAD_BYTES,
            io_bytes,
            tensors,
        };
        Ok(Self {
            weight_bf16: weight_bf16.into_boxed_slice(),
            router,
            receipt,
        })
    }

    pub fn receipt(&self) -> &GateLoadReceipt {
        &self.receipt
    }

    /// Scalar FP32 CPU projection followed by the same top-k selector used by
    /// R700. This is an explicit reference order, not a claim of PyTorch kernel
    /// bitwise parity; the caller owns the hidden-state identity and dtype.
    pub fn route(&self, hidden: &[f32], kind: TokenKind) -> Result<Vec<Choice>> {
        let logits = self.project(hidden)?;
        self.router.route(&logits, kind)
    }

    /// Bind one model-gated token to the validated VoltaFabric routing boundary.
    /// The gate's normalized and scaled coefficients are not recomputed here.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_route_batch(
        &self,
        hidden: &[f32],
        kind: TokenKind,
        stamp: crate::moe::Stamp,
        attempt: u64,
        registry: crate::moe::Registry,
        preferred_owners: BTreeMap<u32, u32>,
        resident_order: BTreeMap<u32, Vec<u32>>,
    ) -> Result<DeepSeekV41RouteBatch> {
        if attempt == 0
            || stamp.model_revision != self.receipt.model_revision
            || stamp.layer != self.receipt.layer
        {
            return Err(invalid(
                "DeepSeek route attempt, model revision or layer identity differs",
            ));
        }
        let route = self.route(hidden, kind)?;
        if route.len() != TOP_K {
            return Err(invalid("DeepSeek gate did not produce the pinned top-k"));
        }
        let experts = route.iter().map(|choice| choice.expert).collect();
        let problem = crate::routing::RoutingProblem {
            hidden: HIDDEN_SIZE,
            experts,
            routes: vec![route],
            registry,
            stamp,
            preferred_owners,
            resident_order,
        };
        crate::routing::validate(&problem)?;
        Ok(DeepSeekV41RouteBatch { problem, attempt })
    }

    pub fn project(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        if hidden.len() != HIDDEN_SIZE {
            return Err(invalid("DeepSeek gate hidden state has wrong width"));
        }
        if hidden.iter().any(|value| !value.is_finite()) {
            return Err(Error::new(
                "NONFINITE_INPUT",
                "gate hidden state contains NaN or infinity",
            ));
        }
        let mut logits = Vec::with_capacity(ROUTED_EXPERTS);
        for row in self.weight_bf16.chunks_exact(HIDDEN_SIZE) {
            let mut sum = 0.0f32;
            for (&encoded, &input) in row.iter().zip(hidden) {
                let weight = f32::from_bits(u32::from(encoded) << 16);
                sum += weight * input;
            }
            if !sum.is_finite() {
                return Err(Error::new(
                    "NUMERICAL_OVERFLOW",
                    "gate projection became nonfinite",
                ));
            }
            logits.push(sum);
        }
        Ok(logits)
    }
}

/// Resident, indexed MoE sublayer for a pinned DeepSeek V4.1 backbone layer.
/// It keeps the gate and expert reads bound to the same checkpoint index, header,
/// shard name, and source-size ceiling across successive token requests.
#[derive(Debug)]
pub struct DeepSeekV41IndexedMoeLayer<'a> {
    root: &'a Path,
    gate: DeepSeekV41Gate,
    expert_source: PinnedMxfp4PayloadSource<'a>,
    ffn_norm: Option<DeepSeekV41FfnRmsNorm>,
    shared_expert: Option<ResidentSharedExpert>,
}

impl<'a> DeepSeekV41IndexedMoeLayer<'a> {
    /// Load only the gate tensors and prove from the pinned index that gate and
    /// expert tensors share the caller-admitted source shard before opening it.
    pub fn load(
        root: &'a Path,
        gate_source: PinnedGateIndexSource<'a>,
        expert_source: PinnedMxfp4PayloadSource<'a>,
        spec: &GateShardSpec,
    ) -> Result<Self> {
        if gate_source.index_bytes != expert_source.index_bytes
            || gate_source.expected_header_sha256 != expert_source.expected_header_sha256
            || gate_source.max_source_shard_bytes != expert_source.max_source_shard_bytes
            || !crate::sha256::is_hex_digest(expert_source.expected_header_sha256)
            || crate::sha256::hex(expert_source.index_bytes)
                != crate::deepseek_v41::PINNED_EXPERT_INDEX_SHA256
            || crate::sha256::hex(expert_source.header_bytes)
                != expert_source.expected_header_sha256
            || expert_source.source_shard_bytes == 0
            || expert_source.source_shard_bytes > expert_source.max_source_shard_bytes
        {
            return Err(invalid(
                "indexed DeepSeek layer gate and expert source identities or bounds differ",
            ));
        }

        let gate = DeepSeekV41Gate::load_indexed_expected_shard(
            root,
            gate_source,
            spec,
            Some(expert_source.header_shard_name),
        )?;
        let expected_header_bytes = u64::try_from(expert_source.header_bytes.len())
            .map_err(|_| invalid("indexed DeepSeek layer header length does not fit u64"))?;
        if gate.receipt().file_bytes != expert_source.source_shard_bytes
            || gate.receipt().header_bytes != expected_header_bytes
        {
            return Err(Error::new(
                "CHECKPOINT_CHANGED",
                "indexed DeepSeek gate and expert source metadata differ",
            ));
        }
        Ok(Self {
            root,
            gate,
            expert_source,
            ffn_norm: None,
            shared_expert: None,
        })
    }

    /// Load the indexed gate/expert boundary together with this layer's resident
    /// `ffn_norm.weight`; the attention norm belongs to the preceding sublayer.
    pub fn load_with_ffn_norm(
        root: &'a Path,
        gate_source: PinnedGateIndexSource<'a>,
        expert_source: PinnedMxfp4PayloadSource<'a>,
        gate_spec: &GateShardSpec,
        norm_spec: &FfnNormTensorSpec,
    ) -> Result<Self> {
        if norm_spec.layer != gate_spec.layer
            || norm_spec.epsilon.to_bits() != PINNED_RMS_NORM_EPSILON.to_bits()
            || !crate::sha256::is_hex_digest(&norm_spec.tensor_sha256)
        {
            return Err(invalid(
                "DeepSeek gate and ffn_norm layer specifications differ",
            ));
        }
        let mut layer = Self::load(root, gate_source, expert_source, gate_spec)?;
        let ffn_norm = DeepSeekV41FfnRmsNorm::load_indexed(root, expert_source, norm_spec)?;
        if ffn_norm.layer() != layer.gate.receipt().layer
            || ffn_norm.receipt().shard_bytes != layer.gate.receipt().file_bytes
            || ffn_norm.receipt().header_bytes != layer.gate.receipt().header_bytes
        {
            return Err(Error::new(
                "CHECKPOINT_CHANGED",
                "indexed DeepSeek gate and ffn_norm imports differ in layer, shard or header",
            ));
        }
        layer.ffn_norm = Some(ffn_norm);
        Ok(layer)
    }

    /// Load the indexed routed gate, ffn_norm, and all six shared-expert FP8 planes
    /// from one pinned layer shard. The shared planes are E4M3FN/E8M0, not MXFP4.
    pub fn load_with_ffn_norm_and_shared_expert(
        root: &'a Path,
        gate_source: PinnedGateIndexSource<'a>,
        expert_source: PinnedMxfp4PayloadSource<'a>,
        gate_spec: &GateShardSpec,
        norm_spec: &FfnNormTensorSpec,
        shared_spec: &SharedExpertTensorSpec,
    ) -> Result<Self> {
        if shared_spec.layer != gate_spec.layer || shared_spec.layer != norm_spec.layer {
            return Err(invalid(
                "DeepSeek gate, ffn_norm and shared-expert layer specifications differ",
            ));
        }
        let mut layer =
            Self::load_with_ffn_norm(root, gate_source, expert_source, gate_spec, norm_spec)?;
        let shared = ResidentSharedExpert::load_indexed(root, expert_source, shared_spec)?;
        let gate_receipt = layer.gate.receipt();
        let norm_receipt = layer
            .ffn_norm
            .as_ref()
            .ok_or_else(|| invalid("indexed DeepSeek layer has no ffn_norm after load"))?
            .receipt();
        if shared.receipt.layer != gate_receipt.layer
            || shared.receipt.layer != norm_receipt.layer
            || shared.receipt.shard_bytes != gate_receipt.file_bytes
            || shared.receipt.shard_bytes != norm_receipt.shard_bytes
            || shared.receipt.header_bytes != gate_receipt.header_bytes
            || shared.receipt.header_bytes != norm_receipt.header_bytes
            || shared.receipt.header_sha256 != expert_source.expected_header_sha256
            || shared.receipt.source_shard_name != expert_source.header_shard_name
        {
            return Err(Error::new(
                "CHECKPOINT_CHANGED",
                "indexed DeepSeek gate, norm and shared-expert imports differ in layer, shard or header",
            ));
        }
        layer.shared_expert = Some(shared);
        Ok(layer)
    }

    pub fn gate_receipt(&self) -> &GateLoadReceipt {
        self.gate.receipt()
    }

    pub fn ffn_norm_receipt(&self) -> Option<&FfnNormLoadReceipt> {
        self.ffn_norm.as_ref().map(DeepSeekV41FfnRmsNorm::receipt)
    }

    pub fn shared_expert_receipt(&self) -> Option<&SharedExpertLoadReceipt> {
        self.shared_expert.as_ref().map(|shared| &shared.receipt)
    }

    /// Evaluate the resident shared path on the exact activation retained by the
    /// gate-produced token plan. The output is raw FP32 and is not route weighted.
    pub fn execute_shared_reference(
        &self,
        plan: &DeepSeekV41IndexedMoeTokenPlan,
    ) -> Result<DeepSeekV41SharedTokenOutput> {
        let shared = self
            .shared_expert
            .as_ref()
            .ok_or_else(|| invalid("indexed DeepSeek layer has no resident shared expert"))?;
        let stamp = &plan.batch.problem.stamp;
        if plan.batch.attempt == 0
            || stamp.model_revision != PINNED_MODEL_REVISION
            || stamp.layer != shared.receipt.layer
            || plan.activation.len() != HIDDEN_SIZE
            || plan.batch.problem.routes.len() != 1
            || plan.batch.problem.routes[0].len() != TOP_K
        {
            return Err(invalid(
                "DeepSeek shared reference requires one identified gate-produced top-6 token",
            ));
        }
        let output_fp32 = shared.expert.forward_reference(&plan.activation)?;
        if output_fp32.len() != HIDDEN_SIZE || output_fp32.iter().any(|value| !value.is_finite()) {
            return Err(Error::new(
                "NUMERICAL_OVERFLOW",
                "DeepSeek shared expert returned an invalid hidden row",
            ));
        }
        Ok(DeepSeekV41SharedTokenOutput {
            stamp: stamp.clone(),
            attempt: plan.batch.attempt,
            output_fp32,
        })
    }

    pub fn normalize_ffn_hidden(&self, hidden_bf16_f32: &[f32]) -> Result<Vec<f32>> {
        self.ffn_norm
            .as_ref()
            .ok_or_else(|| invalid("indexed DeepSeek layer has no resident ffn_norm weight"))?
            .apply(hidden_bf16_f32)
    }

    /// Apply the pinned post-attention `ffn_norm` once, then derive the gate route
    /// and selected expert payload plan from its BF16 output.
    pub fn prepare_ffn_token(
        &self,
        request: DeepSeekV41IndexedFfnTokenRequest<'_>,
    ) -> Result<DeepSeekV41IndexedMoeTokenPlan> {
        let normalized = self.normalize_ffn_hidden(request.hidden)?;
        self.prepare_token(DeepSeekV41IndexedTokenRequest {
            hidden: &normalized,
            token_kind: request.token_kind,
            stamp: request.stamp,
            attempt: request.attempt,
            registry: request.registry,
            preferred_owners: request.preferred_owners,
            resident_order: request.resident_order,
            max_total_selected_payload_bytes: request.max_total_selected_payload_bytes,
        })
    }

    /// Bind one post-normalization MoE activation to the route and exact selected
    /// source ranges. The route coefficients are calculated by the gate once and
    /// retained unchanged by the returned private token plan.
    pub fn prepare_token(
        &self,
        request: DeepSeekV41IndexedTokenRequest<'_>,
    ) -> Result<DeepSeekV41IndexedMoeTokenPlan> {
        let batch = self.gate.prepare_route_batch(
            request.hidden,
            request.token_kind,
            request.stamp,
            request.attempt,
            request.registry,
            request.preferred_owners,
            request.resident_order,
        )?;
        let payloads = batch.plan_selected_mxfp4_payloads(
            self.expert_source,
            request.max_total_selected_payload_bytes,
        )?;
        Ok(DeepSeekV41IndexedMoeTokenPlan {
            batch,
            payloads,
            activation: request.hidden.to_vec(),
        })
    }

    /// Read the full selected payload set for a token plan. The underlying route
    /// reader validates every tensor digest before exposing any resident owner.
    pub fn read_selected_payloads(
        &self,
        plan: &DeepSeekV41IndexedMoeTokenPlan,
        expected_payload_sha256: &BTreeMap<String, String>,
        max_total_selected_payload_bytes: u64,
    ) -> Result<DeepSeekV41RoutePayloadReads> {
        plan.batch.read_selected_mxfp4_indexed_payloads(
            self.root,
            &plan.payloads,
            self.expert_source,
            expected_payload_sha256,
            max_total_selected_payload_bytes,
        )
    }
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() != ROUTED_EXPERTS * 4 {
        return Err(invalid("gate bias payload has wrong byte length"));
    }
    let mut values = Vec::with_capacity(ROUTED_EXPERTS);
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if !value.is_finite() {
            return Err(Error::new(
                "NONFINITE_INPUT",
                "gate bias contains NaN or infinity",
            ));
        }
        values.push(value);
    }
    Ok(values)
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    expert: u32,
    score: f32,
    corrected_score: f32,
}

fn validate_vector(name: &str, values: &[f32]) -> Result<()> {
    if values.len() != ROUTED_EXPERTS {
        return Err(invalid(format!(
            "{name} must contain exactly {ROUTED_EXPERTS} FP32 values"
        )));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::new(
            "NONFINITE_INPUT",
            format!("{name} contains NaN or infinity"),
        ));
    }
    Ok(())
}

/// PyTorch `F.softplus` defaults to threshold 20; preserve that FP32 branch.
fn softplus_fp32(value: f32) -> f32 {
    if value > SOFTPLUS_THRESHOLD {
        value
    } else {
        value.exp().ln_1p()
    }
}

#[cfg(test)]
mod r730_hc_post_tests {
    use super::{
        deepseek_hc_post_reference, hc::HcProjectionBranch, DeepSeekV41HcPostCoefficients,
        DeepSeekV41MoeTokenOutput, DeepSeekV41ResidualStream, HIDDEN_SIZE, PINNED_MODEL_REVISION,
    };

    fn stamp() -> crate::moe::Stamp {
        crate::moe::Stamp {
            run_id: "r730-hc-post".to_owned(),
            model_revision: PINNED_MODEL_REVISION.to_owned(),
            plan_id: "r730-cpu-fixture".to_owned(),
            layer: 0,
            batch: 3,
            epoch: 19,
        }
    }

    fn oracle_bf16_rne(value: f64) -> f32 {
        let value = value as f32;
        let bits = value.to_bits();
        let bias = 0x7fffu32 + ((bits >> 16) & 1);
        f32::from_bits(bits.wrapping_add(bias) & 0xffff_0000)
    }

    #[test]
    fn pinned_hc_post_matches_separate_f64_formula_and_rounds_once() {
        let stamp = stamp();
        let attempt = 7;
        let x = (0..HIDDEN_SIZE)
            .map(|i| ((i % 16) as f32 - 8.0) / 8.0)
            .collect::<Vec<_>>();
        let residual = (0..4)
            .map(|copy| {
                (0..HIDDEN_SIZE)
                    .map(|i| (((i + 3 * copy) % 32) as f32 - 16.0) / 16.0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let post = vec![0.913, 1.107, -0.271, 0.333];
        let comb = vec![
            vec![0.19, -0.31, 0.77, 0.43],
            vec![0.51, 0.23, -0.14, 0.37],
            vec![-0.17, 0.62, 0.29, 0.11],
            vec![0.38, -0.21, 0.56, 0.27],
        ];
        let x = x
            .into_iter()
            .map(|v| crate::deepseek_v41::cast_bf16_rne_reference(v).unwrap())
            .collect::<Vec<_>>();
        let residual = residual
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|v| crate::deepseek_v41::cast_bf16_rne_reference(v).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let sublayer = DeepSeekV41MoeTokenOutput {
            stamp: stamp.clone(),
            attempt,
            output_bf16_f32: x.clone(),
        };
        let residual_stream =
            DeepSeekV41ResidualStream::new(stamp.clone(), attempt, residual.clone()).unwrap();
        let mixes = DeepSeekV41HcPostCoefficients::new(
            stamp.clone(),
            attempt,
            HcProjectionBranch::Ffn,
            post.clone(),
            comb.clone(),
        )
        .unwrap();

        let output = deepseek_hc_post_reference(&sublayer, &residual_stream, &mixes).unwrap();
        assert_eq!(output.stamp(), &stamp);
        assert_eq!(output.attempt(), attempt);
        assert_eq!(output.copies_bf16_f32().len(), 4);
        for destination in 0..4 {
            for hidden in 0..HIDDEN_SIZE {
                let residual_mix = (0..4).fold(0.0f64, |sum, source| {
                    sum + f64::from(comb[destination][source]) * f64::from(residual[source][hidden])
                });
                let expected = oracle_bf16_rne(
                    f64::from(post[destination]) * f64::from(x[hidden]) + residual_mix,
                );
                assert_eq!(
                    output.copies_bf16_f32()[destination][hidden].to_bits(),
                    expected.to_bits(),
                    "copy {destination}, hidden {hidden}"
                );
            }
        }
    }

    #[test]
    fn pinned_hc_post_bf16_cast_uses_ties_to_even_once() {
        let stamp = stamp();
        let mut x = vec![0.0; HIDDEN_SIZE];
        x[0] = 1.0;
        let mut residual = vec![vec![0.0; HIDDEN_SIZE]; 4];
        residual[0][0] = 1.0;
        let mut comb = vec![vec![0.0; 4]; 4];
        // 1 + 2^-8 is exactly halfway between adjacent BF16 numbers near 1.
        comb[0][0] = 1.0 / 256.0;
        let sublayer = DeepSeekV41MoeTokenOutput {
            stamp: stamp.clone(),
            attempt: 9,
            output_bf16_f32: x,
        };
        let residual = DeepSeekV41ResidualStream::new(stamp.clone(), 9, residual).unwrap();
        let coefficients = DeepSeekV41HcPostCoefficients::new(
            stamp,
            9,
            HcProjectionBranch::Ffn,
            vec![1.0; 4],
            comb,
        )
        .unwrap();

        let output = deepseek_hc_post_reference(&sublayer, &residual, &coefficients).unwrap();
        assert_eq!(output.copies_bf16_f32()[0][0].to_bits(), 1.0f32.to_bits());
    }

    #[test]
    fn pinned_hc_post_rejects_wrong_identity_shapes_and_nonfinite_values() {
        let stamp = stamp();
        let copies = vec![vec![0.0; HIDDEN_SIZE]; 4];
        let residual = DeepSeekV41ResidualStream::new(stamp.clone(), 7, copies.clone()).unwrap();
        assert!(
            DeepSeekV41ResidualStream::new(stamp.clone(), 7, vec![vec![0.0; HIDDEN_SIZE]; 3])
                .is_err()
        );
        let mut non_bf16 = copies.clone();
        non_bf16[0][0] = 0.1;
        assert!(DeepSeekV41ResidualStream::new(stamp.clone(), 7, non_bf16).is_err());
        assert!(DeepSeekV41HcPostCoefficients::new(
            stamp.clone(),
            7,
            HcProjectionBranch::Ffn,
            vec![1.0; 3],
            vec![vec![1.0; 4]; 4]
        )
        .is_err());
        assert!(DeepSeekV41HcPostCoefficients::new(
            stamp.clone(),
            7,
            HcProjectionBranch::Ffn,
            vec![f32::NAN; 4],
            vec![vec![1.0; 4]; 4]
        )
        .is_err());

        let sublayer = DeepSeekV41MoeTokenOutput {
            stamp: stamp.clone(),
            attempt: 7,
            output_bf16_f32: vec![0.0; HIDDEN_SIZE],
        };
        let mut other = stamp.clone();
        other.plan_id.push_str("-mismatch");
        let wrong_mix = DeepSeekV41HcPostCoefficients::new(
            other,
            7,
            HcProjectionBranch::Ffn,
            vec![1.0; 4],
            vec![vec![0.25; 4]; 4],
        )
        .unwrap();
        assert!(deepseek_hc_post_reference(&sublayer, &residual, &wrong_mix).is_err());
        let wrong_attempt = DeepSeekV41HcPostCoefficients::new(
            stamp,
            8,
            HcProjectionBranch::Ffn,
            vec![1.0; 4],
            vec![vec![0.25; 4]; 4],
        )
        .unwrap();
        assert!(deepseek_hc_post_reference(&sublayer, &residual, &wrong_attempt).is_err());
        let attention_coefficients = DeepSeekV41HcPostCoefficients::new(
            sublayer.stamp().clone(),
            sublayer.attempt(),
            HcProjectionBranch::Attention,
            vec![1.0; 4],
            vec![vec![0.25; 4]; 4],
        )
        .unwrap();
        assert!(deepseek_hc_post_reference(&sublayer, &residual, &attention_coefficients).is_err());
    }
}
