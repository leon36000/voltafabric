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
    collections::BTreeMap,
    io::{Read, Seek},
};

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

/// A one-token gate decision bound to VoltaFabric's route identity and attempt.
/// The route problem is private so the validated stamp cannot be replaced after
/// construction; its already-scaled coefficients are carried through unchanged.
#[derive(Debug, Clone)]
pub struct DeepSeekV41RouteBatch {
    problem: crate::routing::RoutingProblem,
    attempt: u64,
}

impl DeepSeekV41RouteBatch {
    pub fn problem(&self) -> &crate::routing::RoutingProblem {
        &self.problem
    }

    pub fn attempt(&self) -> u64 {
        self.attempt
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

impl DeepSeekV41Gate {
    pub fn load<R: Read + Seek>(reader: &mut R, spec: &GateShardSpec) -> Result<Self> {
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
        let prefix = format!("layers.{}.ffn.gate", spec.layer);
        let weight_name = format!("{prefix}.weight");
        let text_name = format!("{prefix}.bias");
        let image_name = format!("{prefix}.bias_vl");
        let imported = checkpoint::read_named_tensors(
            reader,
            &[
                NamedTensorSpec {
                    name: weight_name.clone(),
                    dtype: Dtype::BF16,
                    shape: vec![ROUTED_EXPERTS, HIDDEN_SIZE],
                    bytes: GATE_WEIGHT_BYTES,
                    sha256: spec.tensor_sha256.weight.clone(),
                },
                NamedTensorSpec {
                    name: text_name.clone(),
                    dtype: Dtype::F32,
                    shape: vec![ROUTED_EXPERTS],
                    bytes: GATE_BIAS_BYTES,
                    sha256: spec.tensor_sha256.text_bias.clone(),
                },
                NamedTensorSpec {
                    name: image_name.clone(),
                    dtype: Dtype::F32,
                    shape: vec![ROUTED_EXPERTS],
                    bytes: GATE_BIAS_BYTES,
                    sha256: spec.tensor_sha256.image_bias.clone(),
                },
            ],
            spec.max_payload_bytes,
        )?;
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
