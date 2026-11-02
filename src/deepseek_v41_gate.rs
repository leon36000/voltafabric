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
            let read = crate::deepseek_v41::read_pinned_mxfp4_rank_expert(
                reader,
                crate::deepseek_v41::PinnedMxfp4RankExpertReadRequest {
                    route: &self.problem,
                    attempt: self.attempt,
                    expert: choice.expert,
                    rank: binding.rank,
                    index_bytes: source.index_bytes,
                    header_shard_name: source.header_shard_name,
                    expected_header_sha256: source.expected_header_sha256,
                    expected_payload_sha256: &expert_hashes,
                    max_source_shard_bytes: source.max_source_shard_bytes,
                    max_payload_bytes: expert_bytes,
                },
            )?;
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
