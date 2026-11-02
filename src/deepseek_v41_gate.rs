//! CPU reference for the pinned DeepSeek-V4.1-Flash expert-selection boundary.
//!
//! This adapts the official MIT-licensed implementation at revision
//! `314113f8a7dadb388215396ba5fbee2c610a04d6`; it does not copy upstream code.
//! Inputs are the 384 FP32 outputs of the router linear projection, before the
//! model's gate temperature. The projection itself and loading of its checkpoint
//! tensors remain outside this module. The pinned defaults are temperature 1.0,
//! sqrt-softplus scoring, top-6, normalized selected weights, and route scale 1.5.
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
use crate::moe::Choice;
use crate::{invalid, Error, Result};

pub const ROUTED_EXPERTS: usize = 384;
pub const TOP_K: usize = 6;
pub const GATE_TEMPERATURE: f32 = 1.0;
pub const ROUTE_SCALE: f32 = 1.5;
const SOFTPLUS_THRESHOLD: f32 = 20.0;
const ROUTE_EPSILON: f32 = 1.0e-20;

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
}

impl RouterBiases {
    pub fn new(text: Vec<f32>, image: Vec<f32>) -> Result<Self> {
        validate_vector("text correction bias", &text)?;
        validate_vector("image correction bias", &image)?;
        Ok(Self {
            text: text.into_boxed_slice(),
            image: image.into_boxed_slice(),
        })
    }

    /// Route one token from pre-temperature FP32 router logits.
    ///
    /// Returned choices are in descending corrected-score order, matching the
    /// pinned `topk(..., sorted=True)` boundary when no relevant ties exist. Each
    /// coefficient already includes normalization and the pinned route scale; a
    /// downstream executor must apply it exactly once.
    pub fn route(&self, router_logits: &[f32], kind: TokenKind) -> Result<Vec<Choice>> {
        validate_vector("router logits", router_logits)?;

        let correction = match kind {
            TokenKind::Text => &self.text,
            TokenKind::Image => &self.image,
        };
        let mut candidates = Vec::with_capacity(ROUTED_EXPERTS);
        for (expert, (&logit, &bias)) in router_logits.iter().zip(correction.iter()).enumerate() {
            let temperature_scaled = logit / GATE_TEMPERATURE;
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
                let weighted = normalized * ROUTE_SCALE;
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
