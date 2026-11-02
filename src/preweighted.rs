//! Explicit receipt boundary for the R9 laboratory's already-weighted FP32 rows.
//!
//! R9 applies the route coefficient before BF16 rounding and E4M3/UE8M0 packing,
//! then computes W2. Moving that coefficient to the collector is not equivalent.
//! This module preserves those rows and sums in original token/slot order with
//! one binary32 round-to-nearest-even addition at each step, starting at +0.
//! It does not claim this sum order matches an upstream distributed kernel.
//!
//! The digest binds metadata, not authenticity or proof of GPU execution. The
//! producer must separately establish artifact ownership, synchronization, and
//! terminal success. No session capability or retry policy is added here.
use crate::{moe, routing, sha256, Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Format {
    #[serde(rename = "r9_mxfp4_preweighted_fp32_slot_sum_v1")]
    R9Mxfp4PreweightedFp32V1,
    #[serde(rename = "qwen2moe_ffn_postdown_preweighted_fp32_slot_sum_v1")]
    Qwen2MoEFfnPostDownPreweightedFp32V1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Contract {
    pub format: Format,
    pub sha256: String,
}

/// Wire-neutral FP32 bit patterns; no FP16/BF16 reinterpretation or JSON float cast.
/// Coefficients remain part of the original routing plan, never replaced by one.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Completion {
    pub contract: Contract,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempt: Option<u64>,
    pub stamp: moe::Stamp,
    pub token: usize,
    pub slot: usize,
    pub expert: u32,
    pub rank: u32,
    pub local: u32,
    pub weight_bits: u32,
    pub value_bits: Vec<u32>,
}

/// A separate type prevents accidental use of the unweighted `moe::Completion`.
/// The existing collector owns identity, bounded storage and terminal state.
pub struct Collector {
    inner: moe::Collector,
    contract: Contract,
    attempt: Option<u64>,
}

impl Collector {
    pub fn new(problem: &routing::RoutingProblem) -> Result<Self> {
        Self::new_with_scope(problem, Format::R9Mxfp4PreweightedFp32V1, None)
    }

    /// Collector for Qwen2MoE's FP32 post-down route products. The attempt is
    /// included in the arithmetic-contract digest and every completion must carry it.
    pub fn new_qwen2moe_ffn_postdown_fp32(
        problem: &routing::RoutingProblem,
        attempt: u64,
    ) -> Result<Self> {
        if attempt == 0 {
            return Err(Error::new(
                "INVALID_INPUT",
                "Qwen2MoE attempt must be nonzero",
            ));
        }
        Self::new_with_scope(
            problem,
            Format::Qwen2MoEFfnPostDownPreweightedFp32V1,
            Some(attempt),
        )
    }

    fn new_with_scope(
        problem: &routing::RoutingProblem,
        format: Format,
        attempt: Option<u64>,
    ) -> Result<Self> {
        // Use the existing admission/memory contract before building the digest.
        let inner = moe::Collector::from_routing(problem)?;
        for route in &problem.routes {
            for choice in route {
                let fp32 = choice.weight as f32;
                if !fp32.is_finite() || f64::from(fp32).to_bits() != choice.weight.to_bits() {
                    return Err(Error::new(
                        "COEFFICIENT_NOT_FP32",
                        "preweighted R9 requires the original coefficient to be exactly binary32",
                    ));
                }
            }
        }
        let dispatch = routing::dispatch(problem)?;
        // Structured JSON is unambiguous (no concatenated identifiers); dispatch
        // canonicalizes metadata order while retaining physical resident order.
        let encoded = match attempt {
            None => serde_json::to_vec(&("VoltaFabric/preweighted/1", format, &dispatch)),
            Some(attempt) => {
                serde_json::to_vec(&("VoltaFabric/preweighted/2", format, &dispatch, attempt))
            }
        }
        .map_err(|error| Error::new("CONTRACT_ENCODING", error.to_string()))?;
        Ok(Self {
            inner,
            contract: Contract {
                format,
                sha256: sha256::hex(&encoded),
            },
            attempt,
        })
    }

    pub fn contract(&self) -> &Contract {
        &self.contract
    }

    pub fn accept(&mut self, completion: Completion) -> Result<()> {
        if completion.contract != self.contract || completion.attempt != self.attempt {
            self.inner.expire();
            return Err(Error::new(
                "ARITHMETIC_CONTRACT_MISMATCH",
                "preweighted format, routing contract or attempt differs; entire batch aborted",
            ));
        }
        self.inner.accept_preweighted_fp32(completion)
    }

    pub fn expire(&mut self) {
        self.inner.expire();
    }

    pub fn finish(self) -> Result<Vec<Vec<f32>>> {
        self.inner.finish_preweighted_fp32()
    }
}
