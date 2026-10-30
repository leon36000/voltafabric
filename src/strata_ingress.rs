//! CPU admission of a complete Strata-style pool batch into VoltaFabric routing.
//!
//! The multi-token Strata callback does not expose coefficients; a caller must
//! supply the actual gate weights for the same layer and batch. This module
//! neither hooks Strata nor reserves a session, starts a worker, or assigns an
//! attempt. It refuses the -1 hit sentinel. The FP16 path refuses inexact
//! conversion; the separate FP64 path preserves finite FP32 activations.
use crate::{completion, invalid, moe, routing, Result};

#[derive(Debug, Clone)]
pub struct AdmittedPoolBatch {
    pub routing: routing::RoutingProblem,
    pub inputs_fp16_bits: Vec<Vec<u16>>,
}

#[derive(Debug, Clone)]
pub struct AdmittedF64PoolBatch {
    pub routing: routing::RoutingProblem,
    pub inputs: Vec<Vec<f64>>,
}

/// Validate the complete batch before returning data to a Session caller.
#[allow(clippy::too_many_arguments)]
pub fn admit_pool_batch(
    template: &routing::RoutingProblem,
    expected_stamp: &moe::Stamp,
    layer: u32,
    tokens: usize,
    top_k: usize,
    inputs: &[f32],
    ids: &[i32],
    weights: &[f32],
) -> Result<AdmittedPoolBatch> {
    let routing = validate_pool(
        template,
        expected_stamp,
        layer,
        tokens,
        top_k,
        inputs,
        ids,
        weights,
    )?;
    let mut encoded = Vec::with_capacity(tokens);
    for row in inputs.chunks_exact(template.hidden) {
        let mut bits = Vec::with_capacity(template.hidden);
        for &value in row {
            let half = completion::encode_fp16(f64::from(value))?;
            let decoded = completion::decode_fp16(half)? as f32;
            if decoded.to_bits() != value.to_bits() {
                return Err(invalid(
                    "pool activation is not exactly representable in FP16",
                ));
            }
            bits.push(half);
        }
        encoded.push(bits);
    }
    Ok(AdmittedPoolBatch {
        routing,
        inputs_fp16_bits: encoded,
    })
}

/// Admit finite FP32 activations exactly for the CPU FP64 Session contract.
#[allow(clippy::too_many_arguments)]
pub fn admit_pool_batch_f64(
    template: &routing::RoutingProblem,
    expected_stamp: &moe::Stamp,
    layer: u32,
    tokens: usize,
    top_k: usize,
    inputs: &[f32],
    ids: &[i32],
    weights: &[f32],
) -> Result<AdmittedF64PoolBatch> {
    let routing = validate_pool(
        template,
        expected_stamp,
        layer,
        tokens,
        top_k,
        inputs,
        ids,
        weights,
    )?;
    if inputs.iter().any(|value| !value.is_finite()) {
        return Err(invalid("non-finite pool activation"));
    }
    let inputs = inputs
        .chunks_exact(template.hidden)
        .map(|row| row.iter().map(|&value| f64::from(value)).collect())
        .collect();
    Ok(AdmittedF64PoolBatch { routing, inputs })
}

#[allow(clippy::too_many_arguments)]
fn validate_pool(
    template: &routing::RoutingProblem,
    expected_stamp: &moe::Stamp,
    layer: u32,
    tokens: usize,
    top_k: usize,
    inputs: &[f32],
    ids: &[i32],
    weights: &[f32],
) -> Result<routing::RoutingProblem> {
    routing::validate(template)?;
    if template.stamp != *expected_stamp || layer != expected_stamp.layer {
        return Err(invalid("stale pool identity or layer"));
    }
    if tokens == 0 || tokens > 4096 || top_k == 0 || top_k > 64 {
        return Err(invalid("pool dimensions outside contract"));
    }
    let activations = tokens
        .checked_mul(template.hidden)
        .filter(|&n| n <= 16_777_216)
        .ok_or_else(|| invalid("pool activation count exceeds bound"))?;
    let slots = tokens
        .checked_mul(top_k)
        .filter(|&n| n <= 262_144)
        .ok_or_else(|| invalid("pool slot count exceeds bound"))?;
    if inputs.len() != activations || ids.len() != slots || weights.len() != slots {
        return Err(invalid("pool input shape mismatch"));
    }
    let mut routing = template.clone();
    routing.routes.clear();
    for (id_row, weight_row) in ids.chunks_exact(top_k).zip(weights.chunks_exact(top_k)) {
        let mut route = Vec::with_capacity(top_k);
        for (&id, &weight) in id_row.iter().zip(weight_row) {
            let expert = u32::try_from(id).map_err(|_| invalid("negative pool expert ID"))?;
            route.push(moe::Choice {
                expert,
                weight: f64::from(weight),
            });
        }
        routing.routes.push(route);
    }
    routing::validate(&routing)?;
    Ok(routing)
}
