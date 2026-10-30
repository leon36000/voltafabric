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

/// A caller-supplied ownership decision; the caller must bind it to the actual
/// donor plan and must not concurrently write the same destination rows.
pub struct MixedPoolBatch {
    misses: Option<AdmittedF64PoolBatch>,
    positions: Vec<usize>,
    output_values: usize,
    hidden: usize,
}

impl MixedPoolBatch {
    pub fn misses(&self) -> Option<&AdmittedF64PoolBatch> {
        self.misses.as_ref()
    }

    /// Only miss rows are written; externally owned rows are left untouched.
    /// All converted values are staged before any destination row is written.
    pub fn write_misses(&self, rows: &[Vec<Vec<f64>>], output: &mut [f32]) -> Result<()> {
        if output.len() != self.output_values
            || rows.len() != self.positions.len()
            || rows
                .iter()
                .any(|token| token.len() != 1 || token[0].len() != self.hidden)
        {
            return Err(invalid("mixed pool result shape mismatch"));
        }
        let mut values = Vec::with_capacity(self.positions.len() * self.hidden);
        for &value in rows.iter().flatten().flatten() {
            let rounded = value as f32;
            if !value.is_finite() || !rounded.is_finite() {
                return Err(invalid("mixed pool result is non-finite or overflows FP32"));
            }
            values.push(rounded);
        }
        for (i, &position) in self.positions.iter().enumerate() {
            output[position * self.hidden..(position + 1) * self.hidden]
                .copy_from_slice(&values[i * self.hidden..(i + 1) * self.hidden]);
        }
        Ok(())
    }
}

/// Flatten non-external routes into one-slot tokens. The ownership mask is not
/// evidence of a real GPU plan: that must be verified by the integrating caller.
#[allow(clippy::too_many_arguments)]
pub fn admit_mixed_pool_f64(
    template: &routing::RoutingProblem,
    expected_stamp: &moe::Stamp,
    layer: u32,
    tokens: usize,
    top_k: usize,
    inputs: &[f32],
    ids: &[i32],
    weights: &[f32],
    externally_owned: &[bool],
) -> Result<MixedPoolBatch> {
    let full = validate_pool(
        template,
        expected_stamp,
        layer,
        tokens,
        top_k,
        inputs,
        ids,
        weights,
    )?;
    if externally_owned.len() != ids.len() || inputs.iter().any(|x| !x.is_finite()) {
        return Err(invalid("mixed pool mask or activation invalid"));
    }
    let output_values = ids
        .len()
        .checked_mul(template.hidden)
        .filter(|&n| n <= 16_777_216)
        .ok_or_else(|| invalid("mixed pool output exceeds bound"))?;
    let positions: Vec<usize> = externally_owned
        .iter()
        .enumerate()
        .filter_map(|(i, &external)| (!external).then_some(i))
        .collect();
    if positions.len() > 4096
        || positions
            .len()
            .checked_mul(template.hidden)
            .is_none_or(|n| n > 16_777_216)
    {
        return Err(invalid("mixed pool misses exceed bounded session capacity"));
    }
    let misses = if positions.is_empty() {
        None
    } else {
        let mut compact = full.clone();
        compact.routes = positions
            .iter()
            .map(|&i| vec![full.routes[i / top_k][i % top_k].clone()])
            .collect();
        routing::validate(&compact)?;
        let rows = positions
            .iter()
            .map(|&i| {
                let start = (i / top_k) * template.hidden;
                inputs[start..start + template.hidden]
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect()
            })
            .collect();
        Some(AdmittedF64PoolBatch {
            routing: compact,
            inputs: rows,
        })
    };
    Ok(MixedPoolBatch {
        misses,
        positions,
        output_values,
        hidden: template.hidden,
    })
}

/// Convert complete, unweighted Session results to a token/slot-major FP32
/// buffer. Conversion rounds to nearest FP32, rejects non-finite or overflowing
/// values, and commits only after every shape and value has been checked. The
/// caller is responsible for applying each routing coefficient once.
pub fn write_unweighted_f32(
    routing: &routing::RoutingProblem,
    rows: &[Vec<Vec<f64>>],
    output: &mut [f32],
) -> Result<()> {
    routing::validate(routing)?;
    let tokens = routing.routes.len();
    let top_k = routing.routes.first().map_or(0, Vec::len);
    if tokens == 0 || top_k == 0 {
        return Err(invalid("empty pool output"));
    }
    let count = tokens
        .checked_mul(top_k)
        .and_then(|n| n.checked_mul(routing.hidden))
        .filter(|&n| n <= 16_777_216)
        .ok_or_else(|| invalid("pool output exceeds bound"))?;
    if rows.len() != tokens
        || output.len() != count
        || rows.iter().any(|token| {
            token.len() != top_k || token.iter().any(|row| row.len() != routing.hidden)
        })
    {
        return Err(invalid("pool output shape mismatch"));
    }
    let mut converted = Vec::with_capacity(count);
    for &value in rows.iter().flatten().flatten() {
        let rounded = value as f32;
        if !value.is_finite() || !rounded.is_finite() {
            return Err(invalid("pool output is non-finite or overflows FP32"));
        }
        converted.push(rounded);
    }
    output.copy_from_slice(&converted);
    Ok(())
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
