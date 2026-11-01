//! CPU admission of a complete Strata-style pool batch into VoltaFabric routing.
//!
//! The multi-token Strata callback does not expose coefficients. Weighted
//! admission requires actual gate weights; the explicit unweighted path does not.
//! This module
//! neither hooks Strata nor reserves a session, starts a worker, or assigns an
//! attempt. It refuses negative expert IDs. The FP16 path refuses inexact
//! conversion; the separate FP64 path preserves finite FP32 activations.
use crate::{completion, invalid, moe, process, routing, session, Error, Result};
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Duration,
};

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

/// A completed attempt or a no-work result; errors after dispatch retain receipts.
#[derive(Debug)]
pub struct MixedExecutionReport {
    pub attempt: Option<u64>,
    pub receipts: Vec<session::ExecReceipt>,
    pub error: Option<Error>,
}

impl MixedPoolBatch {
    /// Consume this plan, validate the destination before dispatch, and write
    /// only after every worker and every conversion has succeeded. An error after
    /// dispatch retains the attempt and receipts, and is never retried here.
    pub fn execute(
        self,
        sessions: &mut [session::Session],
        artifacts: &[session::CompressedArtifact],
        faults: &BTreeMap<u32, process::Fault>,
        timeout: Duration,
        output: &mut [f32],
    ) -> Result<MixedExecutionReport> {
        if output.len() != self.output_values {
            return Err(invalid("mixed pool output shape mismatch"));
        }
        let Some(misses) = self.misses.as_ref() else {
            if !faults.is_empty() {
                return Err(invalid("fault on an all-external pool batch"));
            }
            return Ok(MixedExecutionReport {
                attempt: None,
                receipts: Vec::new(),
                error: None,
            });
        };
        let report = session::run_compressed_batch_unweighted(
            sessions,
            artifacts,
            &misses.routing,
            &misses.inputs,
            faults,
            timeout,
        )?;
        let mut error = report.error;
        if error.is_none() {
            error = match report.output {
                Some(rows) => self.write_misses(&rows, output).err(),
                None => Some(Error::new(
                    "BATCH_INCOMPLETE",
                    "successful batch lacked output",
                )),
            };
        }
        Ok(MixedExecutionReport {
            attempt: Some(report.attempt),
            receipts: report.receipts,
            error,
        })
    }
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
    admit_mixed_pool_with_model_experts_f64(
        template,
        expected_stamp,
        layer,
        tokens,
        top_k,
        inputs,
        ids,
        weights,
        externally_owned,
        &template.experts,
    )
}

/// The model expert set includes externally hosted experts; only miss experts
/// must appear in the VoltaFabric resident registry. The set and ownership mask
/// must be bound to the actual model and execution plan by the caller.
#[allow(clippy::too_many_arguments)]
pub fn admit_mixed_pool_with_model_experts_f64(
    template: &routing::RoutingProblem,
    expected_stamp: &moe::Stamp,
    layer: u32,
    tokens: usize,
    top_k: usize,
    inputs: &[f32],
    ids: &[i32],
    weights: &[f32],
    externally_owned: &[bool],
    model_experts: &[u32],
) -> Result<MixedPoolBatch> {
    validate_pool_shape(
        template,
        expected_stamp,
        layer,
        tokens,
        top_k,
        inputs,
        ids,
        weights,
    )?;
    let experts: BTreeSet<_> = model_experts.iter().copied().collect();
    if experts.is_empty()
        || experts.len() > 1024
        || experts.len() != model_experts.len()
        || template.experts.iter().any(|e| !experts.contains(e))
    {
        return Err(invalid("invalid model expert set"));
    }
    for (ids_row, weights_row) in ids.chunks_exact(top_k).zip(weights.chunks_exact(top_k)) {
        let mut used = BTreeSet::new();
        for (&id, &weight) in ids_row.iter().zip(weights_row) {
            let expert = u32::try_from(id).map_err(|_| invalid("negative pool expert ID"))?;
            if !experts.contains(&expert)
                || !used.insert(expert)
                || !weight.is_finite()
                || weight < 0.0
            {
                return Err(invalid(
                    "unknown/duplicate expert or invalid routing weight",
                ));
            }
        }
    }
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
        let mut compact = template.clone();
        compact.routes = positions
            .iter()
            .map(|&i| {
                vec![moe::Choice {
                    expert: ids[i] as u32,
                    weight: f64::from(weights[i]),
                }]
            })
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

/// Admit the final per-entry `kind` decisions of the pinned Strata donor:
/// -1 is CPU, 0 is VRAM, 1 is PCIe, and 2 is a peer/helper. The caller must
/// capture this only after all peer/helper decisions, bind it to the batch,
/// and coordinate externally owned rows. This does not verify that provenance.
#[allow(clippy::too_many_arguments)]
pub fn admit_strata_kind_f64(
    template: &routing::RoutingProblem,
    expected_stamp: &moe::Stamp,
    layer: u32,
    tokens: usize,
    top_k: usize,
    inputs: &[f32],
    ids: &[i32],
    weights: &[f32],
    kinds: &[i32],
    model_experts: &[u32],
) -> Result<MixedPoolBatch> {
    if kinds.len() > 262_144 {
        return Err(invalid("Strata kind count exceeds bound"));
    }
    let external: Vec<bool> = kinds
        .iter()
        .map(|&kind| match kind {
            -1 => Ok(false),
            0..=2 => Ok(true),
            _ => Err(invalid("unsupported Strata ownership kind")),
        })
        .collect::<Result<_>>()?;
    admit_mixed_pool_with_model_experts_f64(
        template,
        expected_stamp,
        layer,
        tokens,
        top_k,
        inputs,
        ids,
        weights,
        &external,
        model_experts,
    )
}

/// A Strata-specific unweighted plan. Its internal unit routing coefficients
/// are intentionally not exposed to callers as model coefficients.
pub struct StrataUnweightedBatch(MixedPoolBatch);

impl StrataUnweightedBatch {
    pub fn miss_count(&self) -> usize {
        self.0.positions.len()
    }

    /// Resolve CPU-owned rows through native GGML sessions without applying
    /// gate weights. External rows remain the caller's responsibility.
    #[cfg(feature = "native-ggml-cpu")]
    pub fn execute_native(
        self,
        sessions: &mut [session::Session],
        ledger: &crate::native_ledger::NativeBatchLedger,
        timeout: Duration,
        output: &mut [f32],
    ) -> Result<()> {
        self.execute_native_contract(sessions, ledger, timeout, output, false, None)
    }

    /// Preserve unweighted provider FP32 rows; no Collector reduction is selected.
    #[cfg(feature = "native-ggml-cpu")]
    pub fn execute_native_fp32(
        self,
        sessions: &mut [session::Session],
        ledger: &crate::native_ledger::NativeBatchLedger,
        timeout: Duration,
        output: &mut [f32],
    ) -> Result<()> {
        self.execute_native_contract(sessions, ledger, timeout, output, true, None)
    }

    /// Preserve a caller-selected attempt identity in the local durable ledger.
    #[cfg(feature = "native-ggml-cpu")]
    pub fn execute_native_fp32_reserved(
        self,
        sessions: &mut [session::Session],
        ledger: &crate::native_ledger::NativeBatchLedger,
        ticket: crate::native_ledger::Reservation,
        timeout: Duration,
        output: &mut [f32],
    ) -> Result<()> {
        self.execute_native_contract(sessions, ledger, timeout, output, true, Some(ticket))
    }

    #[cfg(feature = "native-ggml-cpu")]
    fn execute_native_contract(
        self,
        sessions: &mut [session::Session],
        ledger: &crate::native_ledger::NativeBatchLedger,
        timeout: Duration,
        output: &mut [f32],
        unweighted_f32: bool,
        ticket: Option<crate::native_ledger::Reservation>,
    ) -> Result<()> {
        if output.len() != self.0.output_values {
            return Err(invalid("mixed pool output shape mismatch"));
        }
        let Some(misses) = self.0.misses.as_ref() else {
            return Ok(());
        };
        let mut inputs = Vec::with_capacity(misses.inputs.len());
        for row in &misses.inputs {
            let mut values = Vec::with_capacity(row.len());
            for &v in row {
                let x = v as f32;
                if !v.is_finite() || !x.is_finite() || f64::from(x).to_bits() != v.to_bits() {
                    return Err(invalid("native input is not exact finite FP32"));
                }
                values.push(x);
            }
            inputs.push(values);
        }
        let rows = if unweighted_f32 {
            if let Some(ticket) = ticket {
                session::Session::exec_native_ggml_unweighted_f32_reserved(
                    sessions,
                    &misses.routing,
                    &inputs,
                    ledger,
                    ticket,
                    timeout,
                )?
            } else {
                session::Session::exec_native_ggml_unweighted_f32(
                    sessions,
                    &misses.routing,
                    &inputs,
                    ledger,
                    timeout,
                )?
            }
        } else {
            if ticket.is_some() {
                return Err(invalid(
                    "reserved ticket requires unweighted FP32 transport",
                ));
            }
            session::Session::exec_native_ggml_unweighted(
                sessions,
                &misses.routing,
                &inputs,
                ledger,
                timeout,
            )?
        };
        self.0.write_misses(&rows, output)
    }

    pub fn execute(
        self,
        sessions: &mut [session::Session],
        artifacts: &[session::CompressedArtifact],
        faults: &BTreeMap<u32, process::Fault>,
        timeout: Duration,
        output: &mut [f32],
    ) -> Result<MixedExecutionReport> {
        self.0.execute(sessions, artifacts, faults, timeout, output)
    }
}

/// Admit a Strata-style callback without gate coefficients for the unweighted
/// `MixedPoolBatch::execute` contract. The internal unit coefficients are only
/// routing placeholders: they are not Strata gate weights and must never be
/// used to reduce outputs. The donor applies its own weights later, exactly once.
/// The ownership decisions must still be supplied and bound by the caller.
#[allow(clippy::too_many_arguments)]
pub fn admit_strata_unweighted_f64(
    template: &routing::RoutingProblem,
    expected_stamp: &moe::Stamp,
    layer: u32,
    tokens: usize,
    top_k: usize,
    inputs: &[f32],
    ids: &[i32],
    kinds: &[i32],
    model_experts: &[u32],
) -> Result<StrataUnweightedBatch> {
    if ids.len() > 262_144 {
        return Err(invalid("pool slot count exceeds bound"));
    }
    let placeholders = vec![1.0_f32; ids.len()];
    admit_strata_kind_f64(
        template,
        expected_stamp,
        layer,
        tokens,
        top_k,
        inputs,
        ids,
        &placeholders,
        kinds,
        model_experts,
    )
    .map(StrataUnweightedBatch)
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
    validate_pool_shape(
        template,
        expected_stamp,
        layer,
        tokens,
        top_k,
        inputs,
        ids,
        weights,
    )?;
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

#[allow(clippy::too_many_arguments)]
fn validate_pool_shape(
    template: &routing::RoutingProblem,
    expected_stamp: &moe::Stamp,
    layer: u32,
    tokens: usize,
    top_k: usize,
    inputs: &[f32],
    ids: &[i32],
    weights: &[f32],
) -> Result<()> {
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
    Ok(())
}
