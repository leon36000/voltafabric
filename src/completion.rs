//! Convert completed compact FP16 output into the existing unweighted collector receipts.
//! CPU representation boundary only: this does not enqueue, synchronize or prove execution.
use crate::{invalid, launch::Prepared, moe, Error, Result};
use serde::{Deserialize, Serialize};
pub const MAX_OUTPUT_VALUES: usize = 16_777_216;
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawOutput {
    pub stamp: moe::Stamp,
    pub rank: u32,
    pub rows: usize,
    pub columns: usize,
    /// Numeric IEEE binary16 bit patterns, compact token/slot row-major order.
    /// JSON u16 has no byte-order ambiguity; binary staging must read little-endian.
    pub fp16_bits: Vec<u16>,
}

/// Row identity needed to collect an unweighted worker result without repacking its
/// resident weights. The route coefficient is deliberately absent: the coordinator's
/// validated routing plan remains the sole source of coefficients.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputSlot {
    pub token: usize,
    pub slot: usize,
    pub expert: u32,
    pub local: u32,
}

/// Self-contained rows received from a rank worker. Values remain binary16 through
/// admission and move directly into the shared collector; no f64 completion vectors
/// or resident weight copies are constructed at this boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutedRawOutput {
    pub stamp: moe::Stamp,
    pub rank: u32,
    pub attempt: u64,
    pub rows: usize,
    pub columns: usize,
    pub slots: Vec<OutputSlot>,
    pub fp16_bits: Vec<u16>,
}
/// All finite binary16 values are exactly representable in f64, including signed zero.
pub fn decode_fp16(bits: u16) -> Result<f64> {
    moe::decode_fp16(bits)
}

/// Convert a finite f64 directly to IEEE binary16 using round-to-nearest, ties-to-even.
/// Values that round to infinity are rejected; finite underflow rounds to signed zero or
/// a subnormal, as IEEE binary16 requires. The conversion uses the original f64 significand
/// so a binary64 -> binary32 -> binary16 double-rounding edge cannot change the result.
pub fn encode_fp16(value: f64) -> Result<u16> {
    if !value.is_finite() {
        return Err(Error::new(
            "NON_FINITE_RESULT",
            "cannot encode a non-finite completion as FP16",
        ));
    }

    let bits = value.to_bits();
    let sign = ((bits >> 48) as u16) & 0x8000;
    let exponent = ((bits >> 52) & 0x07ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent == 0 {
        // Every f64 subnormal is much smaller than half of the least binary16 subnormal.
        return Ok(sign);
    }

    let unbiased = exponent - 1023;
    if unbiased > 15 {
        return Err(fp16_overflow());
    }
    let significand = (1_u64 << 52) | fraction;
    if unbiased >= -14 {
        let mut exponent = unbiased;
        let rounded = round_right_ties_even(significand, 42);
        let mantissa = if rounded == 2048 {
            exponent += 1;
            1024
        } else {
            rounded
        };
        if exponent > 15 {
            return Err(fp16_overflow());
        }
        return Ok(sign | (((exponent + 15) as u16) << 10) | ((mantissa as u16) & 0x03ff));
    }

    // A binary16 subnormal is an integer multiple of 2^-24. Scale the exact f64
    // significand into those units and round once, without an intermediate f32.
    let shift = u32::try_from(28 - unbiased).expect("subnormal shift is positive");
    let rounded = round_right_ties_even(significand, shift);
    if rounded >= 1024 {
        return Ok(sign | 0x0400); // rounded up to the least normal binary16 value
    }
    Ok(sign | rounded as u16)
}

fn fp16_overflow() -> Error {
    Error::new(
        "FP16_OVERFLOW",
        "finite completion rounds outside the finite FP16 range",
    )
}

fn round_right_ties_even(value: u64, shift: u32) -> u64 {
    if shift == 0 {
        return value;
    }
    // All call-site values are at most 53 bits, so shifts >= 54 round to zero.
    if shift >= 54 {
        return 0;
    }
    let quotient = value >> shift;
    let remainder = value & ((1_u64 << shift) - 1);
    let halfway = 1_u64 << (shift - 1);
    quotient + u64::from(remainder > halfway || (remainder == halfway && quotient & 1 != 0))
}

fn checked_count(rows: usize, columns: usize) -> Result<usize> {
    let count = rows
        .checked_mul(columns)
        .ok_or_else(|| Error::new("OVERFLOW", "compact output shape overflow"))?;
    if count > MAX_OUTPUT_VALUES {
        return Err(Error::new(
            "CAPACITY_EXCEEDED",
            "FP16 completion values exceed the configured value-count cap",
        ));
    }
    Ok(count)
}
fn reserve<T>(count: usize) -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|e| Error::new("ALLOCATION_FAILED", e.to_string()))?;
    Ok(out)
}
fn validate_fp16_bits(bits: &[u16]) -> Result<()> {
    for &value in bits {
        decode_fp16(value)?;
    }
    Ok(())
}
fn validate_output_shape(plan: &Prepared<'_>, raw: &RawOutput) -> Result<()> {
    if &raw.stamp != plan.stamp() || raw.rank != plan.weights().rank() {
        return Err(Error::new(
            "STALE_EXECUTION",
            "raw output differs from prepared full Stamp or rank",
        ));
    }
    if raw.rows != plan.completion_slots().len() || raw.columns != plan.output_columns() {
        return Err(invalid(
            "compact output dimensions differ from prepared rows/hidden",
        ));
    }
    let count = checked_count(raw.rows, raw.columns)?;
    if raw.fp16_bits.len() != count {
        return Err(invalid(
            "compact FP16 output length differs; no padding rows allowed",
        ));
    }
    Ok(())
}
/// Validate the complete packet before allocating receipts. No partial result is
/// exposed if a final value or identity is invalid. Coefficients are NOT applied.
/// The caller must establish GPU completion and trustworthy buffer provenance.
/// Repeated conversion is possible; the existing collector rejects duplicate slots.
pub fn from_output(plan: &Prepared<'_>, raw: &RawOutput) -> Result<Vec<moe::Completion>> {
    validate_output_shape(plan, raw)?;
    for &bits in &raw.fp16_bits {
        decode_fp16(bits)?;
    }
    let mut out = reserve(raw.rows)?;
    for (slot, row) in plan
        .completion_slots()
        .iter()
        .zip(raw.fp16_bits.chunks_exact(raw.columns))
    {
        let mut value = reserve(raw.columns)?;
        for &bits in row {
            value.push(decode_fp16(bits)?);
        }
        out.push(moe::Completion {
            stamp: plan.stamp().clone(),
            token: slot.token,
            slot: slot.slot,
            expert: slot.expert,
            rank: slot.rank,
            local: slot.local,
            value,
        });
    }
    Ok(out)
}
/// Validate and move one externally computed FP16 batch directly into the existing
/// Collector. The resident stack remains owned by the caller and is only borrowed by
/// `plan`; no weights are packed or copied here. The collector retains the compact u16
/// payload until `finish`, then decodes and applies route weights in canonical slot order.
/// This is only a representation/identity boundary: it cannot attest kernel execution,
/// buffer provenance, stream synchronization or the status of a remote attempt.
pub fn accept_output(
    plan: &Prepared<'_>,
    raw: RawOutput,
    collector: &mut moe::Collector,
) -> Result<usize> {
    validate_output_shape(plan, &raw)?;
    validate_fp16_bits(&raw.fp16_bits)?;
    collector.accept_fp16_batch(
        &raw.stamp,
        raw.rank,
        plan.completion_slots(),
        raw.columns,
        raw.fp16_bits,
    )
}

/// Admit one compact worker batch using only row identity and routing already held by
/// the collector. The worker cannot provide or override route coefficients. This API
/// checks the supplied attempt against the caller's admitted attempt and validates
/// shape, identity and finite FP16 values. It cannot prove remote execution, stream
/// synchronization, or whether an uncertain attempt ran before communication failed.
pub fn accept_routed_output(
    expected_attempt: u64,
    raw: RoutedRawOutput,
    collector: &mut moe::Collector,
) -> Result<usize> {
    if expected_attempt == 0 || raw.attempt != expected_attempt {
        return Err(Error::new(
            "STALE_EXECUTION",
            "FP16 output attempt differs from the admitted attempt",
        ));
    }
    let RoutedRawOutput {
        stamp,
        rank,
        attempt: _,
        rows,
        columns,
        slots,
        fp16_bits,
    } = raw;
    if rows != slots.len() {
        return Err(invalid(
            "routed output row count differs from its identity table",
        ));
    }
    let count = checked_count(rows, columns)?;
    if fp16_bits.len() != count {
        return Err(invalid(
            "routed output FP16 length differs; no padding rows allowed",
        ));
    }
    validate_fp16_bits(&fp16_bits)?;
    collector.accept_fp16_batch_with(
        &stamp,
        rank,
        slots.len(),
        |index| {
            let slot = &slots[index];
            moe::Fp16SlotIdentity {
                token: slot.token,
                slot: slot.slot,
                expert: slot.expert,
                rank,
                local: slot.local,
            }
        },
        columns,
        fp16_bits,
    )
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp16_encoder_matches_independent_python_struct_oracle() {
        let cases: serde_json::Value =
            serde_json::from_slice(include_bytes!("../tests/fixtures/a165_fp16_rounding.json"))
                .unwrap();
        for case in cases.as_array().unwrap() {
            let input_bits =
                u64::from_str_radix(case["input_f64_bits"].as_str().unwrap(), 16).unwrap();
            let value = f64::from_bits(input_bits);
            match case["expected_fp16_bits"].as_str() {
                Some(expected) => assert_eq!(
                    encode_fp16(value).unwrap(),
                    u16::from_str_radix(expected, 16).unwrap(),
                    "input f64 bits {input_bits:016x}"
                ),
                None if value.is_finite() => assert_eq!(
                    encode_fp16(value).unwrap_err().code,
                    "FP16_OVERFLOW",
                    "input f64 bits {input_bits:016x}"
                ),
                None => assert_eq!(
                    encode_fp16(value).unwrap_err().code,
                    "NON_FINITE_RESULT",
                    "input f64 bits {input_bits:016x}"
                ),
            }
        }
    }

    #[test]
    fn exact_output_cap_precedes_allocation() {
        assert_eq!(checked_count(4096, 4096).unwrap(), MAX_OUTPUT_VALUES);
        assert_eq!(
            checked_count(4096, 4097).unwrap_err().code,
            "CAPACITY_EXCEEDED"
        );
    }
    #[test]
    fn output_shape_arithmetic_does_not_wrap() {
        assert_eq!(checked_count(usize::MAX, 2).unwrap_err().code, "OVERFLOW");
    }
}
