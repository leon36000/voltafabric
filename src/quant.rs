//! Linear row-major CPU decoder, NOT a QPN2/TensorRT swizzle or hardware kernel.
use crate::memory::nvfp4_layout;
use crate::{invalid, Error, Result};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedRows {
    pub rows: u64,
    pub columns: u64,
    pub codes: Vec<u8>,
    pub scales: Vec<u8>,
    pub global_dequant: f64,
}
pub fn decode_fp4(n: u8) -> Result<f64> {
    if n > 15 {
        return Err(invalid("FP4 nibble outside 0..15"));
    }
    let x = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(n & 7) as usize];
    Ok(if n & 8 != 0 { -x } else { x })
}
#[cfg(test)]
std::thread_local! {
    // Test-only count: threads cannot interfere, and release/service code has no counter.
    pub(crate) static SM70_ADMISSION_SCANS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
/// Validate that QPN2 scale tiles stay inside the SM70 kernel's binary16
/// intermediate range. Each global scale corresponds to one N32 tile and its
/// contiguous K scale bytes.
pub fn validate_sm70_scale_domain(scales: &[u8], gscales: &[f32], k: u64) -> Result<()> {
    #[cfg(test)]
    SM70_ADMISSION_SCANS.with(|count| count.set(count.get() + 1));
    const MAX_K: u64 = 65_536;
    if !(64..=MAX_K).contains(&k) || !k.is_multiple_of(64) {
        return Err(invalid("invalid SM70 scale-domain K dimension"));
    }
    let bytes_per_tile = usize::try_from(k / 16)
        .ok()
        .and_then(|n| n.checked_mul(32))
        .ok_or_else(|| invalid("SM70 scale-domain length overflow"))?;
    let expected_scales = gscales
        .len()
        .checked_mul(bytes_per_tile)
        .ok_or_else(|| invalid("SM70 scale-domain length overflow"))?;
    if scales.len() != expected_scales {
        return Err(invalid("SM70 scale-domain scale length mismatch"));
    }

    for (tile, &gscale) in gscales.iter().enumerate() {
        if !gscale.is_finite() || gscale <= 0.0 {
            return Err(sm70_scale_range("global scale must be positive and finite"));
        }
        let gm_f32 = gscale * 16_384.0_f32;
        if !gm_f32.is_finite() {
            return Err(sm70_scale_range("global scale intermediate is nonfinite"));
        }
        let gm_bits = f32_to_f16_bits_positive_rn(gm_f32)
            .ok_or_else(|| sm70_scale_range("global scale intermediate overflows binary16"))?;
        let gm = f16_positive_to_f64(gm_bits);

        let start = tile * bytes_per_tile;
        let end = start + bytes_per_tile;
        let max_encoded = scales[start..end]
            .iter()
            .copied()
            .max()
            .expect("validated tile lengths are nonzero");
        let max_scale = decode_scale(max_encoded)?;
        if max_scale * gm >= 65_520.0 {
            return Err(sm70_scale_range("scale product overflows binary16"));
        }
    }
    Ok(())
}

fn sm70_scale_range(message: &'static str) -> Error {
    Error::new("SM70_SCALE_RANGE", message)
}

pub(crate) fn f16_positive_to_f64(bits: u16) -> f64 {
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = (bits & 0x03ff) as f64;
    if exponent == 0 {
        fraction * 2.0_f64.powi(-24)
    } else {
        (1024.0 + fraction) * 2.0_f64.powi(exponent - 25)
    }
}

pub(crate) fn f32_to_f16_bits_positive_rn(value: f32) -> Option<u16> {
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x7f_ffff;
    if bits >> 31 != 0 || exponent == 0xff {
        return None;
    }
    if exponent == 0 {
        return Some(0);
    }

    let unbiased = exponent - 127;
    if unbiased > 15 {
        return None;
    }
    if unbiased >= -14 {
        let mut half_exponent = (unbiased + 15) as u16;
        let mut mantissa = fraction >> 13;
        let remainder = fraction & 0x1fff;
        if remainder > 0x1000 || (remainder == 0x1000 && mantissa & 1 != 0) {
            mantissa += 1;
            if mantissa == 0x400 {
                half_exponent += 1;
                mantissa = 0;
            }
        }
        if half_exponent >= 0x1f {
            return None;
        }
        Some((half_exponent << 10) | mantissa as u16)
    } else {
        let shift = (-unbiased - 1) as u32;
        if shift > 24 {
            return Some(0);
        }
        let significand = (0x80_0000 | fraction) as u64;
        let quotient = significand >> shift;
        let remainder = significand & ((1_u64 << shift) - 1);
        let halfway = 1_u64 << (shift - 1);
        let rounded = quotient
            + u64::from(remainder > halfway || (remainder == halfway && quotient & 1 != 0));
        Some(if rounded >= 1024 {
            0x0400
        } else {
            rounded as u16
        })
    }
}

pub fn decode_scale(b: u8) -> Result<f64> {
    if b & 128 != 0 || b == 127 {
        return Err(invalid("NVFP4 scale must be unsigned finite E4M3"));
    }
    let e = (b >> 3) & 15;
    let m = (b & 7) as f64;
    Ok(if e == 0 {
        m * 2.0_f64.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2.0_f64.powi(e as i32 - 7)
    })
}
pub fn dequantize(p: &PackedRows) -> Result<Vec<Vec<f64>>> {
    let layout = nvfp4_layout(p.rows, p.columns, 1, 0)?;
    if !p.global_dequant.is_finite()
        || p.global_dequant <= 0.0
        || p.rows.checked_mul(p.columns).is_none_or(|n| n > 1_048_576)
    {
        return Err(invalid("invalid global dequant factor or CPU size limit"));
    }
    if p.codes.len() as u64 != layout.codes || p.scales.len() as u64 != layout.scales {
        return Err(invalid("packed plane length mismatch"));
    }
    let scales = p
        .scales
        .iter()
        .map(|&b| decode_scale(b))
        .collect::<Result<Vec<_>>>()?;
    let stride = layout.padded_columns as usize;
    let mut out = vec![vec![0.0; p.columns as usize]; p.rows as usize];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, y) in row.iter_mut().enumerate() {
            let byte = p.codes[r * stride / 2 + c / 2];
            let nibble = if c % 2 == 0 { byte & 15 } else { byte >> 4 };
            *y = decode_fp4(nibble)? * scales[r * stride / 16 + c / 16] * p.global_dequant;
            if !y.is_finite() {
                return Err(Error::new("NUMERICAL_OVERFLOW", "decoded value nonfinite"));
            }
        }
    }
    Ok(out)
}
