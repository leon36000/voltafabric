//! CPU reference arithmetic for the SC117 shared-expert boundary.
use crate::{invalid, Result};
fn bf16(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}
/// Scalar gate uses BF16-rounded input and BF16 weights, accumulated in f64.
pub fn scalar_gate(input: &[f32], weights: &[u8]) -> Result<f32> {
    if input.is_empty()
        || input.len() > 16384
        || weights.len() != input.len() * 2
        || input.iter().any(|x| !x.is_finite())
    {
        return Err(invalid("invalid shared scalar gate"));
    }
    let mut sum = 0.0_f64;
    for (&x, w) in input.iter().zip(weights.chunks_exact(2)) {
        let w = f32::from_bits(u32::from(u16::from_le_bytes([w[0], w[1]])) << 16);
        let x = bf16(x);
        if !w.is_finite() || !x.is_finite() {
            return Err(invalid("nonfinite shared gate input"));
        }
        sum += f64::from(x) * f64::from(w);
    }
    let s = if sum >= 0.0 {
        1.0 / (1.0 + (-sum).exp())
    } else {
        let e = sum.exp();
        e / (1.0 + e)
    };
    if !s.is_finite() {
        return Err(invalid("shared gate overflow"));
    }
    Ok(s as f32)
}
/// Elementwise SiLU(gate projection) * up projection.
pub fn activation(gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
    if gate.is_empty()
        || gate.len() > 16384
        || gate.len() != up.len()
        || gate.iter().chain(up).any(|v| !v.is_finite())
    {
        return Err(invalid("invalid shared activation"));
    }
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| {
            let s = if g >= 0.0 {
                1.0 / (1.0 + (-g).exp())
            } else {
                let e = g.exp();
                e / (1.0 + e)
            };
            let x = g * s * u;
            if x.is_finite() {
                Ok(x)
            } else {
                Err(invalid("shared activation overflow"))
            }
        })
        .collect()
}
/// Add already-weighted shared output to the FP16/f64 Collector, once, and cast to f32.
pub fn combine(routed: &[f64], shared: &[f32]) -> Result<Vec<f32>> {
    if routed.is_empty()
        || routed.len() > 16384
        || routed.len() != shared.len()
        || routed.iter().any(|x| !x.is_finite())
        || shared.iter().any(|x| !x.is_finite())
    {
        return Err(invalid("invalid shared combination"));
    }
    routed
        .iter()
        .zip(shared)
        .map(|(&r, &s)| {
            let x = (r + f64::from(s)) as f32;
            if x.is_finite() {
                Ok(x)
            } else {
                Err(invalid("shared combination overflow"))
            }
        })
        .collect()
}
