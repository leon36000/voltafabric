//! Bounded scalar CPU reference for Strata-style hyperconnection read.
//! BF16 activations; f64 dot accumulation; no claim of GPU bit parity.
use crate::{invalid, Result};

#[derive(Debug)]
pub struct ReadOutput {
    pub mixed: Vec<f32>,
    pub inject: Vec<f32>,
}

fn bf16(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}
fn decode(bytes: &[u8]) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for b in bytes.chunks_exact(2) {
        let v = f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16);
        if !v.is_finite() {
            return Err(invalid("non-finite hyperconnection weight"));
        }
        out.push(v);
    }
    Ok(out)
}
fn finite(value: f32) -> Result<f32> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid("non-finite hyperconnection intermediate"))
    }
}
fn dot(row: &[f32], x: &[f32]) -> Result<f32> {
    finite(
        row.iter()
            .zip(x)
            .map(|(&w, &v)| f64::from(w) * f64::from(v))
            .sum::<f64>() as f32,
    )
}
/// RMSNorm per stream, BF16 bottleneck and gate, and pre-attention mixed row.
/// All buffers are validated before calculating; output is returned atomically.
#[allow(clippy::too_many_arguments)]
pub fn read(
    residual: &[f32],
    gamma: &[f32],
    down: &[u8],
    up: &[u8],
    inject: &[u8],
    hidden: usize,
    streams: usize,
    rank: usize,
    epsilon: f32,
) -> Result<ReadOutput> {
    let n = hidden
        .checked_mul(streams)
        .filter(|&x| x > 0 && x <= 16384)
        .ok_or_else(|| invalid("hyperconnection geometry"))?;
    let matrix = n
        .checked_mul(rank)
        .filter(|&x| x <= 4_194_304)
        .ok_or_else(|| invalid("hyperconnection matrix size"))?;
    let imatrix = n
        .checked_mul(streams)
        .ok_or_else(|| invalid("hyperconnection injection size"))?;
    if rank == 0
        || rank > 512
        || streams > 16
        || residual.len() != n
        || gamma.len() != n
        || down.len() != matrix * 2
        || up.len() != matrix * 2
        || inject.len() != imatrix * 2
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || residual.iter().chain(gamma).any(|x| !x.is_finite())
    {
        return Err(invalid("invalid hyperconnection input"));
    }
    let down = decode(down)?;
    let up = decode(up)?;
    let inject = decode(inject)?;
    let mut xn = Vec::with_capacity(n);
    for (r, g) in residual
        .chunks_exact(hidden)
        .zip(gamma.chunks_exact(hidden))
    {
        let ss = r.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>() / hidden as f64;
        let inv = (ss + f64::from(epsilon)).sqrt().recip();
        for (&v, &w) in r.iter().zip(g) {
            xn.push(finite((f64::from(v) * inv * f64::from(w)) as f32)?);
        }
    }
    let xq: Vec<f32> = xn.iter().map(|&x| bf16(x)).collect();
    if xq.iter().any(|x| !x.is_finite()) {
        return Err(invalid("hyperconnection BF16 overflow"));
    }
    let mut lo = Vec::with_capacity(rank);
    for row in down.chunks_exact(n) {
        let x = dot(row, &xq)? / streams as f32;
        lo.push(finite(bf16(x / (1.0 + (-x).exp())))?);
    }
    let mut mixed = vec![0.0_f32; hidden];
    for (i, row) in up.chunks_exact(rank).enumerate() {
        let x = dot(row, &lo)?;
        let gated = finite(xn[i] / (1.0 + (-x).exp()))?;
        mixed[i % hidden] = finite(mixed[i % hidden] + gated)?;
    }
    for x in &mut mixed {
        *x = finite(*x / streams as f32)?;
    }
    let injection = inject
        .chunks_exact(n)
        .map(|row| dot(row, &xq))
        .collect::<Result<Vec<_>>>()?;
    Ok(ReadOutput {
        mixed,
        inject: injection,
    })
}
