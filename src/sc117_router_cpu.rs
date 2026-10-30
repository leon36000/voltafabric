//! Bounded CPU reference for SC117 BF16 router weights; not CUDA bit-parity.
use crate::{invalid, Result};

pub const HIDDEN: usize = 2560;
pub const EXPERTS: usize = 512;
pub const WEIGHT_BYTES: usize = HIDDEN * EXPERTS * 2;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Choice {
    pub expert: u32,
    pub weight: f32,
    pub logit: f32,
}

// Round finite f32 to BF16, nearest-even, then expand to f32.
fn to_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xffff_0000)
}

/// Full softmax, stable descending top-k and clamp/renormalize. A deterministic
/// scalar f64 dot is used, so choices can differ from a GPU kernel near ties.
pub fn route(weights: &[u8], input: &[f32], k: usize) -> Result<Vec<Choice>> {
    if weights.len() != WEIGHT_BYTES
        || input.len() != HIDDEN
        || !(1..=64).contains(&k)
        || input.iter().any(|v| !v.is_finite())
    {
        return Err(invalid("SC117 CPU router dimensions or activation"));
    }
    let rounded: Vec<f32> = input.iter().map(|&v| to_bf16(v)).collect();
    if rounded.iter().any(|v| !v.is_finite()) {
        return Err(invalid("BF16 activation overflow"));
    }
    let mut logits = Vec::with_capacity(EXPERTS);
    for row in weights.chunks_exact(HIDDEN * 2) {
        let mut sum = 0.0_f64;
        for (bytes, &x) in row.chunks_exact(2).zip(&rounded) {
            let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
            let weight = f32::from_bits(u32::from(bits) << 16);
            if !weight.is_finite() {
                return Err(invalid("non-finite BF16 router weight"));
            }
            sum += f64::from(weight) * f64::from(x);
        }
        let value = sum as f32;
        if !value.is_finite() {
            return Err(invalid("non-finite router logit"));
        }
        logits.push(value);
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials: Vec<f64> = logits
        .iter()
        .map(|&v| (f64::from(v) - f64::from(max)).exp())
        .collect();
    let total: f64 = exponentials.iter().sum();
    let inv = (1.0 / total) as f32;
    let probs: Vec<f32> = exponentials
        .iter()
        .map(|v| (v * f64::from(inv)) as f32)
        .collect();
    let mut ids: Vec<usize> = (0..EXPERTS).collect();
    ids.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
    ids.truncate(k);
    let selected = ids
        .iter()
        .map(|&i| f64::from(probs[i]))
        .sum::<f64>()
        .max(2_f64.powi(-14));
    Ok(ids
        .into_iter()
        .map(|i| Choice {
            expert: i as u32,
            weight: (f64::from(probs[i]) / selected) as f32,
            logit: logits[i],
        })
        .collect())
}
