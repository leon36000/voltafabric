//! CPU reference for SC117 PLE post-projection, adapted from MIT Strata
//! fb58e0dbc8399662c0e47c76578c6e878b14f6cf, ple.hpp and ple_parity.cpp.
//! This pure computation does not commit history or qualify a complete layer.
use crate::{invalid, Result};

pub const WIDTH: usize = 2560;
pub const STREAMS: usize = 4;
pub const CHANNELS: usize = WIDTH * STREAMS;
pub const HISTORY: usize = 9;

pub struct Inputs<'a> {
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub hidden: &'a [f32],
    pub norm_key: &'a [f32],
    pub norm_query: &'a [f32],
    pub norm_conv: &'a [f32],
    /// Kernel-fast: kernel index + 4 * channel, decoded from F16.
    pub kernel: &'a [f32],
    /// Oldest first, row-fast: row + 9 * channel.
    pub history: &'a [f32],
}
pub struct Output {
    pub residual: Vec<f32>,
    pub normalized: Vec<f32>,
    /// Candidate history; caller commits it only with the complete layer.
    pub next_history: Vec<f32>,
}
fn norm(x: &[f32], weight: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; CHANNELS];
    for s in 0..STREAMS {
        let range = s * WIDTH..(s + 1) * WIDTH;
        let sum: f64 = x[range.clone()].iter().map(|&v| f64::from(v * v)).sum();
        let scale = 1.0 / ((sum / WIDTH as f64) as f32 + 1e-6).sqrt();
        for i in range {
            out[i] = x[i] * scale * weight[i];
        }
    }
    out
}
pub fn evaluate(input: Inputs<'_>) -> Result<Output> {
    let arrays = [
        input.key,
        input.hidden,
        input.norm_key,
        input.norm_query,
        input.norm_conv,
    ];
    if arrays.iter().any(|x| x.len() != CHANNELS)
        || input.value.len() != WIDTH
        || input.kernel.len() != CHANNELS * 4
        || input.history.len() != CHANNELS * HISTORY
        || arrays
            .iter()
            .chain([input.value, input.kernel, input.history].iter())
            .any(|a| a.iter().any(|x| !x.is_finite()))
    {
        return Err(invalid("PLE post projection shape or nonfinite input"));
    }
    let key = norm(input.key, input.norm_key);
    let query = norm(input.hidden, input.norm_query);
    let mut gated = vec![0.0; CHANNELS];
    for s in 0..STREAMS {
        let begin = s * WIDTH;
        let dot: f64 = (begin..begin + WIDTH)
            .map(|i| f64::from(key[i]) * f64::from(query[i]))
            .sum();
        let score = dot as f32 / (WIDTH as f32).sqrt();
        let sign = if score > 0.0 {
            1.0
        } else if score < 0.0 {
            -1.0
        } else {
            0.0
        };
        let signed = sign * score.abs().max(1e-6).sqrt();
        let gate = 1.0 / (1.0 + (-signed).exp());
        for i in 0..WIDTH {
            gated[begin + i] = input.value[i] * gate;
        }
    }
    let normalized = norm(&gated, input.norm_conv);
    let mut residual = vec![0.0; CHANNELS];
    let mut next_history = vec![0.0; CHANNELS * HISTORY];
    for c in 0..CHANNELS {
        let mut acc = 0.0f32;
        for k in 0..4 {
            let row = k * 3;
            let v = if row == HISTORY {
                normalized[c]
            } else {
                input.history[c * HISTORY + row]
            };
            acc += input.kernel[c * 4 + k] * v;
        }
        residual[c] = input.hidden[c] + gated[c] + acc / (1.0 + (-acc).exp());
        next_history[c * HISTORY..c * HISTORY + HISTORY - 1]
            .copy_from_slice(&input.history[c * HISTORY + 1..c * HISTORY + HISTORY]);
        next_history[c * HISTORY + HISTORY - 1] = normalized[c];
    }
    if residual
        .iter()
        .chain(normalized.iter())
        .chain(next_history.iter())
        .any(|x| !x.is_finite())
    {
        return Err(invalid("PLE post projection nonfinite output"));
    }
    Ok(Output {
        residual,
        normalized,
        next_history,
    })
}
