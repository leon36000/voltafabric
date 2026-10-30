//! Bounded CPU reference for Strata-style GDN preprocessing, not CUDA parity.
//! The caller owns batch ordering, attempt identity, durable state and no-replay admission.
use crate::{invalid, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    pub run_id: String,
    pub model_revision: String,
    pub layer: u32,
    pub epoch: u64,
}
#[derive(Debug)]
pub struct Prepared {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}
pub struct ConvHistory {
    identity: Identity,
    channels: usize,
    history: Vec<f32>,
}

impl ConvHistory {
    /// History starts at zero, with three older samples per channel.
    pub fn new(identity: Identity, channels: usize) -> Result<Self> {
        if identity.run_id.trim().is_empty()
            || identity.run_id.len() > 256
            || identity.model_revision.trim().is_empty()
            || identity.model_revision.len() > 256
            || channels == 0
            || channels > 16384
        {
            return Err(invalid("invalid GDN history identity or size"));
        }
        Ok(Self {
            identity,
            channels,
            history: vec![0.0; channels * 3],
        })
    }
    pub fn identity(&self) -> &Identity {
        &self.identity
    }
    pub fn history(&self) -> &[f32] {
        &self.history
    }
    /// Four-tap channel-major convolution, SiLU, then Q/K L2 normalization.
    /// The history is committed only if all outputs are finite and valid.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        identity: &Identity,
        qkv: &[f32],
        kernel: &[f32],
        k_heads: usize,
        v_heads: usize,
        head_dim: usize,
        epsilon: f32,
    ) -> Result<Prepared> {
        let qk = k_heads
            .checked_mul(head_dim)
            .ok_or_else(|| invalid("GDN geometry overflow"))?;
        let v = v_heads
            .checked_mul(head_dim)
            .ok_or_else(|| invalid("GDN geometry overflow"))?;
        if identity != &self.identity
            || k_heads == 0
            || v_heads == 0
            || head_dim == 0
            || !v_heads.is_multiple_of(k_heads)
            || qk.checked_mul(2).and_then(|n| n.checked_add(v)) != Some(self.channels)
            || qkv.len() != self.channels
            || kernel.len() != self.channels * 4
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || qkv.iter().chain(kernel).any(|x| !x.is_finite())
        {
            return Err(invalid("GDN preprocess identity, geometry or input"));
        }
        let mut values = Vec::with_capacity(self.channels);
        for c in 0..self.channels {
            let h = &self.history[c * 3..c * 3 + 3];
            let w = &kernel[c * 4..c * 4 + 4];
            let mut sum = 0.0_f32;
            for i in 0..3 {
                sum += h[i] * w[i];
            }
            sum += qkv[c] * w[3];
            let y = sum / (1.0 + (-sum).exp());
            if !sum.is_finite() || !y.is_finite() {
                return Err(invalid("GDN convolution overflow"));
            }
            values.push(y);
        }
        for row in values[..2 * qk].chunks_exact_mut(head_dim) {
            let square = row
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>();
            let inv = (square + f64::from(epsilon)).sqrt().recip() as f32;
            for x in row {
                *x *= inv;
                if !x.is_finite() {
                    return Err(invalid("GDN normalization overflow"));
                }
            }
        }
        let output = Prepared {
            q: values[..qk].to_vec(),
            k: values[qk..2 * qk].to_vec(),
            v: values[2 * qk..].to_vec(),
        };
        for (c, &x) in qkv.iter().enumerate() {
            let h = &mut self.history[c * 3..c * 3 + 3];
            h[0] = h[1];
            h[1] = h[2];
            h[2] = x;
        }
        Ok(output)
    }
}
