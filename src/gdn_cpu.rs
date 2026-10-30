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

/// BF16 row-major projections and GDN gates, with full f32 activations and f64 CPU dots.
/// This models the native F32-input projection path, not bitwise CUDA arithmetic.
pub fn project_gates(
    input: &[f32],
    alpha: &[u8],
    beta: &[u8],
    dt: &[f32],
    a: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let rows = dt.len();
    let bytes = input.len().checked_mul(rows).and_then(|n| n.checked_mul(2));
    if input.is_empty()
        || input.len() > 16384
        || rows == 0
        || rows > 1024
        || bytes != Some(alpha.len())
        || bytes != Some(beta.len())
        || a.len() != rows
        || input.iter().chain(dt).chain(a).any(|v| !v.is_finite())
    {
        return Err(invalid("invalid GDN gate projection dimensions or input"));
    }
    let mut gates = Vec::with_capacity(rows);
    let mut betas = Vec::with_capacity(rows);
    for h in 0..rows {
        let mut sums = [0.0_f64; 2];
        for (j, &x) in input.iter().enumerate() {
            let offset = (h * input.len() + j) * 2;
            for (index, weights) in [alpha, beta].iter().enumerate() {
                let value = f32::from_bits(
                    u32::from(u16::from_le_bytes([weights[offset], weights[offset + 1]])) << 16,
                );
                if !value.is_finite() {
                    return Err(invalid("nonfinite GDN BF16 weight"));
                }
                sums[index] += f64::from(value) * f64::from(x);
            }
        }
        let raw_alpha = sums[0] as f32;
        let raw_beta = sums[1] as f32;
        let x = raw_alpha + dt[h];
        let softplus = if x > 20.0 { x } else { x.exp().ln_1p() };
        let sigmoid = if raw_beta >= 0.0 {
            1.0 / (1.0 + (-raw_beta).exp())
        } else {
            let e = raw_beta.exp();
            e / (1.0 + e)
        };
        let gate = softplus * a[h];
        if !raw_alpha.is_finite()
            || !raw_beta.is_finite()
            || !x.is_finite()
            || !gate.is_finite()
            || !sigmoid.is_finite()
        {
            return Err(invalid("GDN gate projection overflow"));
        }
        gates.push(gate);
        betas.push(sigmoid);
    }
    Ok((gates, betas))
}
#[derive(Clone)]
pub struct ConvHistory {
    identity: Identity,
    channels: usize,
    history: Vec<f32>,
}

/// One CPU token transaction. Durable admission and the enclosing layer remain the caller's responsibility.
pub struct GdnState {
    convolution: ConvHistory,
    recurrence: RecurrentState,
}

impl GdnState {
    pub fn new(identity: Identity, k_heads: usize, v_heads: usize, dim: usize) -> Result<Self> {
        let recurrence = RecurrentState::new(identity.clone(), k_heads, v_heads, dim)?;
        let channels = k_heads
            .checked_mul(2)
            .and_then(|n| n.checked_add(v_heads))
            .and_then(|n| n.checked_mul(dim))
            .ok_or_else(|| invalid("GDN geometry overflow"))?;
        Ok(Self {
            convolution: ConvHistory::new(identity, channels)?,
            recurrence,
        })
    }
    pub fn history(&self) -> &[f32] {
        self.convolution.history()
    }
    pub fn state(&self) -> &[f32] {
        self.recurrence.state()
    }

    /// Both internal states advance only after preprocessing and recurrence succeed.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        identity: &Identity,
        qkv: &[f32],
        kernel: &[f32],
        epsilon: f32,
        gate: &[f32],
        beta: &[f32],
    ) -> Result<Vec<f32>> {
        let mut candidate = self.convolution.clone();
        let prepared = candidate.step(
            identity,
            qkv,
            kernel,
            self.recurrence.k_heads,
            self.recurrence.v_heads,
            self.recurrence.dim,
            epsilon,
        )?;
        let output = self.recurrence.step(identity, &prepared, gate, beta)?;
        self.convolution = candidate;
        Ok(output)
    }
}

/// CPU recurrence with state indexed by key dimension, value head, value dimension.
/// The caller must coordinate this state with convolution history and durable attempts.
pub struct RecurrentState {
    identity: Identity,
    k_heads: usize,
    v_heads: usize,
    dim: usize,
    state: Vec<f32>,
}

impl RecurrentState {
    pub fn new(identity: Identity, k_heads: usize, v_heads: usize, dim: usize) -> Result<Self> {
        let size = dim.checked_mul(v_heads).and_then(|n| n.checked_mul(dim));
        if identity.run_id.trim().is_empty()
            || identity.run_id.len() > 256
            || identity.model_revision.trim().is_empty()
            || identity.model_revision.len() > 256
            || k_heads == 0
            || v_heads == 0
            || dim == 0
            || !v_heads.is_multiple_of(k_heads)
            || size.is_none_or(|n| n > 1_048_576)
        {
            return Err(invalid("invalid GDN recurrence identity or geometry"));
        }
        Ok(Self {
            identity,
            k_heads,
            v_heads,
            dim,
            state: vec![0.0; size.unwrap()],
        })
    }
    pub fn state(&self) -> &[f32] {
        &self.state
    }

    /// q and k are L2 normalized but q is not yet scaled; gate is log decay.
    pub fn step(
        &mut self,
        identity: &Identity,
        input: &Prepared,
        gate: &[f32],
        beta: &[f32],
    ) -> Result<Vec<f32>> {
        let qk = self.k_heads * self.dim;
        let values = self.v_heads * self.dim;
        if identity != &self.identity
            || input.q.len() != qk
            || input.k.len() != qk
            || input.v.len() != values
            || gate.len() != self.v_heads
            || beta.len() != self.v_heads
            || input
                .q
                .iter()
                .chain(&input.k)
                .chain(&input.v)
                .chain(gate)
                .chain(beta)
                .any(|x| !x.is_finite())
            || beta.iter().any(|&x| !(0.0..=1.0).contains(&x))
        {
            return Err(invalid("invalid GDN recurrence input"));
        }
        let mut next = self.state.clone();
        let mut output = vec![0.0; values];
        let scale = (self.dim as f32).sqrt().recip();
        for h in 0..self.v_heads {
            let decay = gate[h].exp();
            if !decay.is_finite() {
                return Err(invalid("GDN decay overflow"));
            }
            let kh = h % self.k_heads;
            for j in 0..self.dim {
                let mut sk = 0.0_f32;
                for i in 0..self.dim {
                    let idx = (i * self.v_heads + h) * self.dim + j;
                    next[idx] *= decay;
                    sk += next[idx] * input.k[kh * self.dim + i];
                }
                let delta = (input.v[h * self.dim + j] - sk) * beta[h];
                let mut y = 0.0_f32;
                for i in 0..self.dim {
                    let idx = (i * self.v_heads + h) * self.dim + j;
                    next[idx] += input.k[kh * self.dim + i] * delta;
                    y += next[idx] * input.q[kh * self.dim + i];
                }
                output[h * self.dim + j] = y * scale;
            }
        }
        if next.iter().chain(&output).any(|x| !x.is_finite()) {
            return Err(invalid("GDN recurrence overflow"));
        }
        self.state = next;
        Ok(output)
    }
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
