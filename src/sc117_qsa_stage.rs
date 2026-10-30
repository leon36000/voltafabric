//! Resident short-context SC117 full-attention CPU prefix. Sparse QSA selection,
//! production transport and whole-layer reservation/commit are not implemented.
use crate::{
    gdn_cpu::Identity,
    hyperconnection_cpu, invalid,
    native_ggml_cpu::{MatrixBinding, QuantizedMatrix},
    qsa_cpu,
    sc117_gdn_stage::{self, Span},
    Result,
};
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub model_revision: String,
    pub layer: u32,
    pub dense_bytes: u64,
    pub shard_bytes: u64,
    pub norm: Span,
    pub down: Span,
    pub up: Span,
    pub inject: Span,
    pub qnorm: Span,
    pub knorm: Span,
    pub q0: Span,
    pub q1: Span,
    pub k: Span,
    pub v: Span,
    pub out: Span,
}
pub struct Stage {
    identity: Identity,
    norm: Vec<f32>,
    down: Vec<u8>,
    up: Vec<u8>,
    inject: Vec<u8>,
    qnorm: Vec<f32>,
    knorm: Vec<f32>,
    q0: QuantizedMatrix,
    q1: QuantizedMatrix,
    k: QuantizedMatrix,
    v: QuantizedMatrix,
    out: QuantizedMatrix,
    cache: qsa_cpu::State,
}
pub struct Output {
    pub mixed: Vec<f32>,
    pub inject: Vec<f32>,
    pub raw_query_gate: Vec<f32>,
    pub query: Vec<f32>,
    pub gate: Vec<f32>,
    pub raw_key: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub gated_attention: Vec<f32>,
    pub attention: Vec<f32>,
}
pub struct Proposal {
    cache: qsa_cpu::Proposal,
    output: Output,
}
impl Proposal {
    pub fn output(&self) -> &Output {
        &self.output
    }
}
fn norm_rope(values: &mut [f32], gamma: &[f32], position: usize) -> Result<()> {
    if gamma.len() != 256
        || !values.len().is_multiple_of(256)
        || values.iter().chain(gamma).any(|x| !x.is_finite())
    {
        return Err(invalid("QSA norm dimensions"));
    }
    for row in values.chunks_exact_mut(256) {
        let sum = row
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            / 256.0;
        let inv = (sum + 1e-6).sqrt().recip();
        for (x, &g) in row.iter_mut().zip(gamma) {
            *x = (f64::from(*x) * inv * f64::from(g)) as f32;
            if !x.is_finite() {
                return Err(invalid("QSA norm overflow"));
            }
        }
        for i in 0..32 {
            let angle = position as f64 * 10_000_000f64.powf(-(i as f64) / 32.0);
            let c = angle.cos() as f32;
            let s = angle.sin() as f32;
            let a = row[i];
            let b = row[i + 32];
            row[i] = a * c - b * s;
            row[i + 32] = a * s + b * c;
        }
    }
    Ok(())
}
impl Stage {
    pub fn load(
        m: Manifest,
        identity: Identity,
        dense: &mut std::fs::File,
        shard: &mut std::fs::File,
        capacity: usize,
    ) -> Result<Self> {
        if m.layer >= 48
            || m.layer % 4 != 3
            || identity.layer != m.layer
            || identity.model_revision != m.model_revision
        {
            return Err(invalid("QSA stage identity"));
        }
        let dm = dense.metadata().map_err(|e| invalid(e.to_string()))?;
        let sm = shard.metadata().map_err(|e| invalid(e.to_string()))?;
        if !dm.is_file() || !sm.is_file() || dm.len() != m.dense_bytes || sm.len() != m.shard_bytes
        {
            return Err(invalid("QSA source size"));
        }
        let norm = sc117_gdn_stage::floats(&sc117_gdn_stage::read(dense, &m.norm, 40960)?);
        let down = sc117_gdn_stage::read(dense, &m.down, 6553600)?;
        let up = sc117_gdn_stage::read(dense, &m.up, 6553600)?;
        let inject = sc117_gdn_stage::read(dense, &m.inject, 81920)?;
        let qnorm = sc117_gdn_stage::floats(&sc117_gdn_stage::read(dense, &m.qnorm, 1024)?);
        let knorm = sc117_gdn_stage::floats(&sc117_gdn_stage::read(dense, &m.knorm, 1024)?);
        let q0b = sc117_gdn_stage::read(shard, &m.q0, 8355840)?;
        let q1b = sc117_gdn_stage::read(shard, &m.q1, 8355840)?;
        let kb = sc117_gdn_stage::read(shard, &m.k, 1075200)?;
        let vb = sc117_gdn_stage::read(shard, &m.v, 1075200)?;
        let ob = sc117_gdn_stage::read(shard, &m.out, 16711680)?;
        #[cfg(unix)]
        if !sc117_gdn_stage::same(&dm, &dense.metadata().map_err(|e| invalid(e.to_string()))?)
            || !sc117_gdn_stage::same(&sm, &shard.metadata().map_err(|e| invalid(e.to_string()))?)
        {
            return Err(invalid("QSA source changed"));
        }
        let bind = |name: &str, kind, rows, columns, sha256: String| MatrixBinding {
            model_revision: m.model_revision.clone(),
            layer: m.layer,
            tensor: format!("blk.{}.{name}.weight", m.layer),
            ggml_type: kind,
            rows,
            columns,
            sha256,
        };
        let q0 = QuantizedMatrix::load(bind("attn_q.part0", 23, 6144, 2560, m.q0.sha256), &q0b)?;
        let q1 = QuantizedMatrix::load(bind("attn_q.part1", 23, 6144, 2560, m.q1.sha256), &q1b)?;
        let k = QuantizedMatrix::load(bind("attn_k", 14, 512, 2560, m.k.sha256), &kb)?;
        let v = QuantizedMatrix::load(bind("attn_v", 14, 512, 2560, m.v.sha256), &vb)?;
        let out = QuantizedMatrix::load(bind("attn_output", 8, 2560, 6144, m.out.sha256), &ob)?;
        let cache = qsa_cpu::State::new(identity.clone(), 24, 2, 256, capacity)?;
        Ok(Self {
            identity,
            norm,
            down,
            up,
            inject,
            qnorm,
            knorm,
            q0,
            q1,
            k,
            v,
            out,
            cache,
        })
    }
    pub fn len(&self) -> usize {
        self.cache.len()
    }
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
    pub fn prepare(
        &mut self,
        identity: &Identity,
        position: usize,
        residual: &[f32],
    ) -> Result<Proposal> {
        if identity != &self.identity || position != self.cache.len() || residual.len() != 10240 {
            return Err(invalid("QSA stage identity or position"));
        }
        let hc = hyperconnection_cpu::read(
            residual,
            &self.norm,
            &self.down,
            &self.up,
            &self.inject,
            2560,
            4,
            320,
            1e-6,
        )?;
        let mut qfull = self.q0.forward(&hc.mixed)?;
        qfull.extend(self.q1.forward(&hc.mixed)?);
        let mut query = Vec::with_capacity(6144);
        let mut gate = Vec::with_capacity(6144);
        for h in qfull.chunks_exact(512) {
            query.extend_from_slice(&h[..256]);
            gate.extend_from_slice(&h[256..]);
        }
        let raw_key = self.k.forward(&hc.mixed)?;
        let mut key = raw_key.clone();
        let value = self.v.forward(&hc.mixed)?;
        norm_rope(&mut query, &self.qnorm, position)?;
        norm_rope(&mut key, &self.knorm, position)?;
        let proposal = self
            .cache
            .prepare(identity, position, &query, &key, &value, &gate)?;
        let gated_attention = proposal.output().to_vec();
        let attention = self.out.forward(&gated_attention)?;
        Ok(Proposal {
            cache: proposal,
            output: Output {
                mixed: hc.mixed,
                inject: hc.inject,
                raw_query_gate: qfull,
                query,
                gate,
                raw_key,
                key,
                value,
                gated_attention,
                attention,
            },
        })
    }
    /// Commit only after the caller has validated the rest of the enclosing layer.
    pub fn commit(&mut self, proposal: Proposal) -> Result<Output> {
        self.cache.commit(proposal.cache)?;
        Ok(proposal.output)
    }
}
