//! Resident SC117 CPU GDN attention prefix. Caller manages durable attempt/no-replay admission.
use crate::{
    gdn_cpu::{self, GdnState, Identity},
    hyperconnection_cpu, invalid,
    native_ggml_cpu::{MatrixBinding, QuantizedMatrix},
    sha256, Result,
};
use std::io::{Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicU64, Ordering};
static NEXT_STAGE_ID: AtomicU64 = AtomicU64::new(1);
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Span {
    pub offset: u64,
    pub bytes: usize,
    pub sha256: String,
}
#[derive(Debug, serde::Deserialize)]
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
    pub a: Span,
    pub alpha: Span,
    pub beta: Span,
    pub conv: Span,
    pub dt: Span,
    pub qkv: Span,
    pub qkv_type: i32,
}
pub struct Output {
    pub mixed: Vec<f32>,
    pub inject: Vec<f32>,
    pub qkv: Vec<f32>,
    pub recurrent: Vec<f32>,
    pub gate: Vec<f32>,
    pub beta: Vec<f32>,
}
/// A private candidate state; dropping it leaves the resident state unchanged.
pub struct Proposal {
    stage_id: u64,
    stamp: crate::moe::Stamp,
    attempt: u64,
    token: u64,
    candidate: GdnState,
    output: Output,
}
impl Proposal {
    pub fn output(&self) -> &Output {
        &self.output
    }
}
pub struct Committed {
    pub output: Output,
    pub residual: Vec<f32>,
}
pub struct Stage {
    stage_id: u64,
    identity: Identity,
    norm: Vec<f32>,
    down: Vec<u8>,
    up: Vec<u8>,
    inject: Vec<u8>,
    a: Vec<f32>,
    alpha: Vec<u8>,
    beta: Vec<u8>,
    conv: Vec<f32>,
    dt: Vec<f32>,
    qkv: QuantizedMatrix,
    state: GdnState,
    next_token: u64,
}
fn read(f: &mut std::fs::File, s: &Span, expected: usize) -> Result<Vec<u8>> {
    if s.bytes != expected
        || !sha256::is_hex_digest(&s.sha256)
        || s.offset
            .checked_add(s.bytes as u64)
            .is_none_or(|end| end > f.metadata().map(|m| m.len()).unwrap_or(0))
    {
        return Err(invalid("invalid SC117 span"));
    }
    f.seek(SeekFrom::Start(s.offset))
        .map_err(|e| invalid(e.to_string()))?;
    let mut b = vec![0; s.bytes];
    f.read_exact(&mut b).map_err(|e| invalid(e.to_string()))?;
    if sha256::hex(&b) != s.sha256 {
        return Err(invalid("SC117 span digest mismatch"));
    }
    Ok(b)
}
fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
#[cfg(unix)]
fn same(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    (
        a.dev(),
        a.ino(),
        a.len(),
        a.mtime(),
        a.mtime_nsec(),
        a.ctime(),
        a.ctime_nsec(),
    ) == (
        b.dev(),
        b.ino(),
        b.len(),
        b.mtime(),
        b.mtime_nsec(),
        b.ctime(),
        b.ctime_nsec(),
    )
}
impl Stage {
    pub fn load(
        manifest: Manifest,
        identity: Identity,
        dense: &mut std::fs::File,
        shard: &mut std::fs::File,
    ) -> Result<Self> {
        if identity.layer != manifest.layer
            || identity.model_revision != manifest.model_revision
            || manifest.layer >= 48
            || !matches!(manifest.qkv_type, 13 | 14)
        {
            return Err(invalid("SC117 layer, model or QKV type mismatch"));
        }
        let dm = dense.metadata().map_err(|e| invalid(e.to_string()))?;
        let sm = shard.metadata().map_err(|e| invalid(e.to_string()))?;
        if !dm.is_file()
            || !sm.is_file()
            || dm.len() != manifest.dense_bytes
            || sm.len() != manifest.shard_bytes
        {
            return Err(invalid("SC117 source file identity mismatch"));
        }
        let norm = floats(&read(dense, &manifest.norm, 40960)?);
        let down = read(dense, &manifest.down, 6553600)?;
        let up = read(dense, &manifest.up, 6553600)?;
        let inject = read(dense, &manifest.inject, 81920)?;
        let a = floats(&read(dense, &manifest.a, 192)?);
        let alpha = read(dense, &manifest.alpha, 245760)?;
        let beta = read(dense, &manifest.beta, 245760)?;
        let conv = floats(&read(dense, &manifest.conv, 163840)?);
        let dt = floats(&read(dense, &manifest.dt, 192)?);
        let qlen = if manifest.qkv_type == 13 {
            18022400
        } else {
            21504000
        };
        let w = read(shard, &manifest.qkv, qlen)?;
        #[cfg(unix)]
        if !same(&dm, &dense.metadata().map_err(|e| invalid(e.to_string()))?)
            || !same(&sm, &shard.metadata().map_err(|e| invalid(e.to_string()))?)
        {
            return Err(invalid("SC117 source changed"));
        }
        let binding = MatrixBinding {
            model_revision: manifest.model_revision,
            layer: manifest.layer,
            tensor: format!("blk.{}.attn_qkv.weight", manifest.layer),
            ggml_type: manifest.qkv_type,
            rows: 10240,
            columns: 2560,
            sha256: manifest.qkv.sha256,
        };
        let qkv = QuantizedMatrix::load(binding, &w)?;
        let state = GdnState::new(identity.clone(), 16, 48, 128)?;
        let stage_id = NEXT_STAGE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| x.checked_add(1))
            .map_err(|_| invalid("SC117 stage instance id exhausted"))?;
        Ok(Self {
            stage_id,
            identity,
            norm,
            down,
            up,
            inject,
            a,
            alpha,
            beta,
            conv,
            dt,
            qkv,
            state,
            next_token: 0,
        })
    }
    pub fn next_token(&self) -> u64 {
        self.next_token
    }
    /// Prepare without advancing the resident GDN state. Durable admission remains caller-owned.
    pub fn prepare(
        &mut self,
        stamp: &crate::moe::Stamp,
        attempt: u64,
        token: u64,
        residual: &[f32],
    ) -> Result<Proposal> {
        if stamp.run_id != self.identity.run_id
            || stamp.model_revision != self.identity.model_revision
            || stamp.layer != self.identity.layer
            || stamp.epoch != self.identity.epoch
            || stamp.plan_id.trim().is_empty()
            || stamp.plan_id.len() > 256
            || attempt == 0
            || token != self.next_token
            || token == u64::MAX
        {
            return Err(invalid("SC117 stage identity or token ordering mismatch"));
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
        let qkv = self.qkv.forward(&hc.mixed)?;
        let (gate, beta) =
            gdn_cpu::project_gates(&hc.mixed, &self.alpha, &self.beta, &self.dt, &self.a)?;
        let mut candidate = self.state.clone();
        let recurrent = candidate.step(&self.identity, &qkv, &self.conv, 1e-6, &gate, &beta)?;
        Ok(Proposal {
            stage_id: self.stage_id,
            stamp: stamp.clone(),
            attempt,
            token,
            candidate,
            output: Output {
                mixed: hc.mixed,
                inject: hc.inject,
                qkv,
                recurrent,
                gate,
                beta,
            },
        })
    }
    /// Validate all routed slots with the Collector, combine shared output and write the residual,
    /// then commit the candidate. Caller must establish durable admission and producer provenance.
    pub fn commit_collected(
        &mut self,
        proposal: Proposal,
        inputs: crate::sc117_layer_completion::LayerInputs,
    ) -> Result<Committed> {
        if proposal.stage_id != self.stage_id
            || proposal.stamp != inputs.problem.stamp
            || proposal.attempt != inputs.attempt
            || proposal.token != self.next_token
        {
            return Err(invalid("SC117 candidate or collection identity mismatch"));
        }
        let stamp = inputs.problem.stamp.clone();
        let attempt = inputs.attempt;
        let residual = crate::sc117_layer_completion::finish(inputs)?;
        self.commit_validated(proposal, &stamp, attempt, residual)
    }
    /// Called only after the layer collector and residual calculation succeeded.
    fn commit_validated(
        &mut self,
        proposal: Proposal,
        stamp: &crate::moe::Stamp,
        attempt: u64,
        residual: Vec<f32>,
    ) -> Result<Committed> {
        if proposal.stage_id != self.stage_id
            || proposal.stamp != *stamp
            || proposal.attempt != attempt
            || proposal.token != self.next_token
            || residual.len() != 10240
            || residual.iter().any(|x| !x.is_finite())
        {
            return Err(invalid(
                "SC117 candidate, completion identity or residual mismatch",
            ));
        }
        self.state = proposal.candidate;
        self.next_token += 1;
        Ok(Committed {
            output: proposal.output,
            residual,
        })
    }
}
