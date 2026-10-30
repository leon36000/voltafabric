//! Resident SC117 CPU GDN attention prefix. Caller manages durable attempt/no-replay admission.
use crate::{
    gdn_cpu::{self, GdnState, Identity},
    hyperconnection_cpu, invalid,
    native_ggml_cpu::{MatrixBinding, QuantizedMatrix},
    sha256, Result,
};
use std::io::{Read, Seek, SeekFrom};
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
pub struct Stage {
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
        Ok(Self {
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
    /// Commits GDN state only when this prefix succeeds. Enclosing-layer commit remains caller-owned.
    pub fn step(&mut self, identity: &Identity, token: u64, residual: &[f32]) -> Result<Output> {
        if identity != &self.identity || token != self.next_token || token == u64::MAX {
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
        let recurrent = self
            .state
            .step(identity, &qkv, &self.conv, 1e-6, &gate, &beta)?;
        self.next_token += 1;
        Ok(Output {
            mixed: hc.mixed,
            inject: hc.inject,
            qkv,
            recurrent,
            gate,
            beta,
        })
    }
}
