//! Resident CPU SC117 output head. The caller must authenticate final-layer provenance.
//! This is not a CUDA parity or complete-generation claim.
use crate::{
    gdn_cpu::Identity,
    hyperconnection_cpu, invalid,
    native_ggml_cpu::{MatrixBinding, QuantizedMatrix},
    sc117_gdn_stage::{self, Span},
    Result,
};
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Chunk {
    pub offset: u64,
    pub bytes: usize,
    pub sha256: String,
    pub rows: usize,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub model_revision: String,
    pub layer_count: u32,
    pub vocab: usize,
    pub dense_bytes: u64,
    pub shard_bytes: u64,
    pub norm: Span,
    pub down: Span,
    pub up: Span,
    pub chunks: Vec<Chunk>,
}
pub struct Head {
    identity: Identity,
    norm: Vec<f32>,
    down: Vec<u8>,
    up: Vec<u8>,
    zero_injection: Vec<u8>,
    chunks: Vec<QuantizedMatrix>,
    vocab: usize,
}
pub struct Output {
    pub mixed: Vec<f32>,
    pub logits: Vec<f32>,
    pub greedy: u32,
}
impl Head {
    pub fn load(
        m: Manifest,
        identity: Identity,
        dense: &mut std::fs::File,
        shard: &mut std::fs::File,
    ) -> Result<Self> {
        if m.layer_count != 48
            || m.vocab != 248_320
            || identity.layer != m.layer_count
            || identity.model_revision != m.model_revision
            || identity.run_id.trim().is_empty()
            || identity.run_id.len() > 256
            || identity.model_revision.trim().is_empty()
            || identity.model_revision.len() > 256
            || m.chunks.len() != 31
        {
            return Err(invalid("SC117 head identity or geometry"));
        }
        let mut expected_end = None;
        let mut expected_rows = 0usize;
        for c in &m.chunks {
            if c.rows == 0
                || c.rows > 8192
                || c.bytes != c.rows * 2100
                || expected_end.is_some_and(|end| end != c.offset)
                || !crate::sha256::is_hex_digest(&c.sha256)
            {
                return Err(invalid("SC117 head chunk geometry"));
            }
            expected_rows = expected_rows
                .checked_add(c.rows)
                .ok_or_else(|| invalid("SC117 head row overflow"))?;
            expected_end = Some(
                c.offset
                    .checked_add(c.bytes as u64)
                    .filter(|end| *end <= m.shard_bytes)
                    .ok_or_else(|| invalid("SC117 head offset overflow"))?,
            );
        }
        if expected_rows != m.vocab {
            return Err(invalid("SC117 head incomplete vocabulary"));
        }
        let dm = dense.metadata().map_err(|e| invalid(e.to_string()))?;
        let sm = shard.metadata().map_err(|e| invalid(e.to_string()))?;
        if !dm.is_file() || !sm.is_file() || dm.len() != m.dense_bytes || sm.len() != m.shard_bytes
        {
            return Err(invalid("SC117 head source size"));
        }
        let norm = sc117_gdn_stage::floats(&sc117_gdn_stage::read(dense, &m.norm, 40_960)?);
        let down = sc117_gdn_stage::read(dense, &m.down, 6_553_600)?;
        let up = sc117_gdn_stage::read(dense, &m.up, 6_553_600)?;
        let mut chunks = Vec::with_capacity(m.chunks.len());
        let mut total = 0usize;
        let mut end = None;
        for c in m.chunks {
            if c.rows == 0
                || c.rows > 8192
                || c.bytes != c.rows * 2100
                || end.is_some_and(|x| x != c.offset)
                || total.checked_add(c.rows).is_none_or(|n| n > m.vocab)
            {
                return Err(invalid("SC117 head chunk geometry"));
            }
            let bytes = sc117_gdn_stage::read(
                shard,
                &Span {
                    offset: c.offset,
                    bytes: c.bytes,
                    sha256: c.sha256.clone(),
                },
                c.bytes,
            )?;
            let binding = MatrixBinding {
                model_revision: m.model_revision.clone(),
                layer: m.layer_count,
                tensor: format!("output.weight.rows.{total}"),
                ggml_type: 14,
                rows: c.rows,
                columns: 2560,
                sha256: c.sha256,
            };
            chunks.push(QuantizedMatrix::load(binding, &bytes)?);
            total += c.rows;
            end = c.offset.checked_add(c.bytes as u64);
            if end.is_none() {
                return Err(invalid("SC117 head offset overflow"));
            }
        }
        if total != m.vocab {
            return Err(invalid("SC117 head incomplete vocabulary"));
        }
        #[cfg(unix)]
        if !sc117_gdn_stage::same(&dm, &dense.metadata().map_err(|e| invalid(e.to_string()))?)
            || !sc117_gdn_stage::same(&sm, &shard.metadata().map_err(|e| invalid(e.to_string()))?)
        {
            return Err(invalid("SC117 head source changed"));
        }
        Ok(Self {
            identity,
            norm,
            down,
            up,
            zero_injection: vec![0; 81_920],
            chunks,
            vocab: m.vocab,
        })
    }
    pub fn evaluate(&mut self, identity: &Identity, residual: &[f32]) -> Result<Output> {
        if identity != &self.identity || residual.len() != 10240 {
            return Err(invalid("SC117 head input identity or shape"));
        }
        // Strata passes a null injection matrix for head mixing. An all-zero BF16
        // matrix yields the same mixed output; its zero injection is discarded.
        let mixed = hyperconnection_cpu::read(
            residual,
            &self.norm,
            &self.down,
            &self.up,
            &self.zero_injection,
            2560,
            4,
            320,
            1e-6,
        )?
        .mixed;
        let mut logits = Vec::with_capacity(self.vocab);
        for chunk in &mut self.chunks {
            logits.extend(chunk.forward(&mixed)?);
        }
        if logits.len() != self.vocab || logits.iter().any(|v| !v.is_finite()) {
            return Err(invalid("SC117 head incomplete or nonfinite logits"));
        }
        let mut best = 0usize;
        for i in 1..logits.len() {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        Ok(Output {
            mixed,
            logits,
            greedy: best as u32,
        })
    }
}
