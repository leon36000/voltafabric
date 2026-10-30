//! Resident SC117 CPU post-GDN attention, hyperconnection and routing.
//! The caller must bind the prefix and residual to the same reserved attempt.
use crate::{
    gdn_cpu::{output_norm, Identity},
    hyperconnection_cpu, invalid,
    native_ggml_cpu::{MatrixBinding, QuantizedMatrix},
    sc117_gdn_stage::{self, Output, Span},
    sc117_router_cpu::{self, Choice},
    Result,
};

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub model_revision: String,
    pub layer: u32,
    pub dense_bytes: u64,
    pub shard_bytes: u64,
    pub gamma: Span,
    pub gate: Span,
    pub gate_type: i32,
    pub out: Span,
    pub norm: Span,
    pub down: Span,
    pub up: Span,
    pub inject: Span,
    pub router: Span,
}
pub struct Stage {
    identity: Identity,
    gamma: Vec<f32>,
    gate: QuantizedMatrix,
    out: QuantizedMatrix,
    norm: Vec<f32>,
    down: Vec<u8>,
    up: Vec<u8>,
    inject: Vec<u8>,
    router: Vec<u8>,
}
pub struct Routed {
    pub z: Vec<f32>,
    pub normalized: Vec<f32>,
    pub attention: Vec<f32>,
    pub residual: Vec<f32>,
    pub mixed: Vec<f32>,
    pub inject: Vec<f32>,
    pub choices: Vec<Choice>,
}
impl Stage {
    pub fn load(
        m: Manifest,
        identity: Identity,
        dense: &mut std::fs::File,
        shard: &mut std::fs::File,
    ) -> Result<Self> {
        if identity.model_revision != m.model_revision
            || identity.layer != m.layer
            || m.layer >= 48
            || !matches!(m.gate_type, 12 | 13 | 14 | 23)
        {
            return Err(invalid("post attention identity or type"));
        }
        let dm = dense.metadata().map_err(|e| invalid(e.to_string()))?;
        let sm = shard.metadata().map_err(|e| invalid(e.to_string()))?;
        if !dm.is_file() || !sm.is_file() || dm.len() != m.dense_bytes || sm.len() != m.shard_bytes
        {
            return Err(invalid("post attention source size"));
        }
        let gamma = sc117_gdn_stage::floats(&sc117_gdn_stage::read(dense, &m.gamma, 512)?);
        let gate_bytes = sc117_gdn_stage::read(
            shard,
            &m.gate,
            match m.gate_type {
                12 => 8_847_360,
                13 => 10_813_440,
                14 => 12_902_400,
                23 => 8_355_840,
                _ => return Err(invalid("unsupported SC117 attention gate type")),
            },
        )?;
        let out_bytes = sc117_gdn_stage::read(shard, &m.out, 16_711_680)?;
        let norm = sc117_gdn_stage::floats(&sc117_gdn_stage::read(dense, &m.norm, 40_960)?);
        let down = sc117_gdn_stage::read(dense, &m.down, 6_553_600)?;
        let up = sc117_gdn_stage::read(dense, &m.up, 6_553_600)?;
        let inject = sc117_gdn_stage::read(dense, &m.inject, 81_920)?;
        let router = sc117_gdn_stage::read(dense, &m.router, 2_621_440)?;
        #[cfg(unix)]
        if !sc117_gdn_stage::same(&dm, &dense.metadata().map_err(|e| invalid(e.to_string()))?)
            || !sc117_gdn_stage::same(&sm, &shard.metadata().map_err(|e| invalid(e.to_string()))?)
        {
            return Err(invalid("post attention source changed"));
        }
        let binding = |tensor: &str, ggml_type, rows, columns, sha256: String| MatrixBinding {
            model_revision: m.model_revision.clone(),
            layer: m.layer,
            tensor: format!("blk.{}.{}.weight", m.layer, tensor),
            ggml_type,
            rows,
            columns,
            sha256,
        };
        let gate = QuantizedMatrix::load(
            binding("attn_gate", m.gate_type, 6144, 2560, m.gate.sha256),
            &gate_bytes,
        )?;
        let out =
            QuantizedMatrix::load(binding("ssm_out", 8, 2560, 6144, m.out.sha256), &out_bytes)?;
        Ok(Self {
            identity,
            gamma,
            gate,
            out,
            norm,
            down,
            up,
            inject,
            router,
        })
    }
    pub fn compute(
        &mut self,
        identity: &Identity,
        prefix: &Output,
        residual: &[f32],
    ) -> Result<Routed> {
        if identity != &self.identity
            || prefix.mixed.len() != 2560
            || prefix.inject.len() != 4
            || prefix.recurrent.len() != 6144
            || residual.len() != 10240
        {
            return Err(invalid("post attention identity or dimensions"));
        }
        let z = self.gate.forward(&prefix.mixed)?;
        let normalized = output_norm(&prefix.recurrent, &z, &self.gamma, 1e-6)?;
        let attention = self.out.forward(&normalized)?;
        let residual = hyperconnection_cpu::write(residual, &attention, &prefix.inject)?;
        let hc = hyperconnection_cpu::read(
            &residual,
            &self.norm,
            &self.down,
            &self.up,
            &self.inject,
            2560,
            4,
            320,
            1e-6,
        )?;
        let choices = sc117_router_cpu::route(&self.router, &hc.mixed, 10)?;
        Ok(Routed {
            z,
            normalized,
            attention,
            residual,
            mixed: hc.mixed,
            inject: hc.inject,
            choices,
        })
    }
}
