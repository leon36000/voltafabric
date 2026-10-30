//! Resident SC117 CPU post-attention hyperconnection and expert router.
//! The enclosing transaction must establish input provenance and reservation.
use crate::{
    gdn_cpu::Identity,
    hyperconnection_cpu, invalid,
    sc117_gdn_stage::{self, Span},
    sc117_router_cpu::{self, Choice},
    Result,
};
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub model_revision: String,
    pub layer: u32,
    pub dense_bytes: u64,
    pub norm: Span,
    pub down: Span,
    pub up: Span,
    pub inject: Span,
    pub router: Span,
}
pub struct Stage {
    identity: Identity,
    norm: Vec<f32>,
    down: Vec<u8>,
    up: Vec<u8>,
    inject: Vec<u8>,
    router: Vec<u8>,
}
pub struct Routed {
    pub residual: Vec<f32>,
    pub mixed: Vec<f32>,
    pub inject: Vec<f32>,
    pub choices: Vec<Choice>,
}
impl Stage {
    pub fn load(m: Manifest, identity: Identity, dense: &mut std::fs::File) -> Result<Self> {
        if identity.layer != m.layer
            || identity.model_revision != m.model_revision
            || m.layer >= 48
            || identity.run_id.trim().is_empty()
        {
            return Err(invalid("FFN stage identity"));
        }
        let before = dense.metadata().map_err(|e| invalid(e.to_string()))?;
        if !before.is_file() || before.len() != m.dense_bytes {
            return Err(invalid("FFN source size"));
        }
        let norm = sc117_gdn_stage::floats(&sc117_gdn_stage::read(dense, &m.norm, 40960)?);
        let down = sc117_gdn_stage::read(dense, &m.down, 6553600)?;
        let up = sc117_gdn_stage::read(dense, &m.up, 6553600)?;
        let inject = sc117_gdn_stage::read(dense, &m.inject, 81920)?;
        let router = sc117_gdn_stage::read(dense, &m.router, 2621440)?;
        #[cfg(unix)]
        if !sc117_gdn_stage::same(
            &before,
            &dense.metadata().map_err(|e| invalid(e.to_string()))?,
        ) {
            return Err(invalid("FFN source changed"));
        }
        Ok(Self {
            identity,
            norm,
            down,
            up,
            inject,
            router,
        })
    }
    pub fn compute(
        &self,
        identity: &Identity,
        residual: &[f32],
        attention: &[f32],
        attention_inject: &[f32],
    ) -> Result<Routed> {
        if identity != &self.identity
            || residual.len() != 10240
            || attention.len() != 2560
            || attention_inject.len() != 4
        {
            return Err(invalid("FFN stage input identity or shape"));
        }
        let residual = hyperconnection_cpu::write(residual, attention, attention_inject)?;
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
            residual,
            mixed: hc.mixed,
            inject: hc.inject,
            choices,
        })
    }
}
