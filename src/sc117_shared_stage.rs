//! Resident SC117 CPU shared expert. Caller must bind inputs to a reserved attempt.
use crate::{
    gdn_cpu::Identity,
    invalid,
    native_ggml_cpu::{MatrixBinding, QuantizedMatrix},
    sc117_gdn_stage::{self, Span},
    shared_expert_cpu, Result,
};

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub model_revision: String,
    pub layer: u32,
    pub shard_bytes: u64,
    pub scalar: Span,
    pub gate: Span,
    pub gate_type: i32,
    pub up: Span,
    pub up_type: i32,
    pub down: Span,
}
pub struct Stage {
    identity: Identity,
    scalar: Vec<u8>,
    gate: QuantizedMatrix,
    up: QuantizedMatrix,
    down: QuantizedMatrix,
}
fn projection_bytes(kind: i32) -> Result<usize> {
    let block = match kind {
        8 => 272,
        12 => 144,
        13 => 176,
        14 => 210,
        23 => 136,
        _ => return Err(invalid("unsupported shared projection type")),
    };
    Ok(block * 10 * 640)
}
impl Stage {
    pub fn load(m: Manifest, identity: Identity, shard: &mut std::fs::File) -> Result<Self> {
        if identity.model_revision != m.model_revision || identity.layer != m.layer || m.layer >= 48
        {
            return Err(invalid("shared stage identity"));
        }
        let gm = projection_bytes(m.gate_type)?;
        let um = projection_bytes(m.up_type)?;
        let before = shard.metadata().map_err(|e| invalid(e.to_string()))?;
        if !before.is_file() || before.len() != m.shard_bytes {
            return Err(invalid("shared source size"));
        }
        let scalar = sc117_gdn_stage::read(shard, &m.scalar, 5120)?;
        let gb = sc117_gdn_stage::read(shard, &m.gate, gm)?;
        let ub = sc117_gdn_stage::read(shard, &m.up, um)?;
        let db = sc117_gdn_stage::read(shard, &m.down, 1_740_800)?;
        #[cfg(unix)]
        if !sc117_gdn_stage::same(
            &before,
            &shard.metadata().map_err(|e| invalid(e.to_string()))?,
        ) {
            return Err(invalid("shared source changed"));
        }
        let bind = |name: &str, kind, rows, columns, sha256: String| MatrixBinding {
            model_revision: m.model_revision.clone(),
            layer: m.layer,
            tensor: format!("blk.{}.ffn_{name}_shexp.weight", m.layer),
            ggml_type: kind,
            rows,
            columns,
            sha256,
        };
        let gate = QuantizedMatrix::load(bind("gate", m.gate_type, 640, 2560, m.gate.sha256), &gb)?;
        let up = QuantizedMatrix::load(bind("up", m.up_type, 640, 2560, m.up.sha256), &ub)?;
        let down = QuantizedMatrix::load(bind("down", 8, 2560, 640, m.down.sha256), &db)?;
        Ok(Self {
            identity,
            scalar,
            gate,
            up,
            down,
        })
    }
    pub fn compute(&mut self, identity: &Identity, input: &[f32]) -> Result<Vec<f32>> {
        if identity != &self.identity || input.len() != 2560 {
            return Err(invalid("shared stage identity or dimensions"));
        }
        let scalar = shared_expert_cpu::scalar_gate(input, &self.scalar)?;
        let gate = self.gate.forward(input)?;
        let up = self.up.forward(input)?;
        let mid = shared_expert_cpu::activation(&gate, &up)?;
        let down = self.down.forward(&mid)?;
        down.into_iter()
            .map(|x| {
                let y = x * scalar;
                if y.is_finite() {
                    Ok(y)
                } else {
                    Err(invalid("nonfinite shared output"))
                }
            })
            .collect()
    }
}
