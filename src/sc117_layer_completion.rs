//! In-memory SC117 CPU layer completion. Durable reservation and execution provenance are caller-owned.
use crate::{
    completion::{self, RoutedRawOutput},
    hyperconnection_cpu, invalid,
    moe::{Collector, Stamp},
    routing::RoutingProblem,
    shared_expert_cpu, Result,
};
#[derive(Clone, Debug)]
pub struct StampedValues {
    pub stamp: Stamp,
    pub attempt: u64,
    pub values: Vec<f32>,
}
pub struct LayerInputs {
    pub problem: RoutingProblem,
    pub attempt: u64,
    pub outputs: Vec<RoutedRawOutput>,
    pub shared: StampedValues,
    pub attention_residual: StampedValues,
    pub injection: StampedValues,
}

/// Combine a session-validated result; this does not validate shared/attention producer provenance.
#[cfg(feature = "native-ggml-cpu")]
pub(crate) fn finish_native(
    collected: &crate::session::NativeCollected,
    shared: StampedValues,
    attention_residual: StampedValues,
    injection: StampedValues,
) -> Result<Vec<f32>> {
    let stamp = collected.stamp();
    let attempt = collected.attempt();
    if [&shared, &attention_residual, &injection]
        .iter()
        .any(|x| &x.stamp != stamp || x.attempt != attempt)
        || shared.values.len() != 2560
        || attention_residual.values.len() != 10240
        || injection.values.len() != 4
        || collected.rows().len() != 1
        || collected.rows()[0].len() != 2560
    {
        return Err(invalid("SC117 native completed identity or shape"));
    }
    let combined = shared_expert_cpu::combine(&collected.rows()[0], &shared.values)?;
    hyperconnection_cpu::write(&attention_residual.values, &combined, &injection.values)
}
/// Collect all unweighted FP16 rows in canonical slot order; add the already scaled shared row once,
/// cast f64 to f32 and write the four residual streams. Caller verifies producer provenance and admission.
pub fn finish(inputs: LayerInputs) -> Result<Vec<f32>> {
    let LayerInputs {
        problem,
        attempt,
        outputs,
        shared,
        attention_residual,
        injection,
    } = inputs;
    if attempt == 0
        || problem.hidden != 2560
        || problem.routes.len() != 1
        || outputs.is_empty()
        || outputs.len() > 64
        || outputs.iter().any(|packet| packet.rows == 0)
        || [&shared, &attention_residual, &injection]
            .iter()
            .any(|v| v.stamp != problem.stamp || v.attempt != attempt)
        || shared.values.len() != 2560
        || attention_residual.values.len() != 10240
        || injection.values.len() != 4
        || shared
            .values
            .iter()
            .chain(&attention_residual.values)
            .chain(&injection.values)
            .any(|x| !x.is_finite())
    {
        return Err(invalid(
            "SC117 layer completion identity, shape or finite value",
        ));
    }
    let mut collector = Collector::from_routing(&problem)?;
    for row in outputs {
        completion::accept_routed_output(attempt, row, &mut collector)?;
    }
    let mut rows = collector.finish()?;
    if rows.len() != 1 {
        return Err(invalid("SC117 collector token count"));
    }
    let routed = rows.pop().ok_or_else(|| invalid("SC117 collector empty"))?;
    let combined = shared_expert_cpu::combine(&routed, &shared.values)?;
    hyperconnection_cpu::write(&attention_residual.values, &combined, &injection.values)
}
