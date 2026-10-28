use crate::memory::add;
use crate::{invalid, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layer {
    pub bytes: u64,
    pub compute_us_by_stage: Vec<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub rank: u32,
    pub available_bytes: u64,
    pub boundary_after_us: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineProblem {
    pub layers: Vec<Layer>,
    pub stages: Vec<Stage>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub rank: u32,
    pub first_layer: usize,
    pub end_layer: usize,
    pub bytes: u64,
    pub compute_us: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pipeline {
    pub result_kind: String,
    pub segments: Vec<Segment>,
    pub bottleneck_proxy_us: f64,
    pub serial_latency_us: f64,
}

/// Exact DP for this fixed-stage-order, additive, contiguous-layer cost model only.
/// The bottleneck proxy is not a throughput prediction; serial latency is separate.
/// Segment times are direct left-to-right sums, never differences of prefix sums.
pub fn partition(p: &PipelineProblem) -> Result<Pipeline> {
    let n = p.layers.len();
    let m = p.stages.len();
    if m == 0 || m > 64 || n < m || n > 1024 {
        return Err(invalid(
            "require 1..64 nonempty stages and up to 1024 layers",
        ));
    }
    let mut ranks = BTreeSet::new();
    for s in &p.stages {
        if !ranks.insert(s.rank)
            || s.available_bytes == 0
            || !s.boundary_after_us.is_finite()
            || s.boundary_after_us < 0.0
        {
            return Err(invalid("invalid pipeline stage"));
        }
    }
    if p.stages[m - 1].boundary_after_us != 0.0 {
        return Err(invalid("last stage cannot have a following boundary"));
    }
    let mut bytes = vec![0u64; n + 1];
    let mut row_totals = vec![0.0_f64; m];
    for (i, l) in p.layers.iter().enumerate() {
        if l.compute_us_by_stage.len() != m
            || l.compute_us_by_stage
                .iter()
                .any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err(invalid("invalid layer timing dimensions"));
        }
        bytes[i + 1] = add(bytes[i], l.bytes)?;
        // Legacy input policy: each whole per-stage profile must have a finite sum,
        // including costs on layers never assigned to this stage. Capacity priority
        // below applies only after this profile-domain validation.
        // Left-to-right row sums bound every segment sum below, so segment sums cannot overflow.
        for (total, x) in row_totals.iter_mut().zip(&l.compute_us_by_stage) {
            *total += x;
            if !total.is_finite() {
                return Err(invalid("pipeline timing overflow"));
            }
        }
    }
    let mut dp = vec![vec![f64::INFINITY; n + 1]; m + 1];
    let mut prev = vec![vec![None; n + 1]; m + 1];
    dp[0][0] = 0.0;
    let mut cost_overflow = false;
    for k in 1..=m {
        let stage = &p.stages[k - 1];
        let last_end = n - (m - k);
        for start in (k - 1)..last_end {
            let before = dp[k - 1][start];
            if !before.is_finite() {
                continue;
            }
            let mut segment = 0.0;
            for end in (start + 1)..=last_end {
                if bytes[end] - bytes[start] > stage.available_bytes {
                    break;
                }
                segment += p.layers[end - 1].compute_us_by_stage[k - 1];
                let cost = segment + stage.boundary_after_us;
                if !cost.is_finite() {
                    // Its true cost exceeds every finite candidate, so it is never selected.
                    cost_overflow = true;
                    break;
                }
                let candidate = before.max(cost);
                if candidate < dp[k][end] {
                    dp[k][end] = candidate;
                    prev[k][end] = Some(start);
                }
            }
        }
    }
    if !dp[m][n].is_finite() {
        if cost_overflow && capacity_partition_exists(p, &bytes) {
            return Err(invalid("pipeline timing overflow"));
        }
        return Err(Error::new(
            "PIPELINE_INFEASIBLE",
            "no contiguous split fits this ordered stage model",
        ));
    }
    let mut out = Vec::new();
    let mut end = n;
    for k in (1..=m).rev() {
        let start = prev[k][end].ok_or_else(|| Error::new("INTERNAL", "missing DP predecessor"))?;
        let mut compute_us = 0.0;
        for l in &p.layers[start..end] {
            compute_us += l.compute_us_by_stage[k - 1];
        }
        out.push(Segment {
            rank: p.stages[k - 1].rank,
            first_layer: start,
            end_layer: end,
            bytes: bytes[end] - bytes[start],
            compute_us,
        });
        end = start;
    }
    out.reverse();
    let serial = out.iter().map(|s| s.compute_us).sum::<f64>()
        + p.stages.iter().map(|s| s.boundary_after_us).sum::<f64>();
    if !serial.is_finite() {
        return Err(invalid("pipeline serial sum overflow"));
    }
    Ok(Pipeline {
        result_kind: "analytical_fixed_order_partition_not_gpu_measurement".into(),
        segments: out,
        bottleneck_proxy_us: dp[m][n],
        serial_latency_us: serial,
    })
}

// Capacity-only reachability is needed solely when no finite numerical plan
// survives an overflowing candidate. Otherwise overflow on an irrelevant path
// could hide an ordered capacity failure. Two rolling rows: O(n) extra space,
// O(m*n²) bounded work, executed only on this error path after input validation.
fn capacity_partition_exists(p: &PipelineProblem, bytes: &[u64]) -> bool {
    let n = p.layers.len();
    let m = p.stages.len();
    let mut reachable = vec![false; n + 1];
    reachable[0] = true;
    for (k, stage) in p.stages.iter().enumerate() {
        let last_end = n - (m - k - 1);
        let mut next = vec![false; n + 1];
        for (start, &valid) in reachable.iter().enumerate().take(last_end).skip(k) {
            if !valid {
                continue;
            }
            for (end, slot) in next
                .iter_mut()
                .enumerate()
                .take(last_end + 1)
                .skip(start + 1)
            {
                if bytes[end] - bytes[start] > stage.available_bytes {
                    break;
                }
                *slot = true;
            }
        }
        reachable = next;
    }
    reachable[n]
}
