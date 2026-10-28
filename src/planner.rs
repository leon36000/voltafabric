use crate::cost::CostProvenance;
use crate::memory::{add, mul};
use crate::schedule::{self, schedule_with_budget, Job};
use crate::topology::{PreparedTopology, Topology};
use crate::traffic::{message_sizes, WireFormat};
use crate::work::{WorkBudget, DEFAULT_WORK_LIMIT};
use crate::{invalid, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertProfile {
    pub id: u32,
    pub resident_bytes: u64,
    pub compute_us: f64,
    #[serde(default)]
    pub cost_provenance: CostProvenance,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Demand {
    pub source_rank: u32,
    pub experts: Vec<u32>,
    pub frequency: f64,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WirePolicy {
    Baseline,
    Grouped,
    #[default]
    Adaptive,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireVariant {
    Baseline,
    Grouped,
    ByteAdaptive,
}
fn default_work_limit() -> u64 {
    DEFAULT_WORK_LIMIT
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementProblem {
    pub topology: Topology,
    pub experts: Vec<ExpertProfile>,
    pub demands: Vec<Demand>,
    pub hidden: u64,
    pub wire: WireFormat,
    pub max_iterations: usize,
    #[serde(default)]
    pub wire_policy: WirePolicy,
    #[serde(default = "default_work_limit")]
    pub max_work_units: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantCost {
    pub modeled_latency_us: f64,
    pub wire_bytes: u64,
    pub wire_messages: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemandComparison {
    pub demand_index: usize,
    pub source_rank: u32,
    pub baseline: VariantCost,
    pub grouped: VariantCost,
    pub byte_adaptive: VariantCost,
    pub selected: WireVariant,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Placement {
    pub result_kind: String,
    pub algorithm: String,
    pub owners: BTreeMap<u32, u32>,
    pub modeled_weighted_latency_us: f64,
    pub initial_latency_us: f64,
    pub iterations: usize,
    pub bytes_by_rank: BTreeMap<u32, u64>,
    pub wire_policy: WirePolicy,
    pub demand_comparisons: Vec<DemandComparison>,
    pub compute_profiles: BTreeMap<u32, CostProvenance>,
    pub link_profiles: BTreeMap<String, CostProvenance>,
    pub work_units_used: u64,
    pub termination: String,
}
struct Prepared<'a> {
    p: &'a PlacementProblem,
    topology: PreparedTopology<'a>,
    experts: BTreeMap<u32, &'a ExpertProfile>,
    sources: BTreeMap<u32, BTreeSet<u32>>,
}
impl<'a> Prepared<'a> {
    fn new(p: &'a PlacementProblem, budget: &mut WorkBudget) -> Result<Self> {
        budget.charge(1)?;
        if p.experts.is_empty()
            || p.experts.len() > 256
            || p.demands.len() > 1024
            || p.max_iterations > 4096
        {
            return Err(invalid("bounded planner limits exceeded"));
        }
        message_sizes(p.hidden, 1, p.wire, false)?;
        let topology = PreparedTopology::new(&p.topology, budget)?;
        let mut experts = BTreeMap::new();
        for e in &p.experts {
            budget.charge(1)?;
            e.cost_provenance.validate()?;
            if experts.insert(e.id, e).is_some()
                || e.resident_bytes == 0
                || !e.compute_us.is_finite()
                || e.compute_us < 0.0
            {
                return Err(invalid("invalid expert profile"));
            }
        }
        let mut sources: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
        for d in &p.demands {
            budget.charge(1)?;
            if !topology.has_rank(d.source_rank)
                || d.experts.is_empty()
                || d.experts.len() > 64
                || !d.frequency.is_finite()
                || d.frequency < 0.0
            {
                return Err(invalid("invalid exact demand or nonnegative frequency"));
            }
            budget.charge(d.experts.len())?;
            let mut unique = BTreeSet::new();
            for id in &d.experts {
                if !unique.insert(id) || !experts.contains_key(id) {
                    return Err(invalid("invalid demand expert"));
                }
                sources.entry(*id).or_default().insert(d.source_rank);
            }
        }
        Ok(Self {
            p,
            topology,
            experts,
            sources,
        })
    }
    fn memory(
        &self,
        owners: &BTreeMap<u32, u32>,
        budget: &mut WorkBudget,
    ) -> Result<BTreeMap<u32, u64>> {
        if owners.len() != self.p.experts.len() {
            return Err(invalid("one owner per expert required"));
        }
        budget.charge(self.p.topology.devices.len())?;
        let mut used: BTreeMap<_, _> = self
            .p
            .topology
            .devices
            .iter()
            .map(|d| (d.rank, 0u64))
            .collect();
        for e in &self.p.experts {
            budget.charge(1)?;
            let rank = owners.get(&e.id).ok_or_else(|| invalid("missing owner"))?;
            let x = used
                .get_mut(rank)
                .ok_or_else(|| invalid("unknown owner rank"))?;
            *x = add(*x, e.resident_bytes)?;
        }
        for d in &self.p.topology.devices {
            budget.charge(1)?;
            if used[&d.rank] > d.capacity_bytes - d.reserved_bytes {
                return Err(Error::new(
                    "CAPACITY_EXCEEDED",
                    "resident placement exceeds usable memory",
                ));
            }
        }
        Ok(used)
    }
    fn reachable(&self, expert: u32, rank: u32, budget: &mut WorkBudget) -> Result<bool> {
        if let Some(sources) = self.sources.get(&expert) {
            for &src in sources {
                budget.charge(1)?;
                for (a, b) in [(src, rank), (rank, src)] {
                    match self.topology.transfer(a, b, 1, budget) {
                        Ok(_) => {}
                        Err(e) if e.code == "NO_PATH" => return Ok(false),
                        Err(e) => return Err(e),
                    }
                }
            }
        }
        Ok(true)
    }
    fn graph(
        &self,
        d: &Demand,
        owners: &BTreeMap<u32, u32>,
        variant: WireVariant,
        budget: &mut WorkBudget,
    ) -> Result<VariantCost> {
        let mut by_rank: BTreeMap<u32, Vec<&ExpertProfile>> = BTreeMap::new();
        for id in &d.experts {
            budget.charge(1)?;
            by_rank
                .entry(owners[id])
                .or_default()
                .push(self.experts[id]);
        }
        let mut jobs = Vec::new();
        let mut bytes = 0;
        let mut messages = 0;
        for (rank, experts) in by_rank {
            budget.charge(1)?;
            let gpu = format!("vf:gpu:{rank}");
            if rank == d.source_rank {
                for e in experts {
                    budget.charge(1)?;
                    jobs.push(Job {
                        id: format!("local:{}", e.id),
                        duration_us: e.compute_us,
                        depends_on: vec![],
                        resources: vec![gpu.clone()],
                    });
                }
                continue;
            }
            let c = experts.len() as u64;
            let (bi, bo, bn) = message_sizes(self.p.hidden, c, self.p.wire, false)?;
            let (gi, go, _) = message_sizes(self.p.hidden, c, self.p.wire, true)?;
            let use_group = match variant {
                WireVariant::Baseline => false,
                WireVariant::Grouped => true,
                WireVariant::ByteAdaptive => add(gi, go)? < mul(add(bi, bo)?, bn)?,
            };
            if use_group {
                let (ti, ri) = self.topology.transfer(d.source_rank, rank, gi, budget)?;
                let (to, ro) = self.topology.transfer(rank, d.source_rank, go, budget)?;
                bytes = add(bytes, add(gi, go)?)?;
                messages = add(messages, 2)?;
                budget.charge(2)?;
                // '/' sorts before digits, so numeric IDs keep their old relative order.
                let input = format!("in:{rank}/rank");
                jobs.push(Job {
                    id: input.clone(),
                    duration_us: ti,
                    depends_on: vec![],
                    resources: ri,
                });
                let mut deps = Vec::new();
                for e in experts {
                    budget.charge(1)?;
                    let id = format!("expert:{}", e.id);
                    deps.push(id.clone());
                    jobs.push(Job {
                        id,
                        duration_us: e.compute_us,
                        depends_on: vec![input.clone()],
                        resources: vec![gpu.clone()],
                    });
                }
                jobs.push(Job {
                    id: format!("out:{rank}/rank"),
                    duration_us: to,
                    depends_on: deps,
                    resources: ro,
                });
            } else {
                bytes = add(bytes, mul(add(bi, bo)?, bn)?)?;
                messages = add(messages, mul(2, bn)?)?;
                for e in experts {
                    budget.charge(3)?;
                    let (ti, ri) = self.topology.transfer(d.source_rank, rank, bi, budget)?;
                    let (to, ro) = self.topology.transfer(rank, d.source_rank, bo, budget)?;
                    let input = format!("in:{}/expert", e.id);
                    let compute = format!("expert:{}", e.id);
                    jobs.push(Job {
                        id: input.clone(),
                        duration_us: ti,
                        depends_on: vec![],
                        resources: ri,
                    });
                    jobs.push(Job {
                        id: compute.clone(),
                        duration_us: e.compute_us,
                        depends_on: vec![input],
                        resources: vec![gpu.clone()],
                    });
                    jobs.push(Job {
                        id: format!("out:{}/expert", e.id),
                        duration_us: to,
                        depends_on: vec![compute],
                        resources: ro,
                    });
                }
            }
        }
        Ok(VariantCost {
            modeled_latency_us: schedule_with_budget(&jobs, budget)?.total_us,
            wire_bytes: bytes,
            wire_messages: messages,
        })
    }
    fn evaluate(
        &self,
        owners: &BTreeMap<u32, u32>,
        budget: &mut WorkBudget,
    ) -> Result<(f64, Vec<DemandComparison>)> {
        self.memory(owners, budget)?;
        use num_bigint::BigUint;

        // Accumulate the exact dyadic numerator and denominator from raw inputs.
        // This avoids both a lossy normalized-frequency ratio and intermediate
        // f64 overflow; one correctly rounded f64 conversion happens at the end.
        let mut numerator = BigUint::default();
        let mut denominator = BigUint::default();
        let mut comparisons = Vec::new();
        for (index, d) in self.p.demands.iter().enumerate() {
            budget.charge(1)?;
            let baseline = self.graph(d, owners, WireVariant::Baseline, budget)?;
            let grouped = self.graph(d, owners, WireVariant::Grouped, budget)?;
            let byte_adaptive = self.graph(d, owners, WireVariant::ByteAdaptive, budget)?;
            let (selected, cost) = match self.p.wire_policy {
                WirePolicy::Baseline => (WireVariant::Baseline, baseline.modeled_latency_us),
                WirePolicy::Grouped => (WireVariant::Grouped, grouped.modeled_latency_us),
                WirePolicy::Adaptive => {
                    // Whole graphs, including resources shared across destinations. Baseline wins ties.
                    let mut chosen = (WireVariant::Baseline, baseline.modeled_latency_us);
                    for candidate in [
                        (WireVariant::Grouped, grouped.modeled_latency_us),
                        (WireVariant::ByteAdaptive, byte_adaptive.modeled_latency_us),
                    ] {
                        if candidate.1 < chosen.1 {
                            chosen = candidate;
                        }
                    }
                    chosen
                }
            };
            add_exact_weighted_term(&mut numerator, &mut denominator, d.frequency, cost);
            comparisons.push(DemandComparison {
                demand_index: index,
                source_rank: d.source_rank,
                baseline,
                grouped,
                byte_adaptive,
                selected,
            });
        }
        let weighted_mean = if self.p.demands.is_empty() {
            0.0
        } else {
            exact_weighted_mean(&numerator, &denominator)?
        };
        Ok((weighted_mean, comparisons))
    }
}
// Add one exact dyadic frequency*cost product and its frequency denominator.
// Binary64 exponents guarantee nonnegative shifts in the common bases used here:
// frequency uses 2^-1074; products use 2^-2148.
fn add_exact_weighted_term(
    numerator: &mut num_bigint::BigUint,
    denominator: &mut num_bigint::BigUint,
    frequency: f64,
    cost: f64,
) {
    use num_bigint::BigUint;

    fn dyadic(x: f64) -> (u64, i32) {
        let bits = x.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1_u64 << 52) - 1);
        if exponent == 0 {
            (fraction, -1074)
        } else {
            ((1_u64 << 52) | fraction, exponent - 1023 - 52)
        }
    }

    let (weight, weight_exponent) = dyadic(frequency);
    let (cost_significand, cost_exponent) = dyadic(cost);
    *denominator += BigUint::from(weight) << (weight_exponent + 1074) as usize;
    if cost_significand != 0 {
        *numerator += (BigUint::from(weight) * BigUint::from(cost_significand))
            << (weight_exponent + cost_exponent + 2148) as usize;
    }
}

// Convert N/D units of 2^-1074 directly to nearest-even binary64. This single
// exact path handles subnormals, normal values, and boundary ties without an
// approximate threshold. A positive exact result that rounds to zero is rejected.
fn exact_weighted_mean(
    numerator: &num_bigint::BigUint,
    denominator: &num_bigint::BigUint,
) -> Result<f64> {
    use num_bigint::BigUint;
    use std::cmp::Ordering;

    if denominator == &BigUint::default() {
        return Err(invalid("at least one demand frequency must be positive"));
    }
    if numerator == &BigUint::default() {
        return Ok(0.0);
    }
    let floor = numerator / denominator;
    let shift = floor.bits().saturating_sub(53);
    let scaled_denominator = denominator << shift as usize;
    let quotient = numerator / &scaled_denominator;
    let remainder = numerator % &scaled_denominator;
    let digits = quotient.to_u64_digits();
    if digits.len() > 1 {
        return Err(invalid("weighted objective rounding overflow"));
    }
    let mut significand = digits.first().copied().unwrap_or(0);
    match (&remainder << 1usize).cmp(&scaled_denominator) {
        Ordering::Greater => significand += 1,
        Ordering::Equal if significand & 1 == 1 => significand += 1,
        _ => {}
    }
    let mut shift = shift;
    if significand == (1_u64 << 53) {
        significand >>= 1;
        shift += 1;
    }
    if significand == 0 {
        return Err(invalid(
            "weighted objective is below representable f64 range",
        ));
    }
    if shift == 0 {
        return Ok(f64::from_bits(significand));
    }
    if !((1_u64 << 52)..(1_u64 << 53)).contains(&significand) {
        return Err(invalid("weighted objective rounding invariant failed"));
    }
    let exponent_field = shift + 1;
    if exponent_field > 2046 {
        return Err(invalid("weighted objective overflow"));
    }
    let fraction = significand - (1_u64 << 52);
    Ok(f64::from_bits((exponent_field << 52) | fraction))
}

/// Request-weighted analytical makespan; each demand is an isolated request, not a concurrency model.
pub fn score(p: &PlacementProblem, owners: &BTreeMap<u32, u32>) -> Result<f64> {
    let mut budget = WorkBudget::new(p.max_work_units)?;
    Prepared::new(p, &mut budget)?
        .evaluate(owners, &mut budget)
        .map(|x| x.0)
}
fn consider(
    prepared: &Prepared<'_>,
    candidate: BTreeMap<u32, u32>,
    best: &mut BTreeMap<u32, u32>,
    best_score: &mut f64,
    budget: &mut WorkBudget,
) -> Result<()> {
    let value = match prepared.evaluate(&candidate, budget) {
        Ok((value, _)) => value,
        Err(e) if e.code == "CAPACITY_EXCEEDED" || e.code == "NO_PATH" => return Ok(()),
        // An alternative whose modelled time overflows can never beat the finite current
        // placement: it is discarded like an infeasible one, not fatal to the whole plan.
        Err(e)
            if e.code == "TRANSFER_TIME_OVERFLOW"
                || (e.code == "INVALID_INPUT" && e.message == schedule::TIME_OVERFLOW) =>
        {
            return Ok(())
        }
        Err(e) => return Err(e),
    };
    if value < *best_score - 1e-12 {
        *best_score = value;
        *best = candidate;
    }
    Ok(())
}
/// Best-fit plus bounded local moves/swaps; neither global placement nor list scheduling is optimal.
pub fn plan(p: &PlacementProblem) -> Result<Placement> {
    let mut budget = WorkBudget::new(p.max_work_units)?;
    let prepared = Prepared::new(p, &mut budget)?;
    budget.charge(p.experts.len())?;
    let mut sorted: Vec<_> = p.experts.iter().collect();
    sorted.sort_by_key(|e| (std::cmp::Reverse(e.resident_bytes), e.id));
    budget.charge(p.topology.devices.len())?;
    let mut ranks: Vec<_> = p.topology.devices.iter().map(|d| d.rank).collect();
    ranks.sort_unstable();
    budget.charge(p.topology.devices.len())?;
    let mut remaining: BTreeMap<_, _> = p
        .topology
        .devices
        .iter()
        .map(|d| (d.rank, d.capacity_bytes - d.reserved_bytes))
        .collect();
    let mut total = 0;
    for e in &p.experts {
        budget.charge(1)?;
        total = add(total, e.resident_bytes)?;
    }
    let mut free = 0;
    for &bytes in remaining.values() {
        budget.charge(1)?;
        free = add(free, bytes)?;
    }
    if total > free {
        return Err(Error::new(
            "CAPACITY_EXCEEDED",
            "aggregate resident bytes exceed aggregate usable memory",
        ));
    }
    let mut owners = BTreeMap::new();
    for e in sorted {
        budget.charge(1)?;
        let mut best: Option<(u64, u32)> = None;
        for &rank in &ranks {
            budget.charge(1)?;
            if remaining[&rank] < e.resident_bytes
                || !prepared.reachable(e.id, rank, &mut budget)?
            {
                continue;
            }
            let candidate = (remaining[&rank] - e.resident_bytes, rank);
            if best.is_none_or(|b| candidate < b) {
                best = Some(candidate);
            }
        }
        let rank = best
            .ok_or_else(|| {
                Error::new(
                    "GREEDY_PLACEMENT_FAILED",
                    "no feasible greedy assignment; NOT proof of global infeasibility",
                )
            })?
            .1;
        *remaining
            .get_mut(&rank)
            .ok_or_else(|| invalid("owner missing"))? -= e.resident_bytes;
        owners.insert(e.id, rank);
    }
    let initial = prepared.evaluate(&owners, &mut budget)?.0;
    let mut current = initial;
    let mut evaluated = 0;
    let mut termination = "candidate_limit";
    budget.charge(owners.len())?;
    let ids: Vec<_> = owners.keys().copied().collect();
    while evaluated < p.max_iterations {
        budget.charge(owners.len())?;
        let mut best = owners.clone();
        let mut best_score = current;
        'moves: for &id in &ids {
            for &rank in &ranks {
                budget.charge(1)?;
                if evaluated >= p.max_iterations {
                    break 'moves;
                }
                if owners[&id] != rank {
                    budget.charge(owners.len())?;
                    let mut candidate = owners.clone();
                    candidate.insert(id, rank);
                    evaluated += 1;
                    consider(
                        &prepared,
                        candidate,
                        &mut best,
                        &mut best_score,
                        &mut budget,
                    )?;
                }
            }
        }
        'swaps: for (i, &a) in ids.iter().enumerate() {
            for &b in &ids[i + 1..] {
                budget.charge(1)?;
                if evaluated >= p.max_iterations {
                    break 'swaps;
                }
                if owners[&a] != owners[&b] {
                    budget.charge(owners.len())?;
                    let mut candidate = owners.clone();
                    candidate.insert(a, owners[&b]);
                    candidate.insert(b, owners[&a]);
                    evaluated += 1;
                    consider(
                        &prepared,
                        candidate,
                        &mut best,
                        &mut best_score,
                        &mut budget,
                    )?;
                }
            }
        }
        if best_score >= current - 1e-12 {
            if evaluated < p.max_iterations {
                termination = "local_search_stopped";
            }
            break;
        }
        owners = best;
        current = best_score;
    }
    let bytes_by_rank = prepared.memory(&owners, &mut budget)?;
    let (verified, demand_comparisons) = prepared.evaluate(&owners, &mut budget)?;
    // Final diagnostics consume the same budget; exhausted work never returns a partial placement.
    budget.charge(p.experts.len())?;
    let compute_profiles = p
        .experts
        .iter()
        .map(|e| (e.id, e.cost_provenance.clone()))
        .collect();
    budget.charge(p.topology.links.len())?;
    let link_profiles = p
        .topology
        .links
        .iter()
        .map(|l| (l.id.clone(), l.cost_provenance.clone()))
        .collect();
    Ok(Placement {
        result_kind: "analytical_resident_placement_not_gpu_measurement".into(),
        algorithm: "best_fit_then_bounded_moves_and_swaps_latency_adaptive_wire".into(),
        owners,
        modeled_weighted_latency_us: verified,
        initial_latency_us: initial,
        iterations: evaluated,
        bytes_by_rank,
        wire_policy: p.wire_policy,
        demand_comparisons,
        compute_profiles,
        link_profiles,
        work_units_used: budget.used(),
        termination: termination.into(),
    })
}
