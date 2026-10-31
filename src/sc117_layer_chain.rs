//! Persistent CPU segment of real reservation-bound SC117 layers.
//! The local ledger is not an R53 lease. A segment is not a whole model.
use crate::{
    gdn_cpu::Identity,
    invalid,
    moe::{Registry, Stamp},
    native_ledger::{NativeBatchLedger, Reservation},
    sc117_bound_layer, sc117_ffn_stage, sc117_gdn_stage, sc117_ple, sc117_ple_post,
    sc117_ple_weights, sc117_post_attention, sc117_qsa_bound_layer, sc117_qsa_stage,
    sc117_shared_stage,
    session::Session,
    Result,
};
use std::{collections::BTreeMap, time::Duration};

/// Sessions must already contain the verified resident stacks described by the registry.
pub struct Owners {
    pub sessions: Vec<Session>,
    pub registry: Registry,
    pub preferred: BTreeMap<u32, u32>,
    pub order: BTreeMap<u32, Vec<u32>>,
}
/// Caller-supplied, authenticated material and placement for one missing native expert.
pub struct ExpertAdmission {
    pub binding: crate::native_ggml_cpu::Binding,
    pub index: String,
    pub pack_bytes: u64,
    pub bytes: Vec<u8>,
}
/// A source is consulted only for experts absent from the current registry.
/// It must supply an independently trusted digest and enforce its own resource budget.
pub trait MissingExpertSource {
    /// Optional side-effect-free check before any missing owner is appended.
    /// Implementations which cannot predict admission may leave this as a no-op.
    fn preflight(
        &mut self,
        _stamp: &Stamp,
        _attempt: u64,
        _experts: &[u32],
        _registry: &Registry,
        _order: &BTreeMap<u32, Vec<u32>>,
    ) -> Result<()> {
        Ok(())
    }
    fn load(
        &mut self,
        stamp: &Stamp,
        attempt: u64,
        expert: u32,
        registry: &Registry,
        resident_order: &BTreeMap<u32, Vec<u32>>,
    ) -> Result<ExpertAdmission>;
}
impl Owners {
    fn append(
        &mut self,
        binding: &crate::native_ggml_cpu::Binding,
        index: &str,
        pack_bytes: u64,
        bytes: &[u8],
        timeout: Duration,
    ) -> Result<()> {
        let order = self
            .order
            .get_mut(&binding.rank)
            .ok_or_else(|| invalid("segment append rank order"))?;
        if order.contains(&binding.local)
            || self.registry.copies.iter().any(|r| {
                r.rank == binding.rank && (r.local == binding.local || r.expert == binding.expert)
            })
        {
            return Err(invalid("segment append duplicate expert or local"));
        }
        let mut matches = self
            .sessions
            .iter_mut()
            .filter(|s| s.rank() == binding.rank);
        let session = matches
            .next()
            .ok_or_else(|| invalid("segment append session missing"))?;
        if matches.next().is_some() {
            return Err(invalid("segment append duplicate rank"));
        }
        session.extend_native_ggml(binding, index, pack_bytes, bytes, timeout)?;
        self.registry.copies.push(crate::moe::Replica {
            expert: binding.expert,
            rank: binding.rank,
            local: binding.local,
        });
        order.push(binding.local);
        Ok(())
    }
    fn ensure(
        &mut self,
        stamp: &Stamp,
        attempt: u64,
        choices: &[crate::sc117_router_cpu::Choice],
        source: &mut Option<&mut dyn MissingExpertSource>,
        timeout: Duration,
    ) -> Result<()> {
        let Some(source) = source.as_deref_mut() else {
            return Ok(());
        };
        let mut seen = std::collections::BTreeSet::new();
        let missing: Vec<u32> = choices
            .iter()
            .map(|choice| choice.expert)
            .filter(|expert| {
                !self
                    .registry
                    .copies
                    .iter()
                    .any(|copy| copy.expert == *expert)
            })
            .filter(|expert| seen.insert(*expert))
            .collect();
        if !missing.is_empty() {
            source.preflight(stamp, attempt, &missing, &self.registry, &self.order)?;
        }
        for expert in missing {
            let material = source.load(stamp, attempt, expert, &self.registry, &self.order)?;
            if material.binding.model_revision != stamp.model_revision
                || material.binding.layer != stamp.layer
                || material.binding.epoch != stamp.epoch
                || material.binding.expert != expert
            {
                return Err(invalid(
                    "missing expert source returned a different identity",
                ));
            }
            self.append(
                &material.binding,
                &material.index,
                material.pack_bytes,
                &material.bytes,
                timeout,
            )?;
        }
        Ok(())
    }
}
enum Attention {
    Gdn(sc117_gdn_stage::Stage, sc117_post_attention::Stage),
    Qsa(sc117_qsa_stage::Stage, sc117_ffn_stage::Stage),
}
struct CapturedQsa {
    input: Vec<f32>,
    output: sc117_qsa_stage::Output,
}
pub struct Layer {
    identity: Identity,
    attention: Attention,
    shared: sc117_shared_stage::Stage,
    owners: Owners,
    ple: Option<Ple>,
    last_selected: Vec<u32>,
    last_used: BTreeMap<u32, u64>,
    last_active_ranks: std::collections::BTreeSet<u32>,
}

struct Ple {
    table: sc117_ple::Table,
    weights: sc117_ple_weights::Weights,
    state: PleState,
}
#[derive(Clone)]
struct PleState {
    history: Vec<f32>,
    previous: [i32; 2],
    position: u64,
}
impl PleState {
    fn new() -> Self {
        Self {
            history: vec![0.0; sc117_ple_post::CHANNELS * sc117_ple_post::HISTORY],
            previous: [sc117_ple::NULL_TOKEN; 2],
            position: 0,
        }
    }
    fn prepare<F>(
        &self,
        position: u64,
        token: i32,
        input: &[f32],
        produce: F,
    ) -> Result<sc117_ple_post::Output>
    where
        F: FnOnce(i32, [i32; 2], &[f32], &[f32]) -> Result<sc117_ple_post::Output>,
    {
        if position != self.position
            || position == u64::MAX
            || !(0..sc117_ple::VOCAB).contains(&token)
        {
            return Err(invalid("PLE position or token"));
        }
        let candidate = produce(token, self.previous, input, &self.history)?;
        if candidate.residual.len() != sc117_ple_post::CHANNELS
            || candidate.next_history.len() != sc117_ple_post::CHANNELS * sc117_ple_post::HISTORY
            || candidate
                .residual
                .iter()
                .chain(candidate.next_history.iter())
                .any(|x| !x.is_finite())
        {
            return Err(invalid("invalid PLE candidate"));
        }
        Ok(candidate)
    }
    fn commit(&mut self, token: i32, candidate: sc117_ple_post::Output) {
        self.history = candidate.next_history;
        self.previous = [self.previous[1], token];
        self.position += 1;
    }
    fn prepare_batch<F>(
        &self,
        position: u64,
        tokens: &[i32],
        inputs: &[Vec<f32>],
        mut produce: F,
    ) -> Result<(Self, Vec<Vec<f32>>)>
    where
        F: FnMut(i32, [i32; 2], &[f32], &[f32]) -> Result<sc117_ple_post::Output>,
    {
        if tokens.is_empty()
            || tokens.len() > 128
            || tokens.len() != inputs.len()
            || position != self.position
            || position.checked_add(tokens.len() as u64).is_none()
            || tokens.iter().any(|t| !(0..sc117_ple::VOCAB).contains(t))
        {
            return Err(invalid("PLE batch identity or size"));
        }
        let mut candidate = self.clone();
        let mut residuals = Vec::with_capacity(inputs.len());
        for (&token, input) in tokens.iter().zip(inputs) {
            let next = candidate.prepare(candidate.position, token, input, &mut produce)?;
            residuals.push(next.residual.clone());
            candidate.commit(token, next);
        }
        Ok((candidate, residuals))
    }
    fn finish(
        &mut self,
        token: i32,
        candidate: sc117_ple_post::Output,
        layer_result: Result<Vec<f32>>,
    ) -> Result<Vec<f32>> {
        let output = layer_result?;
        self.commit(token, candidate);
        Ok(output)
    }
}
impl Layer {
    pub fn gdn(
        identity: Identity,
        gdn: sc117_gdn_stage::Stage,
        post: sc117_post_attention::Stage,
        shared: sc117_shared_stage::Stage,
        owners: Owners,
    ) -> Result<Self> {
        if identity.layer >= 48 || identity.layer % 4 == 3 || gdn.next_token() != 0 {
            return Err(invalid("invalid initial GDN layer"));
        }
        Ok(Self {
            identity,
            attention: Attention::Gdn(gdn, post),
            shared,
            owners,
            ple: None,
            last_selected: Vec::new(),
            last_used: BTreeMap::new(),
            last_active_ranks: std::collections::BTreeSet::new(),
        })
    }
    pub fn qsa(
        identity: Identity,
        qsa: sc117_qsa_stage::Stage,
        ffn: sc117_ffn_stage::Stage,
        shared: sc117_shared_stage::Stage,
        owners: Owners,
    ) -> Result<Self> {
        if identity.layer >= 48 || identity.layer % 4 != 3 || !qsa.is_empty() {
            return Err(invalid("invalid initial QSA layer"));
        }
        Ok(Self {
            identity,
            attention: Attention::Qsa(qsa, ffn),
            shared,
            owners,
            ple: None,
            last_selected: Vec::new(),
            last_used: BTreeMap::new(),
            last_active_ranks: std::collections::BTreeSet::new(),
        })
    }
    /// Attach authenticated PLE inputs before constructing a segment. The caller must
    /// authenticate the table and manifest and bind them to this model revision.
    pub fn attach_ple(
        mut self,
        table: sc117_ple::Table,
        weights: sc117_ple_weights::Weights,
    ) -> Result<Self> {
        if self.identity.layer != 1
            || !matches!(self.attention, Attention::Gdn(..))
            || self.ple.is_some()
            || weights.model_revision() != self.identity.model_revision
        {
            return Err(invalid("PLE layer or model identity"));
        }
        self.ple = Some(Ple {
            table,
            weights,
            state: PleState::new(),
        });
        Ok(self)
    }
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        ledger: &NativeBatchLedger,
        ticket: Reservation,
        position: u64,
        input: &[f32],
        token: Option<i32>,
        timeout: Duration,
        source: &mut Option<&mut dyn MissingExpertSource>,
        capture_qsa: bool,
        qsa_trace: &mut Option<CapturedQsa>,
    ) -> Result<Vec<f32>> {
        let stamp = ticket.stamp().clone();
        let attempt = ticket.attempt();
        let before: BTreeMap<_, _> = self
            .owners
            .sessions
            .iter()
            .map(|s| (s.rank(), s.traffic.execs))
            .collect();
        let candidate = if let Some(ple) = &self.ple {
            let token = token.ok_or_else(|| invalid("PLE requires token"))?;
            Some(ple.state.prepare(
                position,
                token,
                input,
                |token, previous, residual, history| {
                    let (_, emb) = ple.table.gather(token, previous)?;
                    ple.weights.evaluate(&emb, residual, history)
                },
            )?)
        } else {
            None
        };
        let input = candidate.as_ref().map_or(input, |p| p.residual.as_slice());
        let mut selected = Vec::new();
        let output = match &mut self.attention {
            Attention::Gdn(gdn, post) => {
                let proposal = gdn.prepare_reserved(ledger, &ticket, position, input)?;
                let routed =
                    sc117_bound_layer::Routed::prepare(post, ledger, &ticket, proposal, input)?;
                selected.extend(routed.choices().iter().map(|choice| choice.expert));
                self.owners
                    .ensure(&stamp, attempt, routed.choices(), source, timeout)?;
                let shared = routed.shared(&mut self.shared)?;
                let collected = routed.dispatch(
                    &mut self.owners.sessions,
                    self.owners.registry.clone(),
                    self.owners.preferred.clone(),
                    self.owners.order.clone(),
                    ledger,
                    ticket,
                    timeout,
                )?;
                Ok(routed.commit(gdn, shared, collected)?.residual)
            }
            Attention::Qsa(qsa, ffn) => {
                let pos =
                    usize::try_from(position).map_err(|_| invalid("QSA position overflow"))?;
                let routed =
                    sc117_qsa_bound_layer::Routed::prepare(qsa, ffn, ledger, &ticket, pos, input)?;
                selected.extend(routed.choices().iter().map(|choice| choice.expert));
                self.owners
                    .ensure(&stamp, attempt, routed.choices(), source, timeout)?;
                let shared = routed.shared(&mut self.shared)?;
                let collected = routed.dispatch(
                    &mut self.owners.sessions,
                    self.owners.registry.clone(),
                    self.owners.preferred.clone(),
                    self.owners.order.clone(),
                    ledger,
                    ticket,
                    timeout,
                )?;
                let committed = routed.commit(qsa, shared, collected)?;
                if capture_qsa {
                    *qsa_trace = Some(CapturedQsa {
                        input: input.to_vec(),
                        output: committed.attention,
                    });
                }
                Ok(committed.residual)
            }
        };
        let output = if let (Some(ple), Some(candidate)) = (&mut self.ple, candidate) {
            // Only after all attention, shared and routed expert work committed.
            ple.state
                .finish(token.expect("PLE token was validated"), candidate, output)
        } else {
            output
        }?;
        for &expert in &selected {
            self.last_used.insert(expert, position + 1);
        }
        self.last_selected = selected;
        self.last_active_ranks = self
            .owners
            .sessions
            .iter()
            .filter(|s| before.get(&s.rank()).is_some_and(|n| s.traffic.execs > *n))
            .map(|s| s.rank())
            .collect();
        Ok(output)
    }
}

struct Cursor {
    position: u64,
    last_batch_start: Option<u64>,
    last_attempt: u64,
    poisoned: bool,
}

fn validate_advance(cursor: &Cursor, count: usize, attempt: u64, input: &[f32]) -> Result<()> {
    if cursor.poisoned
        || count == 0
        || attempt == 0
        || attempt <= cursor.last_attempt
        || cursor.position == u64::MAX
        || input.len() != 10240
        || input.iter().any(|x| !x.is_finite())
    {
        return Err(invalid("segment state, attempt or input"));
    }
    Ok(())
}

fn headroom_target(used: u64, limit: u64, reserve: u64) -> Result<Option<u64>> {
    if used > limit || reserve > limit {
        return Err(invalid("invalid expert headroom"));
    }
    let target = limit - reserve;
    Ok((used > target).then_some(target))
}

// Prefer an already-satisfied rank. Otherwise enumerate eligible ranks in a
// stable order; all alternatives are planned before any eviction is submitted.
fn rank_headroom_options(
    limits: &BTreeMap<u32, u64>,
    used: &BTreeMap<u32, u64>,
    reserve: u64,
) -> Result<Vec<Option<(u32, u64, u64)>>> {
    if limits.is_empty()
        || limits.len() != used.len()
        || limits
            .iter()
            .any(|(rank, &cap)| cap == 0 || used.get(rank).is_none_or(|&n| n > cap))
    {
        return Err(invalid("rank headroom accounting"));
    }
    let mut ranks: Vec<_> = limits.iter().filter(|(_, cap)| **cap >= reserve).collect();
    ranks.sort_by_key(|(rank, cap)| (std::cmp::Reverse(**cap), **rank));
    if ranks.is_empty() {
        return Err(invalid("no rank can reserve expert headroom"));
    }
    let mut options = Vec::with_capacity(ranks.len());
    for (&rank, &cap) in ranks {
        let rank_used = used[&rank];
        match headroom_target(rank_used, cap, reserve)? {
            None => return Ok(vec![None]),
            Some(target) => options.push(Some((rank, rank_used, target))),
        }
    }
    Ok(options)
}

// All intermediate outputs remain private. If a later layer fails, earlier
// resident states may have advanced: the entire segment becomes unusable.
fn advance<F>(
    cursor: &mut Cursor,
    ledger: &NativeBatchLedger,
    base: &Stamp,
    count: usize,
    attempt: u64,
    input: &[f32],
    mut step: F,
) -> Result<Vec<f32>>
where
    F: FnMut(usize, Reservation, &[f32]) -> Result<Vec<f32>>,
{
    validate_advance(cursor, count, attempt, input)?;
    let mut residual = input.to_vec();
    cursor.poisoned = true;
    for i in 0..count {
        let mut stamp = base.clone();
        stamp.layer = base
            .layer
            .checked_add(i as u32)
            .ok_or_else(|| invalid("layer overflow"))?;
        stamp.batch = cursor.position;
        let ticket = ledger.reserve_ticket(&stamp, attempt)?;
        residual = step(i, ticket, &residual)?;
        if residual.len() != 10240 || residual.iter().any(|x| !x.is_finite()) {
            return Err(invalid("invalid segment layer output"));
        }
    }
    cursor.last_batch_start = Some(cursor.position);
    cursor.position += 1;
    cursor.last_attempt = attempt;
    cursor.poisoned = false;
    Ok(residual)
}

// Keep diagnostics private until the entire segment has committed. A failed
// later layer drops the observation along with the partial residual.
#[allow(clippy::too_many_arguments)]
fn advance_with_observation<F, T>(
    cursor: &mut Cursor,
    ledger: &NativeBatchLedger,
    base: &Stamp,
    count: usize,
    attempt: u64,
    input: &[f32],
    mut step: F,
) -> Result<(Vec<f32>, Option<T>)>
where
    F: FnMut(usize, Reservation, &[f32], &mut Option<T>) -> Result<Vec<f32>>,
{
    let mut observation = None;
    let output = advance(
        cursor,
        ledger,
        base,
        count,
        attempt,
        input,
        |i, ticket, values| step(i, ticket, values, &mut observation),
    )?;
    Ok((output, observation))
}

#[cfg(feature = "ple-cpu-qualification")]
#[derive(Debug, serde::Serialize)]
pub struct PleQualificationSnapshot {
    pub position: u64,
    pub previous: [i32; 2],
    pub history_sha256: String,
    pub attention_position: u64,
    pub sessions: Vec<(u32, u64, bool)>,
}
pub struct Segment {
    layers: Vec<Layer>,
    base: Stamp,
    ledger: NativeBatchLedger,
    timeout: Duration,
    cursor: Cursor,
}
/// A bounded diagnostic observation, released only after every layer commits.
#[cfg(feature = "ple-cpu-qualification")]
pub struct QsaQualificationTrace {
    pub stamp: Stamp,
    pub attempt: u64,
    pub input: Vec<f32>,
    pub output: sc117_qsa_stage::Output,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EvictionCandidate {
    layer: u32,
    expert: u32,
    rank: u32,
    bytes: u64,
    // Zero means never selected; otherwise this is the last committed position + 1.
    last_used: u64,
}

fn sort_eviction_candidates(candidates: &mut [EvictionCandidate]) {
    candidates.sort_by_key(|c| (c.last_used, c.layer, c.expert, c.rank));
}

// Plan entirely before sending anything. Input order is deterministic; each
// rank must retain at least one resident and arithmetic is checked.
#[cfg(test)]
fn select_evictions(
    used: u64,
    target: u64,
    max_evictions: usize,
    candidates: &[EvictionCandidate],
    counts: &BTreeMap<(u32, u32), usize>,
) -> Result<Vec<EvictionCandidate>> {
    select_evictions_with_rank(used, target, max_evictions, candidates, counts, None)
}

// Rank headroom is planned first, then global pressure, before any eviction is sent.
fn select_evictions_with_rank(
    used: u64,
    target: u64,
    max_evictions: usize,
    candidates: &[EvictionCandidate],
    counts: &BTreeMap<(u32, u32), usize>,
    rank_target: Option<(u32, u64, u64)>,
) -> Result<Vec<EvictionCandidate>> {
    match select_evictions_ordered(used, target, max_evictions, candidates, counts, rank_target) {
        Ok(plan) => Ok(plan),
        Err(error)
            if error.code == "INVALID_INPUT"
                && error.message == "eviction target cannot be met safely" =>
        {
            // Pure preflight fallback: prefer larger payloads, keeping the
            // original LRU order for ties. This is a heuristic, not an
            // exhaustive feasibility search.
            let mut by_size = candidates.to_vec();
            by_size.sort_by_key(|c| std::cmp::Reverse(c.bytes));
            select_evictions_ordered(used, target, max_evictions, &by_size, counts, rank_target)
        }
        Err(error) => Err(error),
    }
}

fn select_evictions_ordered(
    used: u64,
    target: u64,
    max_evictions: usize,
    candidates: &[EvictionCandidate],
    counts: &BTreeMap<(u32, u32), usize>,
    rank_target: Option<(u32, u64, u64)>,
) -> Result<Vec<EvictionCandidate>> {
    if rank_target.is_some_and(|(_, rank_used, _)| rank_used > used) {
        return Err(invalid("rank eviction accounting"));
    }
    let mut rank_remaining = rank_target.map(|(_, used, _)| used);
    let rank_satisfied = |v: Option<u64>| match (rank_target, v) {
        (Some((_, _, target)), Some(remaining)) => remaining <= target,
        _ => true,
    };
    if used <= target && rank_satisfied(rank_remaining) {
        return Ok(Vec::new());
    }
    if max_evictions == 0 || max_evictions > 768 {
        return Err(invalid("eviction limit"));
    }
    let mut counts = counts.clone();
    let mut remaining = used;
    let mut selected = Vec::new();
    let mut done = vec![false; candidates.len()];
    for phase in 0..2 {
        for (index, &candidate) in candidates.iter().enumerate() {
            if done[index]
                || (phase == 0
                    && (rank_satisfied(rank_remaining)
                        || rank_target.is_none_or(|(rank, _, _)| candidate.rank != rank)))
            {
                continue;
            }
            let count = counts
                .get_mut(&(candidate.layer, candidate.rank))
                .ok_or_else(|| invalid("eviction candidate rank"))?;
            if *count <= 1 {
                continue;
            }
            if candidate.bytes == 0 {
                return Err(invalid("empty expert payload"));
            }
            remaining = remaining
                .checked_sub(candidate.bytes)
                .ok_or_else(|| invalid("eviction plan underflow"))?;
            if rank_target.is_some_and(|(rank, _, _)| candidate.rank == rank) {
                rank_remaining = Some(
                    rank_remaining
                        .unwrap()
                        .checked_sub(candidate.bytes)
                        .ok_or_else(|| invalid("rank eviction plan underflow"))?,
                );
            }
            *count -= 1;
            done[index] = true;
            selected.push(candidate);
            if remaining <= target && rank_satisfied(rank_remaining) {
                return Ok(selected);
            }
            if selected.len() == max_evictions {
                break;
            }
        }
        if selected.len() == max_evictions {
            break;
        }
    }
    Err(invalid("eviction target cannot be met safely"))
}

type PlannedEvictions = (Vec<EvictionCandidate>, Option<(u32, u64, u64)>);

fn select_eviction_alternatives(
    used: u64,
    target: u64,
    max_evictions: usize,
    candidates: &[EvictionCandidate],
    counts: &BTreeMap<(u32, u32), usize>,
    options: &[Option<(u32, u64, u64)>],
) -> Result<PlannedEvictions> {
    if options.is_empty() {
        return Err(invalid("no rank headroom option"));
    }
    for &rank_target in options {
        match select_evictions_with_rank(
            used,
            target,
            max_evictions,
            candidates,
            counts,
            rank_target,
        ) {
            Ok(plan) => return Ok((plan, rank_target)),
            Err(error)
                if error.code == "INVALID_INPUT"
                    && error.message == "eviction target cannot be met safely" => {}
            Err(error) => return Err(error),
        }
    }
    Err(invalid("eviction target cannot be met safely"))
}

/// Receipt is proof of the local acknowledgement, not a distributed lease.
#[derive(Debug, serde::Serialize)]
pub struct EvictionReceipt {
    pub stamp: Stamp,
    pub attempt: u64,
    pub operation_id: String,
    pub binding: crate::native_ggml_cpu::Binding,
    pub released_payload_bytes: u64,
    /// Most recent completed position selecting this expert, if any.
    pub last_used_position: Option<u64>,
}
fn require_supported_layer(layer: u32, has_ple: bool, qualifying: bool) -> Result<()> {
    if layer == 1 && !(qualifying && has_ple) {
        return Err(invalid(
            "SC117 layer 1 PLE transaction requires real-layer qualification",
        ));
    }
    if has_ple && layer != 1 {
        return Err(invalid("PLE is only valid for layer 1"));
    }
    Ok(())
}
impl Segment {
    pub fn new(
        layers: Vec<Layer>,
        plan_id: String,
        ledger: NativeBatchLedger,
        timeout: Duration,
    ) -> Result<Self> {
        Self::build(layers, plan_id, ledger, timeout, false)
    }
    /// Explicitly gated diagnostic only; does not qualify the complete layer or production use.
    #[cfg(feature = "ple-cpu-qualification")]
    pub fn new_ple_cpu_qualification(
        layers: Vec<Layer>,
        plan_id: String,
        ledger: NativeBatchLedger,
        timeout: Duration,
    ) -> Result<Self> {
        Self::build(layers, plan_id, ledger, timeout, true)
    }
    fn build(
        layers: Vec<Layer>,
        plan_id: String,
        ledger: NativeBatchLedger,
        timeout: Duration,
        qualifying: bool,
    ) -> Result<Self> {
        let first = layers.first().ok_or_else(|| invalid("empty segment"))?;
        if plan_id.trim().is_empty()
            || plan_id.len() > 256
            || timeout.is_zero()
            || timeout > Duration::from_secs(60)
            || first.identity.run_id.trim().is_empty()
            || first.identity.run_id.len() > 256
            || first.identity.model_revision.trim().is_empty()
            || first.identity.model_revision.len() > 256
        {
            return Err(invalid("segment identity or timeout"));
        }
        for (i, layer) in layers.iter().enumerate() {
            require_supported_layer(layer.identity.layer, layer.ple.is_some(), qualifying)?;
            if first.identity.layer.checked_add(i as u32) != Some(layer.identity.layer)
                || layer.identity.run_id != first.identity.run_id
                || layer.identity.model_revision != first.identity.model_revision
                || layer.identity.epoch != first.identity.epoch
                || layer.owners.registry.epoch != first.identity.epoch
                || layer.owners.registry.copies.is_empty()
                || layer.owners.sessions.is_empty()
            {
                return Err(invalid("segment layer identity or owners"));
            }
        }
        let base = Stamp {
            run_id: first.identity.run_id.clone(),
            model_revision: first.identity.model_revision.clone(),
            plan_id,
            layer: first.identity.layer,
            batch: 0,
            epoch: first.identity.epoch,
        };
        Ok(Self {
            layers,
            base,
            ledger,
            timeout,
            cursor: Cursor {
                position: 0,
                last_batch_start: None,
                last_attempt: 0,
                poisoned: false,
            },
        })
    }
    /// Read-only diagnostics for the explicitly gated CPU qualification path.
    #[cfg(feature = "ple-cpu-qualification")]
    pub fn ple_qualification_snapshot(&self) -> Result<PleQualificationSnapshot> {
        let layer = self
            .layers
            .iter()
            .find(|layer| layer.ple.is_some())
            .ok_or_else(|| invalid("qualification PLE missing"))?;
        let ple = layer
            .ple
            .as_ref()
            .ok_or_else(|| invalid("qualification PLE missing"))?;
        let attention_position = match &layer.attention {
            Attention::Gdn(gdn, _) => gdn.next_token(),
            _ => return Err(invalid("qualification attention type")),
        };
        let mut bytes = Vec::with_capacity(ple.state.history.len() * 4);
        for value in &ple.state.history {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok(PleQualificationSnapshot {
            position: ple.state.position,
            previous: ple.state.previous,
            history_sha256: crate::sha256::hex(&bytes),
            attention_position,
            sessions: layer
                .owners
                .sessions
                .iter()
                .map(|s| (s.rank(), s.traffic.frames_received, s.is_usable()))
                .collect(),
        })
    }
    pub fn position(&self) -> u64 {
        self.cursor.position
    }
    pub fn is_poisoned(&self) -> bool {
        self.cursor.poisoned
    }

    /// Evict between fully committed positions. The rank must have participated
    /// in the last position. A reserved or transmitted failure poisons the whole
    /// segment; neither the request nor its UNKNOWN claim may be replayed.
    pub fn evict_owner_after_commit(
        &mut self,
        source: &mut crate::sc117_expert_source::Source<'_>,
        binding: &crate::native_ggml_cpu::Binding,
    ) -> Result<EvictionReceipt> {
        if self.cursor.poisoned || self.cursor.position == 0 || self.cursor.last_attempt == 0 {
            return Err(invalid("segment eviction requires committed position"));
        }
        let mut stamp = self.base.clone();
        stamp.batch = self
            .cursor
            .last_batch_start
            .ok_or_else(|| invalid("missing last committed batch identity"))?;
        stamp.layer = binding.layer;
        let layer = self
            .layers
            .iter_mut()
            .find(|l| l.identity.layer == binding.layer)
            .ok_or_else(|| invalid("segment eviction layer"))?;
        if !layer.last_active_ranks.contains(&binding.rank) {
            return Err(invalid("eviction rank did not participate in last batch"));
        }
        let bytes = source.validate_eviction(
            &stamp,
            binding,
            &layer.owners.registry,
            &layer.owners.order,
        )?;
        let mut new_order = layer.owners.order.clone();
        let locals = new_order
            .get_mut(&binding.rank)
            .ok_or_else(|| invalid("segment eviction rank"))?;
        let remaining = locals
            .len()
            .checked_sub(1)
            .filter(|n| *n > 0)
            .ok_or_else(|| invalid("segment eviction last owner"))?;
        locals.retain(|&local| local != binding.local);
        if locals.len() != remaining {
            return Err(invalid("segment eviction local identity"));
        }
        let mut new_registry = layer.owners.registry.clone();
        let old_count = new_registry.copies.len();
        new_registry.copies.retain(|r| {
            !(r.rank == binding.rank && r.local == binding.local && r.expert == binding.expert)
        });
        if new_registry.copies.len().checked_add(1) != Some(old_count) {
            return Err(invalid("segment eviction registry identity"));
        }
        let mut sessions = layer
            .owners
            .sessions
            .iter_mut()
            .filter(|s| s.rank() == binding.rank);
        let session = sessions
            .next()
            .ok_or_else(|| invalid("segment eviction session"))?;
        if sessions.next().is_some()
            || !session.is_usable()
            || !session
                .hello
                .capabilities
                .iter()
                .any(|c| c == crate::session::CAPABILITY_NATIVE_GGML_EVICT)
        {
            return Err(invalid("segment eviction session capability"));
        }
        self.cursor.poisoned = true;
        let attempt = self.cursor.last_attempt;
        let operation_id = self.ledger.reserve_eviction(&stamp, attempt, binding)?;
        session.evict_native_ggml(
            &crate::session::NativeEviction {
                stamp: &stamp,
                attempt,
                operation_id: &operation_id,
                binding,
                expected_bytes: bytes,
                expected_remaining: remaining,
            },
            self.timeout,
        )?;
        source.commit_eviction(binding, bytes)?;
        layer.owners.order = new_order;
        layer.owners.registry = new_registry;
        layer.owners.preferred.remove(&binding.expert);
        let last_used_position = layer.last_used.remove(&binding.expert).map(|p| p - 1);
        self.cursor.poisoned = false;
        Ok(EvictionReceipt {
            stamp,
            attempt,
            operation_id,
            binding: binding.clone(),
            released_payload_bytes: bytes,
            last_used_position,
        })
    }
    /// Explicitly trim accounted payload between committed positions. Protects
    /// each layer's last committed gate choices and evicts least-recently used
    /// candidates first; this does not predict the next
    /// gate, measure RSS, or retry any uncertain operation. If a later eviction
    /// fails, earlier acknowledged evictions remain applied and the segment is
    /// poisoned by the failing operation.
    pub fn evict_to_target(
        &mut self,
        source: &mut crate::sc117_expert_source::Source<'_>,
        target_bytes: u64,
        max_evictions: usize,
    ) -> Result<Vec<EvictionReceipt>> {
        self.evict_to_targets(source, target_bytes, max_evictions, &[None])
    }
    fn evict_to_targets(
        &mut self,
        source: &mut crate::sc117_expert_source::Source<'_>,
        target_bytes: u64,
        max_evictions: usize,
        rank_options: &[Option<(u32, u64, u64)>],
    ) -> Result<Vec<EvictionReceipt>> {
        if self.cursor.poisoned || self.cursor.position == 0 || self.cursor.last_attempt == 0 {
            return Err(invalid("eviction policy requires committed position"));
        }
        let mut candidates = Vec::new();
        let mut counts = BTreeMap::new();
        for layer in &self.layers {
            if layer.last_selected.is_empty() {
                return Err(invalid("missing committed route"));
            }
            let mut copies = layer.owners.registry.copies.clone();
            copies.sort_by_key(|r| (r.expert, r.rank, r.local));
            for copy in &copies {
                *counts
                    .entry((layer.identity.layer, copy.rank))
                    .or_insert(0usize) += 1;
                if !layer.last_selected.contains(&copy.expert)
                    && layer.last_active_ranks.contains(&copy.rank)
                {
                    candidates.push(EvictionCandidate {
                        layer: layer.identity.layer,
                        expert: copy.expert,
                        rank: copy.rank,
                        bytes: source.resident_bytes(layer.identity.layer, copy.expert)?,
                        last_used: layer.last_used.get(&copy.expert).copied().unwrap_or(0),
                    });
                }
            }
        }
        sort_eviction_candidates(&mut candidates);
        let (plan, rank_target) = select_eviction_alternatives(
            source.used_bytes(),
            target_bytes,
            max_evictions,
            &candidates,
            &counts,
            rank_options,
        )?;
        let mut bindings = Vec::with_capacity(plan.len());
        for item in &plan {
            let binding = source.resident_binding(item.layer, item.expert)?;
            if binding.rank != item.rank {
                return Err(invalid("eviction plan binding rank"));
            }
            let layer = self
                .layers
                .iter()
                .find(|l| l.identity.layer == item.layer)
                .ok_or_else(|| invalid("eviction plan layer"))?;
            let mut stamp = self.base.clone();
            stamp.layer = item.layer;
            stamp.batch = self
                .cursor
                .last_batch_start
                .ok_or_else(|| invalid("missing last committed batch identity"))?;
            let bytes = source.validate_eviction(
                &stamp,
                &binding,
                &layer.owners.registry,
                &layer.owners.order,
            )?;
            if bytes != item.bytes {
                return Err(invalid("eviction plan payload size"));
            }
            bindings.push(binding);
        }
        let mut receipts = Vec::with_capacity(bindings.len());
        for binding in &bindings {
            receipts.push(self.evict_owner_after_commit(source, binding)?);
        }
        if source.used_bytes() > target_bytes
            || rank_target.is_some_and(|(rank, _, target)| {
                source
                    .rank_used_bytes()
                    .get(&rank)
                    .is_none_or(|&used| used > target)
            })
        {
            self.cursor.poisoned = true;
            return Err(invalid("eviction target accounting"));
        }
        Ok(receipts)
    }

    /// Append a verified expert to an existing rank between positions. The caller
    /// chooses placement and supplies a trusted digest; no eviction is performed.
    /// Any failure after transmission poisons the segment, including a lost reply.
    pub fn append_owner(
        &mut self,
        binding: &crate::native_ggml_cpu::Binding,
        index: &str,
        pack_bytes: u64,
        bytes: &[u8],
    ) -> Result<()> {
        if self.cursor.poisoned
            || binding.model_revision != self.base.model_revision
            || binding.epoch != self.base.epoch
        {
            return Err(invalid("segment append state or identity"));
        }
        let layer = self
            .layers
            .iter_mut()
            .find(|l| l.identity.layer == binding.layer)
            .ok_or_else(|| invalid("segment append layer"))?;
        // Fail closed even if an error occurred before transmission. This keeps
        // the outer state from guessing whether a worker installed the weight.
        self.cursor.poisoned = true;
        layer
            .owners
            .append(binding, index, pack_bytes, bytes, self.timeout)?;
        self.cursor.poisoned = false;
        Ok(())
    }
    /// Close each resident worker and return its actual shutdown receipt.
    pub fn close(self) -> Vec<crate::session::CloseReceipt> {
        self.layers
            .into_iter()
            .flat_map(|layer| layer.owners.sessions)
            .map(|session| session.close(self.timeout))
            .collect()
    }
    /// Pair each worker's shutdown receipt with that session's successful
    /// coordinator-side execution count. Idle ranks can have fewer executions.
    pub fn close_with_counts(self) -> Vec<(crate::session::CloseReceipt, u64, u64)> {
        self.layers
            .into_iter()
            .flat_map(|layer| layer.owners.sessions)
            .map(|session| {
                let expected = session.traffic.execs;
                let raw = session.traffic.native_ggml_raw_execs;
                (session.close(self.timeout), expected, raw)
            })
            .collect()
    }
    /// Diagnostic-only, layer-major prefill. Earlier layer states may commit
    /// before a later failure; then the entire segment is terminal and no
    /// intermediate residuals are returned. Every layer gets a fresh claim.
    #[cfg(feature = "ple-cpu-qualification")]
    pub fn advance_token_batch_with_source(
        &mut self,
        attempt: u64,
        tokens: &[i32],
        inputs: &[Vec<f32>],
        source: &mut dyn MissingExpertSource,
    ) -> Result<Vec<Vec<f32>>> {
        self.validate_token_batch(attempt, tokens, inputs)?;
        let position = self.cursor.position;
        let original_layer = self.base.layer;
        let previous_attempt = self.cursor.last_attempt;
        let previous_batch_start = self.cursor.last_batch_start;
        let mut residuals = inputs.to_vec();
        for index in 0..self.layers.len() {
            self.base.layer = self.layers[index].identity.layer;
            self.cursor.position = position;
            self.cursor.last_attempt = previous_attempt;
            self.cursor.poisoned = false;
            let result = if matches!(self.layers[index].attention, Attention::Gdn(..)) {
                let ple_tokens = self.layers[index].ple.as_ref().map(|_| tokens);
                self.advance_gdn_batch_inner(index, attempt, &residuals, ple_tokens, source)
            } else {
                self.advance_qsa_batch_inner(index, attempt, &residuals, source)
            };
            self.base.layer = original_layer;
            match result {
                Ok(next) => residuals = next,
                Err(error) => {
                    self.cursor.position = position;
                    self.cursor.last_attempt = previous_attempt;
                    self.cursor.last_batch_start = previous_batch_start;
                    self.cursor.poisoned = true;
                    return Err(error);
                }
            }
        }
        self.cursor.position = position + inputs.len() as u64;
        self.cursor.last_batch_start = Some(position);
        self.cursor.last_attempt = attempt;
        self.cursor.poisoned = false;
        Ok(residuals)
    }
    #[cfg(feature = "ple-cpu-qualification")]
    fn validate_token_batch(
        &self,
        attempt: u64,
        tokens: &[i32],
        inputs: &[Vec<f32>],
    ) -> Result<()> {
        let position = self.cursor.position;
        if self.cursor.poisoned
            || self.layers.is_empty()
            || inputs.is_empty()
            || inputs.len() > 128
            || tokens.len() != inputs.len()
            || tokens.iter().any(|t| !(0..sc117_ple::VOCAB).contains(t))
            || attempt == 0
            || attempt <= self.cursor.last_attempt
            || position.checked_add(inputs.len() as u64).is_none()
            || inputs
                .iter()
                .any(|x| x.len() != 10240 || x.iter().any(|v| !v.is_finite()))
            || self.layers.iter().any(|layer| {
                layer
                    .ple
                    .as_ref()
                    .is_some_and(|p| p.state.position != position)
                    || match &layer.attention {
                        Attention::Gdn(gdn, _) => gdn.next_token() != position,
                        Attention::Qsa(qsa, _) => {
                            u64::try_from(qsa.len()).ok() != Some(position)
                                || !qsa.batch_fits(inputs.len())
                        }
                    }
            })
        {
            return Err(invalid("multi-layer batch state, attempt or input"));
        }
        Ok(())
    }
    /// Between complete layer-major batches, free accounted expert payload to
    /// a caller-supplied headroom target. Preflight precedes any eviction.
    #[cfg(feature = "ple-cpu-qualification")]
    pub fn advance_token_batch_with_headroom(
        &mut self,
        attempt: u64,
        tokens: &[i32],
        inputs: &[Vec<f32>],
        source: &mut crate::sc117_expert_source::Source<'_>,
        headroom_bytes: u64,
        max_evictions: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<EvictionReceipt>)> {
        self.validate_token_batch(attempt, tokens, inputs)?;
        let global_target =
            headroom_target(source.used_bytes(), source.limit_bytes(), headroom_bytes)?;
        let rank_options = if let Some(limits) = source.rank_limits() {
            rank_headroom_options(limits, source.rank_used_bytes(), headroom_bytes)?
        } else {
            vec![None]
        };
        let receipts = if global_target.is_some() || rank_options.iter().all(Option::is_some) {
            self.evict_to_targets(
                source,
                global_target.unwrap_or(source.used_bytes()),
                max_evictions,
                &rank_options,
            )?
        } else {
            Vec::new()
        };
        let output = self.advance_token_batch_with_source(attempt, tokens, inputs, source)?;
        Ok((output, receipts))
    }
    /// Execute a single non-PLE GDN layer over a causal group. Missing experts
    /// are admitted before dispatch; there is no eviction within this call.
    /// Once a reservation is made, any error leaves the segment terminal.
    pub fn advance_gdn_batch_with_source(
        &mut self,
        attempt: u64,
        inputs: &[Vec<f32>],
        source: &mut dyn MissingExpertSource,
    ) -> Result<Vec<Vec<f32>>> {
        if self.layers.len() != 1 {
            return Err(invalid("GDN segment layer count"));
        }
        self.advance_gdn_batch_inner(0, attempt, inputs, None, source)
    }
    /// Diagnostic-only layer-1 PLE/GDN batch. This does not lift the ordinary
    /// constructor's layer-1 qualification guard.
    #[cfg(feature = "ple-cpu-qualification")]
    pub fn advance_gdn_ple_batch_with_source(
        &mut self,
        attempt: u64,
        tokens: &[i32],
        inputs: &[Vec<f32>],
        source: &mut dyn MissingExpertSource,
    ) -> Result<Vec<Vec<f32>>> {
        if self.layers.len() != 1 {
            return Err(invalid("GDN segment layer count"));
        }
        self.advance_gdn_batch_inner(0, attempt, inputs, Some(tokens), source)
    }
    fn advance_gdn_batch_inner(
        &mut self,
        layer_index: usize,
        attempt: u64,
        inputs: &[Vec<f32>],
        tokens: Option<&[i32]>,
        source: &mut dyn MissingExpertSource,
    ) -> Result<Vec<Vec<f32>>> {
        if self.cursor.poisoned
            || layer_index >= self.layers.len()
            || inputs.is_empty()
            || inputs.len() > 128
            || attempt == 0
            || attempt <= self.cursor.last_attempt
            || self
                .cursor
                .position
                .checked_add(inputs.len() as u64)
                .is_none()
            || inputs
                .iter()
                .any(|x| x.len() != 10240 || x.iter().any(|v| !v.is_finite()))
            || !matches!(&self.layers[layer_index].attention, Attention::Gdn(..))
            || match (&self.layers[layer_index].ple, tokens) {
                (None, None) => false,
                (Some(ple), Some(tokens)) => {
                    self.layers[layer_index].identity.layer != 1
                        || ple.state.position != self.cursor.position
                        || tokens.len() != inputs.len()
                        || tokens.iter().any(|t| !(0..sc117_ple::VOCAB).contains(t))
                }
                _ => true,
            }
        {
            return Err(invalid("GDN segment batch state, attempt or input"));
        }
        let position = self.cursor.position;
        let mut stamp = self.base.clone();
        stamp.batch = position;
        self.cursor.poisoned = true;
        let ticket = self.ledger.reserve_ticket(&stamp, attempt)?;
        let layer = &mut self.layers[layer_index];
        let (ple_candidate, prepared) = match (&layer.ple, tokens) {
            (Some(ple), Some(tokens)) => {
                let (candidate, residuals) = ple.state.prepare_batch(
                    position,
                    tokens,
                    inputs,
                    |token, previous, residual, history| {
                        let (_, embedding) = ple.table.gather(token, previous)?;
                        ple.weights.evaluate(&embedding, residual, history)
                    },
                )?;
                (Some(candidate), Some(residuals))
            }
            (None, None) => (None, None),
            _ => unreachable!(),
        };
        let inputs = prepared.as_deref().unwrap_or(inputs);
        let Attention::Gdn(gdn, post) = &mut layer.attention else {
            unreachable!()
        };
        let batch =
            sc117_bound_layer::RoutedBatch::prepare(gdn, post, &self.ledger, &ticket, inputs)?;
        let mut selected = std::collections::BTreeSet::new();
        let mut uses = Vec::with_capacity(inputs.len());
        let mut source = Some(source);
        let union: Vec<_> = batch.choices().flatten().cloned().collect();
        layer
            .owners
            .ensure(&stamp, attempt, &union, &mut source, self.timeout)?;
        for choices in batch.choices() {
            let ids: Vec<_> = choices.iter().map(|c| c.expert).collect();
            selected.extend(ids.iter().copied());
            uses.push(ids);
        }
        let shared = batch.shared(&mut layer.shared)?;
        let before: BTreeMap<_, _> = layer
            .owners
            .sessions
            .iter()
            .map(|s| (s.rank(), s.traffic.execs))
            .collect();
        let collected = batch.dispatch(
            &mut layer.owners.sessions,
            layer.owners.registry.clone(),
            layer.owners.preferred.clone(),
            layer.owners.order.clone(),
            &self.ledger,
            ticket,
            self.timeout,
        )?;
        let committed = batch.commit(shared, collected)?;
        if committed.residuals.len() != inputs.len()
            || committed
                .residuals
                .iter()
                .any(|x| x.len() != 10240 || x.iter().any(|v| !v.is_finite()))
        {
            return Err(invalid("GDN segment batch output"));
        }
        if let Some(candidate) = ple_candidate {
            layer.ple.as_mut().expect("PLE preflight validated").state = candidate;
        }
        for (i, ids) in uses.iter().enumerate() {
            for &expert in ids {
                layer.last_used.insert(expert, position + i as u64 + 1);
            }
        }
        layer.last_selected = selected.into_iter().collect();
        layer.last_active_ranks = layer
            .owners
            .sessions
            .iter()
            .filter(|s| before.get(&s.rank()).is_some_and(|n| s.traffic.execs > *n))
            .map(|s| s.rank())
            .collect();
        self.cursor.last_batch_start = Some(position);
        self.cursor.position += inputs.len() as u64;
        self.cursor.last_attempt = attempt;
        self.cursor.poisoned = false;
        Ok(committed.residuals)
    }
    /// Execute one QSA layer on a causal group of positions. Missing experts are
    /// loaded through the caller's bounded source before any dispatch. There is
    /// no eviction in this operation: insufficient budget fails closed. Once
    /// reservation begins, any error leaves the entire segment terminal.
    pub fn advance_qsa_batch_with_source(
        &mut self,
        attempt: u64,
        inputs: &[Vec<f32>],
        source: &mut dyn MissingExpertSource,
    ) -> Result<Vec<Vec<f32>>> {
        if self.layers.len() != 1 {
            return Err(invalid("QSA segment layer count"));
        }
        self.advance_qsa_batch_inner(0, attempt, inputs, source)
    }
    fn advance_qsa_batch_inner(
        &mut self,
        layer_index: usize,
        attempt: u64,
        inputs: &[Vec<f32>],
        source: &mut dyn MissingExpertSource,
    ) -> Result<Vec<Vec<f32>>> {
        if self.cursor.poisoned
            || layer_index >= self.layers.len()
            || inputs.is_empty()
            || inputs.len() > 128
            || attempt == 0
            || attempt <= self.cursor.last_attempt
            || self
                .cursor
                .position
                .checked_add(inputs.len() as u64)
                .is_none()
            || inputs
                .iter()
                .any(|x| x.len() != 10240 || x.iter().any(|v| !v.is_finite()))
            || !matches!(&self.layers[layer_index].attention, Attention::Qsa(qsa, _) if qsa.batch_fits(inputs.len()))
            || self.layers[layer_index].ple.is_some()
        {
            return Err(invalid("QSA segment batch state, attempt or input"));
        }
        let position = self.cursor.position;
        let mut stamp = self.base.clone();
        stamp.batch = position;
        self.cursor.poisoned = true;
        let ticket = self.ledger.reserve_ticket(&stamp, attempt)?;
        let layer = &mut self.layers[layer_index];
        let Attention::Qsa(qsa, ffn) = &mut layer.attention else {
            unreachable!()
        };
        let batch =
            sc117_qsa_bound_layer::RoutedBatch::prepare(qsa, ffn, &self.ledger, &ticket, inputs)?;
        let mut selected = std::collections::BTreeSet::new();
        let mut uses = Vec::with_capacity(inputs.len());
        let mut source = Some(source);
        let union: Vec<_> = batch.choices().flatten().cloned().collect();
        layer
            .owners
            .ensure(&stamp, attempt, &union, &mut source, self.timeout)?;
        for choices in batch.choices() {
            let ids: Vec<_> = choices.iter().map(|c| c.expert).collect();
            selected.extend(ids.iter().copied());
            uses.push(ids);
        }
        let shared = batch.shared(&mut layer.shared)?;
        let before: BTreeMap<_, _> = layer
            .owners
            .sessions
            .iter()
            .map(|s| (s.rank(), s.traffic.execs))
            .collect();
        let collected = batch.dispatch(
            &mut layer.owners.sessions,
            layer.owners.registry.clone(),
            layer.owners.preferred.clone(),
            layer.owners.order.clone(),
            &self.ledger,
            ticket,
            self.timeout,
        )?;
        let committed = batch.commit(shared, collected)?;
        if committed.residuals.len() != inputs.len()
            || committed
                .residuals
                .iter()
                .any(|x| x.len() != 10240 || x.iter().any(|v| !v.is_finite()))
        {
            return Err(invalid("QSA segment batch output"));
        }
        for (i, ids) in uses.iter().enumerate() {
            for &expert in ids {
                layer.last_used.insert(expert, position + i as u64 + 1);
            }
        }
        layer.last_selected = selected.into_iter().collect();
        layer.last_active_ranks = layer
            .owners
            .sessions
            .iter()
            .filter(|s| before.get(&s.rank()).is_some_and(|n| s.traffic.execs > *n))
            .map(|s| s.rank())
            .collect();
        self.cursor.last_batch_start = Some(position);
        self.cursor.position += inputs.len() as u64;
        self.cursor.last_attempt = attempt;
        self.cursor.poisoned = false;
        Ok(committed.residuals)
    }
    /// Trim between committed batches, then admit and execute a QSA batch.
    /// Headroom is a caller-selected payload bound, not a prediction of routes
    /// or process RSS. Rank-specific headroom follows the existing conservative
    /// single-rank reservation policy. Failed submitted evictions are never retried.
    pub fn advance_qsa_batch_with_headroom(
        &mut self,
        attempt: u64,
        inputs: &[Vec<f32>],
        source: &mut crate::sc117_expert_source::Source<'_>,
        headroom_bytes: u64,
        max_evictions: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<EvictionReceipt>)> {
        if self.cursor.poisoned
            || self.layers.len() != 1
            || inputs.is_empty()
            || inputs.len() > 128
            || attempt == 0
            || attempt <= self.cursor.last_attempt
            || self
                .cursor
                .position
                .checked_add(inputs.len() as u64)
                .is_none()
            || inputs
                .iter()
                .any(|x| x.len() != 10240 || x.iter().any(|v| !v.is_finite()))
            || !matches!(&self.layers[0].attention, Attention::Qsa(qsa, _) if qsa.batch_fits(inputs.len()))
            || self.layers[0].ple.is_some()
        {
            return Err(invalid("QSA segment batch state, attempt or input"));
        }
        let global_target =
            headroom_target(source.used_bytes(), source.limit_bytes(), headroom_bytes)?;
        let rank_options = if let Some(limits) = source.rank_limits() {
            rank_headroom_options(limits, source.rank_used_bytes(), headroom_bytes)?
        } else {
            vec![None]
        };
        let receipts = if global_target.is_some() || rank_options.iter().all(Option::is_some) {
            self.evict_to_targets(
                source,
                global_target.unwrap_or(source.used_bytes()),
                max_evictions,
                &rank_options,
            )?
        } else {
            Vec::new()
        };
        let output = self.advance_qsa_batch_with_source(attempt, inputs, source)?;
        Ok((output, receipts))
    }
    pub fn advance(&mut self, attempt: u64, input: &[f32]) -> Result<Vec<f32>> {
        self.advance_inner(attempt, input, None, &mut None)
    }
    /// After routing, load absent experts through an explicitly supplied source
    /// before dispatch. Any failure leaves this segment terminal and the claim consumed.
    pub fn advance_with_source(
        &mut self,
        attempt: u64,
        input: &[f32],
        source: &mut dyn MissingExpertSource,
    ) -> Result<Vec<f32>> {
        self.advance_inner(attempt, input, None, &mut Some(source))
    }
    /// Advance one real token; token history and PLE convolution state commit with the layer.
    pub fn advance_token(&mut self, attempt: u64, token: i32, input: &[f32]) -> Result<Vec<f32>> {
        self.advance_inner(attempt, input, Some(token), &mut None)
    }
    pub fn advance_token_with_source(
        &mut self,
        attempt: u64,
        token: i32,
        input: &[f32],
        source: &mut dyn MissingExpertSource,
    ) -> Result<Vec<f32>> {
        self.advance_inner(attempt, input, Some(token), &mut Some(source))
    }
    /// Advance with caller-selected pre-position expert-payload headroom. This
    /// does not reserve the next gate's unknown payload, predict its route, or
    /// measure RSS. When trimming is needed the previous committed route stays
    /// protected. A failure never publishes partial token output.
    pub fn advance_token_with_headroom(
        &mut self,
        attempt: u64,
        token: i32,
        input: &[f32],
        source: &mut crate::sc117_expert_source::Source<'_>,
        headroom_bytes: u64,
        max_evictions: usize,
    ) -> Result<(Vec<f32>, Vec<EvictionReceipt>)> {
        let (output, receipts, _) = self.advance_token_with_headroom_inner(
            attempt,
            token,
            input,
            source,
            headroom_bytes,
            max_evictions,
            None,
        )?;
        Ok((output, receipts))
    }

    /// CPU qualification only. No trace escapes if any layer fails.
    #[cfg(feature = "ple-cpu-qualification")]
    #[allow(clippy::too_many_arguments)]
    pub fn advance_token_with_headroom_qsa_trace(
        &mut self,
        attempt: u64,
        token: i32,
        input: &[f32],
        source: &mut crate::sc117_expert_source::Source<'_>,
        headroom_bytes: u64,
        max_evictions: usize,
        layer: u32,
    ) -> Result<(Vec<f32>, Vec<EvictionReceipt>, QsaQualificationTrace)> {
        let mut stamp = self.base.clone();
        stamp.layer = layer;
        stamp.batch = self.cursor.position;
        let (output, receipts, trace) = self.advance_token_with_headroom_inner(
            attempt,
            token,
            input,
            source,
            headroom_bytes,
            max_evictions,
            Some(layer),
        )?;
        let Some(output_trace) = trace else {
            self.cursor.poisoned = true;
            return Err(invalid("committed QSA trace missing"));
        };
        Ok((
            output,
            receipts,
            QsaQualificationTrace {
                stamp,
                attempt,
                input: output_trace.input,
                output: output_trace.output,
            },
        ))
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn advance_token_with_headroom_inner(
        &mut self,
        attempt: u64,
        token: i32,
        input: &[f32],
        source: &mut crate::sc117_expert_source::Source<'_>,
        headroom_bytes: u64,
        max_evictions: usize,
        capture_layer: Option<u32>,
    ) -> Result<(Vec<f32>, Vec<EvictionReceipt>, Option<CapturedQsa>)> {
        validate_advance(&self.cursor, self.layers.len(), attempt, input)?;
        if capture_layer.is_some_and(|wanted| {
            !self.layers.iter().any(|layer| {
                layer.identity.layer == wanted && matches!(layer.attention, Attention::Qsa(..))
            })
        }) {
            return Err(invalid("QSA trace layer missing"));
        }
        if self.layers.iter().any(|layer| layer.ple.is_some())
            && !(0..sc117_ple::VOCAB).contains(&token)
        {
            return Err(invalid("segment PLE token required or invalid"));
        }
        let global_target =
            headroom_target(source.used_bytes(), source.limit_bytes(), headroom_bytes)?;
        let rank_options = if let Some(limits) = source.rank_limits() {
            rank_headroom_options(limits, source.rank_used_bytes(), headroom_bytes)?
        } else {
            vec![None]
        };
        let receipts = if global_target.is_some() || rank_options.iter().all(Option::is_some) {
            self.evict_to_targets(
                source,
                global_target.unwrap_or(source.used_bytes()),
                max_evictions,
                &rank_options,
            )?
        } else {
            Vec::new()
        };
        let (output, trace) = self.advance_inner_capture(
            attempt,
            input,
            Some(token),
            &mut Some(source),
            capture_layer,
        )?;
        Ok((output, receipts, trace))
    }

    fn advance_inner(
        &mut self,
        attempt: u64,
        input: &[f32],
        token: Option<i32>,
        source: &mut Option<&mut dyn MissingExpertSource>,
    ) -> Result<Vec<f32>> {
        self.advance_inner_capture(attempt, input, token, source, None)
            .map(|(output, _)| output)
    }

    fn advance_inner_capture(
        &mut self,
        attempt: u64,
        input: &[f32],
        token: Option<i32>,
        source: &mut Option<&mut dyn MissingExpertSource>,
        capture_layer: Option<u32>,
    ) -> Result<(Vec<f32>, Option<CapturedQsa>)> {
        if self.layers.iter().any(|layer| layer.ple.is_some())
            && token.is_none_or(|t| !(0..sc117_ple::VOCAB).contains(&t))
        {
            return Err(invalid("segment PLE token required or invalid"));
        }
        let position = self.cursor.position;
        let ledger = &self.ledger;
        let timeout = self.timeout;
        let layers = &mut self.layers;
        advance_with_observation(
            &mut self.cursor,
            ledger,
            &self.base,
            layers.len(),
            attempt,
            input,
            |i, ticket, values, trace| {
                let capture = capture_layer == Some(layers[i].identity.layer);
                layers[i].run(
                    ledger, ticket, position, values, token, timeout, source, capture, trace,
                )
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn headroom_preflight_is_overflow_safe_and_only_triggers_on_pressure() {
        assert_eq!(headroom_target(60, 100, 40).unwrap(), None);
        assert_eq!(headroom_target(61, 100, 40).unwrap(), Some(60));
        assert_eq!(
            headroom_target(u64::MAX, u64::MAX, 1).unwrap(),
            Some(u64::MAX - 1)
        );
        assert!(headroom_target(101, 100, 0).is_err());
        assert!(headroom_target(0, 100, 101).is_err());
    }

    #[test]
    fn size_fallback_finds_a_feasible_unequal_payload_plan() {
        let candidates = [
            EvictionCandidate {
                layer: 0,
                expert: 1,
                rank: 2,
                bytes: 2,
                last_used: 0,
            },
            EvictionCandidate {
                layer: 0,
                expert: 2,
                rank: 2,
                bytes: 4,
                last_used: 1,
            },
        ];
        let counts = BTreeMap::from([((0, 2), 3)]);
        // The LRU-only pass fails; the larger candidate is a safe fallback.
        assert!(select_evictions_ordered(10, 6, 1, &candidates, &counts, None).is_err());
        assert_eq!(
            select_evictions(10, 6, 1, &candidates, &counts).unwrap(),
            vec![candidates[1]]
        );
        assert_eq!(
            select_evictions_with_rank(10, 6, 1, &candidates, &counts, Some((2, 10, 6))).unwrap(),
            vec![candidates[1]]
        );
        assert!(select_evictions(10, 6, 1, &candidates, &BTreeMap::new()).is_err());
    }

    #[test]
    fn eviction_plan_is_bounded_and_preserves_last_resident_per_rank() {
        let c = [
            EvictionCandidate {
                layer: 0,
                expert: 1,
                rank: 2,
                bytes: 4,
                last_used: 0,
            },
            EvictionCandidate {
                layer: 0,
                expert: 2,
                rank: 2,
                bytes: 4,
                last_used: 0,
            },
            EvictionCandidate {
                layer: 0,
                expert: 3,
                rank: 3,
                bytes: 4,
                last_used: 0,
            },
            EvictionCandidate {
                layer: 1,
                expert: 4,
                rank: 2,
                bytes: 4,
                last_used: 0,
            },
        ];
        let counts = BTreeMap::from([((0, 2), 2), ((0, 3), 1), ((1, 2), 2)]);
        assert_eq!(
            select_evictions(16, 8, 2, &c, &counts).unwrap(),
            vec![c[0], c[3]]
        );
        assert!(select_evictions(16, 4, 2, &c, &counts).is_err());
        assert!(select_evictions(16, 8, 1, &c, &counts).is_err());
        assert!(select_evictions(16, 8, 769, &c, &counts).is_err());
        assert!(select_evictions(3, 0, 2, &c, &counts).is_err());
        assert!(select_evictions(16, 16, 0, &c, &counts).unwrap().is_empty());
    }

    #[test]
    fn rank_headroom_is_planned_before_global_evictions() {
        let c = [
            EvictionCandidate {
                layer: 0,
                expert: 1,
                rank: 2,
                bytes: 4,
                last_used: 0,
            },
            EvictionCandidate {
                layer: 0,
                expert: 2,
                rank: 3,
                bytes: 4,
                last_used: 1,
            },
            EvictionCandidate {
                layer: 1,
                expert: 3,
                rank: 3,
                bytes: 4,
                last_used: 2,
            },
        ];
        let counts = BTreeMap::from([((0, 2), 2), ((0, 3), 2), ((1, 3), 2)]);
        let plan = select_evictions_with_rank(20, 16, 3, &c, &counts, Some((3, 12, 8))).unwrap();
        assert_eq!(plan, vec![c[1]]);
        let both = select_evictions_with_rank(20, 12, 3, &c, &counts, Some((3, 12, 8))).unwrap();
        assert_eq!(both, vec![c[1], c[0]]);
        assert!(select_evictions_with_rank(20, 12, 1, &c, &counts, Some((3, 12, 8))).is_err());
        assert!(select_evictions_with_rank(20, 20, 3, &c, &counts, Some((3, 21, 8))).is_err());
        let protected = BTreeMap::from([((0, 2), 2), ((0, 3), 1), ((1, 3), 1)]);
        assert!(select_evictions_with_rank(20, 20, 3, &c, &protected, Some((3, 12, 8))).is_err());
    }

    #[test]
    fn headroom_uses_existing_capacity_or_a_feasible_alternate_without_sending_evictions() {
        let limits = BTreeMap::from([(2, 10), (3, 12)]);
        let candidate = EvictionCandidate {
            layer: 0,
            expert: 4,
            rank: 2,
            bytes: 4,
            last_used: 0,
        };
        let counts = BTreeMap::from([((0, 2), 2), ((0, 3), 1)]);
        // Rank 3 has the biggest limit but cannot evict its last resident.
        // Rank 2 already has six bytes free; global pressure can be relieved there.
        let options =
            rank_headroom_options(&limits, &BTreeMap::from([(2, 4), (3, 12)]), 6).unwrap();
        assert_eq!(options, vec![None]);
        let (plan, target) =
            select_eviction_alternatives(16, 12, 1, &[candidate], &counts, &options).unwrap();
        assert_eq!(plan, vec![candidate]);
        assert_eq!(target, None);

        // Neither rank already satisfies the reservation. The first is blocked,
        // but the second can meet it by evicting one non-protected resident.
        let options =
            rank_headroom_options(&limits, &BTreeMap::from([(2, 8), (3, 10)]), 6).unwrap();
        assert_eq!(options, vec![Some((3, 10, 6)), Some((2, 8, 4))]);
        let (plan, target) =
            select_eviction_alternatives(18, 18, 1, &[candidate], &counts, &options).unwrap();
        assert_eq!(plan, vec![candidate]);
        assert_eq!(target, Some((2, 8, 4)));
        assert!(select_eviction_alternatives(18, 18, 1, &[], &counts, &options).is_err());
        assert!(rank_headroom_options(&limits, &BTreeMap::from([(2, 8)]), 6).is_err());
        assert!(rank_headroom_options(&limits, &BTreeMap::from([(2, 11), (3, 0)]), 6).is_err());
        assert!(rank_headroom_options(&limits, &BTreeMap::from([(2, 8), (3, 10)]), 13).is_err());
        // Malformed plans must fail closed rather than trying another rank.
        assert!(
            select_eviction_alternatives(18, 18, 1, &[candidate], &BTreeMap::new(), &options)
                .is_err()
        );
    }

    #[test]
    #[ignore = "requires VF_TEST_RANK_SESSION and creates fresh non-replayable CPU claims"]
    fn alternate_rank_plan_drives_real_local_workers() {
        exercise_local_workers(false);
    }

    #[test]
    #[ignore = "requires VF_TEST_RANK_SESSION and creates fresh non-replayable CPU claims"]
    fn alternate_rank_preserves_nonzero_local_results() {
        exercise_local_workers(true);
    }

    fn exercise_local_workers(nonzero: bool) {
        use crate::{
            moe::{Choice, Registry, Replica, Stamp},
            native_ggml_cpu::Binding,
            routing::RoutingProblem,
            session::{Endpoint, NativeEviction, Session},
        };
        use std::time::{Duration, SystemTime, UNIX_EPOCH};
        const INDEX: &str = include_str!("../fixtures/strata_sc117_native_experts_v3.txt");
        const INDEX_SHA: &str = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24";
        let blob_sha = if nonzero {
            "2c7dcdcf2992bb6127b86407fb43abe67f4ba202acbe005a9bc67d7df9e072b0"
        } else {
            "50eff9144c635807dfdc0cafd1c3bb978d0202d537ae892ec8ebed90dc13e172"
        };
        const PACK_SIZE: u64 = 50_292_326_400;
        const BYTES: u64 = 2_176_000;
        struct VirtualPack<'a> {
            pos: u64,
            bytes: &'a [u8],
        }
        impl std::io::Seek for VirtualPack<'_> {
            fn seek(&mut self, from: std::io::SeekFrom) -> std::io::Result<u64> {
                let pos = match from {
                    std::io::SeekFrom::Start(n) => i128::from(n),
                    std::io::SeekFrom::End(n) => i128::from(PACK_SIZE) + i128::from(n),
                    std::io::SeekFrom::Current(n) => i128::from(self.pos) + i128::from(n),
                };
                self.pos = u64::try_from(pos).map_err(|_| std::io::Error::other("invalid seek"))?;
                Ok(self.pos)
            }
        }
        impl std::io::Read for VirtualPack<'_> {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                let start = usize::try_from(self.pos).unwrap_or(usize::MAX);
                if start >= self.bytes.len() {
                    return Ok(0);
                }
                let n = out.len().min(self.bytes.len() - start);
                out[..n].copy_from_slice(&self.bytes[start..start + n]);
                self.pos += n as u64;
                Ok(n)
            }
        }
        let worker = std::env::var("VF_TEST_RANK_SESSION").expect("explicit local worker path");
        assert!(std::path::Path::new(&worker).is_absolute());
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let attempt = u64::try_from(now).unwrap();
        let name = format!("r513-alt-workers-{}-{now}", std::process::id());
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(".runs")
            .join(&name);
        std::fs::create_dir(&path).unwrap();
        let ledger = NativeBatchLedger::open(&path).unwrap();
        let stamp = Stamp {
            run_id: name,
            model_revision: "model-test".into(),
            plan_id: if nonzero { "r513-nonzero" } else { "r510-plan" }.into(),
            layer: 0,
            batch: 0,
            epoch: 3,
        };
        let binding = |rank, expert, local| Binding {
            model_revision: stamp.model_revision.clone(),
            rank,
            layer: 0,
            expert,
            local,
            epoch: 3,
            index_sha256: INDEX_SHA.into(),
            blob_sha256: blob_sha.into(),
        };
        let mut sessions = Vec::new();
        let mut blob = vec![0u8; BYTES as usize];
        if nonzero {
            for i in (0..1_254_400).step_by(98) {
                blob[i + 1] = 0x28;
            }
            for i in (1_254_400..2_176_000).step_by(18) {
                blob[i + 1] = 0x28;
            }
        }
        assert_eq!(crate::sha256::hex(&blob), blob_sha);
        let input = vec![if nonzero { 0.001_f32 } else { 0.0 }; 2560];
        let layout =
            crate::strata_native_layout::NativeLayout::parse_sc117_v3(INDEX, PACK_SIZE).unwrap();
        let verified = layout
            .read_verified_expert(
                &mut VirtualPack {
                    pos: 0,
                    bytes: &blob,
                },
                0,
                0,
                INDEX_SHA,
                blob_sha,
            )
            .unwrap();
        let mut reference =
            crate::native_ggml_cpu::Resident::load(&layout, verified, binding(2, 0, 0)).unwrap();
        let expected: Vec<_> = reference
            .forward(&input)
            .unwrap()
            .iter()
            .map(|&v| {
                crate::completion::decode_fp16(
                    crate::completion::encode_fp16(f64::from(v)).unwrap(),
                )
                .unwrap()
            })
            .collect();
        if nonzero {
            assert!(expected.iter().any(|v| *v != 0.0));
        }
        let timeout = Duration::from_secs(30);
        for (rank, experts) in [(2, vec![0, 1]), (3, vec![2, 3, 4])] {
            let mut session = Session::connect(
                &Endpoint {
                    rank,
                    argv: vec![worker.clone()],
                },
                timeout,
            )
            .unwrap();
            for (local, expert) in experts.iter().enumerate() {
                session
                    .load_native_ggml(
                        &binding(rank, *expert, local as u32),
                        INDEX,
                        PACK_SIZE,
                        &blob,
                        timeout,
                    )
                    .unwrap();
            }
            session.seal_native_ggml(experts.len(), timeout).unwrap();
            sessions.push(session);
        }
        let mut problem = RoutingProblem {
            hidden: 2560,
            experts: vec![0, 1, 2, 3, 4],
            routes: vec![vec![
                Choice {
                    expert: 0,
                    weight: 0.25,
                },
                Choice {
                    expert: 2,
                    weight: 0.25,
                },
                Choice {
                    expert: 3,
                    weight: 0.25,
                },
                Choice {
                    expert: 4,
                    weight: 0.25,
                },
            ]],
            registry: Registry {
                epoch: 3,
                copies: vec![
                    Replica {
                        expert: 0,
                        rank: 2,
                        local: 0,
                    },
                    Replica {
                        expert: 1,
                        rank: 2,
                        local: 1,
                    },
                    Replica {
                        expert: 2,
                        rank: 3,
                        local: 0,
                    },
                    Replica {
                        expert: 3,
                        rank: 3,
                        local: 1,
                    },
                    Replica {
                        expert: 4,
                        rank: 3,
                        local: 2,
                    },
                ],
            },
            stamp: stamp.clone(),
            preferred_owners: BTreeMap::new(),
            resident_order: BTreeMap::from([(2, vec![0, 1]), (3, vec![0, 1, 2])]),
        };
        let ticket = ledger.reserve_ticket(&stamp, attempt).unwrap();
        let first = Session::exec_native_ggml_reserved(
            &mut sessions,
            &problem,
            std::slice::from_ref(&input),
            &ledger,
            ticket,
            timeout,
        )
        .unwrap();
        assert_eq!(first.rows()[0], expected);
        let candidate = EvictionCandidate {
            layer: 0,
            expert: 1,
            rank: 2,
            bytes: BYTES,
            last_used: 0,
        };
        let options = rank_headroom_options(
            &BTreeMap::from([(2, 4 * BYTES), (3, 5 * BYTES)]),
            &BTreeMap::from([(2, 2 * BYTES), (3, 3 * BYTES)]),
            3 * BYTES,
        )
        .unwrap();
        let traffic: Vec<_> = sessions.iter().map(|s| s.traffic.bytes_sent).collect();
        let (plan, selected) = select_eviction_alternatives(
            5 * BYTES,
            5 * BYTES,
            1,
            &[candidate],
            &BTreeMap::from([((0, 2), 2), ((0, 3), 3)]),
            &options,
        )
        .unwrap();
        assert_eq!(selected, Some((2, 2 * BYTES, BYTES)));
        assert_eq!(plan, vec![candidate]);
        assert_eq!(
            traffic,
            sessions
                .iter()
                .map(|s| s.traffic.bytes_sent)
                .collect::<Vec<_>>()
        );
        let evicted = binding(2, 1, 1);
        let op = ledger.reserve_eviction(&stamp, attempt, &evicted).unwrap();
        sessions[0]
            .evict_native_ggml(
                &NativeEviction {
                    stamp: &stamp,
                    attempt,
                    operation_id: &op,
                    binding: &evicted,
                    expected_bytes: BYTES,
                    expected_remaining: 1,
                },
                timeout,
            )
            .unwrap();
        assert_eq!(sessions[1].traffic.bytes_sent, traffic[1]);
        assert!(ledger.reserve_eviction(&stamp, attempt, &evicted).is_err());
        problem.stamp.batch = 1;
        problem
            .registry
            .copies
            .retain(|copy| !(copy.rank == 2 && copy.expert == 1));
        problem.experts.retain(|&expert| expert != 1);
        problem.resident_order.insert(2, vec![0]);
        let ticket = ledger.reserve_ticket(&problem.stamp, attempt + 1).unwrap();
        let second = Session::exec_native_ggml_reserved(
            &mut sessions,
            &problem,
            &[input],
            &ledger,
            ticket,
            timeout,
        )
        .unwrap();
        assert_eq!(second.rows()[0], expected);
        for session in sessions {
            let receipt = session.close(timeout);
            assert!(!receipt.killed && receipt.exit_code == Some(0) && receipt.error.is_none());
            assert_eq!(receipt.bye_execs, Some(2));
        }
    }

    #[test]
    fn least_recent_order_is_deterministic_and_prefers_unused() {
        let mut c = [
            EvictionCandidate {
                layer: 0,
                expert: 1,
                rank: 2,
                bytes: 1,
                last_used: 3,
            },
            EvictionCandidate {
                layer: 1,
                expert: 8,
                rank: 3,
                bytes: 1,
                last_used: 0,
            },
            EvictionCandidate {
                layer: 1,
                expert: 7,
                rank: 2,
                bytes: 1,
                last_used: 1,
            },
            EvictionCandidate {
                layer: 0,
                expert: 9,
                rank: 3,
                bytes: 1,
                last_used: 1,
            },
        ];
        sort_eviction_candidates(&mut c);
        assert_eq!(
            c.map(|x| (x.layer, x.expert)),
            [(1, 8), (0, 9), (1, 7), (0, 1)]
        );
    }

    #[test]
    fn model_chain_refuses_to_skip_required_layer_one_ple() {
        assert!(require_supported_layer(1, false, false).is_err());
        assert!(require_supported_layer(1, true, false).is_err());
        assert!(require_supported_layer(1, true, true).is_ok());
        assert!(require_supported_layer(1, false, true).is_err());
        assert!(require_supported_layer(0, true, false).is_err());
        assert!(require_supported_layer(0, false, false).is_ok());
        assert!(require_supported_layer(2, false, false).is_ok());
    }

    #[test]
    fn ple_history_commits_only_after_success_and_uses_previous_tokens() {
        let mut state = PleState::new();
        let make = |token, prev, _input: &[f32], history: &[f32]| {
            assert_eq!(token, 42);
            assert_eq!(prev, [sc117_ple::NULL_TOKEN; 2]);
            assert_eq!(history[0], 0.0);
            Ok(sc117_ple_post::Output {
                residual: vec![2.0; 10240],
                normalized: vec![],
                next_history: vec![3.0; 10240 * 9],
            })
        };
        let candidate = state.prepare(0, 42, &vec![0.0; 10240], make).unwrap();
        assert!(state
            .finish(42, candidate, Err(invalid("worker failure")))
            .is_err());
        assert_eq!(state.position, 0);
        assert_eq!(state.history[0], 0.0);
        assert_eq!(state.previous, [-1, -1]);
        let candidate = state.prepare(0, 42, &vec![0.0; 10240], make).unwrap();
        assert_eq!(
            state.finish(42, candidate, Ok(vec![5.0; 10240])).unwrap()[0],
            5.0
        );
        let next = state
            .prepare(1, 73, &vec![0.0; 10240], |token, prev, _, hist| {
                assert_eq!((token, prev, hist[0]), (73, [-1, 42], 3.0));
                Ok(sc117_ple_post::Output {
                    residual: vec![4.0; 10240],
                    normalized: vec![],
                    next_history: vec![6.0; 10240 * 9],
                })
            })
            .unwrap();
        state.finish(73, next, Ok(vec![7.0; 10240])).unwrap();
        assert_eq!(state.previous, [42, 73]);
        assert_eq!(state.position, 2);
        assert!(state
            .prepare(1, 10, &[], |_, _, _, _| unreachable!())
            .is_err());
        assert!(state
            .prepare(2, sc117_ple::VOCAB, &[], |_, _, _, _| unreachable!())
            .is_err());
    }

    #[test]
    fn ple_batch_is_causal_and_rejects_failures_without_mutating_live_state() {
        let state = PleState::new();
        let inputs = vec![vec![0.0; 10240], vec![1.0; 10240]];
        let produce = |token: i32, previous: [i32; 2], _: &[f32], history: &[f32]| {
            if token == 42 {
                assert_eq!((previous, history[0]), ([-1, -1], 0.0));
            } else {
                assert_eq!((token, previous, history[0]), (73, [-1, 42], 3.0));
            }
            Ok(sc117_ple_post::Output {
                residual: vec![token as f32; 10240],
                normalized: vec![],
                next_history: vec![if token == 42 { 3.0 } else { 6.0 }; 10240 * 9],
            })
        };
        let (candidate, outputs) = state.prepare_batch(0, &[42, 73], &inputs, produce).unwrap();
        assert_eq!(
            (candidate.position, candidate.previous, candidate.history[0]),
            (2, [42, 73], 6.0)
        );
        assert_eq!((outputs[0][0], outputs[1][0]), (42.0, 73.0));
        assert_eq!(
            (state.position, state.previous, state.history[0]),
            (0, [-1, -1], 0.0)
        );
        assert!(state
            .prepare_batch(0, &[42, 73], &inputs, |token, _, _, _| {
                if token == 73 {
                    return Err(invalid("injected PLE error"));
                }
                Ok(sc117_ple_post::Output {
                    residual: vec![0.0; 10240],
                    normalized: vec![],
                    next_history: vec![1.0; 10240 * 9],
                })
            })
            .is_err());
        assert_eq!(
            (state.position, state.previous, state.history[0]),
            (0, [-1, -1], 0.0)
        );
        assert!(state
            .prepare_batch(0, &[42], &inputs, |_, _, _, _| unreachable!())
            .is_err());
        assert!(state
            .prepare_batch(
                0,
                &[42, sc117_ple::VOCAB],
                &inputs,
                |_, _, _, _| unreachable!()
            )
            .is_err());
    }

    fn ledger() -> NativeBatchLedger {
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(".runs")
            .join(format!(
                "r434-chain-test-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
        std::fs::create_dir(&directory).unwrap();
        NativeBatchLedger::open(&directory).unwrap()
    }
    fn base() -> Stamp {
        Stamp {
            run_id: "r434-synthetic".into(),
            model_revision: "model".into(),
            plan_id: "plan".into(),
            layer: 3,
            batch: 77,
            epoch: 9,
        }
    }
    fn cursor() -> Cursor {
        Cursor {
            position: 0,
            last_batch_start: None,
            last_attempt: 0,
            poisoned: false,
        }
    }

    struct RefusingPreflight {
        calls: usize,
    }
    impl MissingExpertSource for RefusingPreflight {
        fn preflight(
            &mut self,
            _: &Stamp,
            _: u64,
            experts: &[u32],
            _: &Registry,
            _: &BTreeMap<u32, Vec<u32>>,
        ) -> Result<()> {
            self.calls += 1;
            assert_eq!(experts, &[7, 8]);
            Err(invalid("planned capacity refusal"))
        }
        fn load(
            &mut self,
            _: &Stamp,
            _: u64,
            _: u32,
            _: &Registry,
            _: &BTreeMap<u32, Vec<u32>>,
        ) -> Result<ExpertAdmission> {
            panic!("preflight refusal must precede every worker load")
        }
    }
    #[test]
    fn distinct_missing_experts_are_preflighted_before_any_admission() {
        let mut owners = Owners {
            sessions: vec![],
            registry: Registry {
                epoch: 9,
                copies: vec![],
            },
            preferred: BTreeMap::new(),
            order: BTreeMap::new(),
        };
        let choice = |expert| crate::sc117_router_cpu::Choice {
            expert,
            weight: 1.0,
            logit: 0.0,
        };
        let mut provider = RefusingPreflight { calls: 0 };
        let mut src: Option<&mut dyn MissingExpertSource> = Some(&mut provider);
        let result = owners.ensure(
            &base(),
            11,
            &[choice(7), choice(8), choice(7)],
            &mut src,
            Duration::from_secs(1),
        );
        assert!(result.is_err());
        assert_eq!(provider.calls, 1);
        assert!(owners.registry.copies.is_empty());
    }

    struct WrongSource;
    impl MissingExpertSource for WrongSource {
        fn load(
            &mut self,
            stamp: &Stamp,
            attempt: u64,
            expert: u32,
            _: &Registry,
            _: &BTreeMap<u32, Vec<u32>>,
        ) -> Result<ExpertAdmission> {
            assert_eq!((stamp.layer, attempt, expert), (3, 11, 7));
            Ok(ExpertAdmission {
                binding: crate::native_ggml_cpu::Binding {
                    model_revision: stamp.model_revision.clone(),
                    rank: 2,
                    layer: stamp.layer,
                    expert: 8,
                    local: 0,
                    epoch: stamp.epoch,
                    index_sha256: String::new(),
                    blob_sha256: String::new(),
                },
                index: String::new(),
                pack_bytes: 0,
                bytes: vec![],
            })
        }
    }
    #[test]
    fn source_cannot_substitute_another_expert_and_is_skipped_for_residents() {
        let mut owners = Owners {
            sessions: vec![],
            registry: Registry {
                epoch: 9,
                copies: vec![],
            },
            preferred: BTreeMap::new(),
            order: BTreeMap::new(),
        };
        let choice = crate::sc117_router_cpu::Choice {
            expert: 7,
            weight: 1.0,
            logit: 0.0,
        };
        let mut provider = WrongSource;
        let mut source: Option<&mut dyn MissingExpertSource> = Some(&mut provider);
        let error = owners
            .ensure(
                &base(),
                11,
                std::slice::from_ref(&choice),
                &mut source,
                Duration::from_secs(1),
            )
            .unwrap_err();
        assert!(error.message.contains("different identity"));
        assert!(owners.registry.copies.is_empty());
        owners.registry.copies.push(crate::moe::Replica {
            expert: 7,
            rank: 2,
            local: 0,
        });
        assert!(owners
            .ensure(&base(), 11, &[choice], &mut source, Duration::from_secs(1))
            .is_ok());
        assert_eq!(owners.registry.copies.len(), 1);
    }

    #[test]
    fn sequential_residuals_and_fresh_claims_across_tokens() {
        let ledger = ledger();
        let mut state = cursor();
        for token in 0..2 {
            let mut visited = Vec::new();
            let output = advance(
                &mut state,
                &ledger,
                &base(),
                2,
                token + 10,
                &vec![1.0; 10240],
                |i, ticket, values| {
                    visited.push((ticket.stamp().layer, ticket.stamp().batch, ticket.attempt()));
                    Ok(values
                        .iter()
                        .map(|x| if i == 0 { x + 2.0 } else { x * 3.0 })
                        .collect())
                },
            )
            .unwrap();
            assert_eq!(output, vec![9.0; 10240]);
            assert_eq!(
                visited,
                vec![(3, token, token + 10), (4, token, token + 10)]
            );
            assert_eq!(state.position, token + 1);
        }
        assert!(!state.poisoned);
        let old = Stamp { batch: 0, ..base() };
        assert!(ledger.reserve_ticket(&old, 99).is_err());
    }

    #[test]
    fn later_failure_poisons_segment_and_exposes_no_partial_output() {
        let ledger = ledger();
        let mut state = cursor();
        let mut calls = 0;
        let result = advance(
            &mut state,
            &ledger,
            &base(),
            3,
            11,
            &vec![1.0; 10240],
            |i, _, values| {
                calls += 1;
                if i == 1 {
                    return Err(invalid("worker failed"));
                }
                Ok(values.iter().map(|v| v + 1.0).collect())
            },
        );
        assert!(result.is_err());
        assert_eq!(calls, 2);
        assert!(state.poisoned);
        assert_eq!(state.position, 0);
        assert!(advance(
            &mut state,
            &ledger,
            &base(),
            3,
            12,
            &vec![1.0; 10240],
            |_, _, _| panic!("poisoned segment must not run")
        )
        .is_err());
        for layer in 3..=4 {
            assert!(ledger
                .reserve_ticket(
                    &Stamp {
                        layer,
                        batch: 0,
                        ..base()
                    },
                    12
                )
                .is_err());
        }
        assert!(ledger
            .reserve_ticket(
                &Stamp {
                    layer: 5,
                    batch: 0,
                    ..base()
                },
                12
            )
            .is_ok());
    }

    #[test]
    fn observation_is_released_only_after_every_layer_commits() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        struct Witness(Arc<AtomicUsize>);
        impl Drop for Witness {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(AtomicUsize::new(0));
        let ledger1 = ledger();
        let mut state = cursor();
        let result = advance_with_observation(
            &mut state,
            &ledger1,
            &base(),
            3,
            11,
            &vec![1.0; 10240],
            |i, _, values, trace| {
                if i == 0 {
                    *trace = Some(Witness(dropped.clone()));
                }
                if i == 1 {
                    return Err(invalid("later layer failed"));
                }
                Ok(values.to_vec())
            },
        );
        assert!(result.is_err());
        assert!(state.poisoned);
        assert_eq!(state.position, 0);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);

        let ledger2 = ledger();
        let mut state2 = cursor();
        let (output, trace) = advance_with_observation(
            &mut state2,
            &ledger2,
            &base(),
            3,
            12,
            &vec![1.0; 10240],
            |i, _, values, trace| {
                if i == 0 {
                    *trace = Some(Witness(dropped.clone()));
                }
                Ok(values.to_vec())
            },
        )
        .unwrap();
        assert_eq!(output.len(), 10240);
        assert_eq!(state2.position, 1);
        assert!(!state2.poisoned);
        assert!(trace.is_some());
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        drop(trace);
        assert_eq!(dropped.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn invalid_input_and_repeated_attempt_are_refused_before_reservation() {
        let ledger = ledger();
        let mut state = cursor();
        let mut bad = vec![0.0; 10240];
        bad[0] = f32::NAN;
        assert!(
            advance(&mut state, &ledger, &base(), 1, 5, &bad, |_, _, _| panic!(
                "invalid input reached layer"
            ))
            .is_err()
        );
        assert!(!state.poisoned);
        advance(
            &mut state,
            &ledger,
            &base(),
            1,
            5,
            &vec![1.0; 10240],
            |_, _, x| Ok(x.to_vec()),
        )
        .unwrap();
        assert!(advance(
            &mut state,
            &ledger,
            &base(),
            1,
            5,
            &vec![1.0; 10240],
            |_, _, _| panic!("repeated attempt reached layer")
        )
        .is_err());
        assert!(!state.poisoned);
    }
}
