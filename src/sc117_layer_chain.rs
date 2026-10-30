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
        for choice in choices {
            if self
                .registry
                .copies
                .iter()
                .any(|copy| copy.expert == choice.expert)
            {
                continue;
            }
            let material =
                source.load(stamp, attempt, choice.expert, &self.registry, &self.order)?;
            if material.binding.model_revision != stamp.model_revision
                || material.binding.layer != stamp.layer
                || material.binding.epoch != stamp.epoch
                || material.binding.expert != choice.expert
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
pub struct Layer {
    identity: Identity,
    attention: Attention,
    shared: sc117_shared_stage::Stage,
    owners: Owners,
    ple: Option<Ple>,
    last_selected: Vec<u32>,
}

struct Ple {
    table: sc117_ple::Table,
    weights: sc117_ple_weights::Weights,
    state: PleState,
}
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
    ) -> Result<Vec<f32>> {
        let stamp = ticket.stamp().clone();
        let attempt = ticket.attempt();
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
                Ok(routed.commit(qsa, shared, collected)?.residual)
            }
        };
        let output = if let (Some(ple), Some(candidate)) = (&mut self.ple, candidate) {
            // Only after all attention, shared and routed expert work committed.
            ple.state
                .finish(token.expect("PLE token was validated"), candidate, output)
        } else {
            output
        }?;
        self.last_selected = selected;
        Ok(output)
    }
}

struct Cursor {
    position: u64,
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
    cursor.position += 1;
    cursor.last_attempt = attempt;
    cursor.poisoned = false;
    Ok(residual)
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EvictionCandidate {
    layer: u32,
    expert: u32,
    rank: u32,
    bytes: u64,
}

// Plan entirely before sending anything. Input order is deterministic; each
// rank must retain at least one resident and arithmetic is checked.
fn select_evictions(
    used: u64,
    target: u64,
    max_evictions: usize,
    candidates: &[EvictionCandidate],
    counts: &BTreeMap<(u32, u32), usize>,
) -> Result<Vec<EvictionCandidate>> {
    if used <= target {
        return Ok(Vec::new());
    }
    if max_evictions == 0 || max_evictions > 768 {
        return Err(invalid("eviction limit"));
    }
    let mut counts = counts.clone();
    let mut remaining = used;
    let mut selected = Vec::new();
    for &candidate in candidates {
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
        *count -= 1;
        selected.push(candidate);
        if remaining <= target {
            return Ok(selected);
        }
        if selected.len() == max_evictions {
            break;
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
        stamp.batch = self.cursor.position - 1;
        stamp.layer = binding.layer;
        let layer = self
            .layers
            .iter_mut()
            .find(|l| l.identity.layer == binding.layer)
            .ok_or_else(|| invalid("segment eviction layer"))?;
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
        self.cursor.poisoned = false;
        Ok(EvictionReceipt {
            stamp,
            attempt,
            operation_id,
            binding: binding.clone(),
            released_payload_bytes: bytes,
        })
    }
    /// Explicitly trim accounted payload between committed positions. Protects
    /// each layer's last committed gate choices; this does not predict the next
    /// gate, measure RSS, or retry any uncertain operation. If a later eviction
    /// fails, earlier acknowledged evictions remain applied and the segment is
    /// poisoned by the failing operation.
    pub fn evict_to_target(
        &mut self,
        source: &mut crate::sc117_expert_source::Source<'_>,
        target_bytes: u64,
        max_evictions: usize,
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
                if !layer.last_selected.contains(&copy.expert) {
                    candidates.push(EvictionCandidate {
                        layer: layer.identity.layer,
                        expert: copy.expert,
                        rank: copy.rank,
                        bytes: source.resident_bytes(layer.identity.layer, copy.expert)?,
                    });
                }
            }
        }
        let plan = select_evictions(
            source.used_bytes(),
            target_bytes,
            max_evictions,
            &candidates,
            &counts,
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
            stamp.batch = self.cursor.position - 1;
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
        if source.used_bytes() > target_bytes {
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
        validate_advance(&self.cursor, self.layers.len(), attempt, input)?;
        if self.layers.iter().any(|layer| layer.ple.is_some())
            && !(0..sc117_ple::VOCAB).contains(&token)
        {
            return Err(invalid("segment PLE token required or invalid"));
        }
        let receipts = if let Some(target) =
            headroom_target(source.used_bytes(), source.limit_bytes(), headroom_bytes)?
        {
            self.evict_to_target(source, target, max_evictions)?
        } else {
            Vec::new()
        };
        let output = self.advance_token_with_source(attempt, token, input, source)?;
        Ok((output, receipts))
    }

    fn advance_inner(
        &mut self,
        attempt: u64,
        input: &[f32],
        token: Option<i32>,
        source: &mut Option<&mut dyn MissingExpertSource>,
    ) -> Result<Vec<f32>> {
        if self.layers.iter().any(|layer| layer.ple.is_some())
            && token.is_none_or(|t| !(0..sc117_ple::VOCAB).contains(&t))
        {
            return Err(invalid("segment PLE token required or invalid"));
        }
        let position = self.cursor.position;
        let ledger = &self.ledger;
        let timeout = self.timeout;
        let layers = &mut self.layers;
        advance(
            &mut self.cursor,
            ledger,
            &self.base,
            layers.len(),
            attempt,
            input,
            |i, ticket, values| {
                layers[i].run(ledger, ticket, position, values, token, timeout, source)
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
    fn eviction_plan_is_bounded_and_preserves_last_resident_per_rank() {
        let c = [
            EvictionCandidate {
                layer: 0,
                expert: 1,
                rank: 2,
                bytes: 4,
            },
            EvictionCandidate {
                layer: 0,
                expert: 2,
                rank: 2,
                bytes: 4,
            },
            EvictionCandidate {
                layer: 0,
                expert: 3,
                rank: 3,
                bytes: 4,
            },
            EvictionCandidate {
                layer: 1,
                expert: 4,
                rank: 2,
                bytes: 4,
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
            last_attempt: 0,
            poisoned: false,
        }
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
