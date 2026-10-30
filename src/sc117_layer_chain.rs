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
        let output = match &mut self.attention {
            Attention::Gdn(gdn, post) => {
                let proposal = gdn.prepare_reserved(ledger, &ticket, position, input)?;
                let routed =
                    sc117_bound_layer::Routed::prepare(post, ledger, &ticket, proposal, input)?;
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
        if let (Some(ple), Some(candidate)) = (&mut self.ple, candidate) {
            // Only after all attention, shared and routed expert work committed.
            ple.state
                .finish(token.expect("PLE token was validated"), candidate, output)
        } else {
            output
        }
    }
}

struct Cursor {
    position: u64,
    last_attempt: u64,
    poisoned: bool,
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

pub struct Segment {
    layers: Vec<Layer>,
    base: Stamp,
    ledger: NativeBatchLedger,
    timeout: Duration,
    cursor: Cursor,
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
    pub fn position(&self) -> u64 {
        self.cursor.position
    }
    pub fn is_poisoned(&self) -> bool {
        self.cursor.poisoned
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
