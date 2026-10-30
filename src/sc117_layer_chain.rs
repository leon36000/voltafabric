//! Persistent CPU segment of real reservation-bound SC117 layers.
//! The local ledger is not an R53 lease. A segment is not a whole model.
use crate::{
    gdn_cpu::Identity,
    invalid,
    moe::{Registry, Stamp},
    native_ledger::{NativeBatchLedger, Reservation},
    sc117_bound_layer, sc117_ffn_stage, sc117_gdn_stage, sc117_post_attention,
    sc117_qsa_bound_layer, sc117_qsa_stage, sc117_shared_stage,
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
enum Attention {
    Gdn(sc117_gdn_stage::Stage, sc117_post_attention::Stage),
    Qsa(sc117_qsa_stage::Stage, sc117_ffn_stage::Stage),
}
pub struct Layer {
    identity: Identity,
    attention: Attention,
    shared: sc117_shared_stage::Stage,
    owners: Owners,
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
        })
    }
    fn run(
        &mut self,
        ledger: &NativeBatchLedger,
        ticket: Reservation,
        position: u64,
        input: &[f32],
        timeout: Duration,
    ) -> Result<Vec<f32>> {
        match &mut self.attention {
            Attention::Gdn(gdn, post) => {
                let proposal = gdn.prepare_reserved(ledger, &ticket, position, input)?;
                let routed =
                    sc117_bound_layer::Routed::prepare(post, ledger, &ticket, proposal, input)?;
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
impl Segment {
    pub fn new(
        layers: Vec<Layer>,
        plan_id: String,
        ledger: NativeBatchLedger,
        timeout: Duration,
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
        let order = layer
            .owners
            .order
            .get_mut(&binding.rank)
            .ok_or_else(|| invalid("segment append rank order"))?;
        if order.contains(&binding.local)
            || layer.owners.registry.copies.iter().any(|r| {
                r.rank == binding.rank && (r.local == binding.local || r.expert == binding.expert)
            })
        {
            return Err(invalid("segment append duplicate expert or local"));
        }
        let mut matches = layer
            .owners
            .sessions
            .iter_mut()
            .filter(|s| s.rank() == binding.rank);
        let session = matches
            .next()
            .ok_or_else(|| invalid("segment append session missing"))?;
        if matches.next().is_some() {
            return Err(invalid("segment append duplicate rank"));
        }
        // Fail closed even if an error occurred before transmission. This keeps
        // the outer state from guessing whether a worker installed the weight.
        self.cursor.poisoned = true;
        session.extend_native_ggml(binding, index, pack_bytes, bytes, self.timeout)?;
        layer.owners.registry.copies.push(crate::moe::Replica {
            expert: binding.expert,
            rank: binding.rank,
            local: binding.local,
        });
        order.push(binding.local);
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
            |i, ticket, values| layers[i].run(ledger, ticket, position, values, timeout),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

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
