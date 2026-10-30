//! Opt-in CPU QSA whole-layer transaction: candidate cache commits after a
//! complete local Collector result. This local ledger is not a hardware lease.
use crate::{
    gdn_cpu::Identity,
    invalid,
    moe::{Choice, Registry, Stamp},
    native_ledger::{NativeBatchLedger, Reservation},
    routing::{self, RoutingProblem},
    sc117_ffn_stage,
    sc117_layer_completion::{self, StampedValues},
    sc117_qsa_stage, sc117_shared_stage,
    session::{NativeCollected, Session},
    Result,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Duration,
};
pub struct Routed {
    proposal: sc117_qsa_stage::Proposal,
    route: sc117_ffn_stage::Routed,
    identity: Identity,
    stamp: Stamp,
    attempt: u64,
    ledger_id: (u64, u64),
    marker: Arc<()>,
}
pub struct Shared {
    value: Vec<f32>,
    marker: Arc<()>,
}
pub struct Collected {
    value: NativeCollected,
    marker: Arc<()>,
}
pub struct Committed {
    pub attention: sc117_qsa_stage::Output,
    pub residual: Vec<f32>,
}
impl Shared {
    pub fn values(&self) -> &[f32] {
        &self.value
    }
}
impl Collected {
    pub fn rows(&self) -> &[Vec<f64>] {
        self.value.rows()
    }
}
impl Routed {
    pub fn prepare(
        qsa: &mut sc117_qsa_stage::Stage,
        ffn: &sc117_ffn_stage::Stage,
        ledger: &NativeBatchLedger,
        ticket: &Reservation,
        position: usize,
        residual: &[f32],
    ) -> Result<Self> {
        ledger.validate_ticket(ticket, ticket.stamp())?;
        let stamp = ticket.stamp().clone();
        let identity = Identity {
            run_id: stamp.run_id.clone(),
            model_revision: stamp.model_revision.clone(),
            layer: stamp.layer,
            epoch: stamp.epoch,
        };
        let proposal = qsa.prepare(&identity, position, residual)?;
        let route = ffn.compute(
            &identity,
            residual,
            &proposal.output().attention,
            &proposal.output().inject,
        )?;
        Ok(Self {
            proposal,
            route,
            identity,
            stamp,
            attempt: ticket.attempt(),
            ledger_id: ticket.directory_id(),
            marker: Arc::new(()),
        })
    }
    pub fn attention_prefix(&self) -> &sc117_qsa_stage::Output {
        self.proposal.output()
    }
    pub fn choices(&self) -> &[crate::sc117_router_cpu::Choice] {
        &self.route.choices
    }
    pub fn activation(&self) -> &[f32] {
        &self.route.mixed
    }
    pub fn attention_residual(&self) -> &[f32] {
        &self.route.residual
    }
    pub fn injection(&self) -> &[f32] {
        &self.route.inject
    }
    pub fn shared(&self, stage: &mut sc117_shared_stage::Stage) -> Result<Shared> {
        Ok(Shared {
            value: stage.compute(&self.identity, &self.route.mixed)?,
            marker: self.marker.clone(),
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        sessions: &mut [Session],
        registry: Registry,
        preferred_owners: BTreeMap<u32, u32>,
        resident_order: BTreeMap<u32, Vec<u32>>,
        ledger: &NativeBatchLedger,
        ticket: Reservation,
        timeout: Duration,
    ) -> Result<Collected> {
        ledger.validate_ticket(&ticket, &self.stamp)?;
        if ticket.attempt() != self.attempt || ticket.directory_id() != self.ledger_id {
            return Err(invalid("QSA bound attempt mismatch"));
        }
        let experts = registry
            .copies
            .iter()
            .map(|r| r.expert)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let choices = self
            .route
            .choices
            .iter()
            .map(|c| Choice {
                expert: c.expert,
                weight: f64::from(c.weight),
            })
            .collect();
        let problem = RoutingProblem {
            hidden: 2560,
            experts,
            routes: vec![choices],
            registry,
            stamp: self.stamp.clone(),
            preferred_owners,
            resident_order,
        };
        routing::validate(&problem)?;
        let value = Session::exec_native_ggml_reserved(
            sessions,
            &problem,
            std::slice::from_ref(&self.route.mixed),
            ledger,
            ticket,
            timeout,
        )?;
        Ok(Collected {
            value,
            marker: self.marker.clone(),
        })
    }
    pub fn commit(
        self,
        qsa: &mut sc117_qsa_stage::Stage,
        shared: Shared,
        collected: Collected,
    ) -> Result<Committed> {
        if !Arc::ptr_eq(&self.marker, &shared.marker)
            || !Arc::ptr_eq(&self.marker, &collected.marker)
            || collected.value.stamp() != &self.stamp
            || collected.value.attempt() != self.attempt
            || collected.value.ledger_id() != self.ledger_id
        {
            return Err(invalid("QSA bound producer mismatch"));
        }
        let wrap = |values| StampedValues {
            stamp: self.stamp.clone(),
            attempt: self.attempt,
            values,
        };
        let residual = sc117_layer_completion::finish_native(
            &collected.value,
            wrap(shared.value),
            wrap(self.route.residual),
            wrap(self.route.inject),
        )?;
        let attention = qsa.commit(self.proposal)?;
        Ok(Committed {
            attention,
            residual,
        })
    }
}
