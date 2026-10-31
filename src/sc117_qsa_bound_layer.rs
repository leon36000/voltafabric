//! Opt-in CPU QSA whole-layer transaction: candidate cache commits after a
//! complete local Collector result. This local ledger is not a hardware lease.
use crate::{
    gdn_cpu::Identity,
    hyperconnection_cpu, invalid,
    moe::{Choice, Registry, Stamp},
    native_ledger::{NativeBatchLedger, Reservation},
    routing::{self, RoutingProblem},
    sc117_ffn_stage,
    sc117_layer_completion::{self, StampedValues},
    sc117_qsa_stage, sc117_shared_stage,
    session::{NativeCollected, Session},
    shared_expert_cpu, Result,
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

/// One reserved multi-position QSA layer. Its cache stays provisional until
/// every shared and routed result has been validated and all residuals exist.
pub struct RoutedBatch<'a> {
    qsa: sc117_qsa_stage::Batch<'a>,
    routes: Vec<sc117_ffn_stage::Routed>,
    identity: Identity,
    stamp: Stamp,
    attempt: u64,
    ledger_id: (u64, u64),
    marker: Arc<()>,
}
pub struct SharedBatch {
    values: Vec<Vec<f32>>,
    marker: Arc<()>,
}
pub struct CollectedBatch {
    value: NativeCollected,
    marker: Arc<()>,
}
pub struct CommittedBatch {
    pub attention: Vec<sc117_qsa_stage::Output>,
    pub residuals: Vec<Vec<f32>>,
}

impl<'a> RoutedBatch<'a> {
    /// Tentative routes for residency planning; no output is committed here.
    pub fn choices(&self) -> impl Iterator<Item = &[crate::sc117_router_cpu::Choice]> {
        self.routes.iter().map(|r| r.choices.as_slice())
    }
    pub fn activations(&self) -> impl Iterator<Item = &[f32]> {
        self.routes.iter().map(|r| r.mixed.as_slice())
    }
    pub fn prepare(
        qsa: &'a mut sc117_qsa_stage::Stage,
        ffn: &sc117_ffn_stage::Stage,
        ledger: &NativeBatchLedger,
        ticket: &Reservation,
        residuals: &[Vec<f32>],
    ) -> Result<Self> {
        ledger.validate_ticket(ticket, ticket.stamp())?;
        if residuals.is_empty() || residuals.len() > 128 {
            return Err(invalid("QSA batch count"));
        }
        let stamp = ticket.stamp().clone();
        let identity = Identity {
            run_id: stamp.run_id.clone(),
            model_revision: stamp.model_revision.clone(),
            layer: stamp.layer,
            epoch: stamp.epoch,
        };
        let mut batch = qsa.begin_batch(residuals.len())?;
        let mut routes = Vec::with_capacity(residuals.len());
        for residual in residuals {
            let attention = batch.prepare_next(&identity, residual)?;
            routes.push(ffn.compute(
                &identity,
                residual,
                &attention.attention,
                &attention.inject,
            )?);
        }
        Ok(Self {
            qsa: batch,
            routes,
            identity,
            stamp,
            attempt: ticket.attempt(),
            ledger_id: ticket.directory_id(),
            marker: Arc::new(()),
        })
    }
    pub fn shared(&self, stage: &mut sc117_shared_stage::Stage) -> Result<SharedBatch> {
        let mut values = Vec::with_capacity(self.routes.len());
        for route in &self.routes {
            values.push(stage.compute(&self.identity, &route.mixed)?);
        }
        Ok(SharedBatch {
            values,
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
    ) -> Result<CollectedBatch> {
        ledger.validate_ticket(&ticket, &self.stamp)?;
        if ticket.attempt() != self.attempt || ticket.directory_id() != self.ledger_id {
            return Err(invalid("QSA batch attempt mismatch"));
        }
        let experts = registry
            .copies
            .iter()
            .map(|r| r.expert)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let routes = self
            .routes
            .iter()
            .map(|r| {
                r.choices
                    .iter()
                    .map(|c| Choice {
                        expert: c.expert,
                        weight: f64::from(c.weight),
                    })
                    .collect()
            })
            .collect();
        let problem = RoutingProblem {
            hidden: 2560,
            experts,
            routes,
            registry,
            stamp: self.stamp.clone(),
            preferred_owners,
            resident_order,
        };
        routing::validate(&problem)?;
        let inputs = self
            .routes
            .iter()
            .map(|r| r.mixed.clone())
            .collect::<Vec<_>>();
        let value = Session::exec_native_ggml_reserved(
            sessions, &problem, &inputs, ledger, ticket, timeout,
        )?;
        Ok(CollectedBatch {
            value,
            marker: self.marker.clone(),
        })
    }
    pub fn commit(self, shared: SharedBatch, collected: CollectedBatch) -> Result<CommittedBatch> {
        if !Arc::ptr_eq(&self.marker, &shared.marker)
            || !Arc::ptr_eq(&self.marker, &collected.marker)
            || collected.value.stamp() != &self.stamp
            || collected.value.attempt() != self.attempt
            || collected.value.ledger_id() != self.ledger_id
            || shared.values.len() != self.routes.len()
            || collected.value.rows().len() != self.routes.len()
        {
            return Err(invalid("QSA batch producer identity or size"));
        }
        let mut residuals = Vec::with_capacity(self.routes.len());
        for ((route, shared_row), routed_row) in self
            .routes
            .iter()
            .zip(&shared.values)
            .zip(collected.value.rows())
        {
            let combined = shared_expert_cpu::combine(routed_row, shared_row)?;
            residuals.push(hyperconnection_cpu::write(
                &route.residual,
                &combined,
                &route.inject,
            )?);
        }
        let attention = self.qsa.commit()?;
        Ok(CommittedBatch {
            attention,
            residuals,
        })
    }
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
