//! Opaque CPU transaction tying the GDN proposal, post-attention route, shared
//! expert and Rust collector to one local durable reservation. Not an R53 lease.
use crate::{
    gdn_cpu::Identity,
    hyperconnection_cpu, invalid,
    moe::{Choice, Registry, Stamp},
    native_ledger::{NativeBatchLedger, Reservation},
    routing::{self, RoutingProblem},
    sc117_gdn_stage::{self, Proposal},
    sc117_layer_completion::StampedValues,
    sc117_post_attention, sc117_shared_stage,
    session::{NativeCollected, Session},
    shared_expert_cpu, Result,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Duration,
};

pub struct Routed {
    proposal: Proposal,
    stamp: Stamp,
    attempt: u64,
    ledger_id: (u64, u64),
    identity: Identity,
    route: sc117_post_attention::Routed,
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
/// One reserved local GDN layer batch. Both attention state and all residuals
/// remain provisional until the same reservation's complete collection.
pub struct RoutedBatch<'a> {
    gdn: sc117_gdn_stage::Batch<'a>,
    routes: Vec<sc117_post_attention::Routed>,
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
    pub attention: Vec<sc117_gdn_stage::Output>,
    pub residuals: Vec<Vec<f32>>,
}
impl<'a> RoutedBatch<'a> {
    pub fn choices(&self) -> impl Iterator<Item = &[crate::sc117_router_cpu::Choice]> {
        self.routes.iter().map(|route| route.choices.as_slice())
    }
    pub fn activations(&self) -> impl Iterator<Item = &[f32]> {
        self.routes.iter().map(|route| route.mixed.as_slice())
    }
    pub fn prepare(
        gdn: &'a mut sc117_gdn_stage::Stage,
        post: &mut sc117_post_attention::Stage,
        ledger: &NativeBatchLedger,
        ticket: &Reservation,
        residuals: &[Vec<f32>],
    ) -> Result<Self> {
        ledger.validate_ticket(ticket, ticket.stamp())?;
        if residuals.is_empty() || residuals.len() > 128 {
            return Err(invalid("GDN batch count"));
        }
        let stamp = ticket.stamp().clone();
        let identity = Identity {
            run_id: stamp.run_id.clone(),
            model_revision: stamp.model_revision.clone(),
            layer: stamp.layer,
            epoch: stamp.epoch,
        };
        let mut batch = gdn.begin_batch(residuals.len())?;
        let mut routes = Vec::with_capacity(residuals.len());
        for residual in residuals {
            let output = batch.prepare_next(&identity, residual)?;
            routes.push(post.compute(&identity, output, residual)?);
        }
        Ok(Self {
            gdn: batch,
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
            return Err(invalid("GDN batch attempt mismatch"));
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
            return Err(invalid("GDN batch producer identity or size"));
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
        let attention = self.gdn.commit()?;
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
    /// Consumes a reserved proposal; the residual must be exactly the bytes used to prepare it.
    pub fn prepare(
        post: &mut sc117_post_attention::Stage,
        ledger: &NativeBatchLedger,
        ticket: &Reservation,
        proposal: Proposal,
        residual: &[f32],
    ) -> Result<Self> {
        proposal.check_reservation(ledger, ticket, residual)?;
        let stamp = ticket.stamp().clone();
        let attempt = ticket.attempt();
        let identity = Identity {
            run_id: stamp.run_id.clone(),
            model_revision: stamp.model_revision.clone(),
            layer: stamp.layer,
            epoch: stamp.epoch,
        };
        let route = post.compute(&identity, proposal.output(), residual)?;
        Ok(Self {
            proposal,
            stamp,
            attempt,
            ledger_id: ticket.directory_id(),
            identity,
            route,
            marker: Arc::new(()),
        })
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
    pub fn choices(&self) -> &[crate::sc117_router_cpu::Choice] {
        &self.route.choices
    }
    pub fn shared(&self, stage: &mut sc117_shared_stage::Stage) -> Result<Shared> {
        let value = stage.compute(&self.identity, &self.route.mixed)?;
        Ok(Shared {
            value,
            marker: self.marker.clone(),
        })
    }
    /// The route and activation are taken exclusively from this object; caller
    /// chooses the verified placement and must have loaded matching owners.
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
            return Err(invalid("bound attempt mismatch"));
        }
        // A rank worker keeps its entire stack resident across requests. The
        // registry must describe that stack, including experts not selected by
        // this token, so physical indices remain stable when routes change.
        let experts: Vec<_> = registry
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
        gdn: &mut sc117_gdn_stage::Stage,
        shared: Shared,
        collected: Collected,
    ) -> Result<sc117_gdn_stage::Committed> {
        if !Arc::ptr_eq(&self.marker, &shared.marker)
            || !Arc::ptr_eq(&self.marker, &collected.marker)
        {
            return Err(invalid("bound producer mismatch"));
        }
        let wrap = |values| StampedValues {
            stamp: self.stamp.clone(),
            attempt: self.attempt,
            values,
        };
        gdn.commit_native(
            self.proposal,
            collected.value,
            wrap(shared.value),
            wrap(self.route.residual),
            wrap(self.route.inject),
        )
    }
}
