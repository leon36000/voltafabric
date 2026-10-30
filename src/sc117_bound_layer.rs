//! Opaque CPU transaction tying the GDN proposal, post-attention route, shared
//! expert and Rust collector to one local durable reservation. Not an R53 lease.
use crate::{
    gdn_cpu::Identity,
    invalid,
    moe::{Choice, Registry, Stamp},
    native_ledger::{NativeBatchLedger, Reservation},
    routing::{self, RoutingProblem},
    sc117_gdn_stage::{self, Proposal},
    sc117_layer_completion::StampedValues,
    sc117_post_attention, sc117_shared_stage,
    session::{NativeCollected, Session},
    Result,
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

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
        let experts: Vec<_> = self.route.choices.iter().map(|c| c.expert).collect();
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
