//! Metadata-only routing boundary; no dense expert weights are needed.
use crate::{invalid, moe, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingProblem {
    pub hidden: usize,
    pub experts: Vec<u32>,
    pub routes: Vec<Vec<moe::Choice>>,
    pub registry: moe::Registry,
    pub stamp: moe::Stamp,
    #[serde(default)]
    pub preferred_owners: BTreeMap<u32, u32>,
    pub resident_order: BTreeMap<u32, Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RankSlot {
    pub slot: moe::Slot,
    pub physical_expert: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RankDispatch {
    pub stamp: moe::Stamp,
    pub rank: u32,
    pub hidden: usize,
    pub tokens: usize,
    pub resident_order: Vec<u32>,
    pub slots: Vec<RankSlot>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Dispatch {
    pub stamp: moe::Stamp,
    pub hidden: usize,
    pub tokens: usize,
    pub slots: Vec<moe::Slot>,
    pub ranks: Vec<RankDispatch>,
}

/// Validate all metadata before constructing dispatches or CPU collection buffers.
pub fn validate(p: &RoutingProblem) -> Result<()> {
    let tokens = p.routes.len();
    let k = p.routes.first().map_or(0, Vec::len);
    if p.hidden == 0 || p.hidden > 65536 || p.experts.is_empty() || p.experts.len() > 1024 {
        return Err(invalid("routing dimensions outside bounded contract"));
    }
    if tokens > 4096 || (tokens != 0 && (k == 0 || k > 64)) {
        return Err(invalid("batch length or top-k outside routing contract"));
    }
    if tokens.checked_mul(k).is_none_or(|n| n > 262144)
        || tokens.checked_mul(p.hidden).is_none_or(|n| n > 16_777_216)
    {
        return Err(invalid(
            "routing slots or CPU collection buffer limit exceeded",
        ));
    }
    let experts: BTreeSet<_> = p.experts.iter().copied().collect();
    if experts.len() != p.experts.len() {
        return Err(invalid("duplicate global expert"));
    }
    for route in &p.routes {
        if route.len() != k {
            return Err(invalid("inconsistent top-k"));
        }
        let mut used = BTreeSet::new();
        for choice in route {
            if !experts.contains(&choice.expert)
                || !used.insert(choice.expert)
                || !choice.weight.is_finite()
                || choice.weight < 0.0
            {
                return Err(invalid(
                    "unknown/duplicate expert or invalid routing weight",
                ));
            }
        }
    }
    if p.registry.epoch != p.stamp.epoch
        || [&p.stamp.run_id, &p.stamp.model_revision, &p.stamp.plan_id]
            .iter()
            .any(|s| s.trim().is_empty() || s.len() > 256)
    {
        return Err(invalid(
            "epoch mismatch or empty/oversized execution identity",
        ));
    }
    if p.registry.copies.len() > 65536 || p.resident_order.len() > 64 {
        return Err(invalid("replica or worker limit exceeded"));
    }
    let mut locals: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    let mut pairs = BTreeSet::new();
    let mut covered = BTreeSet::new();
    for replica in &p.registry.copies {
        if !experts.contains(&replica.expert)
            || !locals
                .entry(replica.rank)
                .or_default()
                .insert(replica.local)
            || !pairs.insert((replica.expert, replica.rank))
        {
            return Err(invalid("invalid replica or rank/local collision"));
        }
        covered.insert(replica.expert);
    }
    if covered != experts || locals.len() > 64 {
        return Err(invalid("unowned expert or worker limit exceeded"));
    }
    for (&expert, &rank) in &p.preferred_owners {
        if !pairs.contains(&(expert, rank)) {
            return Err(invalid("preferred owner has no resident replica"));
        }
    }
    if locals
        .keys()
        .any(|rank| !p.resident_order.contains_key(rank))
    {
        return Err(invalid("resident rank missing from physical order"));
    }
    for (&rank, order) in &p.resident_order {
        let expected = locals.get(&rank);
        if order.len() != expected.map_or(0, BTreeSet::len) {
            return Err(invalid("physical order does not cover resident local IDs"));
        }
        let actual: BTreeSet<_> = order.iter().copied().collect();
        if actual.len() != order.len() || expected.is_some_and(|ids| ids != &actual) {
            return Err(invalid("duplicate or foreign local ID in physical order"));
        }
    }
    Ok(())
}

/// Select one resident owner per logical expert, preserving token/slot and weight.
/// Rank-local logical IDs remain separate from physical stack indices.
pub fn dispatch(p: &RoutingProblem) -> Result<Dispatch> {
    validate(p)?;
    let mut selected: BTreeMap<u32, &moe::Replica> = BTreeMap::new();
    for replica in &p.registry.copies {
        if p.preferred_owners
            .get(&replica.expert)
            .is_some_and(|&rank| rank != replica.rank)
        {
            continue;
        }
        if selected
            .get(&replica.expert)
            .is_none_or(|old| replica.rank < old.rank)
        {
            selected.insert(replica.expert, replica);
        }
    }
    let mut ranks: BTreeMap<_, _> = p
        .resident_order
        .iter()
        .map(|(&rank, order)| {
            (
                rank,
                RankDispatch {
                    stamp: p.stamp.clone(),
                    rank,
                    hidden: p.hidden,
                    tokens: p.routes.len(),
                    resident_order: order.clone(),
                    slots: Vec::new(),
                },
            )
        })
        .collect();
    let physical: BTreeMap<_, _> = p
        .resident_order
        .iter()
        .flat_map(|(&rank, order)| {
            order
                .iter()
                .enumerate()
                .map(move |(index, &local)| ((rank, local), index))
        })
        .collect();
    let mut slots = Vec::new();
    for (token, route) in p.routes.iter().enumerate() {
        for (slot, choice) in route.iter().enumerate() {
            let replica = selected
                .get(&choice.expert)
                .ok_or_else(|| invalid("missing selected resident replica"))?;
            let routed = moe::Slot {
                token,
                slot,
                expert: choice.expert,
                rank: replica.rank,
                local: replica.local,
                weight: choice.weight,
            };
            let physical_expert = *physical
                .get(&(replica.rank, replica.local))
                .ok_or_else(|| invalid("selected local ID missing from physical order"))?;
            ranks
                .get_mut(&replica.rank)
                .ok_or_else(|| invalid("selected rank missing from dispatch"))?
                .slots
                .push(RankSlot {
                    slot: routed.clone(),
                    physical_expert,
                });
            slots.push(routed);
        }
    }
    Ok(Dispatch {
        stamp: p.stamp.clone(),
        hidden: p.hidden,
        tokens: p.routes.len(),
        slots,
        ranks: ranks.into_values().collect(),
    })
}

/// Oracle adapter only. Copies routes and metadata, never model/input matrices.
/// Dense fixtures have no physical-order declaration: use ascending declared local
/// IDs. Call dispatch/validate on the result; malformed metadata is not repaired.
pub fn from_dense(p: &moe::Problem) -> RoutingProblem {
    let mut resident_order: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for replica in &p.registry.copies {
        resident_order
            .entry(replica.rank)
            .or_default()
            .push(replica.local);
    }
    for order in resident_order.values_mut() {
        order.sort_unstable();
    }
    RoutingProblem {
        hidden: p.model.hidden,
        experts: p.model.experts.iter().map(|e| e.id).collect(),
        routes: p.batch.routes.clone(),
        registry: p.registry.clone(),
        stamp: p.stamp.clone(),
        preferred_owners: p.preferred_owners.clone(),
        resident_order,
    }
}
