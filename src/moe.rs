//! CPU SwiGLU oracle and versioned dispatch/collection contracts, not a GPU backend.
use crate::{invalid, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expert {
    pub id: u32,
    pub gate: Vec<f64>,
    pub up: Vec<f64>,
    pub down: Vec<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub hidden: usize,
    pub intermediate: usize,
    pub experts: Vec<Expert>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Choice {
    pub expert: u32,
    pub weight: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Batch {
    pub inputs: Vec<Vec<f64>>,
    pub routes: Vec<Vec<Choice>>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Stamp {
    pub run_id: String,
    pub model_revision: String,
    pub plan_id: String,
    pub layer: u32,
    pub batch: u64,
    pub epoch: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Replica {
    pub expert: u32,
    pub rank: u32,
    pub local: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Registry {
    pub epoch: u64,
    pub copies: Vec<Replica>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Problem {
    pub model: Model,
    pub batch: Batch,
    pub registry: Registry,
    pub stamp: Stamp,
    #[serde(default)]
    pub preferred_owners: BTreeMap<u32, u32>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Slot {
    pub token: usize,
    pub slot: usize,
    pub expert: u32,
    pub rank: u32,
    pub local: u32,
    pub weight: f64,
}
#[derive(Debug, Clone, Copy)]
pub(crate) struct Fp16SlotIdentity {
    pub token: usize,
    pub slot: usize,
    pub expert: u32,
    pub rank: u32,
    pub local: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Completion {
    pub stamp: Stamp,
    pub token: usize,
    pub slot: usize,
    pub expert: u32,
    pub rank: u32,
    pub local: u32,
    pub value: Vec<f64>,
}

/// Explicit CPU-probe bounds prevent accidental full-model evaluation on the host.
pub fn validate(p: &Problem) -> Result<()> {
    let m = &p.model;
    let n = m
        .hidden
        .checked_mul(m.intermediate)
        .ok_or_else(|| invalid("matrix size overflow"))?;
    if m.hidden == 0
        || m.intermediate == 0
        || n > 1_048_576
        || m.experts.is_empty()
        || m.experts.len() > 1024
    {
        return Err(invalid(
            "CPU model dimensions outside bounded probe contract",
        ));
    }
    let weight_count = n
        .checked_mul(m.experts.len())
        .and_then(|x| x.checked_mul(3))
        .ok_or_else(|| invalid("weight count overflow"))?;
    if weight_count > 16_777_216 {
        return Err(invalid("CPU probe weight limit"));
    }
    let mut experts = BTreeSet::new();
    for e in &m.experts {
        if !experts.insert(e.id) || e.gate.len() != n || e.up.len() != n || e.down.len() != n {
            return Err(invalid("duplicate global expert or invalid matrix shape"));
        }
        if e.gate
            .iter()
            .chain(&e.up)
            .chain(&e.down)
            .any(|x| !x.is_finite())
        {
            return Err(invalid("nonfinite model weights"));
        }
    }
    let k = p.batch.routes.first().map_or(0, Vec::len);
    if p.batch.inputs.len() != p.batch.routes.len()
        || p.batch.inputs.len() > 4096
        || (!p.batch.inputs.is_empty() && (k == 0 || k > 64))
    {
        return Err(invalid("batch lengths or top-k outside CPU contract"));
    }
    let work = n
        .checked_mul(p.batch.inputs.len())
        .and_then(|x| x.checked_mul(k))
        .and_then(|x| x.checked_mul(3))
        .ok_or_else(|| invalid("CPU work overflow"))?;
    if work > 100_000_000 {
        return Err(invalid("CPU work budget exceeded"));
    }
    for (x, route) in p.batch.inputs.iter().zip(&p.batch.routes) {
        if x.len() != m.hidden || x.iter().any(|v| !v.is_finite()) || route.len() != k {
            return Err(invalid("input width/finite value or inconsistent top-k"));
        }
        let mut used = BTreeSet::new();
        for c in route {
            if !experts.contains(&c.expert)
                || !used.insert(c.expert)
                || !c.weight.is_finite()
                || c.weight < 0.0
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
    if p.registry.copies.len() > 65536 {
        return Err(invalid("too many replicas"));
    }
    let mut local = BTreeSet::new();
    let mut pairs = BTreeSet::new();
    let mut covered = BTreeSet::new();
    let mut ranks = BTreeSet::new();
    for r in &p.registry.copies {
        if !experts.contains(&r.expert)
            || !local.insert((r.rank, r.local))
            || !pairs.insert((r.expert, r.rank))
        {
            return Err(invalid("invalid replica or rank/local collision"));
        }
        covered.insert(r.expert);
        ranks.insert(r.rank);
    }
    if covered != experts || ranks.len() > 64 {
        return Err(invalid("unowned expert or CPU worker limit exceeded"));
    }
    for (&e, &rank) in &p.preferred_owners {
        if !pairs.contains(&(e, rank)) {
            return Err(invalid("preferred owner has no resident replica"));
        }
    }
    Ok(())
}

pub fn dispatch(p: &Problem) -> Result<Vec<Slot>> {
    validate(p)?;
    let mut selected: BTreeMap<u32, &Replica> = BTreeMap::new();
    for r in &p.registry.copies {
        if p.preferred_owners
            .get(&r.expert)
            .is_some_and(|&rank| rank != r.rank)
        {
            continue;
        }
        if selected.get(&r.expert).is_none_or(|old| r.rank < old.rank) {
            selected.insert(r.expert, r);
        }
    }
    let mut slots = Vec::new();
    for (token, route) in p.batch.routes.iter().enumerate() {
        for (slot, c) in route.iter().enumerate() {
            let r = selected
                .get(&c.expert)
                .ok_or_else(|| invalid("missing selected resident copy"))?;
            slots.push(Slot {
                token,
                slot,
                expert: c.expert,
                rank: r.rank,
                local: r.local,
                weight: c.weight,
            });
        }
    }
    Ok(slots)
}

pub(crate) fn finite(x: f64) -> Result<f64> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(Error::new(
            "NUMERICAL_OVERFLOW",
            "CPU intermediate is nonfinite",
        ))
    }
}
/// Decode one finite IEEE binary16 value exactly to f64.
pub fn decode_fp16(bits: u16) -> Result<f64> {
    let exponent = (bits >> 10) & 31;
    let fraction = bits & 1023;
    if exponent == 31 {
        return Err(Error::new(
            "NON_FINITE_RESULT",
            "FP16 output contains NaN or infinity",
        ));
    }
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let value = if exponent == 0 {
        f64::from(fraction) * 2.0f64.powi(-24)
    } else {
        f64::from(1024 + fraction) * 2.0f64.powi(i32::from(exponent) - 25)
    };
    Ok(sign * value)
}
pub(crate) fn silu(x: f64) -> f64 {
    if x >= 0.0 {
        x / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        x * z / (1.0 + z)
    }
}
/// One expert's unweighted SwiGLU output; only `m.hidden` and `m.intermediate` are read
/// from `m`. Gate and up accumulate over the hidden index, down over the intermediate one.
pub fn expert_value(m: &Model, e: &Expert, x: &[f64]) -> Result<Vec<f64>> {
    let mut activated = vec![0.0; m.intermediate];
    for (j, a) in activated.iter_mut().enumerate() {
        let mut g = 0.0;
        let mut u = 0.0;
        for (i, &v) in x.iter().enumerate() {
            g = finite(g + v * e.gate[i * m.intermediate + j])?;
            u = finite(u + v * e.up[i * m.intermediate + j])?;
        }
        *a = finite(silu(g) * u)?;
    }
    let mut out = vec![0.0; m.hidden];
    for (i, y) in out.iter_mut().enumerate() {
        for (j, &a) in activated.iter().enumerate() {
            *y = finite(*y + a * e.down[j * m.hidden + i])?;
        }
    }
    Ok(out)
}

/// Sequential f64 reference; independent external oracle is in scripts/crosscheck.py.
pub fn reference(p: &Problem) -> Result<Vec<Vec<f64>>> {
    validate(p)?;
    let mut out = vec![vec![0.0; p.model.hidden]; p.batch.inputs.len()];
    for (t, (x, route)) in p.batch.inputs.iter().zip(&p.batch.routes).enumerate() {
        for c in route {
            let e = p
                .model
                .experts
                .iter()
                .find(|e| e.id == c.expert)
                .ok_or_else(|| invalid("expert missing"))?;
            let value = expert_value(&p.model, e, x)?;
            for (y, v) in out[t].iter_mut().zip(value) {
                *y = finite(*y + c.weight * v)?;
            }
        }
    }
    Ok(out)
}
fn compute_validated(p: &Problem, s: &Slot) -> Result<Completion> {
    let e = p
        .model
        .experts
        .iter()
        .find(|e| e.id == s.expert)
        .ok_or_else(|| invalid("unknown expert"))?;
    Ok(Completion {
        stamp: p.stamp.clone(),
        token: s.token,
        slot: s.slot,
        expert: s.expert,
        rank: s.rank,
        local: s.local,
        value: expert_value(&p.model, e, &p.batch.inputs[s.token])?,
    })
}
pub fn compute_slot(p: &Problem, s: &Slot) -> Result<Completion> {
    if !dispatch(p)?.iter().any(|expected| expected == s) {
        return Err(invalid("slot does not belong to the current dispatch"));
    }
    compute_validated(p, s)
}
/// Shared-memory CPU worker per participating rank. Not an inter-process/GPU transport.
pub fn parallel(p: &Problem) -> Result<Vec<Vec<f64>>> {
    let slots = dispatch(p)?;
    let mut by_rank: BTreeMap<u32, Vec<Slot>> = BTreeMap::new();
    for s in slots {
        by_rank.entry(s.rank).or_default().push(s);
    }
    let mut collector = Collector::new(p)?;
    let all = std::thread::scope(|scope| {
        let handles: Vec<_> = by_rank
            .into_values()
            .map(|ss| {
                scope.spawn(move || {
                    ss.iter()
                        .map(|s| compute_validated(p, s))
                        .collect::<Result<Vec<_>>>()
                })
            })
            .collect();
        let mut results = Vec::new();
        for h in handles {
            results.extend(
                h.join()
                    .map_err(|_| Error::new("WORKER_PANIC", "CPU worker panicked"))??,
            );
        }
        Ok::<_, Error>(results)
    })?;
    for c in all.into_iter().rev() {
        collector.accept(c)?;
    }
    collector.finish()
}

/// One batch collector with a terminal failure state. No partial output and no silent retry.
/// This is a receipt contract, not a claim of exactly-once network execution.
pub struct Collector {
    stamp: Stamp,
    expected: BTreeMap<(usize, usize), Slot>,
    received: BTreeMap<(usize, usize), ReceivedValues>,
    fp16_batches: Vec<Vec<u16>>,
    tokens: usize,
    hidden: usize,
    failed: bool,
}
enum ReceivedValues {
    F64(Vec<f64>),
    Fp16 { batch: usize, offset: usize },
    PreweightedFp32(Vec<u32>),
}
impl Collector {
    pub fn new(p: &Problem) -> Result<Self> {
        let slots = dispatch(p)?;
        Ok(Self::from_slots(
            &p.stamp,
            p.batch.inputs.len(),
            p.model.hidden,
            slots,
        ))
    }
    /// Create the same strict collector from routing metadata alone.
    /// Incoming Completion.value is the unweighted expert output. This collector
    /// applies the original coefficient once, in original token/slot order.
    /// The bounded CPU receipt oracle stores f64 values, not resident model weights.
    pub fn from_routing(p: &crate::routing::RoutingProblem) -> Result<Self> {
        crate::routing::validate(p)?;
        let topk = p.routes.first().map_or(0, Vec::len);
        let values = p
            .routes
            .len()
            .checked_mul(topk)
            .and_then(|n| n.checked_mul(p.hidden))
            .ok_or_else(|| Error::new("COLLECTION_TOO_LARGE", "pending result count overflow"))?;
        if values > 16_777_216 {
            return Err(Error::new(
                "COLLECTION_TOO_LARGE",
                "pending slot results exceed 16777216 f64 values",
            ));
        }
        // Validation and the memory bound precede dispatch construction.
        let d = crate::routing::dispatch(p)?;
        Ok(Self::from_slots(&d.stamp, d.tokens, d.hidden, d.slots))
    }
    fn from_slots(stamp: &Stamp, tokens: usize, hidden: usize, slots: Vec<Slot>) -> Self {
        Self {
            stamp: stamp.clone(),
            expected: slots.into_iter().map(|s| ((s.token, s.slot), s)).collect(),
            received: BTreeMap::new(),
            fp16_batches: Vec::new(),
            tokens,
            hidden,
            failed: false,
        }
    }
    fn expected_slot(
        &self,
        key: (usize, usize),
        expert: u32,
        rank: u32,
        local: u32,
    ) -> Option<&Slot> {
        self.expected
            .get(&key)
            .filter(|s| s.expert == expert && s.rank == rank && s.local == local)
    }
    pub fn accept(&mut self, c: Completion) -> Result<()> {
        if self.failed {
            return Err(Error::new("BATCH_ABORTED", "collector is terminal"));
        }
        let key = (c.token, c.slot);
        let valid = c.stamp == self.stamp
            && c.value.len() == self.hidden
            && c.value.iter().all(|x| x.is_finite())
            && !self.received.contains_key(&key)
            && self.expected_slot(key, c.expert, c.rank, c.local).is_some();
        if !valid {
            self.failed = true;
            return Err(Error::new(
                "INVALID_COMPLETION",
                "duplicate, stale, unexpected or malformed result; entire batch aborted",
            ));
        }
        self.received.insert(key, ReceivedValues::F64(c.value));
        Ok(())
    }
    /// Store one unweighted row-major FP16 batch after completion::accept_output has
    /// validated its complete shape and every value. The bits are moved into bounded
    /// resident storage and decoded only during `finish`, avoiding per-slot f64 vectors.
    /// This validates collector identity but cannot attest who produced the bits, whether
    /// a GPU kernel ran, or whether its stream synced. Keep crate-private so callers use
    /// the representation boundary that performs numeric preflight before collector state.
    pub(crate) fn accept_fp16_batch(
        &mut self,
        stamp: &Stamp,
        rank: u32,
        slots: &[Slot],
        columns: usize,
        bits: Vec<u16>,
    ) -> Result<usize> {
        self.accept_fp16_batch_with(
            stamp,
            rank,
            slots.len(),
            |index| {
                let slot = &slots[index];
                Fp16SlotIdentity {
                    token: slot.token,
                    slot: slot.slot,
                    expert: slot.expert,
                    rank: slot.rank,
                    local: slot.local,
                }
            },
            columns,
            bits,
        )
    }

    /// Identity is borrowed by index so routed FP16 collection need not clone per-row
    /// metadata before the collector validates and records the compact payload.
    pub(crate) fn accept_fp16_batch_with(
        &mut self,
        stamp: &Stamp,
        rank: u32,
        slot_count: usize,
        slot_at: impl Fn(usize) -> Fp16SlotIdentity,
        columns: usize,
        bits: Vec<u16>,
    ) -> Result<usize> {
        if self.failed {
            return Err(Error::new("BATCH_ABORTED", "collector is terminal"));
        }
        if stamp != &self.stamp || (0..slot_count).any(|index| slot_at(index).rank != rank) {
            self.failed = true;
            return Err(Error::new(
                "STALE_EXECUTION",
                "FP16 completion batch differs from collector Stamp or rank",
            ));
        }
        let count = slot_count
            .checked_mul(columns)
            .filter(|&count| count <= 16_777_216);
        if columns != self.hidden || count != Some(bits.len()) {
            self.failed = true;
            return Err(Error::new(
                "INVALID_COMPLETION",
                "FP16 completion batch shape exceeds or differs from the collector",
            ));
        }

        let mut keys = BTreeSet::new();
        for index in 0..slot_count {
            let slot = slot_at(index);
            let key = (slot.token, slot.slot);
            let matches = self
                .expected_slot(key, slot.expert, slot.rank, slot.local)
                .is_some();
            if !matches || self.received.contains_key(&key) || !keys.insert(key) {
                self.failed = true;
                return Err(Error::new(
                    "INVALID_COMPLETION",
                    "duplicate, stale, unexpected or malformed FP16 slot; entire batch aborted",
                ));
            }
        }
        if slot_count == 0 {
            return Ok(0);
        }
        if let Err(error) = self.fp16_batches.try_reserve(1) {
            self.failed = true;
            return Err(Error::new(
                "ALLOCATION_FAILED",
                format!("cannot retain FP16 completion batch: {error}"),
            ));
        }
        let batch = self.fp16_batches.len();
        self.fp16_batches.push(bits);
        for row in 0..slot_count {
            let slot = slot_at(row);
            self.received.insert(
                (slot.token, slot.slot),
                ReceivedValues::Fp16 {
                    batch,
                    offset: row * columns,
                },
            );
        }
        Ok(slot_count)
    }
    pub fn finish(self) -> Result<Vec<Vec<f64>>> {
        if self.failed || self.received.len() != self.expected.len() {
            return Err(Error::new("BATCH_INCOMPLETE", "aborted or missing results"));
        }
        let mut out = vec![vec![0.0; self.hidden]; self.tokens];
        // BTreeMap restores original token/slot order irrespective of arrival order.
        for (key, s) in self.expected {
            let values = self
                .received
                .get(&key)
                .ok_or_else(|| Error::new("BATCH_INCOMPLETE", "missing slot"))?;
            match values {
                ReceivedValues::F64(values) => {
                    for (y, &v) in out[s.token].iter_mut().zip(values) {
                        *y = finite(*y + s.weight * v)?;
                    }
                }
                ReceivedValues::Fp16 { batch, offset } => {
                    let bits = &self.fp16_batches[*batch][*offset..*offset + self.hidden];
                    for (y, &value) in out[s.token].iter_mut().zip(bits) {
                        let value = decode_fp16(value)?;
                        *y = finite(*y + s.weight * value)?;
                    }
                }
                ReceivedValues::PreweightedFp32(_) => {
                    return Err(Error::new(
                        "ARITHMETIC_CONTRACT_MISMATCH",
                        "preweighted FP32 values cannot enter the unweighted collector finish",
                    ));
                }
            }
        }
        Ok(out)
    }
    // Only the explicit preweighted type can reach these methods. In particular
    // neither the old FP16 session nor an unweighted Completion can select them.
    pub(crate) fn accept_preweighted_fp32(
        &mut self,
        c: crate::preweighted::Completion,
    ) -> Result<()> {
        if self.failed {
            return Err(Error::new("BATCH_ABORTED", "collector is terminal"));
        }
        let key = (c.token, c.slot);
        let duplicate = self.received.contains_key(&key);
        let valid = self.stamp == c.stamp
            && c.value_bits.len() == self.hidden
            && c.value_bits.iter().all(|v| v & 0x7f80_0000 != 0x7f80_0000)
            && (c.weight_bits & 0x7fff_ffff != 0
                || c.value_bits.iter().all(|v| v & 0x7fff_ffff == 0))
            && !duplicate
            && self
                .expected_slot(key, c.expert, c.rank, c.local)
                .is_some_and(|s| (s.weight as f32).to_bits() == c.weight_bits);
        if !valid {
            self.failed = true;
            return Err(Error::new(
                "INVALID_COMPLETION",
                "preweighted identity, coefficient, finite values or shape differ; entire batch aborted",
            ));
        }
        self.received
            .insert(key, ReceivedValues::PreweightedFp32(c.value_bits));
        Ok(())
    }
    pub(crate) fn finish_preweighted_fp32(self) -> Result<Vec<Vec<f32>>> {
        if self.received.len() != self.expected.len() || self.failed {
            return Err(Error::new("BATCH_INCOMPLETE", "aborted or missing results"));
        }
        let mut out = vec![vec![0.0_f32; self.hidden]; self.tokens];
        // Keys are ordered by original (token, slot), never by arrival or rank.
        // Keep one explicit f32 addition per slot; coefficients were already
        // applied before R9's BF16/activation quantization/W2 and are not moved.
        for (key, slot) in self.expected {
            let Some(ReceivedValues::PreweightedFp32(bits)) = self.received.get(&key) else {
                return Err(Error::new(
                    "ARITHMETIC_CONTRACT_MISMATCH",
                    "expected preweighted FP32 rows",
                ));
            };
            for (sum, &bits) in out[slot.token].iter_mut().zip(bits) {
                *sum += f32::from_bits(bits);
                if !sum.is_finite() {
                    return Err(Error::new(
                        "NUMERICAL_OVERFLOW",
                        "FP32 combine overflow; no batch published",
                    ));
                }
            }
        }
        Ok(out)
    }
    pub fn expire(&mut self) {
        self.failed = true;
    }
}

#[cfg(test)]
mod fp16_collector_tests {
    use super::*;

    fn stamp() -> Stamp {
        Stamp {
            run_id: "run".into(),
            model_revision: "model".into(),
            plan_id: "plan".into(),
            layer: 0,
            batch: 1,
            epoch: 2,
        }
    }

    fn slot() -> Slot {
        Slot {
            token: 0,
            slot: 0,
            expert: 7,
            rank: 3,
            local: 9,
            weight: 0.5,
        }
    }

    #[test]
    fn fp16_batch_retains_quarter_of_f64_completion_payload() {
        let stamp = stamp();
        let slot = slot();
        let mut f64_collector = Collector::from_slots(&stamp, 1, 2, vec![slot.clone()]);
        f64_collector
            .accept(Completion {
                stamp: stamp.clone(),
                token: 0,
                slot: 0,
                expert: 7,
                rank: 3,
                local: 9,
                value: vec![1.0, 2.0],
            })
            .unwrap();
        let f64_bytes = match f64_collector.received.get(&(0, 0)).unwrap() {
            ReceivedValues::F64(values) => values.len() * std::mem::size_of::<f64>(),
            ReceivedValues::Fp16 { .. } | ReceivedValues::PreweightedFp32(_) => {
                panic!("expected f64 completion")
            }
        };

        let mut fp16_collector = Collector::from_slots(&stamp, 1, 2, vec![slot.clone()]);
        assert_eq!(
            fp16_collector
                .accept_fp16_batch(&stamp, 3, &[slot], 2, vec![0x3c00, 0x4000])
                .unwrap(),
            1
        );
        let fp16_bytes = fp16_collector.fp16_batches[0].len() * std::mem::size_of::<u16>();
        assert_eq!(f64_bytes, 16);
        assert_eq!(fp16_bytes, 4);
        assert_eq!(f64_bytes / fp16_bytes, 4);
        assert_eq!(
            f64_collector.finish().unwrap(),
            fp16_collector.finish().unwrap()
        );
    }
}
