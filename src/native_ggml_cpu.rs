//! Experimental native GGML CPU owner, separate from MXFP4/QPN2.
//! Bindings are caller-provided evidence, not a distributed lease.
use crate::{
    invalid, sha256,
    strata_native_layout::{NativeLayout, VerifiedNativeExpert},
    Result,
};

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub model_revision: String,
    pub rank: u32,
    pub layer: u32,
    pub expert: u32,
    pub local: u32,
    pub epoch: u64,
    pub index_sha256: String,
    pub blob_sha256: String,
}

pub struct Resident {
    binding: Binding,
    owner: vf_ggml_native_ffi::NativeResident,
}
impl Resident {
    /// Recheck the complete local binding and payload hash before native load.
    /// Callers must obtain the expected digests from a trusted model manifest.
    pub fn load(
        layout: &NativeLayout,
        expert: VerifiedNativeExpert,
        binding: Binding,
    ) -> Result<Self> {
        let expected = layout.expert(binding.layer as usize, u64::from(binding.expert))?;
        if binding.model_revision.trim().is_empty()
            || binding.model_revision.len() > 256
            || expert.layer != binding.layer as usize
            || expert.expert != u64::from(binding.expert)
            || expert.span != expected
            || binding.index_sha256 != layout.index_sha256()
            || expert.bytes.len() as u64 != expected.bytes
            || sha256::hex(&expert.bytes) != binding.blob_sha256
        {
            return Err(invalid("native expert binding or payload mismatch"));
        }
        let owner = vf_ggml_native_ffi::NativeResident::load(
            expected.gate_up_type as i32,
            expected.down_type as i32,
            &expert.bytes,
        )
        .map_err(invalid)?;
        Ok(Self { binding, owner })
    }
    pub fn binding(&self) -> &Binding {
        &self.binding
    }
    /// Returns one unweighted FP32 expert row; no gate is applied here.
    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        self.owner.forward(input).map_err(invalid)
    }
}

/// Rank-local CPU owner boundary. A transport must supply trusted bindings and
/// provide its own durable UNKNOWN/lease handling; this guard is process-local.
pub struct RankWorker {
    owners: Vec<Resident>,
    attempted: std::collections::BTreeSet<String>,
    scope: Option<(String, String)>,
    last_attempt: u64,
    terminal: bool,
}

/// Unweighted finite FP32 rows; this is not the preweighted FP32 reduction contract.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnweightedF32Output {
    pub stamp: crate::moe::Stamp,
    pub rank: u32,
    pub attempt: u64,
    pub slots: Vec<crate::completion::OutputSlot>,
    pub rows: Vec<Vec<f32>>,
}

impl RankWorker {
    pub fn new(owners: Vec<Resident>) -> Result<Self> {
        use std::collections::BTreeSet;
        if owners.is_empty() || owners.len() > 1024 {
            return Err(invalid("native rank owner count"));
        }
        let base = owners[0].binding();
        let mut locals = BTreeSet::new();
        let mut experts = BTreeSet::new();
        for owner in &owners {
            let b = owner.binding();
            if b.rank != base.rank
                || b.model_revision != base.model_revision
                || b.layer != base.layer
                || b.epoch != base.epoch
                || b.index_sha256 != base.index_sha256
                || !locals.insert(b.local)
                || !experts.insert(b.expert)
            {
                return Err(invalid("native rank binding mismatch or duplicate"));
            }
        }
        Ok(Self {
            owners,
            attempted: BTreeSet::new(),
            scope: None,
            last_attempt: 0,
            terminal: false,
        })
    }

    /// Returns compact, unweighted FP16 rows only after all rank slots succeed.
    /// The coordinator must validate and collect them against its routing plan.
    pub fn execute(
        &mut self,
        dispatch: &crate::routing::RankDispatch,
        inputs: &[(usize, Vec<f32>)],
        attempt: u64,
    ) -> Result<crate::completion::RoutedRawOutput> {
        use crate::completion::{self, RoutedRawOutput};
        let result = self.execute_f32(dispatch, inputs, attempt)?;
        let converted = result
            .rows
            .iter()
            .flatten()
            .map(|&v| completion::encode_fp16(f64::from(v)))
            .collect::<Result<Vec<_>>>();
        match converted {
            Ok(bits) => Ok(RoutedRawOutput {
                stamp: result.stamp,
                rank: result.rank,
                attempt,
                rows: result.slots.len(),
                columns: 2560,
                slots: result.slots,
                fp16_bits: bits,
            }),
            Err(error) => {
                self.terminal = true;
                Err(error)
            }
        }
    }

    /// Stage all unweighted FP32 outputs atomically for a slot-oriented consumer.
    pub fn execute_f32(
        &mut self,
        dispatch: &crate::routing::RankDispatch,
        inputs: &[(usize, Vec<f32>)],
        attempt: u64,
    ) -> Result<UnweightedF32Output> {
        use crate::completion::OutputSlot;
        use std::collections::{BTreeMap, BTreeSet};
        if self.terminal
            || attempt <= self.last_attempt
            || dispatch.hidden != 2560
            || dispatch.tokens == 0
            || dispatch.tokens > 4096
            || dispatch.slots.is_empty()
            || (dispatch.slots.len() as u64)
                .checked_mul(3 * 2560 * 640)
                .is_none_or(|n| n > 100_000_000)
        {
            return Err(invalid("native rank state, dimensions or work budget"));
        }
        let b = self.owners[0].binding();
        let stamp = &dispatch.stamp;
        if dispatch.rank != b.rank
            || stamp.model_revision != b.model_revision
            || stamp.layer != b.layer
            || stamp.epoch != b.epoch
            || [&stamp.run_id, &stamp.plan_id]
                .iter()
                .any(|x| x.trim().is_empty() || x.len() > 256)
            || self
                .scope
                .as_ref()
                .is_some_and(|s| s.0 != stamp.run_id || s.1 != stamp.plan_id)
        {
            return Err(invalid("native rank execution identity differs"));
        }
        let order: Vec<_> = self.owners.iter().map(|o| o.binding().local).collect();
        if dispatch.resident_order != order {
            return Err(invalid("native rank resident order differs"));
        }
        let mut tokens = BTreeMap::new();
        for (token, row) in inputs {
            if *token >= dispatch.tokens
                || row.len() != 2560
                || row.iter().any(|x| !x.is_finite())
                || tokens.insert(*token, row).is_some()
            {
                return Err(invalid("invalid native rank token input"));
            }
        }
        let mut used = BTreeSet::new();
        let mut seen = BTreeSet::new();
        let mut plan = Vec::with_capacity(dispatch.slots.len());
        for entry in &dispatch.slots {
            let s = &entry.slot;
            let owner = self
                .owners
                .get(entry.physical_expert)
                .ok_or_else(|| invalid("native physical expert missing"))?;
            if s.rank != dispatch.rank
                || s.token >= dispatch.tokens
                || s.slot >= 64
                || !s.weight.is_finite()
                || s.weight < 0.0
                || owner.binding().expert != s.expert
                || owner.binding().local != s.local
                || !seen.insert((s.token, s.slot))
                || !tokens.contains_key(&s.token)
            {
                return Err(invalid("invalid native rank slot"));
            }
            used.insert(s.token);
            plan.push(entry.physical_expert);
        }
        if used.len() != tokens.len() {
            return Err(invalid("unused native token input"));
        }
        let key = serde_json::to_string(stamp).map_err(|_| invalid("native identity encoding"))?;
        if self.attempted.len() >= 4096 || self.attempted.contains(&key) {
            return Err(invalid(
                "native identity already attempted or limit reached",
            ));
        }
        self.attempted.insert(key);
        self.last_attempt = attempt;
        if self.scope.is_none() {
            self.scope = Some((stamp.run_id.clone(), stamp.plan_id.clone()));
        }
        let work = (|| -> Result<UnweightedF32Output> {
            let mut rows = Vec::with_capacity(dispatch.slots.len());
            let mut slots = Vec::with_capacity(dispatch.slots.len());
            for (entry, &owner) in dispatch.slots.iter().zip(&plan) {
                let s = &entry.slot;
                rows.push(self.owners[owner].forward(tokens[&s.token])?);
                slots.push(OutputSlot {
                    token: s.token,
                    slot: s.slot,
                    expert: s.expert,
                    local: s.local,
                });
            }
            Ok(UnweightedF32Output {
                stamp: stamp.clone(),
                rank: dispatch.rank,
                attempt,
                slots,
                rows,
            })
        })();
        if work.is_err() {
            self.terminal = true;
        }
        work
    }
}

/// In-process experimental fleet. This is not a Session transport or durable
/// lease; attempts are remembered only for this fleet's lifetime.
pub struct Fleet {
    owners: Vec<Resident>,
    attempted: std::collections::BTreeSet<String>,
    scope: Option<(String, String)>,
    last_attempt: u64,
    terminal: bool,
}
#[derive(Debug)]
pub struct BatchReceipt {
    pub stamp: crate::moe::Stamp,
    pub attempt: u64,
    pub completed_slots: usize,
    pub output: Vec<Vec<f64>>,
}
impl Fleet {
    pub fn new(owners: Vec<Resident>) -> Result<Self> {
        use std::collections::BTreeSet;
        if owners.is_empty() || owners.len() > 1024 {
            return Err(invalid("native fleet owner count"));
        }
        let mut pairs = BTreeSet::new();
        let mut locals = BTreeSet::new();
        let base = owners[0].binding();
        for owner in &owners {
            let b = owner.binding();
            if b.model_revision != base.model_revision
                || b.layer != base.layer
                || b.epoch != base.epoch
                || b.index_sha256 != base.index_sha256
                || !pairs.insert((b.rank, b.expert))
                || !locals.insert((b.rank, b.local))
            {
                return Err(invalid("native fleet binding mismatch or duplicate owner"));
            }
        }
        Ok(Self {
            owners,
            attempted: BTreeSet::new(),
            scope: None,
            last_attempt: 0,
            terminal: false,
        })
    }

    /// Evaluate all routed slots, round each unweighted output to FP16 and use
    /// the ordinary Rust collector's ordered FP64 weighted reduction. No result
    /// is returned unless every slot succeeds; work is never replayed here.
    pub fn run_fp16_collected(
        &mut self,
        p: &crate::routing::RoutingProblem,
        inputs: &[Vec<f32>],
        attempt: u64,
    ) -> Result<BatchReceipt> {
        use crate::{completion, moe, routing};
        if self.terminal {
            return Err(invalid("native fleet terminal"));
        }
        if attempt <= self.last_attempt
            || p.hidden != 2560
            || inputs.len() != p.routes.len()
            || inputs
                .iter()
                .any(|row| row.len() != 2560 || row.iter().any(|x| !x.is_finite()))
        {
            return Err(invalid("invalid native batch input or attempt"));
        }
        let dispatch = routing::dispatch(p)?;
        if dispatch.slots.is_empty() {
            return Err(invalid("empty native batch"));
        }
        let base = self.owners[0].binding();
        if p.stamp.model_revision != base.model_revision
            || p.stamp.layer != base.layer
            || p.stamp.epoch != base.epoch
        {
            return Err(invalid("native fleet execution identity differs"));
        }
        if self
            .scope
            .as_ref()
            .is_some_and(|s| s.0 != p.stamp.run_id || s.1 != p.stamp.plan_id)
        {
            return Err(invalid("native fleet run or plan differs"));
        }
        let mut by_rank = std::collections::BTreeMap::<u32, usize>::new();
        let mut plan = Vec::with_capacity(dispatch.slots.len());
        for slot in &dispatch.slots {
            let owner = self
                .owners
                .iter()
                .position(|r| {
                    let b = r.binding();
                    b.rank == slot.rank && b.expert == slot.expert && b.local == slot.local
                })
                .ok_or_else(|| invalid("missing native resident owner"))?;
            let n = by_rank.entry(slot.rank).or_default();
            *n += 1;
            if (*n as u64)
                .checked_mul(3 * 2560 * 640)
                .is_none_or(|n| n > 100_000_000)
            {
                return Err(invalid("native rank CPU work budget exceeded"));
            }
            plan.push(owner);
        }
        let key =
            serde_json::to_string(&p.stamp).map_err(|_| invalid("native identity encoding"))?;
        if self.attempted.len() >= 4096 || self.attempted.contains(&key) {
            return Err(invalid(
                "native identity already attempted or limit reached",
            ));
        }
        let mut collector = moe::Collector::from_routing(p)?;
        self.attempted.insert(key);
        self.last_attempt = attempt;
        if self.scope.is_none() {
            self.scope = Some((p.stamp.run_id.clone(), p.stamp.plan_id.clone()));
        }
        let work = (|| -> Result<Vec<Vec<f64>>> {
            for (slot, &owner) in dispatch.slots.iter().zip(&plan) {
                let output = self.owners[owner].forward(&inputs[slot.token])?;
                let values = output
                    .into_iter()
                    .map(|x| {
                        completion::encode_fp16(f64::from(x)).and_then(completion::decode_fp16)
                    })
                    .collect::<Result<Vec<_>>>()?;
                collector.accept(moe::Completion {
                    stamp: p.stamp.clone(),
                    token: slot.token,
                    slot: slot.slot,
                    expert: slot.expert,
                    rank: slot.rank,
                    local: slot.local,
                    value: values,
                })?;
            }
            collector.finish()
        })();
        match work {
            Ok(output) => Ok(BatchReceipt {
                stamp: p.stamp.clone(),
                attempt,
                completed_slots: dispatch.slots.len(),
                output,
            }),
            Err(e) => {
                self.terminal = true;
                Err(e)
            }
        }
    }
}
