//! Experimental native GGML CPU owner, separate from MXFP4/QPN2.
//! Bindings are caller-provided evidence, not a distributed lease.
use crate::{
    invalid, sha256,
    strata_native_layout::{NativeLayout, VerifiedNativeExpert},
    Result,
};

// Group computation by physical owner while retaining each original slot index.
// This does not reorder the externally returned rows or the Collector reduction.
fn expert_execution_order(owners: &[usize]) -> Vec<usize> {
    let mut indices: Vec<_> = (0..owners.len()).collect();
    indices.sort_unstable_by_key(|&i| (owners[i], i));
    indices
}

/// Per coordinator and per rank: bounds only resident payload bytes, not total RSS.
pub const MAX_NATIVE_EXPERTS: usize = 512;
pub const MAX_NATIVE_PAYLOAD_BYTES: u64 = 1_250_000_000;
pub const DEFAULT_NATIVE_PAYLOAD_BYTES: u64 = 16 * 1024 * 1024;

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

/// Diagnostic reference only: produces unweighted f64 values, never worker output.
/// The caller supplies an authenticated binding and is responsible for any stamp,
/// f32/FP16 conversion and Collector admission. No remote execution is qualified.
#[cfg(feature = "native-scalar-reference")]
pub fn scalar_reference(
    layout: &NativeLayout,
    expert: &VerifiedNativeExpert,
    binding: &Binding,
    input: &[f32],
) -> Result<Vec<f64>> {
    let span = layout.expert(binding.layer as usize, u64::from(binding.expert))?;
    if binding.model_revision.trim().is_empty()
        || binding.model_revision.len() > 256
        || binding.epoch == 0
        || expert.layer != binding.layer as usize
        || expert.expert != u64::from(binding.expert)
        || expert.span != span
        || binding.index_sha256 != layout.index_sha256()
        || expert.bytes.len() as u64 != span.bytes
        || sha256::hex(&expert.bytes) != binding.blob_sha256
    {
        return Err(invalid("scalar reference binding or payload mismatch"));
    }
    vf_ggml_native_ffi::scalar_reference(
        span.gate_up_type as i32,
        span.down_type as i32,
        &expert.bytes,
        input,
    )
    .map_err(invalid)
}

pub struct Resident {
    binding: Binding,
    owner: ResidentBackend,
    payload_bytes: u64,
}
enum ResidentBackend {
    Vector(vf_ggml_native_ffi::NativeResident),
    #[cfg(feature = "native-scalar-reference")]
    Scalar {
        gu: i32,
        down: i32,
        bytes: Vec<u8>,
    },
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
        Ok(Self {
            binding,
            owner: ResidentBackend::Vector(owner),
            payload_bytes: expected.bytes,
        })
    }
    /// Explicit CPU diagnostic backend; no lease, trusted manifest, or remote execution.
    #[cfg(feature = "native-scalar-reference")]
    pub fn load_scalar(
        layout: &NativeLayout,
        expert: VerifiedNativeExpert,
        binding: Binding,
    ) -> Result<Self> {
        let expected = layout.expert(binding.layer as usize, u64::from(binding.expert))?;
        if binding.model_revision.trim().is_empty()
            || binding.model_revision.len() > 256
            || binding.epoch == 0
            || expert.layer != binding.layer as usize
            || expert.expert != u64::from(binding.expert)
            || expert.span != expected
            || binding.index_sha256 != layout.index_sha256()
            || expert.bytes.len() as u64 != expected.bytes
            || sha256::hex(&expert.bytes) != binding.blob_sha256
        {
            return Err(invalid("scalar resident binding or payload mismatch"));
        }
        Ok(Self {
            binding,
            owner: ResidentBackend::Scalar {
                gu: expected.gate_up_type as i32,
                down: expected.down_type as i32,
                bytes: expert.bytes,
            },
            payload_bytes: expected.bytes,
        })
    }
    pub fn binding(&self) -> &Binding {
        &self.binding
    }
    pub(crate) fn uses_vector(&self) -> bool {
        matches!(self.owner, ResidentBackend::Vector(_))
    }
    fn gate_up_type(&self) -> i32 {
        match &self.owner {
            ResidentBackend::Vector(owner) => owner.gate_up_type(),
            #[cfg(feature = "native-scalar-reference")]
            ResidentBackend::Scalar { gu, .. } => *gu,
        }
    }
    fn prepare(&self, input: &[f32]) -> Result<vf_ggml_native_ffi::NativePrepared> {
        match &self.owner {
            ResidentBackend::Vector(owner) => owner.prepare(input).map_err(invalid),
            #[cfg(feature = "native-scalar-reference")]
            ResidentBackend::Scalar { .. } => {
                Err(invalid("scalar resident cannot use vector preparation"))
            }
        }
    }
    fn forward_prepared(
        &mut self,
        prepared: &vf_ggml_native_ffi::NativePrepared,
    ) -> Result<Vec<f32>> {
        match &mut self.owner {
            ResidentBackend::Vector(owner) => owner.forward_prepared(prepared).map_err(invalid),
            #[cfg(feature = "native-scalar-reference")]
            ResidentBackend::Scalar { .. } => {
                Err(invalid("scalar resident cannot use vector preparation"))
            }
        }
    }
    #[cfg(feature = "native-scalar-reference")]
    fn prepare_scalar(input: &[f32]) -> Result<vf_ggml_native_ffi::NativeScalarPrepared> {
        vf_ggml_native_ffi::scalar_prepare(input).map_err(invalid)
    }
    #[cfg(feature = "native-scalar-reference")]
    fn forward_scalar_prepared(
        &mut self,
        prepared: &vf_ggml_native_ffi::NativeScalarPrepared,
    ) -> Result<Vec<f32>> {
        match &mut self.owner {
            ResidentBackend::Scalar { gu, down, bytes } => {
                let values =
                    vf_ggml_native_ffi::scalar_reference_prepared(*gu, *down, bytes, prepared)
                        .map_err(invalid)?;
                let values: Vec<f32> = values.into_iter().map(|v| v as f32).collect();
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("scalar resident FP32 overflow"));
                }
                Ok(values)
            }
            ResidentBackend::Vector(_) => {
                Err(invalid("vector resident cannot use scalar preparation"))
            }
        }
    }
    /// Returns one unweighted FP32 expert row; no gate is applied here.
    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        match &mut self.owner {
            ResidentBackend::Vector(owner) => owner.forward(input).map_err(invalid),
            #[cfg(feature = "native-scalar-reference")]
            ResidentBackend::Scalar { gu, down, bytes } => {
                let values = vf_ggml_native_ffi::scalar_reference(*gu, *down, bytes, input)
                    .map_err(invalid)?;
                let values: Vec<f32> = values.into_iter().map(|v| v as f32).collect();
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("scalar resident FP32 overflow"));
                }
                Ok(values)
            }
        }
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
    last_completed: Option<(crate::moe::Stamp, u64)>,
    evictions: std::collections::BTreeSet<String>,
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
        let vector_backend = owners[0].uses_vector();
        let mut locals = BTreeSet::new();
        let mut experts = BTreeSet::new();
        for owner in &owners {
            let b = owner.binding();
            if b.rank != base.rank
                || owner.uses_vector() != vector_backend
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
            last_completed: None,
            evictions: std::collections::BTreeSet::new(),
        })
    }

    /// Append a verified resident between executions without resetting attempt or scope history.
    /// This does not evict existing weights or change their physical indices.
    pub fn add_owner(&mut self, owner: Resident) -> Result<()> {
        let base = self.owners[0].binding();
        let b = owner.binding();
        if self.terminal
            || self.owners.len() >= MAX_NATIVE_EXPERTS
            || owner.uses_vector() != self.owners[0].uses_vector()
            || b.rank != base.rank
            || b.model_revision != base.model_revision
            || b.layer != base.layer
            || b.epoch != base.epoch
            || b.index_sha256 != base.index_sha256
            || self.owners.iter().any(|existing| {
                existing.binding().local == b.local || existing.binding().expert == b.expert
            })
        {
            return Err(invalid(
                "native appended owner identity, duplicate or limit",
            ));
        }
        self.owners.push(owner);
        Ok(())
    }

    pub fn owner_count(&self) -> usize {
        self.owners.len()
    }

    /// Remove a fully identified owner only after a completed local execution.
    /// The caller must establish that the enclosing batch committed globally;
    /// this in-memory operation set is not a durable replay ledger.
    pub fn evict_owner(
        &mut self,
        stamp: &crate::moe::Stamp,
        attempt: u64,
        operation_id: &str,
        binding: &Binding,
    ) -> Result<(u64, usize)> {
        if self.terminal
            || self.owners.len() <= 1
            || !sha256::is_hex_digest(operation_id)
            || self.evictions.len() >= 4096
            || self.evictions.contains(operation_id)
            || self.last_completed.as_ref() != Some(&(stamp.clone(), attempt))
        {
            return Err(invalid("native eviction state or identity"));
        }
        let index = self
            .owners
            .iter()
            .position(|owner| owner.binding() == binding)
            .ok_or_else(|| invalid("native eviction binding differs"))?;
        self.evictions.insert(operation_id.to_owned());
        let owner = self.owners.remove(index);
        let bytes = owner.payload_bytes;
        drop(owner);
        Ok((bytes, self.owners.len()))
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
            let mut rows = vec![None; dispatch.slots.len()];
            // Scoped to this batch. Never reuse an activation across tokens or
            // gate/up formats. Compute by physical owner, then restore slot order.
            let mut prepared = BTreeMap::new();
            #[cfg(feature = "native-scalar-reference")]
            let mut scalar_prepared = BTreeMap::new();
            for index in expert_execution_order(&plan) {
                let s = &dispatch.slots[index].slot;
                let resident = &mut self.owners[plan[index]];
                rows[index] = Some(if resident.uses_vector() {
                    let key = (s.token, resident.gate_up_type());
                    if let std::collections::btree_map::Entry::Vacant(entry) = prepared.entry(key) {
                        entry.insert(resident.prepare(tokens[&s.token])?);
                    }
                    resident.forward_prepared(&prepared[&key])?
                } else {
                    #[cfg(feature = "native-scalar-reference")]
                    {
                        let key = (s.token, resident.gate_up_type());
                        if let std::collections::btree_map::Entry::Vacant(entry) =
                            scalar_prepared.entry(key)
                        {
                            entry.insert(Resident::prepare_scalar(tokens[&s.token])?);
                        }
                        resident.forward_scalar_prepared(&scalar_prepared[&key])?
                    }
                    #[cfg(not(feature = "native-scalar-reference"))]
                    {
                        return Err(invalid("scalar backend is unavailable"));
                    }
                });
            }
            let rows = rows
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| invalid("native grouped output incomplete"))?;
            let slots = dispatch
                .slots
                .iter()
                .map(|entry| {
                    let s = &entry.slot;
                    OutputSlot {
                        token: s.token,
                        slot: s.slot,
                        expert: s.expert,
                        local: s.local,
                    }
                })
                .collect();
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
        } else {
            self.last_completed = Some((stamp.clone(), attempt));
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
        let vector_backend = owners[0].uses_vector();
        for owner in &owners {
            let b = owner.binding();
            if owner.uses_vector() != vector_backend
                || b.model_revision != base.model_revision
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

/// Explicit identity for a bounded GGML CPU projection, distinct from expert formats.
/// The caller must bind this digest to an independently trusted model manifest.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatrixBinding {
    pub model_revision: String,
    pub layer: u32,
    pub tensor: String,
    pub ggml_type: i32,
    pub rows: usize,
    pub columns: usize,
    pub sha256: String,
}

pub struct QuantizedMatrix {
    binding: MatrixBinding,
    inner: vf_ggml_native_ffi::NativeMatrix,
}
impl QuantizedMatrix {
    pub fn load(binding: MatrixBinding, bytes: &[u8]) -> Result<Self> {
        if binding.model_revision.trim().is_empty()
            || binding.model_revision.len() > 256
            || binding.tensor.trim().is_empty()
            || binding.tensor.len() > 256
            || !sha256::is_hex_digest(&binding.sha256)
            || sha256::hex(bytes) != binding.sha256
        {
            return Err(invalid("native matrix binding or digest mismatch"));
        }
        let inner = vf_ggml_native_ffi::NativeMatrix::load(
            binding.ggml_type,
            binding.rows,
            binding.columns,
            bytes,
        )
        .map_err(invalid)?;
        Ok(Self { binding, inner })
    }
    pub fn binding(&self) -> &MatrixBinding {
        &self.binding
    }
    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        self.inner.forward(input).map_err(invalid)
    }
}

#[cfg(test)]
mod grouped_execution_tests {
    use super::expert_execution_order;

    #[test]
    fn grouping_is_stable_and_preserves_every_original_index() {
        assert_eq!(expert_execution_order(&[]), Vec::<usize>::new());
        assert_eq!(
            expert_execution_order(&[2, 0, 2, 1, 0]),
            vec![1, 4, 3, 0, 2]
        );
        assert_eq!(expert_execution_order(&[3, 3, 3]), vec![0, 1, 2]);
    }

    #[test]
    fn grouping_matches_independent_bucket_oracle_for_all_small_plans() {
        for len in 0..=7_u32 {
            for mut code in 0..3_usize.pow(len) {
                let mut owners = Vec::new();
                for _ in 0..len {
                    owners.push(code % 3);
                    code /= 3;
                }
                let mut expected = Vec::new();
                for owner in 0..3 {
                    for (index, actual) in owners.iter().enumerate() {
                        if *actual == owner {
                            expected.push(index);
                        }
                    }
                }
                assert_eq!(expert_execution_order(&owners), expected);
            }
        }
    }
}
