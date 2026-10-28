//! Owned, immutable QPN2 buffers for one rank. CPU construction only; no GPU residency.
use crate::{invalid, moe, qpn2, quant, routing, Result};
use serde::{Deserialize, Serialize};

pub const MAX_PAYLOAD_BYTES: u64 = 128 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidentExpert {
    pub id: u32,
    pub local: u32,
    pub gate: quant::PackedRows,
    pub up: quant::PackedRows,
    pub down: quant::PackedRows,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidentInput {
    pub model_revision: String,
    pub layer: u32,
    pub rank: u32,
    pub registry: moe::Registry,
    pub resident_order: Vec<u32>,
    pub experts: Vec<ResidentExpert>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PhysicalExpert {
    id: u32,
    local: u32,
}
impl PhysicalExpert {
    pub fn id(&self) -> u32 {
        self.id
    }
    pub fn local(&self) -> u32 {
        self.local
    }
}
/// Serialization exposes a snapshot. Deserialization is deliberately absent:
/// all owned stacks must pass the complete constructor validation.
/// The retained SM70 admission is valid only because neither payload replacement
/// nor mutable slices are exposed. Internal epoch rebinding refreshes admission.
///
/// ```
/// fn inspect(stack: &voltafabric::resident::ResidentStack) -> serde_json::Result<String> {
///     let _ = stack.scales13();
///     serde_json::to_string(stack)
/// }
/// ```
/// ```compile_fail
/// fn overwrite(stack: &mut voltafabric::resident::ResidentStack) {
///     stack.scales13()[0] = 126;
/// }
/// ```
/// ```compile_fail
/// fn replace(stack: &mut voltafabric::resident::ResidentStack) {
///     stack.gscales13 = vec![4.0];
/// }
/// ```
/// ```compile_fail
/// let _: voltafabric::resident::ResidentStack = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct ResidentStack {
    // Stable alphabetical snapshot order matches the existing JSON CLI.
    codes13: Vec<u8>,
    codes2: Vec<u8>,
    epoch: u64,
    #[serde(serialize_with = "serialize_f32_as_f64")]
    gscales13: Vec<f32>,
    #[serde(serialize_with = "serialize_f32_as_f64")]
    gscales2: Vec<f32>,
    layer: u32,
    model_revision: String,
    payload_bytes: u64,
    physical_experts: Vec<PhysicalExpert>,
    rank: u32,
    resident_order: Vec<u32>,
    scales13: Vec<u8>,
    scales2: Vec<u8>,
    shape: (usize, u64, u64, u64, u64),
    // Backend admission belongs to these exact owned bytes, never a tensor pointer.
    // CPU ownership remains valid if the SM70 backend refuses their numeric range.
    #[serde(skip)]
    sm70_domains: Option<[Result<()>; 2]>,
}
// The old serde_json::Value snapshot promoted each F32 to F64. Preserve those
// exact JSON decimals while streaming, without allocating a second factor table.
fn serialize_f32_as_f64<S: serde::Serializer>(
    values: &[f32],
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(values.len()))?;
    for &value in values {
        seq.serialize_element(&f64::from(value))?;
    }
    seq.end()
}
impl ResidentStack {
    // Establish admission at both owned construction boundaries and on an epoch change.
    // Payloads have no mutable accessor or Deserialize path. Clear old authorization
    // first: an interrupted refresh or a constructor omitting the seal fails closed.
    fn refresh_sm70(&mut self) {
        self.sm70_domains = None;
        if !self.is_empty() {
            self.sm70_domains = Some([
                quant::validate_sm70_scale_domain(&self.scales13, &self.gscales13, self.shape.2),
                quant::validate_sm70_scale_domain(&self.scales2, &self.gscales2, self.shape.4),
            ]);
        }
    }

    fn seal_sm70(mut self) -> Self {
        self.refresh_sm70();
        self
    }

    pub(crate) fn sm70_domain(&self, down: bool) -> Result<()> {
        self.sm70_domains.as_ref().ok_or_else(|| {
            Error::new(
                "SM70_ADMISSION_MISSING",
                "resident stack has no SM70 numeric admission",
            )
        })?[usize::from(down)]
        .clone()
    }

    pub fn model_revision(&self) -> &str {
        &self.model_revision
    }
    pub fn layer(&self) -> u32 {
        self.layer
    }
    pub fn rank(&self) -> u32 {
        self.rank
    }
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
    pub fn physical_experts(&self) -> &[PhysicalExpert] {
        &self.physical_experts
    }
    pub fn resident_order(&self) -> &[u32] {
        &self.resident_order
    }
    /// (experts, gate+up rows, gate+up columns, down rows, down columns).
    /// A rank without experts has the dimensionless sentinel (0,0,0,0,0).
    pub fn shape(&self) -> (usize, u64, u64, u64, u64) {
        self.shape
    }
    pub fn is_empty(&self) -> bool {
        self.physical_experts.is_empty()
    }
    pub fn codes13(&self) -> &[u8] {
        &self.codes13
    }
    pub fn scales13(&self) -> &[u8] {
        &self.scales13
    }
    pub fn gscales13(&self) -> &[f32] {
        &self.gscales13
    }
    pub fn codes2(&self) -> &[u8] {
        &self.codes2
    }
    pub fn scales2(&self) -> &[u8] {
        &self.scales2
    }
    pub fn gscales2(&self) -> &[f32] {
        &self.gscales2
    }
    pub fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    /// Validate a rank-local dispatch against static resident identity and physical
    /// placement. Owner selection and cross-rank completeness remain routing's
    /// responsibility. Run, plan and batch are intentionally not bound to weights.
    /// Empty ranks have no H to compare: only routing bounds are established.
    pub fn bind_dispatch(&self, d: &routing::RankDispatch) -> Result<()> {
        if d.stamp.model_revision != self.model_revision
            || d.stamp.layer != self.layer
            || d.stamp.epoch != self.epoch
            || d.rank != self.rank
            || d.resident_order != self.resident_order
        {
            return Err(invalid(
                "dispatch does not match static resident identity or physical order",
            ));
        }
        if d.hidden == 0
            || d.hidden > 65536
            || d.tokens > 4096
            || d.slots.len() > 262144
            || d.tokens
                .checked_mul(d.hidden)
                .is_none_or(|n| n > 16_777_216)
            || [&d.stamp.run_id, &d.stamp.plan_id]
                .iter()
                .any(|s| s.trim().is_empty() || s.len() > 256)
        {
            return Err(invalid(
                "dispatch identity or dimensions outside routing bounds",
            ));
        }
        if self.is_empty() {
            return if d.slots.is_empty() {
                Ok(())
            } else {
                Err(invalid("empty rank cannot bind compute slots"))
            };
        }
        if d.hidden as u64 != self.shape.2 {
            return Err(invalid(
                "dispatch hidden dimension differs from resident weights",
            ));
        }
        let mut slots = std::collections::BTreeSet::new();
        let mut experts = std::collections::BTreeSet::new();
        for routed in &d.slots {
            let s = &routed.slot;
            let resident = self
                .physical_experts
                .get(routed.physical_expert)
                .ok_or_else(|| invalid("physical expert outside resident stack"))?;
            if s.rank != self.rank
                || s.expert != resident.id
                || s.local != resident.local
                || s.token >= d.tokens
                || s.slot >= 64
                || !s.weight.is_finite()
                || s.weight < 0.0
                || !slots.insert((s.token, s.slot))
                || !experts.insert((s.token, s.expert))
            {
                return Err(invalid("foreign, duplicate or malformed resident slot"));
            }
        }
        Ok(())
    }
}

use crate::memory::{add, mul};
use crate::Error;
use std::collections::{BTreeMap, BTreeSet};

/// CPU construction/oracle ceiling. Resident GPU artifacts use the separate payload bound.
const MAX_CPU_MATRIX_ELEMENTS: u64 = 1_048_576;
/// A single projection cannot exceed the resident payload budget even before the
/// complete code/scale/factor byte sum is checked below. Three QPN2 projections need
/// 27/16 bytes per logical matrix element, ignoring the much smaller factor vectors.
const MAX_RESIDENT_MATRIX_ELEMENTS: u64 = MAX_PAYLOAD_BYTES * 16 / 27;
#[derive(Default)]
struct Layout {
    shape: (usize, u64, u64, u64, u64),
    codes13: u64,
    scales13: u64,
    gscales13: u64,
    codes2: u64,
    scales2: u64,
    gscales2: u64,
    payload_bytes: u64,
}

fn matrix_elements(p: &quant::PackedRows) -> Result<u64> {
    if p.rows == 0 || p.columns == 0 || !p.rows.is_multiple_of(32) || !p.columns.is_multiple_of(64)
    {
        return Err(invalid(
            "resident QPN2 planes require positive rows/32 and columns/64 alignment",
        ));
    }
    let elements = mul(p.rows, p.columns)?;
    if elements > MAX_CPU_MATRIX_ELEMENTS {
        return Err(invalid(
            "resident plane exceeds bounded QPN2 CPU work limit",
        ));
    }
    Ok(elements)
}
fn total_bytes(
    codes13: u64,
    scales13: u64,
    codes2: u64,
    scales2: u64,
    gscales13: u64,
    gscales2: u64,
) -> Result<u64> {
    let bytes = add(
        add(add(codes13, scales13)?, add(codes2, scales2)?)?,
        mul(add(gscales13, gscales2)?, 4)?,
    )?;
    if bytes > MAX_PAYLOAD_BYTES {
        return Err(Error::new(
            "CAPACITY_EXCEEDED",
            "resident compressed buffers and tile factors exceed 128 MiB CPU payload cap",
        ));
    }
    Ok(bytes)
}

/// Scalar preflight does not inspect/allocate payloads or run any permutations.
fn layout(p: &ResidentInput) -> Result<Layout> {
    if p.model_revision.trim().is_empty()
        || p.model_revision.len() > 256
        || p.experts.len() > 1024
        || p.resident_order.len() > 1024
        || p.registry.copies.len() > 65536
    {
        return Err(invalid(
            "resident identity or metadata exceeds bounded contract",
        ));
    }
    let mut shape = None;
    for expert in &p.experts {
        matrix_elements(&expert.gate)?;
        matrix_elements(&expert.up)?;
        matrix_elements(&expert.down)?;
        let this = (
            expert.gate.rows,
            expert.gate.columns,
            expert.down.rows,
            expert.down.columns,
        );
        if (expert.up.rows, expert.up.columns) != (this.0, this.1)
            || this.1 != this.2
            || this.0 != this.3
            || shape.is_some_and(|old| old != this)
        {
            return Err(invalid(
                "gate/up/down or inter-expert shapes do not form a uniform SwiGLU stack",
            ));
        }
        shape = Some(this);
    }
    let Some((gate_rows, gate_columns, down_rows, down_columns)) = shape else {
        return Ok(Layout::default());
    };
    let e = p.experts.len() as u64;
    let n13 = mul(gate_rows, 2)?;
    let elements13 = mul(n13, gate_columns)?;
    let elements2 = mul(down_rows, down_columns)?;
    let codes13 = mul(e, elements13 / 2)?;
    let scales13 = mul(e, elements13 / 16)?;
    let codes2 = mul(e, elements2 / 2)?;
    let scales2 = mul(e, elements2 / 16)?;
    let gscales13 = mul(e, n13 / 32)?;
    let gscales2 = mul(e, down_rows / 32)?;
    Ok(Layout {
        shape: (p.experts.len(), n13, gate_columns, down_rows, down_columns),
        codes13,
        scales13,
        gscales13,
        codes2,
        scales2,
        gscales2,
        payload_bytes: total_bytes(codes13, scales13, codes2, scales2, gscales13, gscales2)?,
    })
}
fn exact_f32(factor: f64) -> Result<f32> {
    let value = factor as f32;
    if !factor.is_finite()
        || factor <= 0.0
        || !value.is_finite()
        || value <= 0.0
        || f64::from(value) != factor
    {
        return Err(invalid(
            "global dequant factor must be positive, finite and exactly representable as F32",
        ));
    }
    Ok(value)
}
fn validate_plane(p: &quant::PackedRows) -> Result<()> {
    let elements = mul(p.rows, p.columns)?;
    if p.codes.len() as u64 != elements / 2 || p.scales.len() as u64 != elements / 16 {
        return Err(invalid("resident packed plane length mismatch"));
    }
    exact_f32(p.global_dequant)?;
    for &scale in &p.scales {
        // Reuse unsigned finite NVFP4 E4M3 validation, including valid zero.
        // The scalar result is discarded; no floating weight plane is built.
        quant::decode_scale(scale)?;
    }
    Ok(())
}
fn validate_registry(p: &ResidentInput) -> Result<BTreeMap<u32, &ResidentExpert>> {
    let mut global = BTreeSet::new();
    let mut by_local = BTreeMap::new();
    for expert in &p.experts {
        if !global.insert(expert.id) || by_local.insert(expert.local, expert).is_some() {
            return Err(invalid("duplicate resident global expert or rank-local ID"));
        }
    }
    let mut pairs = BTreeSet::new();
    let mut locals = BTreeSet::new();
    let mut resident = BTreeMap::new();
    for replica in &p.registry.copies {
        if !pairs.insert((replica.expert, replica.rank))
            || !locals.insert((replica.rank, replica.local))
        {
            return Err(invalid(
                "registry has duplicate replica or rank-local collision",
            ));
        }
        if replica.rank == p.rank {
            resident.insert(replica.local, replica.expert);
        }
    }
    if resident.len() != by_local.len()
        || by_local
            .iter()
            .any(|(&local, expert)| resident.get(&local) != Some(&expert.id))
    {
        return Err(invalid(
            "registry rank does not exactly match local resident experts",
        ));
    }
    let order: BTreeSet<_> = p.resident_order.iter().copied().collect();
    if order.len() != p.resident_order.len() || order != by_local.keys().copied().collect() {
        return Err(invalid(
            "explicit physical order does not cover resident local IDs exactly",
        ));
    }
    Ok(by_local)
}
fn buffer<T>(count: u64) -> Result<Vec<T>> {
    let count = usize::try_from(count)
        .map_err(|_| Error::new("OVERFLOW", "resident buffer length exceeds usize"))?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|e| Error::new("ALLOCATION_FAILED", e.to_string()))?;
    Ok(result)
}

/// Metadata-only preflight for binary checkpoint import; no payload scans.
pub(crate) fn preflight(p: &ResidentInput) -> Result<()> {
    layout(p)?;
    validate_registry(p)?;
    Ok(())
}

// The constructor validated every source and reserved the complete final payload.
// Appended segments are initialized and exclusively borrowed by pack_into.
fn append_plane(p: &quant::PackedRows, codes: &mut Vec<u8>, scales: &mut Vec<u8>) -> Result<()> {
    let cstart = codes.len();
    let sstart = scales.len();
    let cend = cstart
        .checked_add(p.codes.len())
        .ok_or_else(|| Error::new("OVERFLOW", "code append overflow"))?;
    let send = sstart
        .checked_add(p.scales.len())
        .ok_or_else(|| Error::new("OVERFLOW", "scale append overflow"))?;
    codes.resize(cend, 0);
    scales.resize(send, 0);
    qpn2::pack_into(p, &mut codes[cstart..cend], &mut scales[sstart..send])
}

/// Own compressed expert-major buffers with gate tiles followed by up tiles.
/// Limits are checked from metadata before payload scans, allocations or packing.
/// The 128 MiB cap describes final CPU payload, not the process peak or GPU memory.
/// Each plane is permuted directly into the final pre-reserved initialized buffers.
pub fn pack(p: &ResidentInput) -> Result<ResidentStack> {
    let sizes = layout(p)?;
    let experts = validate_registry(p)?;
    for expert in &p.experts {
        for plane in [&expert.gate, &expert.up, &expert.down] {
            validate_plane(plane)?;
        }
    }
    let mut stack = ResidentStack {
        model_revision: p.model_revision.clone(),
        layer: p.layer,
        rank: p.rank,
        epoch: p.registry.epoch,
        physical_experts: buffer(p.experts.len() as u64)?,
        resident_order: p.resident_order.clone(),
        shape: sizes.shape,
        codes13: buffer(sizes.codes13)?,
        scales13: buffer(sizes.scales13)?,
        gscales13: buffer(sizes.gscales13)?,
        codes2: buffer(sizes.codes2)?,
        scales2: buffer(sizes.scales2)?,
        gscales2: buffer(sizes.gscales2)?,
        payload_bytes: sizes.payload_bytes,
        sm70_domains: None,
    };
    for &local in &p.resident_order {
        let expert = experts[&local];
        stack.physical_experts.push(PhysicalExpert {
            id: expert.id,
            local,
        });
        for plane in [&expert.gate, &expert.up] {
            append_plane(plane, &mut stack.codes13, &mut stack.scales13)?;
            stack.gscales13.extend(std::iter::repeat_n(
                exact_f32(plane.global_dequant)?,
                (plane.rows / 32) as usize,
            ));
        }
        append_plane(&expert.down, &mut stack.codes2, &mut stack.scales2)?;
        stack.gscales2.extend(std::iter::repeat_n(
            exact_f32(expert.down.global_dequant)?,
            (expert.down.rows / 32) as usize,
        ));
    }
    Ok(stack.seal_sm70())
}

/// The buffers and identity an exported resident artifact carries for one stack.
pub(crate) struct Parts {
    pub model_revision: String,
    pub layer: u32,
    pub rank: u32,
    pub epoch: u64,
    /// (global expert ID, rank-local ID) in physical order.
    pub physical_experts: Vec<(u32, u32)>,
    pub resident_order: Vec<u32>,
    pub shape: [usize; 5],
    pub payload_bytes: u64,
    pub codes13: Vec<u8>,
    pub scales13: Vec<u8>,
    pub gscales13: Vec<f32>,
    pub codes2: Vec<u8>,
    pub scales2: Vec<u8>,
    pub gscales2: Vec<f32>,
}

fn artifact_invalid(message: &str) -> Error {
    Error::new("ARTIFACT_INVALID", message)
}

/// Each plane's N32 tiles share the one positive, finite global factor `pack` repeats.
fn uniform_factors(factors: &[f32], per_plane: usize) -> Result<()> {
    for plane in factors.chunks_exact(per_plane) {
        if plane
            .iter()
            .any(|&f| !f.is_finite() || f <= 0.0 || f != plane[0])
        {
            return Err(artifact_invalid(
                "tile factors must be one positive finite F32 per plane",
            ));
        }
    }
    Ok(())
}

/// Rebuild a stack from exported buffers. Everything `pack` establishes about its output
/// is checked again: bounded identity, distinct experts in the declared physical order,
/// the uniform SwiGLU shape and its alignment, exact buffer lengths and payload size,
/// unsigned finite E4M3 scale bytes and one positive finite factor per plane.
pub(crate) fn from_parts(p: Parts) -> Result<ResidentStack> {
    let e = p.physical_experts.len();
    if p.model_revision.trim().is_empty() || p.model_revision.len() > 256 || e > 1024 {
        return Err(artifact_invalid(
            "resident identity or metadata exceeds bounded contract",
        ));
    }
    let mut ids = BTreeSet::new();
    let mut locals = BTreeSet::new();
    if p.physical_experts
        .iter()
        .any(|&(id, local)| !ids.insert(id) || !locals.insert(local))
        || p.resident_order.len() != e
        || p.resident_order
            .iter()
            .zip(&p.physical_experts)
            .any(|(&order, &(_, local))| order != local)
    {
        return Err(artifact_invalid(
            "duplicate expert or physical order differing from the experts' local IDs",
        ));
    }
    let [count, n13, k13, n2, k2] = p.shape;
    let (n13, k13, n2, k2) = (n13 as u64, k13 as u64, n2 as u64, k2 as u64);
    let sizes = if e == 0 {
        if p.shape != [0; 5] {
            return Err(artifact_invalid("an empty rank has the shape (0,0,0,0,0)"));
        }
        [0; 6]
    } else {
        // gate rows = up rows = down columns = k2, and down rows = hidden = k13.
        if count != e
            || k2 == 0
            || k13 == 0
            || !k2.is_multiple_of(64)
            || !k13.is_multiple_of(64)
            || n13 != mul(k2, 2)?
            || n2 != k13
            || mul(k2, k13)? > MAX_RESIDENT_MATRIX_ELEMENTS
        {
            return Err(artifact_invalid(
                "shape is not an aligned, bounded, uniform SwiGLU stack",
            ));
        }
        let e = e as u64;
        let elements13 = mul(n13, k13)?;
        let elements2 = mul(n2, k2)?;
        [
            mul(e, elements13 / 2)?,
            mul(e, elements13 / 16)?,
            mul(e, n13 / 32)?,
            mul(e, elements2 / 2)?,
            mul(e, elements2 / 16)?,
            mul(e, n2 / 32)?,
        ]
    };
    let lengths = [
        p.codes13.len(),
        p.scales13.len(),
        p.gscales13.len(),
        p.codes2.len(),
        p.scales2.len(),
        p.gscales2.len(),
    ];
    if lengths
        .iter()
        .zip(sizes)
        .any(|(&have, want)| have as u64 != want)
        || total_bytes(sizes[0], sizes[1], sizes[3], sizes[4], sizes[2], sizes[5])?
            != p.payload_bytes
    {
        return Err(artifact_invalid(
            "buffer lengths or payload size differ from the declared shape",
        ));
    }
    if p.scales13
        .iter()
        .chain(&p.scales2)
        .any(|&b| quant::decode_scale(b).is_err())
    {
        return Err(artifact_invalid(
            "scale byte is not an unsigned finite E4M3 value",
        ));
    }
    if e != 0 {
        uniform_factors(&p.gscales13, (n13 / 64) as usize)?;
        uniform_factors(&p.gscales2, (n2 / 32) as usize)?;
    }
    Ok(ResidentStack {
        codes13: p.codes13,
        codes2: p.codes2,
        epoch: p.epoch,
        gscales13: p.gscales13,
        gscales2: p.gscales2,
        layer: p.layer,
        model_revision: p.model_revision,
        payload_bytes: p.payload_bytes,
        physical_experts: p
            .physical_experts
            .into_iter()
            .map(|(id, local)| PhysicalExpert { id, local })
            .collect(),
        rank: p.rank,
        resident_order: p.resident_order,
        scales13: p.scales13,
        scales2: p.scales2,
        shape: (count, n13, k13, n2, k2),
        sm70_domains: None,
    }
    .seal_sm70())
}

/// QPN2 fragment slot of each column within a K16 group (inverse of qpn2's K order).
const SLOT_OF_K: [usize; 16] = [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15];

/// One QPN2 plane of a stack: fragment codes, one scale byte per lane and K16 group,
/// one global factor per N32 tile.
struct Plane<'a> {
    codes: &'a [u8],
    scales: &'a [u8],
    factors: &'a [f32],
    columns: usize,
}
impl Plane<'_> {
    /// Row `row` of the plane times `x`. Each weight is decoded where it is used and
    /// summed in column order, as the dense oracle sums.
    fn dot(&self, row: usize, x: &[f64]) -> Result<f64> {
        let groups = self.columns / 16;
        let tile = row / 32;
        let r = row % 32;
        let lane = (r & 3) | (((r >> 3) & 3) << 2) | (((r >> 2) & 1) << 4);
        let factor = f64::from(self.factors[tile]);
        let mut sum = 0.0;
        for group in 0..groups {
            let fragment = (tile * groups + group) * 32 + lane;
            let scale = quant::decode_scale(self.scales[fragment])?;
            let codes = &self.codes[fragment * 8..fragment * 8 + 8];
            for (k, &slot) in SLOT_OF_K.iter().enumerate() {
                let nibble = (codes[slot / 2] >> (4 * (slot % 2))) & 15;
                let weight = quant::decode_fp4(nibble)? * scale * factor;
                sum = moe::finite(sum + x[group * 16 + k] * weight)?;
            }
        }
        Ok(sum)
    }
}

impl ResidentStack {
    /// Bind to a new ownership epoch and refresh its numeric admission once.
    pub(crate) fn set_epoch(&mut self, epoch: u64) {
        if self.epoch != epoch {
            self.epoch = epoch;
            self.refresh_sm70();
        }
    }

    /// Whether physical expert `physical` here and `other_physical` in `other` hold the
    /// same quantized planes: codes, scale bytes and global factors, plane for plane.
    pub(crate) fn same_expert(
        &self,
        physical: usize,
        other: &ResidentStack,
        other_physical: usize,
    ) -> bool {
        match (self.planes(physical), other.planes(other_physical)) {
            (Ok(a), Ok(b)) => a.iter().zip(&b).all(|(x, y)| {
                x.columns == y.columns
                    && x.codes == y.codes
                    && x.scales == y.scales
                    && x.factors == y.factors
            }),
            _ => false,
        }
    }

    /// (gate, up, down) planes of one physical expert.
    fn planes(&self, physical: usize) -> Result<[Plane<'_>; 3]> {
        let (e, n13, k13, n2, k2) = self.shape;
        if physical >= e {
            return Err(invalid("physical expert outside the resident stack"));
        }
        let (n13, k13, n2, k2) = (n13 as usize, k13 as usize, n2 as usize, k2 as usize);
        let (c13, s13, t13) = (n13 * k13 / 2, n13 * k13 / 16, n13 / 32);
        let (c2, s2, t2) = (n2 * k2 / 2, n2 * k2 / 16, n2 / 32);
        let half = |start: usize, len: usize, second: bool| {
            let begin = start + if second { len / 2 } else { 0 };
            begin..begin + len / 2
        };
        let gate_up = |second| Plane {
            codes: &self.codes13[half(physical * c13, c13, second)],
            scales: &self.scales13[half(physical * s13, s13, second)],
            factors: &self.gscales13[half(physical * t13, t13, second)],
            columns: k13,
        };
        Ok([
            gate_up(false),
            gate_up(true),
            Plane {
                codes: &self.codes2[physical * c2..(physical + 1) * c2],
                scales: &self.scales2[physical * s2..(physical + 1) * s2],
                factors: &self.gscales2[physical * t2..(physical + 1) * t2],
                columns: k2,
            },
        ])
    }

    /// Unweighted SwiGLU output of one physical expert, computed on the compressed
    /// planes: no weight plane is ever decoded as a whole. Gate and up rows accumulate over
    /// the hidden index and down rows over the intermediate one, term for term as
    /// `moe::expert_value` does, so the result equals the dense f64 oracle on the
    /// dequantized weights bit for bit (every decoded NVFP4 weight is exact in f64).
    pub fn expert_value(&self, physical: usize, x: &[f64]) -> Result<Vec<f64>> {
        let [gate, up, down] = self.planes(physical)?;
        if x.len() != gate.columns || x.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "activation length differs from the resident hidden size or is nonfinite",
            ));
        }
        let intermediate = down.columns;
        let mut activated = Vec::with_capacity(intermediate);
        for j in 0..intermediate {
            let g = gate.dot(j, x)?;
            let u = up.dot(j, x)?;
            activated.push(moe::finite(moe::silu(g) * u)?);
        }
        (0..gate.columns).map(|i| down.dot(i, &activated)).collect()
    }

    /// Dense f64 copy of one resident expert in the dense oracle's layout, by the other
    /// path: QPN2 unpack to row-major NVFP4, then the linear decoder. A bounded CPU
    /// oracle; it materializes the three planes the compressed kernel never builds.
    pub fn dense_expert(&self, physical: usize) -> Result<moe::Expert> {
        let planes = self.planes(physical)?;
        let id = self.physical_experts[physical];
        let mut dense = Vec::with_capacity(3);
        for plane in &planes {
            let rows = plane.factors.len() * 32;
            let fragments = qpn2::FragmentRows {
                rows: rows as u64,
                columns: plane.columns as u64,
                codes: plane.codes.to_vec(),
                scales: plane.scales.to_vec(),
                global_dequant: f64::from(plane.factors[0]),
            };
            let w = quant::dequantize(&qpn2::unpack(&fragments)?)?;
            // Dense layout: input index major, w[row][column] -> out[column * rows + row].
            let mut out = vec![0.0; rows * plane.columns];
            for (row, values) in w.iter().enumerate() {
                for (column, &v) in values.iter().enumerate() {
                    out[column * rows + row] = v;
                }
            }
            dense.push(out);
        }
        let down = dense.pop().unwrap_or_default();
        let up = dense.pop().unwrap_or_default();
        let gate = dense.pop().unwrap_or_default();
        Ok(moe::Expert {
            id: id.id,
            gate,
            up,
            down,
        })
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    #[test]
    fn exact_cap_includes_f32_tile_factor_bytes_without_allocating_payloads() {
        assert_eq!(
            total_bytes(MAX_PAYLOAD_BYTES - 8, 0, 0, 0, 1, 1).unwrap(),
            MAX_PAYLOAD_BYTES
        );
        assert_eq!(
            total_bytes(MAX_PAYLOAD_BYTES - 7, 0, 0, 0, 1, 1)
                .unwrap_err()
                .code,
            "CAPACITY_EXCEEDED"
        );
    }
    #[test]
    fn payload_addition_and_factor_multiplication_are_checked() {
        assert_eq!(
            total_bytes(u64::MAX, 1, 0, 0, 0, 0).unwrap_err().code,
            "OVERFLOW"
        );
        assert_eq!(
            total_bytes(0, 0, 0, 0, u64::MAX, 0).unwrap_err().code,
            "OVERFLOW"
        );
    }
}

#[cfg(test)]
mod a145_admission_tests {
    use super::*;
    use crate::{launch, resident_artifact};

    fn input() -> ResidentInput {
        let plane = || quant::PackedRows {
            rows: 64,
            columns: 64,
            codes: vec![0x22; 2048],
            scales: vec![56; 256],
            global_dequant: 1.0,
        };
        ResidentInput {
            model_revision: "a145-model".into(),
            layer: 3,
            rank: 17,
            registry: moe::Registry {
                epoch: 12,
                copies: vec![moe::Replica {
                    expert: 23,
                    rank: 17,
                    local: 900,
                }],
            },
            resident_order: vec![900],
            experts: vec![ResidentExpert {
                id: 23,
                local: 900,
                gate: plane(),
                up: plane(),
                down: plane(),
            }],
        }
    }
    fn dispatch(w: &ResidentStack, batch: u64) -> routing::RankDispatch {
        routing::RankDispatch {
            stamp: moe::Stamp {
                run_id: "a145-run".into(),
                model_revision: w.model_revision().into(),
                plan_id: "a145-plan".into(),
                layer: w.layer(),
                batch,
                epoch: w.epoch(),
            },
            rank: w.rank(),
            hidden: 64,
            tokens: 1,
            resident_order: vec![900],
            slots: vec![routing::RankSlot {
                physical_expert: 0,
                slot: moe::Slot {
                    token: 0,
                    slot: 0,
                    expert: 23,
                    rank: 17,
                    local: 900,
                    weight: 0.25,
                },
            }],
        }
    }
    fn prepare(w: &ResidentStack, batch: u64) -> Result<()> {
        let d = dispatch(w, batch);
        let o = launch::Options {
            splitk13: 4,
            splitk2: 4,
            max_workspace_bytes: 1 << 20,
        };
        launch::prepare(w, &d, &d.stamp, &o).map(|_| ())
    }
    fn scans() -> usize {
        quant::SM70_ADMISSION_SCANS.with(|count| count.get())
    }
    fn reset() {
        quant::SM70_ADMISSION_SCANS.with(|count| count.set(0));
    }

    #[test]
    fn a145_load_scans_once_per_projection_and_batches_never_rescan() {
        reset();
        let w = pack(&input()).unwrap();
        assert_eq!(scans(), 2, "load must establish both numeric admissions");
        reset();
        for batch in 0..32 {
            prepare(&w, batch).unwrap();
        }
        let cloned = w.clone();
        prepare(&cloned, 33).unwrap();
        assert_eq!(
            scans(),
            0,
            "immutable owners must not rescan on batches or clone"
        );
        let bytes = resident_artifact::encode(&w).unwrap();
        let loaded = resident_artifact::import(&bytes).unwrap();
        assert_eq!(
            scans(),
            2,
            "artifact reconstruction must admit its own bytes"
        );
        reset();
        prepare(&loaded, 34).unwrap();
        assert_eq!(scans(), 0);
        assert_eq!(resident_artifact::encode(&loaded).unwrap(), bytes);
        assert_eq!(
            serde_json::to_value(&w).unwrap(),
            serde_json::to_value(&loaded).unwrap()
        );
    }

    #[test]
    fn a145_bad_gate_up_and_down_keep_the_exact_refusal_without_rescans() {
        for plane in 0..3 {
            for (factor, scale, reason) in [
                (4.0, 56, "global scale intermediate overflows binary16"),
                (0.5, 126, "scale product overflows binary16"),
            ] {
                let mut p = input();
                let e = &mut p.experts[0];
                let target = match plane {
                    0 => &mut e.gate,
                    1 => &mut e.up,
                    _ => &mut e.down,
                };
                target.global_dequant = factor;
                target.scales.fill(scale);
                // CPU format ownership is valid even when the SM70 backend must refuse it.
                let w = pack(&p).unwrap();
                let bytes = resident_artifact::encode(&w).unwrap();
                let loaded = resident_artifact::import(&bytes).unwrap();
                for owner in [&w, &loaded, &w.clone()] {
                    reset();
                    for batch in [1, 99] {
                        let err = prepare(owner, batch).unwrap_err();
                        assert_eq!(err.code, "SM70_SCALE_RANGE");
                        assert_eq!(err.message, reason);
                    }
                    assert_eq!(scans(), 0, "negative admission must be retained as well");
                }
            }
        }
    }

    #[test]
    fn a145_empty_rank_retains_zero_payload_and_needs_no_numeric_scan() {
        let mut p = input();
        p.registry.copies.clear();
        p.resident_order.clear();
        p.experts.clear();
        reset();
        let w = pack(&p).unwrap();
        let loaded = resident_artifact::import(&resident_artifact::encode(&w).unwrap()).unwrap();
        assert_eq!(scans(), 0);
        assert_eq!(loaded.payload_bytes(), 0);
        assert!(
            loaded.sm70_domain(false).is_err(),
            "no projection exists to authorize"
        );
        assert!(
            loaded.sm70_domain(true).is_err(),
            "no projection exists to authorize"
        );
        let mut d = dispatch(&loaded, 1);
        d.resident_order.clear();
        d.slots.clear();
        let o = launch::Options {
            splitk13: 4,
            splitk2: 4,
            max_workspace_bytes: 1 << 20,
        };
        assert!(launch::prepare(&loaded, &d, &d.stamp, &o).unwrap().skip());
        assert_eq!(scans(), 0);
    }

    #[test]
    fn a145_epoch_rebinding_readmits_and_keeps_old_dispatch_ineligible() {
        for invalid in [false, true] {
            let mut p = input();
            if invalid {
                p.experts[0].down.global_dequant = 4.0;
            }
            let mut w = pack(&p).unwrap();
            let old_dispatch = dispatch(&w, 1);
            reset();
            w.set_epoch(13);
            assert_eq!(scans(), 2, "new epoch must establish a fresh admission");
            reset();
            w.set_epoch(13);
            assert_eq!(scans(), 0, "unchanged epoch is not a new admission");
            let result = prepare(&w, 2);
            if invalid {
                assert_eq!(result.unwrap_err().code, "SM70_SCALE_RANGE");
            } else {
                result.unwrap();
            }
            assert_eq!(scans(), 0, "batches still use the rebound admission");
            let o = launch::Options {
                splitk13: 4,
                splitk2: 4,
                max_workspace_bytes: 1 << 20,
            };
            assert_eq!(
                launch::prepare(&w, &old_dispatch, &old_dispatch.stamp, &o)
                    .unwrap_err()
                    .code,
                "INVALID_INPUT",
                "old ownership epoch must fail at identity binding"
            );
        }
    }

    #[test]
    fn a145_gate_up_failure_still_precedes_a_distinct_down_failure() {
        let mut p = input();
        p.experts[0].gate.global_dequant = 4.0;
        p.experts[0].down.global_dequant = 0.5;
        p.experts[0].down.scales.fill(126);
        let w = pack(&p).unwrap();
        let err = prepare(&w, 1).unwrap_err();
        assert_eq!(err.code, "SM70_SCALE_RANGE");
        assert_eq!(err.message, "global scale intermediate overflows binary16");
    }

    #[test]
    fn a145_source_changes_revision_and_epoch_do_not_reuse_an_old_admission() {
        let mut p = input();
        let old = pack(&p).unwrap();
        p.experts[0].up.global_dequant = 4.0;
        let same_identity_new_content = pack(&p).unwrap();
        assert_eq!(
            prepare(&same_identity_new_content, 1).unwrap_err().code,
            "SM70_SCALE_RANGE",
            "equal model/epoch tags cannot authorize different content"
        );
        p.model_revision.push_str("-new");
        p.registry.epoch += 1;
        let new = pack(&p).unwrap();
        // The original owner cannot see edits to the caller's mutable input.
        prepare(&old, 1).unwrap();
        assert_eq!(prepare(&new, 1).unwrap_err().code, "SM70_SCALE_RANGE");
        let d = dispatch(&old, 1);
        let o = launch::Options {
            splitk13: 4,
            splitk2: 4,
            max_workspace_bytes: 1 << 20,
        };
        assert_eq!(
            launch::prepare(&new, &d, &d.stamp, &o).unwrap_err().code,
            "INVALID_INPUT",
            "identity binding precedes numeric admission"
        );
        p.experts[0].up.global_dequant = 1.0;
        let repaired = pack(&p).unwrap();
        prepare(&repaired, 2).unwrap();
        assert_eq!(prepare(&new, 2).unwrap_err().code, "SM70_SCALE_RANGE");
    }
}
