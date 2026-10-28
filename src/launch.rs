//! CPU preparation for the existing QPN2-MoE FFI. No device pointers or CUDA calls.
use crate::{
    invalid,
    memory::{add, mul},
    moe,
    resident::ResidentStack,
    routing::RankDispatch,
    Error, Result,
};
use serde::{Deserialize, Serialize};
pub const MAX_WORKSPACE_BYTES: u64 = 128 * 1024 * 1024;
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Options {
    pub splitk13: u32,
    pub splitk2: u32,
    pub max_workspace_bytes: u64,
}
#[derive(Debug, Clone, Serialize)]
pub struct Projection {
    experts: usize,
    n: u64,
    k: u64,
    slots: usize,
    input_rows: usize,
    in_div: u32,
    splitk: u32,
    nacc: u32,
    codes_bytes: usize,
    scales_bytes: usize,
    gscales_count: usize,
    x_count: u64,
    y_count: u64,
}
#[derive(Debug, Clone, Serialize)]
pub struct Prepared<'a> {
    #[serde(skip)]
    weights: &'a ResidentStack,
    stamp: moe::Stamp,
    rank: u32,
    output_columns: usize,
    resident_order: Vec<u32>,
    source_tokens: Vec<usize>,
    completion_slots: Vec<moe::Slot>,
    block_expert: Vec<i32>,
    sorted_slots: Vec<i32>,
    ntpp: i32,
    projection13: Option<Projection>,
    projection2: Option<Projection>,
    workspace_bytes: u64,
    skip: bool,
    evidence_level: &'static str,
    cuda_executed: bool,
}
impl<'a> Prepared<'a> {
    pub fn output_columns(&self) -> usize {
        self.output_columns
    }

    pub fn weights(&self) -> &'a ResidentStack {
        self.weights
    }
    pub fn source_tokens(&self) -> &[usize] {
        &self.source_tokens
    }
    pub fn completion_slots(&self) -> &[moe::Slot] {
        &self.completion_slots
    }
    pub fn block_expert(&self) -> &[i32] {
        &self.block_expert
    }
    pub fn sorted_slots(&self) -> &[i32] {
        &self.sorted_slots
    }
    pub fn ntpp(&self) -> i32 {
        self.ntpp
    }
    pub fn workspace_bytes(&self) -> u64 {
        self.workspace_bytes
    }
    pub fn skip(&self) -> bool {
        self.skip
    }
    pub fn stamp(&self) -> &moe::Stamp {
        &self.stamp
    }
    /// Synchronous CPU validation through the exact existing C ABI.
    /// Opaque DEVICE declarations are checked arithmetically; success does not
    /// prove allocation/accessibility, upload equality, CUDA completion or values.
    /// Metadata is borrowed from this immutable prepared plan; no weight copies.
    /// Four activation spans are distinct and disjoint from both stages weights
    /// and metadata. Buffer reuse/in-place activation requires separate qualification.
    #[cfg(feature = "c-abi-validation")]
    pub fn validate_c(
        &self,
        current: &moe::Stamp,
        gate_up: &vf_qpn2_ffi::DeviceBuffers,
        down: &vf_qpn2_ffi::DeviceBuffers,
    ) -> Result<()> {
        if current != &self.stamp {
            return Err(Error::new(
                "STALE_EXECUTION",
                "prepared plan differs from current full Stamp",
            ));
        }
        if self.skip {
            return Ok(());
        }
        let (a, b) = self.projections();
        let a = a.ok_or_else(|| invalid("nonempty prepared plan lacks gate/up projection"))?;
        let b = b.ok_or_else(|| invalid("nonempty prepared plan lacks down projection"))?;
        for (name, projection, buffers) in [("gate/up", a, gate_up), ("down", b, down)] {
            let cast =
                |x: u64| u32::try_from(x).map_err(|_| invalid("projection exceeds C u32 contract"));
            let dimensions = vf_qpn2_ffi::Dimensions {
                experts: u32::try_from(projection.experts)
                    .map_err(|_| invalid("expert count exceeds C u32"))?,
                n: cast(projection.n)?,
                k: cast(projection.k)?,
                slots: u32::try_from(projection.slots)
                    .map_err(|_| invalid("slots exceed C u32"))?,
                input_rows: u32::try_from(projection.input_rows)
                    .map_err(|_| invalid("input rows exceed C u32"))?,
                in_div: projection.in_div,
                splitk: projection.splitk,
                nacc: projection.nacc,
            };
            vf_qpn2_ffi::validate(
                dimensions,
                buffers,
                vf_qpn2_ffi::HostMetadata {
                    block_expert: self.block_expert(),
                    sorted_slots: self.sorted_slots(),
                    ntpp: self.ntpp(),
                },
            )
            .map_err(|status| {
                Error::new(
                    "C_ABI_VALIDATION",
                    format!("{name} rejected by C CPU validator: {}", status.code()),
                )
            })?;
        }
        // The existing workspace reserves four distinct FP16 buffers. Reject
        // unqualified reuse and cross-stage corruption of persistent weights/tables.
        let byte_count = |count: u64, width: u64| -> Result<usize> {
            usize::try_from(mul(count, width)?).map_err(|_| invalid("layer span exceeds usize"))
        };
        let activation = [
            (gate_up.x.address, byte_count(a.x_count, 2)?),
            (gate_up.y.address, byte_count(a.y_count, 2)?),
            (down.x.address, byte_count(b.x_count, 2)?),
            (down.y.address, byte_count(b.y_count, 2)?),
        ];
        let protected = [
            (gate_up.codes.address, a.codes_bytes),
            (gate_up.scales.address, a.scales_bytes),
            (
                gate_up.gscales.address,
                byte_count(a.gscales_count as u64, 4)?,
            ),
            (down.codes.address, b.codes_bytes),
            (down.scales.address, b.scales_bytes),
            (down.gscales.address, byte_count(b.gscales_count as u64, 4)?),
            (
                gate_up.block_expert.address,
                byte_count(self.block_expert.len() as u64, 4)?,
            ),
            (
                gate_up.sorted_slots.address,
                byte_count(self.sorted_slots.len() as u64, 4)?,
            ),
            (gate_up.ntpp.address, 4),
            (
                down.block_expert.address,
                byte_count(self.block_expert.len() as u64, 4)?,
            ),
            (
                down.sorted_slots.address,
                byte_count(self.sorted_slots.len() as u64, 4)?,
            ),
            (down.ntpp.address, 4),
            (
                self.block_expert.as_ptr() as usize,
                byte_count(self.block_expert.len() as u64, 4)?,
            ),
            (
                self.sorted_slots.as_ptr() as usize,
                byte_count(self.sorted_slots.len() as u64, 4)?,
            ),
        ];
        let overlaps = |left: (usize, usize), right: (usize, usize)| -> Result<bool> {
            let end = |span: (usize, usize)| {
                span.0.checked_add(span.1).ok_or_else(|| {
                    Error::new("C_ABI_VALIDATION", "layer span wraps address space: -4")
                })
            };
            Ok(left.0 < end(right)? && right.0 < end(left)?)
        };
        for (i, span) in activation.iter().copied().enumerate() {
            for other in activation[..i].iter().chain(&protected).copied() {
                if overlaps(span, other)? {
                    return Err(Error::new("C_ABI_VALIDATION",
                        "four-buffer layer workspace overlaps activation or persistent weight/metadata span: -4"));
                }
            }
        }

        Ok(())
    }

    pub fn projections(&self) -> (Option<&Projection>, Option<&Projection>) {
        (self.projection13.as_ref(), self.projection2.as_ref())
    }
}
fn projection(w: &ResidentStack, slots: usize, down: bool, splitk: u32) -> Result<Projection> {
    let (experts, n13, k13, n2, k2) = w.shape();
    let (n, k, codes, scales, globals) = if down {
        (n2, k2, w.codes2(), w.scales2(), w.gscales2())
    } else {
        (n13, k13, w.codes13(), w.scales13(), w.gscales13())
    };
    if n == 0 || n > 65536 || k == 0 || k > 65536 || !(k / 16).is_multiple_of(u64::from(splitk)) {
        return Err(invalid(
            "projection dimensions or splitk outside compiled FFI contract",
        ));
    }
    // Exact numeric admission was established at owned load. All buffers exposed
    // by ResidentStack are immutable borrows; each reconstructed owner is readmitted.
    w.sm70_domain(down)?;
    Ok(Projection {
        experts,
        n,
        k,
        slots,
        input_rows: slots,
        in_div: 1,
        splitk,
        nacc: 1,
        codes_bytes: codes.len(),
        scales_bytes: scales.len(),
        gscales_count: globals.len(),
        x_count: mul(slots as u64, k)?,
        y_count: mul(slots as u64, n)?,
    })
}
/// Bind the FULL caller-supplied current Stamp, then compact rows in original
/// token/slot order and group those compact indices into expert-major blocks8.
/// Padding uses sentinel slots (never a valid compact index). No weight copies.
/// Workspace is a conservative FP16 allocation plan for compact x, h1, activation,
/// h2 and one shared device metadata set. Excludes resident payload and host/RSS.
pub fn prepare<'a>(
    w: &'a ResidentStack,
    d: &RankDispatch,
    current: &moe::Stamp,
    o: &Options,
) -> Result<Prepared<'a>> {
    if o.max_workspace_bytes == 0
        || o.max_workspace_bytes > MAX_WORKSPACE_BYTES
        || ![2, 4, 8, 16].contains(&o.splitk13)
        || ![2, 4, 8, 16].contains(&o.splitk2)
    {
        return Err(invalid("launch options outside bounded compiled contract"));
    }
    if &d.stamp != current {
        return Err(Error::new(
            "STALE_EXECUTION",
            "rank dispatch differs from current full Stamp",
        ));
    }
    w.bind_dispatch(d)?;
    let slots = d.slots.len();
    let mut p = Prepared {
        weights: w,
        stamp: current.clone(),
        rank: d.rank,
        output_columns: d.hidden,
        resident_order: w.resident_order().to_vec(),
        source_tokens: Vec::new(),
        completion_slots: Vec::new(),
        block_expert: Vec::new(),
        sorted_slots: Vec::new(),
        ntpp: 0,
        projection13: None,
        projection2: None,
        workspace_bytes: 0,
        skip: slots == 0,
        evidence_level: "CPU_FFI_LAUNCH_METADATA",
        cuda_executed: false,
    };
    if slots == 0 {
        return Ok(p);
    }
    let (e, _, _, _, _) = w.shape();
    let mut counts = vec![0usize; e];
    for s in &d.slots {
        counts[s.physical_expert] += 1;
    }
    let blocks = counts.iter().map(|&n| n.div_ceil(8)).sum::<usize>();
    if blocks == 0 || blocks > 65535 {
        return Err(invalid("launch block count outside FFI contract"));
    }
    let a = projection(w, slots, false, o.splitk13)?;
    let b = projection(w, slots, true, o.splitk2)?;
    let activation_bytes = mul(
        add(add(a.x_count, a.y_count)?, add(b.x_count, b.y_count)?)?,
        2,
    )?;
    let metadata_bytes = add(mul(blocks as u64, 36)?, 4)?;
    let total = add(activation_bytes, metadata_bytes)?;
    if total > o.max_workspace_bytes {
        return Err(Error::new(
            "CAPACITY_EXCEEDED",
            "launch FP16 workspace and metadata exceed requested cap",
        ));
    }
    // Metadata bounded by routing's slots/expert limits; activation buffers are NOT allocated here.
    let mut rows = d.slots.iter().collect::<Vec<_>>();
    rows.sort_unstable_by_key(|r| (r.slot.token, r.slot.slot));
    let mut groups = vec![Vec::<i32>::new(); e];
    for (index, r) in rows.into_iter().enumerate() {
        p.source_tokens.push(r.slot.token);
        p.completion_slots.push(r.slot.clone());
        groups[r.physical_expert].push(index as i32);
    }
    let sentinel = slots as i32;
    for (expert, group) in groups.into_iter().enumerate() {
        for chunk in group.chunks(8) {
            p.block_expert.push(expert as i32);
            p.sorted_slots.extend_from_slice(chunk);
            p.sorted_slots
                .extend(std::iter::repeat_n(sentinel, 8 - chunk.len()));
        }
    }
    p.ntpp = p.sorted_slots.len() as i32;
    p.workspace_bytes = total;
    p.projection13 = Some(a);
    p.projection2 = Some(b);
    Ok(p)
}
