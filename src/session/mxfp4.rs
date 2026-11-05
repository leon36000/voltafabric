//! The actual persistent session, with an explicit CPU R9 producer and FP32 receipts.
//! Authentication is supplied by the existing Endpoint (local process or SSH).
//! CPU remains the default. A separate native GPU executor is explicitly negotiated; no fallback or retry.
use super::*;
use crate::deepseek_v41::{self, r9_fp32, Mxfp4ExpertShard};
use crate::preweighted as arithmetic;

mod series;
pub use series::{run_series, Batch, BatchReport, Report, SeriesInput, WorkerSource};

pub const CAPABILITY: &str = "cpu_mxfp4_r9_preweighted_fp32_v1";
pub const GPU_CAPABILITY: &str = "gpu_mxfp4_r13_preweighted_fp32_v1";
/// Lossless finite BF16 input wire encoding, independent of the arithmetic producer.
pub const BF16_ACTIVATION_CAPABILITY: &str = "mxfp4_exec_bf16_exact_raw_v1";

/// Only the dedicated GPU binary implements this seam. CPU clients/Artifacts
/// retain their existing Send/Sync properties; device owners stay on the serve thread.
pub trait Executor {
    /// Immutable, verified native identity for the whole session; never changes after HELLO.
    fn producer(&self) -> Producer;
    /// Atomic LOAD: a nonfatal error leaves no owners resident. Roll back staged
    /// owners before returning; any unknown allocation/cleanup outcome is fatal.
    fn load(&mut self, artifact: &Artifact) -> Result<()>;
    fn forward(&mut self, expert: u32, input: &[f32], coefficient: f32) -> Result<Vec<f32>>;
    /// Release every owner before success. Empty/repeated eviction is a no-op.
    /// Any cleanup refusal or uncertain result must latch fatal, never be retried.
    fn evict(&mut self) -> Result<()>;
    fn fatal(&self) -> bool;
}
pub const MAX_ARTIFACT_BYTES: usize = deepseek_v41::MAX_MXFP4_EXPERT_ARTIFACT_BYTES as usize;
const MAX_EXEC_MACS: u64 = 100_000_000; // same explicit per-rank budget as QPN2 CPU EXEC

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expert {
    pub expert: u32,
    pub local: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub artifact_sha256: String,
    pub model_revision: String,
    pub rank: u32,
    pub layer: u32,
    pub epoch: u64,
    /// Physical resident order, with logical expert and rank-local IDs kept distinct.
    pub experts: Vec<Expert>,
}

/// The one compressed shard owner; its planes are neither copied nor decompressed.
pub struct Artifact {
    shard: Arc<Mxfp4ExpertShard>,
    binding: Binding,
    hidden: usize,
}

impl Artifact {
    pub fn new(bytes: Vec<u8>, binding: Binding) -> Result<Self> {
        let shard = Mxfp4ExpertShard::from_safetensors_shard_artifact(
            bytes,
            &binding.artifact_sha256,
            MAX_ARTIFACT_BYTES,
        )?;
        Self::from_shard(shard, binding)
    }

    pub fn from_bounded_reader(reader: impl Read, binding: Binding) -> Result<Self> {
        let shard = Mxfp4ExpertShard::from_bounded_reader(
            reader,
            &binding.artifact_sha256,
            MAX_ARTIFACT_BYTES,
        )?;
        Self::from_shard(shard, binding)
    }

    /// The series client constrains new owners by its remaining aggregate budget.
    /// A single sentinel byte detects overflow; this is not a process-RSS bound.
    fn from_client_bounded_reader(
        reader: impl Read,
        binding: Binding,
        remaining_bytes: usize,
    ) -> Result<Self> {
        if remaining_bytes == 0 {
            return Err(series::client_artifact_budget_error());
        }
        let max_bytes = remaining_bytes.min(MAX_ARTIFACT_BYTES);
        let mut source = reader.take((max_bytes + 1) as u64);
        let shard =
            Mxfp4ExpertShard::from_bounded_reader(&mut source, &binding.artifact_sha256, max_bytes);
        if remaining_bytes < MAX_ARTIFACT_BYTES && source.limit() == 0 {
            return Err(series::client_artifact_budget_error());
        }
        Self::from_shard(shard?, binding)
    }

    fn from_shard(shard: Mxfp4ExpertShard, binding: Binding) -> Result<Self> {
        Self::from_shared_shard(Arc::new(shard), binding)
    }

    /// Reuse only this immutable admitted snapshot; identities are validated anew.
    fn sharing_binding(&self, binding: Binding) -> Result<Self> {
        Self::from_shared_shard(Arc::clone(&self.shard), binding)
    }

    fn from_shared_shard(shard: Arc<Mxfp4ExpertShard>, binding: Binding) -> Result<Self> {
        if binding.artifact_sha256 != shard.artifact_sha256() {
            return Err(invalid("MXFP4 shared snapshot digest differs from binding"));
        }
        if binding.model_revision.trim().is_empty()
            || binding.model_revision.len() > 256
            || binding.experts.is_empty()
            || binding.experts.len() > 1024
            || shard.expert_count() != binding.experts.len()
        {
            return Err(invalid(
                "MXFP4 binding must name every resident expert of one layer",
            ));
        }
        let mut ids = BTreeSet::new();
        let mut locals = BTreeSet::new();
        let mut hidden = None;
        for item in &binding.experts {
            let e = shard.expert(binding.layer, item.expert)?;
            if !ids.insert(item.expert)
                || !locals.insert(item.local)
                || hidden.is_some_and(|h| h != e.w1().cols() as usize)
            {
                return Err(invalid("MXFP4 resident IDs or hidden shapes differ"));
            }
            hidden = Some(e.w1().cols() as usize);
        }
        Ok(Self {
            shard,
            binding,
            hidden: hidden.unwrap(),
        })
    }

    pub fn binding(&self) -> &Binding {
        &self.binding
    }
    pub fn expert(&self, expert: u32) -> Result<&deepseek_v41::Mxfp4Expert> {
        self.shard.expert(self.binding.layer, expert)
    }
    pub fn bytes(&self) -> &[u8] {
        self.shard.artifact_bytes()
    }

    pub(super) fn rebind(&mut self, epoch: u64) -> usize {
        self.binding.epoch = epoch;
        self.binding.experts.len()
    }

    fn validate_dispatch(&self, d: &routing::RankDispatch) -> Result<()> {
        validate_binding(&self.binding, self.hidden, d)?;
        let mut macs = 0_u64;
        for routed in &d.slots {
            let e = self.shard.expert(self.binding.layer, routed.slot.expert)?;
            let work = e
                .w1()
                .rows()
                .checked_mul(e.w1().cols())
                .and_then(|n| n.checked_mul(3))
                .filter(|&n| n <= deepseek_v41::MAX_DEEPSEEK_EXPERT_REFERENCE_MACS);
            macs = work
                .and_then(|n| n.checked_add(macs))
                .filter(|&n| n <= MAX_EXEC_MACS)
                .ok_or_else(|| invalid("MXFP4 EXEC exceeds the 100000000-MAC CPU rank budget"))?;
        }
        Ok(())
    }
}

fn validate_binding(binding: &Binding, hidden: usize, d: &routing::RankDispatch) -> Result<()> {
    let order: Vec<_> = binding.experts.iter().map(|e| e.local).collect();
    if d.rank != binding.rank
        || d.stamp.model_revision != binding.model_revision
        || d.stamp.layer != binding.layer
        || d.stamp.epoch != binding.epoch
        || d.hidden != hidden
        || d.tokens > 4096
        || d.slots.is_empty()
        || d.slots.len() > 262144
        || d.resident_order != order
        || d.slots
            .len()
            .checked_mul(hidden)
            .is_none_or(|n| n > crate::completion::MAX_OUTPUT_VALUES)
    {
        return Err(invalid(
            "MXFP4 dispatch differs from its resident binding or bounds",
        ));
    }
    let mut seen = BTreeSet::new();
    let mut logical_experts = BTreeSet::new();
    for routed in &d.slots {
        let slot = &routed.slot;
        if slot.token >= d.tokens
            || slot.slot >= 64
            || slot.rank != d.rank
            || !seen.insert((slot.token, slot.slot))
            || !logical_experts.insert((slot.token, slot.expert))
            || !slot.weight.is_finite()
            || slot.weight < 0.0
            || f64::from(slot.weight as f32).to_bits() != slot.weight.to_bits()
            || binding
                .experts
                .get(routed.physical_expert)
                .is_none_or(|e| e.expert != slot.expert || e.local != slot.local)
        {
            return Err(invalid(
                "MXFP4 slot identity, coefficient or physical index differs",
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenInput {
    pub token: usize,
    /// Binary32 bit identity; wire encoding is negotiated separately. Every value
    /// must already be finite and BF16-exact, including signed zero/subnormals.
    pub value_bits: Vec<u32>,
}

fn encode_bf16_inputs(inputs: &[TokenInput]) -> Result<Vec<u8>> {
    let count = inputs
        .iter()
        .try_fold(0_usize, |n, row| n.checked_add(row.value_bits.len()));
    if inputs.len() > 4096
        || inputs.iter().any(|row| row.value_bits.len() > 65536)
        || count.is_none_or(|n| n > MAX_RAW_ACTIVATION_BYTES / 4)
    {
        return Err(invalid("MXFP4 BF16 decoded activation bound exceeded"));
    }
    // Validate before allocating/publishing. Never cast f32 to f16 or round values.
    if inputs
        .iter()
        .flat_map(|row| &row.value_bits)
        .any(|&bits| bits & 0xffff != 0 || bits & 0x7f80_0000 == 0x7f80_0000)
    {
        return Err(invalid(
            "MXFP4 BF16 wire requires finite exactly-BF16 inputs",
        ));
    }
    let mut body = vec![0_u8; count.unwrap() * 2];
    let mut offset = 0;
    for row in inputs {
        let end = offset + row.value_bits.len() * 2;
        for (dst, bits) in body[offset..end].chunks_exact_mut(2).zip(&row.value_bits) {
            dst.copy_from_slice(&((bits >> 16) as u16).to_le_bytes());
        }
        offset = end;
    }
    Ok(body)
}

#[cfg(test)]
mod bf16_wire_tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn all_finite_patterns_roundtrip_without_numeric_conversion() {
        let bits: Vec<_> = (0_u32..=65535)
            .map(|v| v << 16)
            .filter(|v| f32::from_bits(*v).is_finite())
            .collect();
        assert_eq!(bits.len(), 65280);
        let body = encode_bf16_inputs(&[TokenInput {
            token: 31,
            value_bits: bits.clone(),
        }])
        .unwrap();
        assert_eq!(body.len(), bits.len() * 2);
        let rows = read_inputs(
            &mut Cursor::new(body),
            0,
            (bits.len() * 2) as u64,
            vec![RawTokenInputMeta {
                token: 31,
                values: bits.len() as u64,
            }],
            true,
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].token, 31);
        assert_eq!(rows[0].value_bits, bits);
        assert!(rows[0].value_bits.contains(&0x8000_0000));
        assert!(rows[0].value_bits.contains(&0x0001_0000));
    }

    #[test]
    fn nonfinite_and_inexact_values_are_rejected_atomically() {
        for high in 0_u32..=65535 {
            let bits = high << 16;
            if !f32::from_bits(bits).is_finite() {
                assert!(encode_bf16_inputs(&[TokenInput {
                    token: 0,
                    value_bits: vec![0, bits]
                }])
                .is_err());
                let body = [0, 0, high as u8, (high >> 8) as u8];
                assert!(read_inputs(
                    &mut Cursor::new(body),
                    0,
                    4,
                    vec![RawTokenInputMeta {
                        token: 0,
                        values: 2
                    }],
                    true
                )
                .is_err());
            }
        }
        for bits in [1_u32, 0x3f80_0001, 0x7f7f_ffff, 0x8000_0001] {
            assert!(encode_bf16_inputs(&[TokenInput {
                token: 0,
                value_bits: vec![0, bits]
            }])
            .is_err());
        }
        let body = encode_bf16_inputs(&[TokenInput {
            token: 1,
            value_bits: vec![0x3f80_0000, 0x8001_0000],
        }])
        .unwrap();
        assert_eq!(body, [0x80, 0x3f, 1, 0x80]);
    }

    struct NeverRead;
    impl Read for NeverRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            panic!("invalid metadata read a body")
        }
    }
    impl BufRead for NeverRead {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            panic!("invalid metadata read a body")
        }
        fn consume(&mut self, _: usize) {
            panic!("invalid metadata consumed a body")
        }
    }

    #[test]
    fn original_decoded_cap_and_metadata_reject_before_any_body_read() {
        // A BF16 body of this size fits the raw wire cap but its decoded values
        // exceed the original cap. Check before read/materialization, not EXEC.
        let metadata: Vec<_> = (0..64)
            .map(|token| RawTokenInputMeta {
                token,
                values: 65536,
            })
            .collect();
        let error = read_inputs(&mut NeverRead, 0, 64 * 65536 * 2, metadata, true).unwrap_err();
        assert!(error.message.contains("decoded activation bound"));
        for (declared, values) in [(3, 1), (2, 2), (2, u64::MAX)] {
            assert!(read_inputs(
                &mut NeverRead,
                0,
                declared,
                vec![RawTokenInputMeta { token: 0, values }],
                true
            )
            .is_err());
        }
        assert!(read_inputs(
            &mut NeverRead,
            0,
            0,
            (0..4097)
                .map(|token| RawTokenInputMeta { token, values: 0 })
                .collect(),
            true
        )
        .is_err());
        let error = read_inputs(
            &mut Cursor::new([0_u8]),
            0,
            2,
            vec![RawTokenInputMeta {
                token: 0,
                values: 1,
            }],
            true,
        )
        .unwrap_err();
        // The shared raw-body reader uses this existing code for input truncation.
        assert_eq!(error.code, "ARTIFACT_TRUNCATED");
    }
}

fn checked_inputs(
    d: &routing::RankDispatch,
    inputs: Vec<TokenInput>,
) -> Result<BTreeMap<usize, Vec<f64>>> {
    let count = inputs
        .iter()
        .try_fold(0_usize, |n, row| n.checked_add(row.value_bits.len()));
    if inputs.len() > 4096
        || inputs
            .iter()
            .any(|row| row.token >= d.tokens || row.value_bits.len() != d.hidden)
        || count.is_none_or(|n| n > MAX_RAW_ACTIVATION_BYTES / 4)
    {
        return Err(invalid(
            "MXFP4 compact inputs exceed the shape or byte bound",
        ));
    }
    let mut rows = Vec::with_capacity(inputs.len());
    for row in inputs {
        let mut value = Vec::with_capacity(row.value_bits.len());
        for bits in row.value_bits {
            let x = f32::from_bits(bits);
            if deepseek_v41::cast_bf16_rne_reference(x)?.to_bits() != bits {
                return Err(invalid("MXFP4 EXEC inputs must already be exactly BF16"));
            }
            value.push(f64::from(x));
        }
        rows.push(crate::process::TokenInput {
            token: row.token,
            value,
        });
    }
    token_inputs(d, rows)
}

fn check_contract(contract: &arithmetic::Contract) -> Result<()> {
    if contract.format != arithmetic::Format::R9Mxfp4PreweightedFp32V1
        || !sha256::is_hex_digest(&contract.sha256)
    {
        return Err(invalid(
            "MXFP4 EXEC requires the explicit R9 arithmetic contract",
        ));
    }
    Ok(())
}

/// Read exactly one declared body. This LOAD alone uses the existing 64-MiB MXFP4
/// owner bound (a real three-plane expert is 18.8 MB); EXEC/reply frames keep 16 MiB.
pub(super) fn read_artifact(input: &mut impl Read, declared: u64) -> Result<Vec<u8>> {
    let length = usize::try_from(declared)
        .ok()
        .filter(|&n| n > 0 && n <= MAX_ARTIFACT_BYTES)
        .ok_or_else(|| {
            Error::new(
                "MESSAGE_TOO_LARGE",
                "MXFP4 LOAD exceeds its 64-MiB owner bound",
            )
        })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|e| invalid(format!("MXFP4 LOAD allocation: {e}")))?;
    input
        .take(declared)
        .read_to_end(&mut bytes)
        .map_err(|e| Error::new("WORKER_PROTOCOL", format!("MXFP4 LOAD body: {e}")))?;
    if bytes.len() != length {
        return Err(Error::new("WORKER_PROTOCOL", "truncated MXFP4 LOAD"));
    }
    Ok(bytes)
}

pub(super) fn load(
    resident: &mut Option<Resident>,
    rank: u32,
    binding: Binding,
    bytes: Vec<u8>,
    executor: &mut Option<&mut dyn Executor>,
) -> Reply {
    let result = (|| {
        if resident.is_some() {
            return Err(Error::new(
                "ARTIFACT_LOADED",
                "evict the resident before another LOAD",
            ));
        }
        if binding.rank != rank {
            return Err(invalid("MXFP4 LOAD names a foreign worker rank"));
        }
        let artifact = Artifact::new(bytes, binding.clone())?;
        if let Some(provider) = executor.as_deref_mut() {
            provider.load(&artifact)?;
        }
        let experts = binding.experts.len();
        *resident = Some(Resident {
            digest: binding.artifact_sha256.clone(),
            model_revision: binding.model_revision,
            layer: binding.layer,
            epoch: binding.epoch,
            resident_order: binding.experts.iter().map(|e| e.local).collect(),
            weights: Weights::Mxfp4(Box::new(artifact)),
        });
        Ok(Reply::Loaded {
            artifact_sha256: binding.artifact_sha256,
            epoch: binding.epoch,
            experts,
            transferred: true,
        })
    })();
    result.unwrap_or_else(|error| Reply::Refused { error })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowMeta {
    pub token: usize,
    pub slot: usize,
    pub expert: u32,
    pub local: u32,
    pub weight_bits: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Producer {
    #[default]
    CpuR9Fp32WarpReferenceV1,
    GpuR13ResidentFp32V1 {
        library_sha256: String,
    },
}
impl Producer {
    pub fn validate(&self) -> Result<()> {
        if let Self::GpuR13ResidentFp32V1 { library_sha256 } = self {
            if !sha256::is_hex_digest(library_sha256)
                || library_sha256.bytes().any(|b| b.is_ascii_uppercase())
            {
                return Err(invalid("GPU producer requires a canonical library SHA-256"));
            }
        }
        Ok(())
    }
    pub fn capability(&self) -> String {
        match self {
            Self::CpuR9Fp32WarpReferenceV1 => CAPABILITY.into(),
            Self::GpuR13ResidentFp32V1 { library_sha256 } => {
                format!("{GPU_CAPABILITY}:{library_sha256}")
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionHeader {
    pub producer: Producer,
    pub artifact_sha256: String,
    pub contract: arithmetic::Contract,
    pub stamp: moe::Stamp,
    pub completion_stamp: moe::Stamp,
    pub rank: u32,
    pub attempt: u64,
    pub hidden: usize,
    pub rows: Vec<RowMeta>,
}

/// Shared with the reader only while one typed EXEC is pending. Identity is checked
/// before allocating or consuming a raw body, not merely when collecting it later.
pub(super) struct Admission {
    producer: Producer,
    stamp: moe::Stamp,
    rank: u32,
    attempt: u64,
    artifact_sha256: String,
    contract: arithmetic::Contract,
    hidden: usize,
    slots: usize,
}

pub(super) fn read_completion(
    input: &mut impl Read,
    header: CompletionHeader,
    values: u64,
    allowed: &Mutex<Option<Admission>>,
    header_bytes: usize,
) -> Result<Reply> {
    {
        let mut guard = allowed
            .lock()
            .map_err(|_| invalid("MXFP4 reader admission poisoned"))?;
        let admitted = guard.as_mut().is_some_and(|a| {
            let valid = header.producer == a.producer
                && header.stamp == a.stamp
                && header.completion_stamp == a.stamp
                && header.rank == a.rank
                && header.attempt == a.attempt
                && header.artifact_sha256 == a.artifact_sha256
                && header.contract == a.contract
                && header.hidden == a.hidden
                && !header.rows.is_empty()
                && header.rows.len() <= a.slots;
            if valid {
                // Consume the total row budget before reading a body. Per-frame bounds
                // alone let a fast corrupt peer queue arbitrarily many bounded bodies.
                a.slots -= header.rows.len();
            }
            valid
        });
        if !admitted {
            return Err(Error::new(
                "INVALID_COMPLETION",
                "unsolicited or foreign preweighted FP32 body",
            ));
        }
    }
    check_contract(&header.contract)?;
    let count = header
        .rows
        .len()
        .checked_mul(header.hidden)
        .filter(|&n| n > 0 && n <= crate::completion::MAX_OUTPUT_VALUES)
        .ok_or_else(|| invalid("MXFP4 completion shape outside bound"))?;
    if header.rows.len() > 262144
        || header.hidden == 0
        || header.hidden > 65536
        || values != count as u64
        || !sha256::is_hex_digest(&header.artifact_sha256)
    {
        return Err(invalid(
            "MXFP4 completion header differs from its declared body",
        ));
    }
    let length = count
        .checked_mul(4)
        .filter(|&n| {
            header_bytes
                .checked_add(1)
                .and_then(|h| h.checked_add(n))
                .is_some_and(|n| n <= MAX_MESSAGE_BYTES)
        })
        .ok_or_else(|| Error::new("MESSAGE_TOO_LARGE", "MXFP4 completion frame exceeds 16 MiB"))?;
    let mut bytes = vec![0; length];
    input
        .read_exact(&mut bytes)
        .map_err(|e| Error::new("WORKER_PROTOCOL", format!("truncated FP32 completion: {e}")))?;
    let bits = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    Ok(Reply::PreweightedFp32 { header, bits })
}

pub(super) fn read_inputs(
    input: &mut impl BufRead,
    header_bytes: usize,
    declared: u64,
    metadata: Vec<RawTokenInputMeta>,
    bf16_activations: bool,
) -> Result<Vec<TokenInput>> {
    if metadata.len() > 4096 || metadata.iter().any(|m| m.values > 65536) {
        return Err(invalid("MXFP4 activation metadata exceeds its bound"));
    }
    let count = metadata
        .iter()
        .try_fold(0_u64, |n, m| n.checked_add(m.values))
        .filter(|&n| n <= (MAX_RAW_ACTIVATION_BYTES / 4) as u64)
        .ok_or_else(|| invalid("MXFP4 decoded activation bound exceeded"))?;
    let width = if bf16_activations { 2 } else { 4 };
    // The shorter wire body must not double logical/decoded memory admission.
    // Check the original cap before reading a body or materializing any values.
    if count.checked_mul(width as u64) != Some(declared) {
        return Err(invalid("MXFP4 activation bytes differ from metadata"));
    }
    let body = read_raw_body(input, header_bytes, declared, true)?;
    if bf16_activations
        && body
            .chunks_exact(2)
            .any(|p| u16::from_le_bytes([p[0], p[1]]) & 0x7f80 == 0x7f80)
    {
        return Err(invalid("MXFP4 BF16 wire contains a nonfinite activation"));
    }
    let mut rows = Vec::with_capacity(metadata.len());
    let mut offset = 0;
    for item in metadata {
        let end = offset + item.values as usize * width;
        let value_bits = body[offset..end]
            .chunks_exact(width)
            .map(|b| {
                if bf16_activations {
                    u32::from(u16::from_le_bytes(b.try_into().unwrap())) << 16
                } else {
                    u32::from_le_bytes(b.try_into().unwrap())
                }
            })
            .collect();
        rows.push(TokenInput {
            token: item.token,
            value_bits,
        });
        offset = end;
    }
    Ok(rows)
}

struct Events<'a, W> {
    output: &'a mut W,
    artifact: &'a Artifact,
    producer: Producer,
    contract: &'a arithmetic::Contract,
    dispatch: &'a routing::RankDispatch,
}

impl<W: Write> EventSink for Events<'_, W> {
    fn send(&mut self, event: &WorkerEvent) -> Result<()> {
        if matches!(event.event, EventKind::Completion { .. }) {
            return Err(invalid("MXFP4 producer cannot emit a legacy completion"));
        }
        write_frame(
            self.output,
            &encode(&EventReply::Event { event })?,
            "MXFP4 control",
        )
    }
    fn send_completions(&mut self, batch: WorkerCompletions) -> Result<()> {
        if batch.rows.is_empty() {
            return Ok(());
        }
        let completion_stamp = batch.rows[0].stamp.clone();
        let expected: BTreeMap<_, _> = self
            .dispatch
            .slots
            .iter()
            .map(|r| ((r.slot.token, r.slot.slot), &r.slot))
            .collect();
        let mut header = CompletionHeader {
            producer: self.producer.clone(),
            artifact_sha256: self.artifact.binding.artifact_sha256.clone(),
            contract: self.contract.clone(),
            stamp: batch.stamp,
            completion_stamp,
            rank: batch.rank,
            attempt: batch.attempt,
            hidden: self.dispatch.hidden,
            rows: Vec::new(),
        };
        let mut body = Vec::new();
        for row in batch.rows {
            let slot = expected
                .get(&(row.token, row.slot))
                .ok_or_else(|| invalid("unknown producer slot"))?;
            if row.stamp != header.completion_stamp {
                return Err(invalid("mixed completion stamps"));
            }
            header.rows.push(RowMeta {
                token: row.token,
                slot: row.slot,
                expert: row.expert,
                local: row.local,
                weight_bits: (slot.weight as f32).to_bits(),
            });
            for x in row.value {
                let fp32 = x as f32;
                if !fp32.is_finite() || f64::from(fp32).to_bits() != x.to_bits() {
                    return Err(invalid("MXFP4 producer output is not finite exact FP32"));
                }
                body.extend_from_slice(&fp32.to_bits().to_le_bytes());
            }
        }
        let frame = encode(&Reply::PreweightedFp32Raw {
            header,
            values: (body.len() / 4) as u64,
        })?;
        if frame
            .len()
            .checked_add(body.len())
            .is_none_or(|n| n > MAX_MESSAGE_BYTES)
        {
            return Err(Error::new(
                "MESSAGE_TOO_LARGE",
                "MXFP4 result needs a bounded completion frame",
            ));
        }
        self.output
            .write_all(&frame)
            .and_then(|()| self.output.write_all(&body))
            .and_then(|()| self.output.flush())
            .map_err(|e| Error::new("WORKER_IO", format!("MXFP4 output: {e}")))
    }
}

#[allow(clippy::too_many_arguments)] // the existing EXEC envelope plus its explicit executor
pub(super) fn execute(
    resident: &Resident,
    dispatch: routing::RankDispatch,
    contract: arithmetic::Contract,
    attempt: u64,
    inputs: Vec<TokenInput>,
    fault: Fault,
    output: &mut impl Write,
    mut executor: Option<&mut dyn Executor>,
) -> Result<bool> {
    let stamp = dispatch.stamp.clone();
    let rank = dispatch.rank;
    let admitted = (|| {
        let Weights::Mxfp4(artifact) = &resident.weights else {
            return Err(Error::new(
                "ARITHMETIC_CONTRACT_MISMATCH",
                "resident is not an R9 MXFP4 producer",
            ));
        };
        check_contract(&contract)?;
        artifact.validate_dispatch(&dispatch)?;
        check_fault(&fault)?;
        let inputs = checked_inputs(&dispatch, inputs)?;
        Ok((
            artifact,
            Job {
                dispatch: dispatch.clone(),
                inputs,
                fault,
            },
        ))
    })();
    let (artifact, job) = match admitted {
        Ok(value) => value,
        Err(error) => {
            send_reply(
                output,
                &Reply::Event {
                    event: WorkerEvent {
                        stamp,
                        rank,
                        attempt,
                        event: EventKind::Rejected { error },
                    },
                },
            )?;
            return Ok(false);
        }
    };
    let mut events = Events {
        output,
        artifact,
        producer: executor
            .as_ref()
            .map_or(Producer::default(), |p| p.producer()),
        contract: &contract,
        dispatch: &dispatch,
    };
    execute_batch_with(job, &stamp, rank, attempt, &mut events, |d, inputs| {
        d.slots
            .iter()
            .map(|r| {
                let expert = artifact
                    .shard
                    .expert(artifact.binding.layer, r.slot.expert)?;
                let input: Vec<f32> = inputs[&r.slot.token].iter().map(|&x| x as f32).collect();
                let value = if let Some(provider) = executor.as_deref_mut() {
                    provider.forward(r.slot.expert, &input, r.slot.weight as f32)
                } else {
                    r9_fp32::forward(expert, &input, r.slot.weight as f32)
                }?;
                Ok(value.into_iter().map(f64::from).collect())
            })
            .collect()
    })?;
    Ok(true)
}

pub struct Pending {
    owner: Arc<()>,
    producer: Producer,
    dispatch: routing::RankDispatch,
    contract: arithmetic::Contract,
    artifact_sha256: String,
    receipt: ExecReceipt,
}

impl Session {
    /// Explicitly prefer the lossless BF16 activation encoding for future MXFP4
    /// requests. An old peer still uses raw32, chosen before any EXEC bytes.
    /// Does not alter an in-flight request or its failure/retry semantics.
    pub fn prefer_mxfp4_bf16_activations(&mut self) {
        self.prefer_mxfp4_bf16_activations = true;
    }

    /// Deadline-bounded writes borrow the immutable compressed owner. No thread can
    /// retain it after a timeout, even if a peer descendant keeps stdin open.
    fn write_preweighted_until(
        &mut self,
        header: Vec<u8>,
        body: Option<Arc<Vec<u8>>>,
        deadline: Instant,
    ) -> Result<()> {
        if !self.usable {
            return Err(Error::new("SESSION_UNUSABLE", "session must be reopened"));
        }
        if Instant::now() >= deadline {
            return Err(Error::new(
                "SESSION_DEADLINE",
                "deadline elapsed before transmission; nothing sent",
            ));
        }
        #[cfg(not(unix))]
        return Err(Error::new(
            "UNSUPPORTED_PLATFORM",
            "bounded MXFP4 writes require Unix; nothing sent",
        ));
        #[cfg(unix)]
        {
            let length = header
                .len()
                .checked_add(body.as_ref().map_or(0, |b| b.len()))
                .ok_or_else(|| invalid("MXFP4 write length overflow"))?;
            let Some(SessionInput::Socket(stream)) = self.stdin.as_mut() else {
                return Err(closed());
            };
            let result = (|| -> io::Result<()> {
                for part in [
                    header.as_slice(),
                    body.as_ref().map_or(&[][..], |b| b.as_slice()),
                ] {
                    let mut offset = 0;
                    while offset < part.len() {
                        let remaining = deadline.saturating_duration_since(Instant::now());
                        if remaining.is_zero() {
                            return Err(io::Error::new(
                                io::ErrorKind::TimedOut,
                                "MXFP4 write deadline",
                            ));
                        }
                        stream.set_write_timeout(Some(remaining))?;
                        match stream.write(&part[offset..]) {
                            Ok(0) => {
                                return Err(io::Error::new(
                                    io::ErrorKind::WriteZero,
                                    "MXFP4 write closed",
                                ))
                            }
                            Ok(n) => offset += n,
                            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                            Err(e)
                                if matches!(
                                    e.kind(),
                                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                                ) =>
                            {
                                // Socket timeout granularity may wake slightly early. Recompute
                                // the remaining total deadline rather than restarting it.
                                continue;
                            }
                            Err(e) => return Err(e),
                        }
                    }
                }
                stream.set_write_timeout(None)?;
                Ok(())
            })();
            match result {
                Ok(()) => {
                    self.traffic.bytes_sent += length as u64;
                    self.traffic.frames_sent += 1;
                    Ok(())
                }
                Err(error) => {
                    self.usable = false;
                    let _ = stream.shutdown(std::net::Shutdown::Both);
                    let _ = self.child.kill();
                    Err(Error::new(
                        if error.kind() == io::ErrorKind::TimedOut { "SESSION_WRITE_TIMEOUT" } else { "WORKER_IO" },
                        format!("MXFP4 write lost contact ({error}); partial admission is UNKNOWN, never replayed"),
                    ))
                }
            }
        }
    }

    pub fn connect_mxfp4(
        endpoint: &Endpoint,
        timeout: Duration,
        producer: &Producer,
    ) -> Result<Self> {
        producer.validate()?;
        match producer {
            Producer::CpuR9Fp32WarpReferenceV1 => Self::connect(endpoint, timeout),
            Producer::GpuR13ResidentFp32V1 { .. } => {
                let mut session =
                    Self::connect_with_capability(endpoint, timeout, &producer.capability())?;
                session.mxfp4_producer = producer.clone();
                Ok(session)
            }
        }
    }
    pub fn load_mxfp4(
        &mut self,
        artifact: &Artifact,
        epoch: u64,
        timeout: Duration,
    ) -> Result<LoadReceipt> {
        self.load_mxfp4_as(artifact, epoch, timeout, &Producer::default())
    }
    pub fn load_mxfp4_as(
        &mut self,
        artifact: &Artifact,
        epoch: u64,
        timeout: Duration,
        producer: &Producer,
    ) -> Result<LoadReceipt> {
        producer.validate()?;
        if self.artifact.is_some() && self.mxfp4_producer != *producer {
            return Err(invalid(
                "MXFP4 producer cannot change within a loaded session",
            ));
        }
        if !self
            .hello
            .capabilities
            .iter()
            .any(|c| c == &producer.capability())
        {
            return Err(Error::new(
                "CAPABILITY_MISSING",
                "worker lacks the requested MXFP4 producer; nothing sent",
            ));
        }
        if self.preweighted_inflight {
            return Err(invalid("an MXFP4 EXEC is still pending"));
        }
        if artifact.binding.rank != self.rank {
            return Err(invalid("foreign MXFP4 artifact rank"));
        }
        if epoch < artifact.binding.epoch {
            return Err(Error::new(
                "EPOCH_ROLLBACK",
                "MXFP4 epoch precedes artifact admission",
            ));
        }
        let mut binding = artifact.binding.clone();
        binding.epoch = epoch;
        let rebind = self
            .artifact
            .as_ref()
            .is_some_and(|(d, _)| *d == binding.artifact_sha256);
        if self.artifact.is_some() && !rebind {
            return Err(Error::new(
                "ARTIFACT_LOADED",
                "evict the loaded artifact before another LOAD",
            ));
        }
        if rebind
            && self
                .artifact
                .as_ref()
                .is_some_and(|(_, current)| epoch <= *current)
        {
            return Err(Error::new(
                "EPOCH_NOT_NEWER",
                "MXFP4 rebind requires an epoch newer than this session's",
            ));
        }
        if rebind
            && self.mxfp4_binding.as_ref().is_none_or(|(b, _)| {
                b.model_revision != binding.model_revision
                    || b.layer != binding.layer
                    || b.experts != binding.experts
            })
        {
            return Err(invalid("MXFP4 rebind cannot change the resident binding"));
        }
        let request = if rebind {
            Request::Load {
                artifact_sha256: binding.artifact_sha256.clone(),
                model_revision: binding.model_revision.clone(),
                layer: binding.layer,
                epoch,
                resident_order: binding.experts.iter().map(|e| e.local).collect(),
                model: None,
            }
        } else {
            Request::LoadMxfp4Raw {
                binding: binding.clone(),
                artifact_bytes: artifact.bytes().len() as u64,
            }
        };
        // The specific, prevalidated LOAD body has the existing MXFP4 owner's 64-MiB cap.
        // No legacy message limit is enlarged and subsequent EXECs carry no weight bytes.
        let header = encode(&request)?;
        let bytes = header.len() + if rebind { 0 } else { artifact.bytes().len() };
        let deadline = Instant::now() + timeout;
        let body = (!rebind).then(|| artifact.shard.shared_artifact_bytes());
        self.write_preweighted_until(header, body, deadline)?;
        match self.reply_until(deadline)? {
            Reply::Loaded {
                artifact_sha256,
                epoch: bound,
                experts,
                transferred,
            } if artifact_sha256 == binding.artifact_sha256
                && bound == epoch
                && experts == binding.experts.len()
                && transferred != rebind =>
            {
                if transferred {
                    self.traffic.loads_transferred += 1;
                    self.traffic.weight_bytes_sent += bytes as u64;
                } else {
                    self.traffic.rebinds += 1;
                }
                self.artifact = Some((artifact_sha256.clone(), epoch));
                self.resident = None;
                self.mxfp4_binding = Some((binding, artifact.hidden));
                self.mxfp4_producer = producer.clone();
                Ok(LoadReceipt {
                    rank: self.rank,
                    artifact_sha256,
                    epoch,
                    transferred,
                    request_bytes: bytes as u64,
                })
            }
            Reply::Refused { error } => {
                if !rebind {
                    self.usable = false;
                }
                Err(error)
            }
            _ => {
                self.usable = false;
                Err(Error::new(
                    "SESSION_PROTOCOL",
                    "unexpected MXFP4 LOAD reply",
                ))
            }
        }
    }

    /// Fully validate the original plan and collector before any request byte is sent.
    /// This read-only boundary lets a multi-rank adapter validate every Session before
    /// any rank receives EXEC. `start_preweighted_fp32_until` repeats the checks at the
    /// actual write boundary so callers cannot rely on this result after mutation.
    pub(crate) fn preflight_preweighted_fp32_until(
        &mut self,
        problem: &routing::RoutingProblem,
        collector: &arithmetic::Collector,
        attempt: u64,
        inputs: &[TokenInput],
        fault: &Fault,
        deadline: Instant,
    ) -> Result<routing::RankDispatch> {
        if !self.usable {
            return Err(Error::new("SESSION_UNUSABLE", "session must be reopened"));
        }
        if self.preweighted_inflight {
            return Err(invalid("an MXFP4 EXEC is still pending"));
        }
        match self.frames.try_recv() {
            Err(mpsc::TryRecvError::Empty) => {}
            _ => {
                self.usable = false;
                return Err(Error::new(
                    "SESSION_PROTOCOL",
                    "unsolicited reply before MXFP4 EXEC",
                ));
            }
        }
        if !self
            .hello
            .capabilities
            .iter()
            .any(|c| c == &self.mxfp4_producer.capability())
        {
            return Err(Error::new(
                "CAPABILITY_MISSING",
                "worker lacks the requested MXFP4 producer; nothing sent",
            ));
        }
        let expected = arithmetic::Collector::new(problem)?;
        if collector.contract() != expected.contract() || attempt == 0 {
            return Err(invalid(
                "MXFP4 plan differs from the collector or attempt is zero",
            ));
        }
        let dispatch = routing::dispatch(problem)?
            .ranks
            .into_iter()
            .find(|d| d.rank == self.rank)
            .ok_or_else(|| invalid("MXFP4 rank has no selected work"))?;
        let (binding, hidden) = self
            .mxfp4_binding
            .as_ref()
            .ok_or_else(|| invalid("no MXFP4 artifact is loaded"))?;
        validate_binding(binding, *hidden, &dispatch)?;
        check_fault(fault)?;
        checked_inputs(&dispatch, inputs.to_vec())?;
        if Instant::now() >= deadline {
            return Err(Error::new(
                "SESSION_DEADLINE",
                "deadline elapsed before transmission; nothing sent",
            ));
        }
        Ok(dispatch)
    }

    pub fn start_preweighted_fp32(
        &mut self,
        problem: &routing::RoutingProblem,
        collector: &arithmetic::Collector,
        attempt: u64,
        inputs: Vec<TokenInput>,
        fault: Fault,
    ) -> Result<Pending> {
        self.start_preweighted_fp32_until(
            problem,
            collector,
            attempt,
            inputs,
            fault,
            Instant::now() + Duration::from_secs(60),
        )
    }

    /// Caller-controlled deadline bounds both transmission and later collection.
    pub fn start_preweighted_fp32_until(
        &mut self,
        problem: &routing::RoutingProblem,
        collector: &arithmetic::Collector,
        attempt: u64,
        inputs: Vec<TokenInput>,
        fault: Fault,
        deadline: Instant,
    ) -> Result<Pending> {
        let dispatch = self.preflight_preweighted_fp32_until(
            problem, collector, attempt, &inputs, &fault, deadline,
        )?;
        let (binding, _) = self
            .mxfp4_binding
            .as_ref()
            .ok_or_else(|| invalid("no MXFP4 artifact is loaded"))?;
        let bf16_activations = self.prefer_mxfp4_bf16_activations
            && self
                .hello
                .capabilities
                .iter()
                .any(|c| c == BF16_ACTIVATION_CAPABILITY);
        let mut body = if bf16_activations {
            encode_bf16_inputs(&inputs)?
        } else {
            Vec::new()
        };
        let mut metadata = Vec::new();
        for row in inputs {
            metadata.push(RawTokenInputMeta {
                token: row.token,
                values: row.value_bits.len() as u64,
            });
            if !bf16_activations {
                for bits in row.value_bits {
                    body.extend_from_slice(&bits.to_le_bytes());
                }
            }
        }
        let contract = collector.contract().clone();
        let digest = binding.artifact_sha256.clone();
        let request = if bf16_activations {
            Request::ExecPreweightedBf16Raw {
                artifact_sha256: digest.clone(),
                dispatch: dispatch.clone(),
                contract: contract.clone(),
                attempt,
                inputs: metadata,
                activation_bytes: body.len() as u64,
                fault,
            }
        } else {
            Request::ExecPreweightedFp32Raw {
                artifact_sha256: digest.clone(),
                dispatch: dispatch.clone(),
                contract: contract.clone(),
                attempt,
                inputs: metadata,
                activation_bytes: body.len() as u64,
                fault,
            }
        };
        let header = encode(&request)?;
        let bytes = header
            .len()
            .checked_add(body.len())
            .filter(|&n| n <= MAX_MESSAGE_BYTES)
            .ok_or_else(|| Error::new("MESSAGE_TOO_LARGE", "MXFP4 EXEC exceeds 16 MiB"))?;
        let mut receipt = ExecReceipt {
            rank: self.rank,
            attempt,
            state: ReceiptState::Unknown,
            started: false,
            expected_slots: dispatch.slots.len(),
            received_slots: 0,
            request_bytes: bytes as u64,
            error: None,
        };
        *self
            .raw_preweighted
            .lock()
            .map_err(|_| invalid("MXFP4 admission poisoned"))? = Some(Admission {
            producer: self.mxfp4_producer.clone(),
            stamp: dispatch.stamp.clone(),
            rank: dispatch.rank,
            attempt,
            artifact_sha256: digest.clone(),
            contract: contract.clone(),
            hidden: dispatch.hidden,
            slots: dispatch.slots.len(),
        });
        self.preweighted_inflight = true;
        if let Err(error) = self.write_preweighted_until(header, Some(Arc::new(body)), deadline) {
            if error.code == "SESSION_DEADLINE" {
                self.preweighted_inflight = false;
                if let Ok(mut admission) = self.raw_preweighted.lock() {
                    *admission = None;
                }
                return Err(error);
            }
            receipt.error = Some(error);
        } else {
            self.traffic.execs += 1;
        }
        Ok(Pending {
            owner: self.preweighted_owner.clone(),
            producer: self.mxfp4_producer.clone(),
            dispatch,
            contract,
            artifact_sha256: digest,
            receipt,
        })
    }

    /// A completion is staged until Finished. Every unsuccessful terminal path expires
    /// the whole collector, including loss after all rows but before Finished.
    pub fn finish_preweighted_fp32(
        &mut self,
        pending: Pending,
        collector: &mut arithmetic::Collector,
        deadline: Instant,
    ) -> ExecReceipt {
        let receipt = self.receive_preweighted(pending, collector, deadline);
        self.preweighted_inflight = false;
        if let Ok(mut admission) = self.raw_preweighted.lock() {
            *admission = None;
        }
        if receipt.state != ReceiptState::Completed || receipt.error.is_some() {
            collector.expire();
        }
        receipt
    }

    fn receive_preweighted(
        &mut self,
        pending: Pending,
        collector: &mut arithmetic::Collector,
        deadline: Instant,
    ) -> ExecReceipt {
        let Pending {
            owner,
            producer,
            dispatch,
            contract,
            artifact_sha256,
            mut receipt,
        } = pending;
        if receipt.error.is_some() {
            return receipt;
        }
        let mut rows = Vec::new();
        let mut seen = BTreeSet::new();
        let expected: BTreeMap<_, _> = dispatch
            .slots
            .iter()
            .map(|r| ((r.slot.token, r.slot.slot), &r.slot))
            .collect();
        if !Arc::ptr_eq(&owner, &self.preweighted_owner) || collector.contract() != &contract {
            self.usable = false;
            receipt.error = Some(invalid(
                "MXFP4 pending belongs to another session/collector",
            ));
            return receipt;
        }
        loop {
            let reply = match self.reply_until(deadline) {
                Ok(reply) => reply,
                Err(error) => {
                    receipt.state = ReceiptState::Unknown;
                    receipt.error = Some(error);
                    return receipt;
                }
            };
            let step: Result<bool> = (|| match reply {
                Reply::Refused { error } if !receipt.started => {
                    self.usable = false;
                    receipt.state = ReceiptState::NotStarted;
                    receipt.error = Some(error);
                    Ok(true)
                }
                Reply::Event { event } => {
                    if event.stamp != dispatch.stamp
                        || event.rank != dispatch.rank
                        || event.attempt != receipt.attempt
                    {
                        return Err(Error::new(
                            "INVALID_COMPLETION",
                            "MXFP4 control has foreign full identity",
                        ));
                    }
                    match event.event {
                        EventKind::Started if !receipt.started => {
                            receipt.started = true;
                            receipt.state = ReceiptState::Started;
                            Ok(false)
                        }
                        EventKind::Rejected { error } if !receipt.started => {
                            receipt.state = ReceiptState::NotStarted;
                            receipt.error = Some(error);
                            Ok(true)
                        }
                        EventKind::Failed { error } if receipt.started => {
                            receipt.error = Some(error);
                            Ok(true)
                        }
                        EventKind::Finished { slots } if receipt.started => {
                            if slots != receipt.expected_slots || seen.len() != slots {
                                return Err(Error::new(
                                    "BATCH_INCOMPLETE",
                                    "MXFP4 rank finished without exact slot coverage",
                                ));
                            }
                            // Finished and exact coverage establish that this execution
                            // completed even if another rank already aborted collection.
                            receipt.state = ReceiptState::Completed;
                            for row in rows.drain(..) {
                                if let Err(error) = collector.accept(row) {
                                    receipt.error = Some(error);
                                    break;
                                }
                            }
                            Ok(true)
                        }
                        _ => Err(Error::new(
                            "WORKER_PROTOCOL",
                            "MXFP4 control outside its terminal lifecycle",
                        )),
                    }
                }
                Reply::PreweightedFp32 { header, bits } => {
                    if !receipt.started
                        || header.producer != producer
                        || header.contract != contract
                        || header.artifact_sha256 != artifact_sha256
                        || header.stamp != dispatch.stamp
                        || header.completion_stamp != dispatch.stamp
                        || header.rank != dispatch.rank
                        || header.attempt != receipt.attempt
                        || header.hidden != dispatch.hidden
                        || header.rows.len().checked_mul(dispatch.hidden) != Some(bits.len())
                    {
                        return Err(Error::new(
                            "INVALID_COMPLETION",
                            "MXFP4 completion arithmetic/artifact/full identity differs",
                        ));
                    }
                    for (meta, value) in header
                        .rows
                        .into_iter()
                        .zip(bits.chunks_exact(dispatch.hidden))
                    {
                        let Some(slot) = expected.get(&(meta.token, meta.slot)) else {
                            return Err(invalid("foreign MXFP4 slot"));
                        };
                        if slot.expert != meta.expert
                            || slot.local != meta.local
                            || (slot.weight as f32).to_bits() != meta.weight_bits
                            || !seen.insert((meta.token, meta.slot))
                            || value.iter().any(|v| v & 0x7f80_0000 == 0x7f80_0000)
                            || (meta.weight_bits & 0x7fff_ffff == 0
                                && value.iter().any(|v| v & 0x7fff_ffff != 0))
                        {
                            return Err(Error::new(
                                "INVALID_COMPLETION",
                                "MXFP4 slot duplicated, nonfinite or foreign",
                            ));
                        }
                        rows.push(arithmetic::Completion {
                            contract: contract.clone(),
                            attempt: None,
                            stamp: dispatch.stamp.clone(),
                            token: meta.token,
                            slot: meta.slot,
                            expert: meta.expert,
                            rank: header.rank,
                            local: meta.local,
                            weight_bits: meta.weight_bits,
                            value_bits: value.to_vec(),
                        });
                        receipt.received_slots += 1;
                    }
                    Ok(false)
                }
                _ => Err(Error::new(
                    "WORKER_PROTOCOL",
                    "unexpected legacy reply for preweighted FP32 EXEC",
                )),
            })();
            match step {
                Ok(true) => return receipt,
                Ok(false) => {}
                Err(error) => {
                    self.usable = false;
                    receipt.state = ReceiptState::Unknown;
                    receipt.error = Some(error);
                    return receipt;
                }
            }
        }
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    struct CountForwards(usize);
    impl Executor for CountForwards {
        fn producer(&self) -> Producer {
            Producer::CpuR9Fp32WarpReferenceV1
        }
        fn load(&mut self, _: &Artifact) -> Result<()> {
            Ok(())
        }
        fn forward(&mut self, _: u32, input: &[f32], _: f32) -> Result<Vec<f32>> {
            self.0 += 1;
            Ok(vec![0.0; input.len()])
        }
        fn evict(&mut self) -> Result<()> {
            Ok(())
        }
        fn fatal(&self) -> bool {
            false
        }
    }

    fn raw_route_fixture() -> (Resident, routing::RankDispatch) {
        let bytes = include_bytes!("../../fixtures/preweighted_mxfp4_tiny.safetensors").to_vec();
        let binding = Binding {
            artifact_sha256: sha256::hex(&bytes),
            model_revision: "raw-route-boundary".into(),
            rank: 7,
            layer: 0,
            epoch: 1,
            experts: vec![
                Expert {
                    expert: 17,
                    local: 4,
                },
                Expert {
                    expert: 93,
                    local: 1,
                },
                Expert {
                    expert: 241,
                    local: 9,
                },
            ],
        };
        let artifact = Artifact::new(bytes, binding.clone()).unwrap();
        let dispatch = serde_json::from_value(serde_json::json!({
            "stamp": {"run_id":"raw-route-boundary", "model_revision":binding.model_revision,
                      "plan_id":"raw-route", "layer":0, "batch":1, "epoch":1},
            "rank":7,"hidden":32,"tokens":1,"resident_order":[4,1,9],
            "slots":[{"slot":{"token":0,"slot":0,"expert":17,"rank":7,"local":4,"weight":0.375},"physical_expert":0}]
        })).unwrap();
        let resident = Resident {
            digest: binding.artifact_sha256,
            model_revision: binding.model_revision,
            layer: 0,
            epoch: 1,
            resident_order: vec![4, 1, 9],
            weights: Weights::Mxfp4(Box::new(artifact)),
        };
        (resident, dispatch)
    }

    #[test]
    fn duplicate_logical_expert_refuses_before_start_and_any_provider_call() {
        let (resident, base) = raw_route_fixture();
        // Sparse slot IDs and either order must not evade logical uniqueness.
        for expert in 0..3 {
            for (first, second) in [(0, 1), (63, 2), (2, 63)] {
                let mut dispatch = base.clone();
                let mut route = dispatch.slots[0].clone();
                route.slot.expert = [17, 93, 241][expert];
                route.slot.local = [4, 1, 9][expert];
                route.physical_expert = expert;
                route.slot.slot = first;
                dispatch.slots = vec![route.clone(), route];
                dispatch.slots[1].slot.slot = second;
                let inputs = vec![TokenInput {
                    token: 0,
                    value_bits: vec![1.0_f32.to_bits(); 32],
                }];
                let contract = arithmetic::Contract {
                    format: arithmetic::Format::R9Mxfp4PreweightedFp32V1,
                    sha256: "f".repeat(64),
                };
                let mut provider = CountForwards(0);
                let mut output = Vec::new();
                assert!(!execute(
                    &resident,
                    dispatch,
                    contract,
                    1,
                    inputs,
                    Fault::None,
                    &mut output,
                    Some(&mut provider)
                )
                .unwrap());
                assert_eq!(
                    provider.0, 0,
                    "logical refusal must precede every provider call"
                );
                let replies: Vec<Reply> = output
                    .split(|&b| b == b'\n')
                    .filter(|s| !s.is_empty())
                    .map(|s| serde_json::from_slice(s).unwrap())
                    .collect();
                assert!(matches!(
                    &replies[..],
                    [Reply::Event {
                        event: WorkerEvent {
                            event: EventKind::Rejected { .. },
                            ..
                        }
                    }]
                ));
            }
        }
    }

    #[test]
    fn logical_route_uniqueness_keeps_shared_experts_and_selected_replicas_valid() {
        let (resident, base) = raw_route_fixture();
        let Weights::Mxfp4(artifact) = &resident.weights else {
            unreachable!()
        };
        let mut shared = base.clone();
        shared.tokens = 2;
        shared.slots.push(shared.slots[0].clone());
        shared.slots[1].slot.token = 1;
        // Same logical expert on different tokens is legal; physical index 0 maps local4.
        artifact.validate_dispatch(&shared).unwrap();
        shared.slots.reverse();
        artifact.validate_dispatch(&shared).unwrap();
        let mut different = base.clone();
        let mut route = different.slots[0].clone();
        route.slot.slot = 63;
        route.slot.expert = 93;
        route.slot.local = 1;
        route.physical_expert = 1;
        different.slots.push(route);
        artifact.validate_dispatch(&different).unwrap();
        for mutation in 0..4 {
            let mut invalid = different.clone();
            match mutation {
                0 => invalid.slots[1].physical_expert = 0,
                1 => invalid.slots[1].slot.slot = 0,
                2 => invalid.slots[1].slot.token = 1,
                _ => invalid.slots[1].slot.slot = 64,
            }
            assert!(artifact.validate_dispatch(&invalid).is_err());
        }
    }

    struct NoBody;
    impl Read for NoBody {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            panic!("a foreign or oversized header must be refused before reading its body")
        }
    }
    fn header() -> CompletionHeader {
        CompletionHeader {
            producer: Producer::CpuR9Fp32WarpReferenceV1,
            artifact_sha256: "1".repeat(64),
            contract: arithmetic::Contract {
                format: arithmetic::Format::R9Mxfp4PreweightedFp32V1,
                sha256: "2".repeat(64),
            },
            stamp: moe::Stamp {
                run_id: "r12-admission".into(),
                model_revision: "model".into(),
                plan_id: "plan".into(),
                layer: 0,
                batch: 0,
                epoch: 1,
            },
            completion_stamp: moe::Stamp {
                run_id: "r12-admission".into(),
                model_revision: "model".into(),
                plan_id: "plan".into(),
                layer: 0,
                batch: 0,
                epoch: 1,
            },
            rank: 3,
            attempt: 9,
            hidden: 32,
            rows: vec![RowMeta {
                token: 0,
                slot: 0,
                expert: 17,
                local: 7,
                weight_bits: 0x3ec0_0000,
            }],
        }
    }
    fn admission(h: &CompletionHeader) -> Mutex<Option<Admission>> {
        Mutex::new(Some(Admission {
            producer: h.producer.clone(),
            stamp: h.stamp.clone(),
            rank: h.rank,
            attempt: h.attempt,
            artifact_sha256: h.artifact_sha256.clone(),
            contract: h.contract.clone(),
            hidden: h.hidden,
            slots: 1,
        }))
    }
    #[test]
    fn full_identity_is_checked_before_a_raw_body_read() {
        for field in 0..6 {
            let mut h = header();
            let active = admission(&h);
            match field {
                0 => h.artifact_sha256 = "f".repeat(64),
                1 => h.attempt += 1,
                2 => h.completion_stamp.epoch += 1,
                3 => h.contract.sha256 = "f".repeat(64),
                4 => h.rank += 1,
                _ => h.stamp.model_revision = "other".into(),
            };
            assert_eq!(
                read_completion(&mut NoBody, h, 32, &active, 200)
                    .unwrap_err()
                    .code,
                "INVALID_COMPLETION"
            );
        }
    }
    #[test]
    fn producer_and_library_identity_refuse_before_read_without_consuming_budget() {
        let h = header();
        let gpu = Producer::GpuR13ResidentFp32V1 {
            library_sha256: "a".repeat(64),
        };
        for (expected, observed) in [
            (h.producer.clone(), gpu.clone()),
            (gpu.clone(), h.producer.clone()),
            (
                gpu.clone(),
                Producer::GpuR13ResidentFp32V1 {
                    library_sha256: "b".repeat(64),
                },
            ),
        ] {
            let mut expected_header = h.clone();
            expected_header.producer = expected;
            let active = admission(&expected_header);
            let mut observed_header = h.clone();
            observed_header.producer = observed;
            assert_eq!(
                read_completion(&mut NoBody, observed_header, 32, &active, 200)
                    .unwrap_err()
                    .code,
                "INVALID_COMPLETION"
            );
            assert_eq!(active.lock().unwrap().as_ref().unwrap().slots, 1);
            // A refused producer must not consume the real completion's body allowance.
            assert!(read_completion(
                &mut io::Cursor::new(vec![0; 128]),
                expected_header,
                32,
                &active,
                200
            )
            .is_ok());
            assert_eq!(active.lock().unwrap().as_ref().unwrap().slots, 0);
        }
        for bad in ["A".repeat(64), "a".repeat(63), "g".repeat(64)] {
            assert!(Producer::GpuR13ResidentFp32V1 {
                library_sha256: bad
            }
            .validate()
            .is_err());
        }
        assert!(gpu.validate().is_ok());
    }

    #[test]
    fn actual_header_length_and_value_overflow_refuse_before_body_read() {
        let h = header();
        let active = admission(&h);
        assert_eq!(
            read_completion(&mut NoBody, h.clone(), 32, &active, MAX_MESSAGE_BYTES - 1)
                .unwrap_err()
                .code,
            "MESSAGE_TOO_LARGE"
        );
        assert!(read_completion(&mut NoBody, h, u64::MAX, &active, 200).is_err());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn inherited_output_handles_cannot_keep_reader_threads_after_close_or_drop() {
        use std::os::linux::net::SocketAddrExt;
        use std::os::unix::net::{SocketAddr, UnixListener};
        for explicit in [false, true] {
            let name = format!("vf-r12-outputs-{}-{explicit}", std::process::id());
            let listener =
                UnixListener::bind_addr(&SocketAddr::from_abstract_name(name.as_bytes()).unwrap())
                    .unwrap();
            let endpoint = Endpoint {
                rank: 7,
                argv: vec![
                    "python3".into(),
                    "-B".into(),
                    concat!(
                        env!("CARGO_MANIFEST_DIR"),
                        "/tests/reference/preweighted_fault_peer.py"
                    )
                    .into(),
                    "fork_hold_outputs".into(),
                    format!("@{name}"),
                ],
            };
            let mut session = Session::connect(&endpoint, Duration::from_secs(5)).unwrap();
            let (control, _) = listener.accept().unwrap();
            control
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut control = BufReader::new(control);
            let mut pid = String::new();
            control.read_line(&mut pid).unwrap();
            assert!(pid.trim().parse::<u32>().is_ok());
            // Taking handles lets the test observe thread termination independently of
            // dropping JoinHandle. Cancellation must work even while the descendant
            // still has both output descriptors open.
            let reader = session.reader.take().unwrap();
            let stderr = session.stderr.take().unwrap();
            if explicit {
                let close = session.close(Duration::from_millis(100));
                assert_eq!(close.exit_code, Some(0));
            } else {
                drop(session);
            }
            let deadline = Instant::now() + Duration::from_secs(1);
            while !(reader.is_finished() && stderr.is_finished()) && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(1));
            }
            let stopped_before_descendant = reader.is_finished() && stderr.is_finished();
            control.get_mut().write_all(b"R").unwrap();
            let mut done = [0];
            control.read_exact(&mut done).unwrap();
            assert_eq!(done, [b'D']);
            reader.join().unwrap();
            stderr.join().unwrap();
            assert!(
                stopped_before_descendant,
                "output reader survived closed session (explicit={explicit})"
            );
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn an_idle_collector_applies_backpressure_to_control_frame_floods() {
        use std::os::linux::net::SocketAddrExt;
        use std::os::unix::net::{SocketAddr, UnixListener};
        let name = format!("vf-r12-control-flood-{}", std::process::id());
        let listener =
            UnixListener::bind_addr(&SocketAddr::from_abstract_name(name.as_bytes()).unwrap())
                .unwrap();
        let endpoint = Endpoint {
            rank: 7,
            argv: vec![
                "python3".into(),
                "-B".into(),
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/reference/preweighted_fault_peer.py"
                )
                .into(),
                "control_flood".into(),
                format!("@{name}"),
            ],
        };
        let mut session = Session::connect(&endpoint, Duration::from_secs(5)).unwrap();
        let (mut control, _) = listener.accept().unwrap();
        control
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let mut marker = [0];
        control.read_exact(&mut marker).unwrap();
        assert_eq!(marker, [b'S']);
        // No frame is collected after HELLO. A bounded reader must backpressure the
        // actual peer instead of consuming and retaining its entire control flood.
        let done = control.read_exact(&mut marker);
        let received = session.received.load(Ordering::Relaxed);
        session.usable = false;
        let close = session.close(Duration::from_millis(300));
        assert!(close.exit_code.is_some() || close.killed);
        assert!(
            matches!(done, Err(ref e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut)),
            "an idle collector let the peer stream every control frame"
        );
        assert!(
            received <= 128 * 1024,
            "idle reader consumed {received} bytes"
        );
        assert!(
            close.error.is_none(),
            "bounded channel prevented reader shutdown: {close:?}"
        );
    }

    #[test]
    fn cumulative_body_budget_is_consumed_before_a_second_allocation() {
        let h = header();
        let active = admission(&h);
        let mut body = io::Cursor::new(vec![0; 32 * 4]);
        assert!(read_completion(&mut body, h.clone(), 32, &active, 200).is_ok());
        assert_eq!(
            read_completion(&mut NoBody, h, 32, &active, 200)
                .unwrap_err()
                .code,
            "INVALID_COMPLETION"
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn a_descendant_holding_stdin_does_not_retain_the_payload_after_timeout() {
        use safetensors::tensor::{serialize, Dtype, SafeTensors, TensorView};
        use std::os::linux::net::SocketAddrExt;
        use std::os::unix::net::{SocketAddr, UnixListener};
        let tiny = include_bytes!("../../fixtures/preweighted_mxfp4_tiny.safetensors");
        let tensors = SafeTensors::deserialize(tiny).unwrap();
        let pad = vec![0; 8 * 1024 * 1024];
        let mut entries: Vec<_> = tensors
            .names()
            .into_iter()
            .map(|name| (name.to_owned(), tensors.tensor(name).unwrap()))
            .collect();
        entries.push((
            "r12_padding".into(),
            TensorView::new(Dtype::U8, vec![pad.len()], &pad).unwrap(),
        ));
        let bytes = serialize(entries, None).unwrap();
        let artifact = Artifact::new(
            bytes.clone(),
            Binding {
                artifact_sha256: sha256::hex(&bytes),
                model_revision: "r12-timeout".into(),
                rank: 7,
                layer: 0,
                epoch: 1,
                experts: [17, 93, 241]
                    .into_iter()
                    .enumerate()
                    .map(|(local, expert)| Expert {
                        expert,
                        local: local as u32,
                    })
                    .collect(),
            },
        )
        .unwrap();
        let shared = artifact.shard.shared_artifact_bytes();
        let before = Arc::strong_count(&shared);
        let name = format!("vf-r12-descendant-{}", std::process::id());
        let listener =
            UnixListener::bind_addr(&SocketAddr::from_abstract_name(name.as_bytes()).unwrap())
                .unwrap();
        let endpoint = Endpoint {
            rank: 7,
            argv: vec![
                "python3".into(),
                "-B".into(),
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/reference/preweighted_fault_peer.py"
                )
                .into(),
                "fork_no_load_read".into(),
                format!("@{name}"),
            ],
        };
        let mut session = Session::connect(&endpoint, Duration::from_secs(5)).unwrap();
        let (control, _) = listener.accept().unwrap();
        control
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let error = session
            .load_mxfp4(&artifact, 1, Duration::from_millis(300))
            .unwrap_err();
        let held_after_timeout = Arc::strong_count(&shared);
        let mut control = BufReader::new(control);
        let mut pid = String::new();
        control.read_line(&mut pid).unwrap();
        assert!(pid.trim().parse::<u32>().is_ok());
        control.get_mut().write_all(b"R").unwrap();
        let mut done = [0];
        control.read_exact(&mut done).unwrap();
        assert_eq!(done, [b'D']);
        let deadline = Instant::now() + Duration::from_secs(5);
        while Arc::strong_count(&shared) != before && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
        session.close(Duration::from_secs(5));
        assert_eq!(error.code, "SESSION_WRITE_TIMEOUT");
        assert_eq!(
            Arc::strong_count(&shared),
            before,
            "owned fault descendant did not release its pipe"
        );
        assert_eq!(
            held_after_timeout, before,
            "a detached writer retained the compressed payload after timeout"
        );
    }
}

#[cfg(test)]
mod reader_budget_tests;
