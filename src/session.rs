//! Persistent rank-worker sessions (A52.1) over one byte stream: a local pipe or an
//! already authenticated `ssh -T` channel. CPU f64 dense MoE oracle only; this is not a
//! CUDA, RDMA or GPUDirect backend and makes no throughput claim. The explicit raw-FP16
//! activation option accepts caller-owned finite binary16 bits; the default remains f64.
//!
//! HELLO binds the session to one rank and reports what the worker really is. LOAD sends
//! the rank's resident experts once, identified by the SHA-256 of their serialization;
//! a later LOAD without weights only rebinds a newer ownership epoch. Each EXEC carries
//! the rank dispatch and its compacted activations, never weights, and is answered by the
//! single-request worker's own events (Started, unweighted Completions, Finished).
//! When both peers advertise batch-completion support, validated rows may share bounded raw
//! frames; older peers retain one completion event per slot. The coordinator applies routing
//! coefficients once, in the collector. EVICT drops the
//! artifact; SHUTDOWN is confirmed by BYE before the worker exits.
//! Engram owner queries negotiate compact varint row IDs when both peers advertise support;
//! older workers retain JSON row IDs without changing the authenticated transport.
//!
//! An execution is reported as not started (refused or rejected), started (and failed),
//! completed, or unknown (contact lost or deadline passed after admission). Nothing is
//! retried, and a session whose stream position can no longer be trusted is unusable:
//! it must be closed, never reused. There is no exactly-once promise across nodes.
use crate::engram::{self, EngramRowCache, EngramShard, OwnerRows, TableIdentity};
use crate::process::{
    attempts_left, check_fault, claim_identity, encode, execute, execute_batch_with, local_work,
    next_attempt, rank_request, read_message_with_len, token_inputs, write_frame, EventKind,
    EventSink, Fault, Job, RankRequest, ReceiptState, TokenInput, WorkerCompletions, WorkerEvent,
    MAX_MESSAGE_BYTES,
};
use crate::resident::ResidentStack;
use crate::{base64, invalid, moe, resident_artifact, routing, sha256, Error, Result};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub mod mxfp4;

pub const PROTOCOL: u32 = 1;
pub const CAPABILITY: &str = "cpu_f64_dense_moe_v1";
/// LOAD of a `QPN2_RESIDENT_V1` artifact's bytes, EXEC on its compressed planes (A53).
pub const CAPABILITY_QPN2: &str = "cpu_qpn2_resident_v1";
/// Raw artifact body follows a bounded JSON LOAD header (A54); older workers use base64.
pub const CAPABILITY_QPN2_RAW: &str = "cpu_qpn2_resident_raw_v1";
/// Explicitly negotiated multipart resident loading; legacy raw LOAD is unchanged.
pub const CAPABILITY_QPN2_MULTIPART: &str = "cpu_qpn2_multipart_v1";
/// Little-endian f64 activation body follows a bounded JSON EXEC header (A55).
pub const CAPABILITY_EXEC_F64_RAW: &str = "cpu_exec_f64_raw_v1";
/// Little-endian binary16 activation body follows a bounded JSON EXEC header.
pub const CAPABILITY_EXEC_FP16_RAW: &str = "cpu_exec_fp16_raw_v1";
/// Opt-in (A56): an EXEC_RAW that sets `raw_completions` is answered with
/// `completion_raw` frames, a JSON identity header followed by little-endian f64 values.
pub const CAPABILITY_COMPLETION_F64_RAW: &str = "cpu_completion_f64_raw_v1";
/// Opt-in compact framing for all validated f64 completions from one rank attempt.
pub const CAPABILITY_COMPLETION_BATCH_F64_RAW: &str = "cpu_completion_batch_f64_raw_v1";
/// Session-latched completion batching: one EXEC opts in, subsequent raw EXECs reuse it.
pub const CAPABILITY_COMPLETION_BATCH_SESSION_F64_RAW: &str =
    "cpu_completion_batch_f64_raw_session_v1";
/// Opt-in: one rank EXEC may return bounded little-endian binary16 completion batches.
pub const CAPABILITY_COMPLETION_BATCH_FP16_RAW: &str = "cpu_completion_batch_fp16_raw_v1";
/// Separate immutable Engram shard and bounded owner-row exchange (A64).
pub const CAPABILITY_ENGRAM_ROWS_RAW: &str = "cpu_engram_owner_rows_raw_v1";
/// Owner row-ID requests may append a canonical unsigned-LEB128 body (A78).
pub const CAPABILITY_ENGRAM_ROW_IDS_VARINT: &str = "cpu_engram_row_ids_varint_v1";
#[cfg(feature = "native-ggml-cpu")]
pub const CAPABILITY_NATIVE_GGML_CPU: &str = "cpu_strata_sc117_ggml_fp16_v1";
pub const MAX_RAW_ACTIVATION_BYTES: usize = MAX_MESSAGE_BYTES - 64 * 1024;
const MAX_STDERR_KEPT: usize = 64 * 1024;
/// How long close() still waits for the output of a worker it has just killed.
const PIPE_GRACE: Duration = Duration::from_millis(200);

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerHello {
    pub protocol: u32,
    pub rank: u32,
    pub arch: String,
    pub os: String,
    pub endian: String,
    pub build: String,
    pub capabilities: Vec<String>,
    pub max_message_bytes: usize,
    pub pid: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawTokenInputMeta {
    pub token: usize,
    pub values: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawCompletionMeta {
    pub stamp: moe::Stamp,
    pub token: usize,
    pub slot: usize,
    pub expert: u32,
    pub rank: u32,
    pub local: u32,
    pub values: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawBatchCompletionMeta {
    pub token: usize,
    pub slot: usize,
    pub expert: u32,
    pub local: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Request {
    Hello {
        protocol: u32,
        rank: u32,
    },
    #[cfg(feature = "native-ggml-cpu")]
    LoadNativeGgmlRaw {
        binding: crate::native_ggml_cpu::Binding,
        index: String,
        pack_bytes: u64,
        artifact_bytes: u64,
    },
    #[cfg(feature = "native-ggml-cpu")]
    SealNativeGgml,
    #[cfg(feature = "native-ggml-cpu")]
    ExecNativeGgml {
        dispatch: routing::RankDispatch,
        attempt: u64,
        inputs: Vec<(usize, Vec<f32>)>,
    },
    Load {
        artifact_sha256: String,
        model_revision: String,
        layer: u32,
        epoch: u64,
        resident_order: Vec<u32>,
        /// Present for the one transfer; absent to rebind a newer epoch.
        #[serde(default)]
        model: Option<moe::Model>,
    },
    /// The one transfer of a compressed resident (A53): canonical base64 of the bytes of a
    /// `QPN2_RESIDENT_V1` SafeTensors artifact whose SHA-256 is `artifact_sha256`. A
    /// weightless `Load` rebinds it to a newer epoch, as for a dense model.
    LoadArtifact {
        artifact_sha256: String,
        model_revision: String,
        layer: u32,
        epoch: u64,
        resident_order: Vec<u32>,
        artifact: String,
    },
    /// The bounded JSON header is followed immediately by exactly `artifact_bytes` raw bytes.
    /// The digest and epoch bind the transfer; rank, model revision, layer and resident order
    /// come from the canonical content-addressed artifact itself.
    LoadArtifactRaw {
        artifact_sha256: String,
        epoch: u64,
        artifact_bytes: u64,
    },
    LoadMxfp4Raw {
        binding: mxfp4::Binding,
        artifact_bytes: u64,
    },
    ExecPreweightedFp32Raw {
        artifact_sha256: String,
        dispatch: routing::RankDispatch,
        contract: crate::preweighted::Contract,
        attempt: u64,
        inputs: Vec<RawTokenInputMeta>,
        activation_bytes: u64,
        #[serde(default)]
        fault: Fault,
    },
    /// Lossless BF16 inputs; accumulation/results retain the R9 FP32 contract.
    /// Sent only after mxfp4::BF16_ACTIVATION_CAPABILITY is advertised.
    ExecPreweightedBf16Raw {
        artifact_sha256: String,
        dispatch: routing::RankDispatch,
        contract: crate::preweighted::Contract,
        attempt: u64,
        inputs: Vec<RawTokenInputMeta>,
        activation_bytes: u64,
        #[serde(default)]
        fault: Fault,
    },
    #[serde(skip)]
    ExecPreweightedFp32 {
        artifact_sha256: String,
        dispatch: routing::RankDispatch,
        contract: crate::preweighted::Contract,
        attempt: u64,
        inputs: Vec<mxfp4::TokenInput>,
        fault: Fault,
    },
    LoadEngramShardRaw {
        table: TableIdentity,
        world_size: u32,
        worker_rank: u32,
        owner_rank: u32,
        artifact_sha256: String,
        artifact_bytes: u64,
    },
    FetchEngramRows {
        stamp: moe::Stamp,
        table: TableIdentity,
        world_size: u32,
        owner_rank: u32,
        row_ids: Vec<u64>,
        attempt: u64,
    },
    /// The bounded JSON header is followed by `row_id_bytes` canonical unsigned-LEB128 IDs.
    /// Older sessions keep using FetchEngramRows with its JSON ID array.
    FetchEngramRowsVarint {
        stamp: moe::Stamp,
        table: TableIdentity,
        world_size: u32,
        owner_rank: u32,
        row_id_count: u64,
        row_id_bytes: u64,
        attempt: u64,
    },
    EvictEngram {
        artifact_sha256: String,
    },
    /// The JSON header is followed by `activation_bytes` little-endian f64 values.
    /// `inputs` identifies each token and its value count; values are concatenated in order.
    ExecRaw {
        artifact_sha256: String,
        dispatch: routing::RankDispatch,
        attempt: u64,
        inputs: Vec<RawTokenInputMeta>,
        activation_bytes: u64,
        #[serde(default)]
        fault: Fault,
        /// Absent (false) for workers without `cpu_completion_f64_raw_v1`.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        raw_completions: bool,
        /// Absent (false) for workers without `cpu_completion_batch_f64_raw_v1`.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        raw_completion_batch: bool,
        /// Explicit opt-in to the lossy, finite-only binary16 batch response format.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        raw_completion_fp16_batch: bool,
    },
    /// The JSON header is followed by `activation_bytes` little-endian finite binary16 values.
    /// Unlike `ExecRaw`, this path accepts caller-owned half bits and never casts f64 inputs.
    ExecRawFp16 {
        artifact_sha256: String,
        dispatch: routing::RankDispatch,
        attempt: u64,
        inputs: Vec<RawTokenInputMeta>,
        activation_bytes: u64,
        #[serde(default)]
        fault: Fault,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        raw_completions: bool,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        raw_completion_batch: bool,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        raw_completion_fp16_batch: bool,
    },
    Exec {
        artifact_sha256: String,
        dispatch: routing::RankDispatch,
        attempt: u64,
        inputs: Vec<TokenInput>,
        #[serde(default)]
        fault: Fault,
    },
    Evict {
        artifact_sha256: String,
    },
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "reply", rename_all = "snake_case", deny_unknown_fields)]
pub enum Reply {
    #[cfg(feature = "native-ggml-cpu")]
    NativeGgmlLoaded {
        binding: crate::native_ggml_cpu::Binding,
    },
    #[cfg(feature = "native-ggml-cpu")]
    NativeGgmlReady {
        experts: usize,
    },
    #[cfg(feature = "native-ggml-cpu")]
    NativeGgmlResult {
        output: crate::completion::RoutedRawOutput,
    },
    MultipartReady {
        offset: u64,
    },
    ChunkAccepted {
        offset: u64,
    },
    Hello {
        hello: ServerHello,
    },
    Loaded {
        artifact_sha256: String,
        epoch: u64,
        experts: usize,
        transferred: bool,
    },
    EngramLoaded {
        artifact_sha256: String,
        table: TableIdentity,
        transferred: bool,
    },
    EngramEvicted {
        artifact_sha256: String,
    },
    EngramRowsRaw {
        attempt: u64,
        frame_bytes: u64,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        frame: Vec<u8>,
    },
    Event {
        event: WorkerEvent,
    },
    PreweightedFp32Raw {
        header: mxfp4::CompletionHeader,
        values: u64,
    },
    #[serde(skip)]
    PreweightedFp32 {
        header: mxfp4::CompletionHeader,
        bits: Vec<u32>,
    },
    /// A completion event whose header is followed by `completion.values` little-endian
    /// f64 values; the coordinator rebuilds exactly the `Event` it stands for.
    CompletionRaw {
        stamp: moe::Stamp,
        rank: u32,
        attempt: u64,
        completion: RawCompletionMeta,
    },
    /// A raw batch header followed by `values` little-endian f64 values, row-major.
    CompletionBatchRaw {
        stamp: moe::Stamp,
        completion_stamp: moe::Stamp,
        rank: u32,
        attempt: u64,
        hidden: u64,
        values: u64,
        completions: Vec<RawBatchCompletionMeta>,
    },
    /// A raw batch header followed by `values` little-endian IEEE binary16 values.
    CompletionBatchFp16Raw {
        stamp: moe::Stamp,
        completion_stamp: moe::Stamp,
        rank: u32,
        attempt: u64,
        hidden: u64,
        values: u64,
        completions: Vec<RawBatchCompletionMeta>,
    },
    /// Decoded local form of a negotiated `CompletionBatchRaw` frame; never serialized.
    #[serde(skip)]
    CompletionBatch {
        stamp: moe::Stamp,
        rank: u32,
        attempt: u64,
        completions: Vec<moe::Completion>,
    },
    /// Decoded compact form of `CompletionBatchFp16Raw`; bits stay u16 until collection.
    #[serde(skip)]
    CompletionBatchFp16 {
        stamp: moe::Stamp,
        completion_stamp: moe::Stamp,
        rank: u32,
        attempt: u64,
        hidden: usize,
        completions: Vec<RawBatchCompletionMeta>,
        fp16_bits: Vec<u16>,
    },
    Evicted {
        artifact_sha256: String,
    },
    Bye {
        execs: u64,
        /// The worker's own peak resident set (Linux VmHWM), when readable.
        #[serde(default)]
        peak_rss_kib: Option<u64>,
    },
    Refused {
        error: Error,
    },
}

/// The same frame as `Reply::Event`, serialized without cloning the event.
#[derive(Serialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
enum EventReply<'a> {
    Event { event: &'a WorkerEvent },
}

/// SHA-256 of the canonical JSON serialization of a resident model. Both ends compute
/// it: the coordinator from what it sends, the worker from what it parsed.
pub fn artifact_digest(model: &moe::Model) -> Result<String> {
    let bytes = serde_json::to_vec(model).map_err(|e| invalid(e.to_string()))?;
    Ok(sha256::hex(&bytes))
}

// ------------------------------------------------------------------ worker side

/// What a session holds: the dense test model, or a compressed resident stack.
enum Weights {
    Dense(moe::Model),
    Compressed(Box<ResidentStack>),
    Mxfp4(Box<mxfp4::Artifact>),
}

struct Resident {
    digest: String,
    weights: Weights,
    model_revision: String,
    layer: u32,
    epoch: u64,
    resident_order: Vec<u32>,
}

struct SessionEvents<'a, W: Write> {
    output: &'a mut W,
    raw_completions: bool,
    raw_completion_batch: bool,
    raw_completion_fp16_batch: bool,
}
impl<W: Write> EventSink for SessionEvents<'_, W> {
    fn send(&mut self, event: &WorkerEvent) -> Result<()> {
        match &event.event {
            EventKind::Completion { completion } if self.raw_completions => write_frame(
                self.output,
                &smaller_completion_frame(event, completion)?,
                "session event",
            ),
            _ => write_frame(
                self.output,
                &encode(&EventReply::Event { event })?,
                "session event",
            ),
        }
    }

    fn send_completions(&mut self, batch: WorkerCompletions) -> Result<()> {
        if self.raw_completion_fp16_batch {
            return write_raw_completion_fp16_batch(self.output, batch);
        }
        if !self.raw_completion_batch || !self.raw_completions || batch.rows.is_empty() {
            for completion in batch.rows {
                self.send(&WorkerEvent {
                    stamp: batch.stamp.clone(),
                    rank: batch.rank,
                    attempt: batch.attempt,
                    event: EventKind::Completion { completion },
                })?;
            }
            return Ok(());
        }
        write_raw_completion_batch(self.output, batch)
    }
}
/// The shorter of the JSON `event` frame and the `completion_raw` frame (JSON on a tie).
/// Both are lossless for finite values (`float_roundtrip` keeps JSON bit-exact, `-0.0`
/// included) and the coordinator always accepts JSON, so zero or short decimal values never
/// cost more than the JSON protocol did.
fn smaller_completion_frame(event: &WorkerEvent, completion: &moe::Completion) -> Result<Vec<u8>> {
    let json = encode(&EventReply::Event { event })?;
    let raw = encode_raw_completion(event, completion)?;
    Ok(if raw.len() < json.len() { raw } else { json })
}
/// One `completion_raw` frame: header, newline, then the values' little-endian bytes.
fn encode_raw_completion(event: &WorkerEvent, completion: &moe::Completion) -> Result<Vec<u8>> {
    let mut frame = encode(&Reply::CompletionRaw {
        stamp: event.stamp.clone(),
        rank: event.rank,
        attempt: event.attempt,
        completion: RawCompletionMeta {
            stamp: completion.stamp.clone(),
            token: completion.token,
            slot: completion.slot,
            expert: completion.expert,
            rank: completion.rank,
            local: completion.local,
            values: u64::try_from(completion.value.len())
                .map_err(|_| invalid("raw completion length does not fit u64"))?,
        },
    })?;
    let body_len = completion
        .value
        .len()
        .checked_mul(std::mem::size_of::<f64>())
        .filter(|&n| {
            n <= MAX_RAW_ACTIVATION_BYTES && frame.len().saturating_add(n) <= MAX_MESSAGE_BYTES
        })
        .ok_or_else(|| {
            Error::new(
                "MESSAGE_TOO_LARGE",
                "raw completion exceeds the frame bound",
            )
        })?;
    frame.try_reserve_exact(body_len).map_err(|e| {
        Error::new(
            "EXEC_ALLOC",
            format!("cannot allocate raw completion frame: {e}"),
        )
    })?;
    for value in &completion.value {
        frame.extend_from_slice(&value.to_le_bytes());
    }
    Ok(frame)
}

#[derive(Serialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
enum CompletionBatchHeader<'a> {
    CompletionBatchRaw {
        stamp: &'a moe::Stamp,
        completion_stamp: &'a moe::Stamp,
        rank: u32,
        attempt: u64,
        hidden: u64,
        values: u64,
        completions: CompletionMetadata<'a>,
    },
}

#[derive(Serialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
enum CompletionBatchFp16Header<'a> {
    CompletionBatchFp16Raw {
        stamp: &'a moe::Stamp,
        completion_stamp: &'a moe::Stamp,
        rank: u32,
        attempt: u64,
        hidden: u64,
        values: u64,
        completions: CompletionMetadata<'a>,
    },
}

/// Serializes metadata directly from the completion slice, avoiding a second unbounded
/// vector whose size could exceed the frame bound for low-width, high-slot batches.
struct CompletionMetadata<'a>(&'a [moe::Completion]);

impl Serialize for CompletionMetadata<'_> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for completion in self.0 {
            sequence.serialize_element(&RawBatchCompletionMeta {
                token: completion.token,
                slot: completion.slot,
                expert: completion.expert,
                local: completion.local,
            })?;
        }
        sequence.end()
    }
}

fn encode_raw_completion_batch_header(
    stamp: &moe::Stamp,
    completion_stamp: &moe::Stamp,
    rank: u32,
    attempt: u64,
    hidden: usize,
    completions: &[moe::Completion],
) -> Result<Vec<u8>> {
    let values = completions
        .len()
        .checked_mul(hidden)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| Error::new("MESSAGE_TOO_LARGE", "batch value count overflows"))?;
    encode(&CompletionBatchHeader::CompletionBatchRaw {
        stamp,
        completion_stamp,
        rank,
        attempt,
        hidden: u64::try_from(hidden)
            .map_err(|_| Error::new("MESSAGE_TOO_LARGE", "hidden width does not fit u64"))?,
        values,
        completions: CompletionMetadata(completions),
    })
}

fn write_raw_completion_batch(output: &mut impl Write, batch: WorkerCompletions) -> Result<()> {
    write_raw_completion_batch_bounded(output, batch, MAX_MESSAGE_BYTES)
}

fn write_raw_completion_batch_bounded(
    output: &mut impl Write,
    batch: WorkerCompletions,
    max_frame_bytes: usize,
) -> Result<()> {
    let Some(first) = batch.rows.first() else {
        return Ok(());
    };
    let hidden = first.value.len();
    if hidden == 0 || hidden > 65536 {
        return Err(invalid(
            "completion batch hidden width is outside the admitted domain",
        ));
    }
    let completion_stamp = first.stamp.clone();
    for completion in &batch.rows {
        if completion.stamp != completion_stamp
            || completion.rank != batch.rank
            || completion.value.len() != hidden
            || completion.value.iter().any(|x| !x.is_finite())
        {
            return Err(invalid(
                "completion batch rows differ in identity, shape or finite-value contract",
            ));
        }
    }

    let row_bytes = hidden
        .checked_mul(std::mem::size_of::<f64>())
        .filter(|&n| n <= MAX_RAW_ACTIVATION_BYTES)
        .ok_or_else(|| {
            Error::new(
                "MESSAGE_TOO_LARGE",
                "one completion row exceeds the raw batch body bound",
            )
        })?;
    let mut start = 0;
    while start < batch.rows.len() {
        let mut low = start + 1;
        let mut high = batch.rows.len();
        let mut best: Option<(usize, Vec<u8>)> = None;
        while low <= high {
            let middle = low + (high - low) / 2;
            let header = match encode_raw_completion_batch_header(
                &batch.stamp,
                &completion_stamp,
                batch.rank,
                batch.attempt,
                hidden,
                &batch.rows[start..middle],
            ) {
                Ok(header) => header,
                Err(error) if error.code == "MESSAGE_TOO_LARGE" => {
                    high = middle.saturating_sub(1);
                    continue;
                }
                Err(error) => return Err(error),
            };
            let rows = middle - start;
            let fits = rows
                .checked_mul(row_bytes)
                .filter(|&body| body <= MAX_RAW_ACTIVATION_BYTES)
                .and_then(|body| header.len().checked_add(body))
                .is_some_and(|total| total <= max_frame_bytes);
            if fits {
                best = Some((middle, header));
                low = middle.saturating_add(1);
            } else {
                high = middle.saturating_sub(1);
            }
        }
        let Some((end, header)) = best else {
            // A long identity header can make the batch envelope larger than the normal
            // completion envelope even for one row. Preserve compatibility by using the
            // existing smallest single-row encoding in that case.
            let completion = &batch.rows[start];
            let event = WorkerEvent {
                stamp: batch.stamp.clone(),
                rank: batch.rank,
                attempt: batch.attempt,
                event: EventKind::Completion {
                    completion: completion.clone(),
                },
            };
            write_frame(
                output,
                &smaller_completion_frame(&event, completion)?,
                "session event",
            )?;
            start += 1;
            continue;
        };
        let body_len = (end - start) * row_bytes;
        let header_len = header.len();
        let mut frame = header;
        frame.try_reserve_exact(body_len).map_err(|e| {
            Error::new(
                "EXEC_ALLOC",
                format!("cannot allocate bounded completion batch frame: {e}"),
            )
        })?;
        for completion in &batch.rows[start..end] {
            for value in &completion.value {
                frame.extend_from_slice(&value.to_le_bytes());
            }
        }
        debug_assert_eq!(frame.len(), header_len + body_len);
        write_frame(output, &frame, "session completion batch")?;
        start = end;
    }
    Ok(())
}

fn encode_raw_completion_fp16_batch_header(
    stamp: &moe::Stamp,
    completion_stamp: &moe::Stamp,
    rank: u32,
    attempt: u64,
    hidden: usize,
    completions: &[moe::Completion],
) -> Result<Vec<u8>> {
    let values = completions
        .len()
        .checked_mul(hidden)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| Error::new("MESSAGE_TOO_LARGE", "FP16 batch value count overflows"))?;
    encode(&CompletionBatchFp16Header::CompletionBatchFp16Raw {
        stamp,
        completion_stamp,
        rank,
        attempt,
        hidden: u64::try_from(hidden)
            .map_err(|_| Error::new("MESSAGE_TOO_LARGE", "hidden width does not fit u64"))?,
        values,
        completions: CompletionMetadata(completions),
    })
}

fn write_raw_completion_fp16_batch(
    output: &mut impl Write,
    batch: WorkerCompletions,
) -> Result<()> {
    write_raw_completion_fp16_batch_bounded(output, batch, MAX_MESSAGE_BYTES)
}

fn write_raw_completion_fp16_batch_bounded(
    output: &mut impl Write,
    batch: WorkerCompletions,
    max_frame_bytes: usize,
) -> Result<()> {
    let Some(first) = batch.rows.first() else {
        return Ok(());
    };
    let hidden = first.value.len();
    if hidden == 0 || hidden > 65_536 {
        return Err(invalid(
            "FP16 completion batch hidden width is outside the admitted domain",
        ));
    }
    let completion_stamp = first.stamp.clone();
    for completion in &batch.rows {
        if completion.stamp != completion_stamp
            || completion.rank != batch.rank
            || completion.value.len() != hidden
        {
            return Err(invalid(
                "FP16 completion batch rows differ in identity or shape",
            ));
        }
    }

    let value_count = batch
        .rows
        .len()
        .checked_mul(hidden)
        .filter(|&n| n <= crate::completion::MAX_OUTPUT_VALUES)
        .ok_or_else(|| {
            Error::new(
                "CAPACITY_EXCEEDED",
                "FP16 completion batch exceeds the configured value-count cap",
            )
        })?;
    let mut fp16_bits = Vec::new();
    fp16_bits.try_reserve_exact(value_count).map_err(|error| {
        Error::new(
            "EXEC_ALLOC",
            format!("cannot allocate FP16 completion payload: {error}"),
        )
    })?;
    for completion in &batch.rows {
        for &value in &completion.value {
            fp16_bits.push(crate::completion::encode_fp16(value)?);
        }
    }

    let row_bytes = hidden
        .checked_mul(std::mem::size_of::<u16>())
        .filter(|&n| n <= MAX_RAW_ACTIVATION_BYTES)
        .ok_or_else(|| {
            Error::new(
                "MESSAGE_TOO_LARGE",
                "one FP16 completion row exceeds the raw body bound",
            )
        })?;

    // Validate every bounded header before writing the first byte. A later identity or
    // size failure must not leave a partial set of frames on an otherwise reusable stream.
    let mut frame_headers = Vec::new();
    frame_headers
        .try_reserve(batch.rows.len().min(1024))
        .map_err(|error| Error::new("EXEC_ALLOC", format!("cannot plan FP16 frames: {error}")))?;
    let mut start = 0;
    while start < batch.rows.len() {
        let mut low = start + 1;
        let mut high = batch.rows.len();
        let mut best: Option<(usize, Vec<u8>)> = None;
        while low <= high {
            let middle = low + (high - low) / 2;
            let header = match encode_raw_completion_fp16_batch_header(
                &batch.stamp,
                &completion_stamp,
                batch.rank,
                batch.attempt,
                hidden,
                &batch.rows[start..middle],
            ) {
                Ok(header) => header,
                Err(error) if error.code == "MESSAGE_TOO_LARGE" => {
                    high = middle.saturating_sub(1);
                    continue;
                }
                Err(error) => return Err(error),
            };
            let rows = middle - start;
            let fits = rows
                .checked_mul(row_bytes)
                .filter(|&body| body <= MAX_RAW_ACTIVATION_BYTES)
                .and_then(|body| header.len().checked_add(body))
                .is_some_and(|total| total <= max_frame_bytes);
            if fits {
                best = Some((middle, header));
                low = middle.saturating_add(1);
            } else {
                high = middle.saturating_sub(1);
            }
        }
        let Some((end, header)) = best else {
            return Err(Error::new(
                "MESSAGE_TOO_LARGE",
                "one FP16 completion row cannot fit the configured frame bound",
            ));
        };
        frame_headers.push((start, end, header));
        start = end;
    }

    for (start, end, header) in frame_headers {
        let value_start = start * hidden;
        let value_end = end * hidden;
        let body_len = (end - start) * row_bytes;
        let mut frame = header;
        frame.try_reserve_exact(body_len).map_err(|error| {
            Error::new(
                "EXEC_ALLOC",
                format!("cannot allocate bounded FP16 completion frame: {error}"),
            )
        })?;
        for &bits in &fp16_bits[value_start..value_end] {
            frame.extend_from_slice(&bits.to_le_bytes());
        }
        write_frame(output, &frame, "session FP16 completion batch")?;
    }
    Ok(())
}
/// Read the body of a `completion_raw` header and rebuild the event it stands for. Bounds
/// are checked before allocating; truncated or non-finite bodies are protocol errors, as a
/// non-finite JSON value (serialized `null`) already is.
fn read_raw_completion(
    input: &mut impl BufRead,
    header_bytes: usize,
    stamp: moe::Stamp,
    rank: u32,
    attempt: u64,
    meta: RawCompletionMeta,
) -> Result<Reply> {
    let body_len = usize::try_from(meta.values)
        .ok()
        .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
        .filter(|&n| {
            n <= MAX_RAW_ACTIVATION_BYTES
                && header_bytes
                    .checked_add(1)
                    .and_then(|h| h.checked_add(n))
                    .is_some_and(|total| total <= MAX_MESSAGE_BYTES)
        })
        .ok_or_else(|| {
            Error::new(
                "MESSAGE_TOO_LARGE",
                "raw completion header and body exceed the frame bound",
            )
        })?;
    let mut body = Vec::new();
    body.try_reserve_exact(body_len).map_err(|e| {
        Error::new(
            "EXEC_ALLOC",
            format!("cannot allocate raw completion body: {e}"),
        )
    })?;
    body.resize(body_len, 0);
    input.read_exact(&mut body).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            Error::new(
                "WORKER_PROTOCOL",
                "raw completion body ended before its declared length",
            )
        } else {
            Error::new("WORKER_IO", format!("cannot read raw completion body: {e}"))
        }
    })?;
    let mut value = Vec::new();
    value
        .try_reserve_exact(body_len / std::mem::size_of::<f64>())
        .map_err(|e| {
            Error::new(
                "EXEC_ALLOC",
                format!("cannot allocate raw completion values: {e}"),
            )
        })?;
    for bytes in body.chunks_exact(std::mem::size_of::<f64>()) {
        let x = f64::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| invalid("raw completion f64 value is not eight bytes"))?,
        );
        if !x.is_finite() {
            return Err(Error::new(
                "WORKER_PROTOCOL",
                "raw completion value is not finite",
            ));
        }
        value.push(x);
    }
    Ok(Reply::Event {
        event: WorkerEvent {
            stamp,
            rank,
            attempt,
            event: EventKind::Completion {
                completion: moe::Completion {
                    stamp: meta.stamp,
                    token: meta.token,
                    slot: meta.slot,
                    expert: meta.expert,
                    rank: meta.rank,
                    local: meta.local,
                    value,
                },
            },
        },
    })
}

struct RawCompletionBatchHeader {
    stamp: moe::Stamp,
    completion_stamp: moe::Stamp,
    rank: u32,
    attempt: u64,
    hidden: u64,
    values: u64,
    completions: Vec<RawBatchCompletionMeta>,
}

struct RawCompletionFp16BatchHeader {
    stamp: moe::Stamp,
    completion_stamp: moe::Stamp,
    rank: u32,
    attempt: u64,
    hidden: u64,
    values: u64,
    completions: Vec<RawBatchCompletionMeta>,
}

fn read_raw_completion_batch(
    input: &mut impl BufRead,
    header_bytes: usize,
    header: RawCompletionBatchHeader,
) -> Result<Reply> {
    let hidden = usize::try_from(header.hidden)
        .ok()
        .filter(|&n| (1..=65536).contains(&n))
        .ok_or_else(|| Error::new("WORKER_PROTOCOL", "invalid completion batch hidden width"))?;
    if header.completions.is_empty() {
        return Err(Error::new(
            "WORKER_PROTOCOL",
            "completion batch must contain at least one row",
        ));
    }
    let expected_values = header
        .completions
        .len()
        .checked_mul(hidden)
        .ok_or_else(|| Error::new("MESSAGE_TOO_LARGE", "completion batch shape overflows"))?;
    if usize::try_from(header.values).ok() != Some(expected_values) {
        return Err(Error::new(
            "WORKER_PROTOCOL",
            "completion batch value count differs from its row metadata",
        ));
    }
    let body_len = expected_values
        .checked_mul(std::mem::size_of::<f64>())
        .filter(|&n| n <= MAX_RAW_ACTIVATION_BYTES)
        .filter(|&n| {
            header_bytes
                .checked_add(1)
                .and_then(|h| h.checked_add(n))
                .is_some_and(|total| total <= MAX_MESSAGE_BYTES)
        })
        .ok_or_else(|| {
            Error::new(
                "MESSAGE_TOO_LARGE",
                "completion batch header and body exceed the session bound",
            )
        })?;
    let mut body = Vec::new();
    body.try_reserve_exact(body_len).map_err(|e| {
        Error::new(
            "EXEC_ALLOC",
            format!("cannot allocate completion batch body: {e}"),
        )
    })?;
    body.resize(body_len, 0);
    input.read_exact(&mut body).map_err(|e| {
        Error::new(
            "WORKER_PROTOCOL",
            format!("completion batch body ended before its declared length: {e}"),
        )
    })?;

    let mut completions = Vec::new();
    completions
        .try_reserve_exact(header.completions.len())
        .map_err(|e| Error::new("EXEC_ALLOC", format!("cannot allocate decoded rows: {e}")))?;
    let mut offset = 0;
    for meta in header.completions {
        let mut row = Vec::new();
        row.try_reserve_exact(hidden)
            .map_err(|e| Error::new("EXEC_ALLOC", format!("cannot allocate decoded row: {e}")))?;
        for _ in 0..hidden {
            let x = f64::from_le_bytes(
                body[offset..offset + std::mem::size_of::<f64>()]
                    .try_into()
                    .expect("checked completion value width"),
            );
            offset += std::mem::size_of::<f64>();
            if !x.is_finite() {
                return Err(Error::new(
                    "WORKER_PROTOCOL",
                    "raw completion batch contains a non-finite value",
                ));
            }
            row.push(x);
        }
        completions.push(moe::Completion {
            stamp: header.completion_stamp.clone(),
            token: meta.token,
            slot: meta.slot,
            expert: meta.expert,
            rank: header.rank,
            local: meta.local,
            value: row,
        });
    }
    debug_assert_eq!(offset, body_len);
    Ok(Reply::CompletionBatch {
        stamp: header.stamp,
        rank: header.rank,
        attempt: header.attempt,
        completions,
    })
}

fn read_raw_completion_fp16_batch(
    input: &mut impl BufRead,
    header_bytes: usize,
    header: RawCompletionFp16BatchHeader,
) -> Result<Reply> {
    let hidden = usize::try_from(header.hidden)
        .ok()
        .filter(|&n| (1..=65_536).contains(&n))
        .ok_or_else(|| Error::new("WORKER_PROTOCOL", "invalid FP16 batch hidden width"))?;
    if header.completions.is_empty() {
        return Err(Error::new(
            "WORKER_PROTOCOL",
            "FP16 completion batch must contain at least one row",
        ));
    }
    let expected_values = header
        .completions
        .len()
        .checked_mul(hidden)
        .filter(|&count| count <= crate::completion::MAX_OUTPUT_VALUES)
        .ok_or_else(|| {
            Error::new(
                "MESSAGE_TOO_LARGE",
                "FP16 completion batch shape exceeds its value-count cap",
            )
        })?;
    if usize::try_from(header.values).ok() != Some(expected_values) {
        return Err(Error::new(
            "WORKER_PROTOCOL",
            "FP16 completion batch value count differs from its row metadata",
        ));
    }
    let body_len = expected_values
        .checked_mul(std::mem::size_of::<u16>())
        .filter(|&n| n <= MAX_RAW_ACTIVATION_BYTES)
        .filter(|&n| {
            header_bytes
                .checked_add(1)
                .and_then(|h| h.checked_add(n))
                .is_some_and(|total| total <= MAX_MESSAGE_BYTES)
        })
        .ok_or_else(|| {
            Error::new(
                "MESSAGE_TOO_LARGE",
                "FP16 completion header and body exceed the session bound",
            )
        })?;
    let mut fp16_bits = Vec::new();
    fp16_bits
        .try_reserve_exact(expected_values)
        .map_err(|error| {
            Error::new(
                "EXEC_ALLOC",
                format!("cannot allocate decoded FP16 values: {error}"),
            )
        })?;
    let mut scratch = [0_u8; 8192];
    let mut remaining = body_len;
    while remaining > 0 {
        let chunk_len = remaining.min(scratch.len());
        input
            .read_exact(&mut scratch[..chunk_len])
            .map_err(|error| {
                Error::new(
                    "WORKER_PROTOCOL",
                    format!("FP16 completion body ended before its declared length: {error}"),
                )
            })?;
        for bytes in scratch[..chunk_len].chunks_exact(std::mem::size_of::<u16>()) {
            let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
            if crate::completion::decode_fp16(bits).is_err() {
                return Err(Error::new(
                    "WORKER_PROTOCOL",
                    "FP16 completion batch contains a non-finite value",
                ));
            }
            fp16_bits.push(bits);
        }
        remaining -= chunk_len;
    }
    Ok(Reply::CompletionBatchFp16 {
        stamp: header.stamp,
        completion_stamp: header.completion_stamp,
        rank: header.rank,
        attempt: header.attempt,
        hidden,
        completions: header.completions,
        fp16_bits,
    })
}

/// The coordinator's frame reader admits raw completion frames only after this session
/// requested the matching capability, then decodes them into checked local completions.
#[cfg(test)]
fn read_reply_with_engram(
    input: &mut impl BufRead,
    raw_allowed: &AtomicBool,
    engram_attempt: &AtomicU64,
) -> Result<Option<Reply>> {
    read_reply_with_flags(input, raw_allowed, &AtomicBool::new(false), engram_attempt)
}

#[cfg(test)]
fn read_reply_with_flags(
    input: &mut impl BufRead,
    raw_allowed: &AtomicBool,
    raw_batch_allowed: &AtomicBool,
    engram_attempt: &AtomicU64,
) -> Result<Option<Reply>> {
    read_reply_with_fp16_flags(
        input,
        raw_allowed,
        raw_batch_allowed,
        &AtomicBool::new(false),
        engram_attempt,
    )
}

#[cfg(test)]
fn read_reply_with_fp16_flags(
    input: &mut impl BufRead,
    raw_allowed: &AtomicBool,
    raw_batch_allowed: &AtomicBool,
    raw_fp16_batch_allowed: &AtomicBool,
    engram_attempt: &AtomicU64,
) -> Result<Option<Reply>> {
    read_reply_with_preweighted_flags(
        input,
        raw_allowed,
        raw_batch_allowed,
        raw_fp16_batch_allowed,
        engram_attempt,
        &Mutex::new(None),
    )
}

fn read_reply_with_preweighted_flags(
    input: &mut impl BufRead,
    raw_allowed: &AtomicBool,
    raw_batch_allowed: &AtomicBool,
    raw_fp16_batch_allowed: &AtomicBool,
    engram_attempt: &AtomicU64,
    preweighted_allowed: &Mutex<Option<mxfp4::Admission>>,
) -> Result<Option<Reply>> {
    match read_message_with_len::<Reply>(input)? {
        Some((Reply::PreweightedFp32Raw { header, values }, header_bytes)) => {
            mxfp4::read_completion(input, header, values, preweighted_allowed, header_bytes)
                .map(Some)
        }
        Some((
            Reply::CompletionRaw {
                stamp,
                rank,
                attempt,
                completion,
            },
            header_bytes,
        )) => {
            if !raw_allowed.load(Ordering::SeqCst) {
                return Err(Error::new(
                    "WORKER_PROTOCOL",
                    "unsolicited raw completion frame",
                ));
            }
            read_raw_completion(input, header_bytes, stamp, rank, attempt, completion).map(Some)
        }
        Some((
            Reply::CompletionBatchRaw {
                stamp,
                completion_stamp,
                rank,
                attempt,
                hidden,
                values,
                completions,
            },
            header_bytes,
        )) => {
            if !raw_batch_allowed.load(Ordering::SeqCst) {
                return Err(Error::new(
                    "WORKER_PROTOCOL",
                    "unsolicited raw completion batch frame",
                ));
            }
            read_raw_completion_batch(
                input,
                header_bytes,
                RawCompletionBatchHeader {
                    stamp,
                    completion_stamp,
                    rank,
                    attempt,
                    hidden,
                    values,
                    completions,
                },
            )
            .map(Some)
        }
        Some((
            Reply::CompletionBatchFp16Raw {
                stamp,
                completion_stamp,
                rank,
                attempt,
                hidden,
                values,
                completions,
            },
            header_bytes,
        )) => {
            if !raw_fp16_batch_allowed.load(Ordering::SeqCst) {
                return Err(Error::new(
                    "WORKER_PROTOCOL",
                    "unsolicited raw FP16 completion batch frame",
                ));
            }
            read_raw_completion_fp16_batch(
                input,
                header_bytes,
                RawCompletionFp16BatchHeader {
                    stamp,
                    completion_stamp,
                    rank,
                    attempt,
                    hidden,
                    values,
                    completions,
                },
            )
            .map(Some)
        }
        Some((
            Reply::EngramRowsRaw {
                attempt,
                frame_bytes,
                ..
            },
            header_bytes,
        )) => {
            let expected = engram_attempt.load(Ordering::SeqCst);
            if expected == 0 || attempt != expected {
                return Err(Error::new(
                    "WORKER_PROTOCOL",
                    "unsolicited or stale Engram owner frame",
                ));
            }
            let len = usize::try_from(frame_bytes).map_err(|_| {
                Error::new(
                    "MESSAGE_TOO_LARGE",
                    "Engram frame length does not fit this host",
                )
            })?;
            if len == 0
                || len > MAX_MESSAGE_BYTES
                || header_bytes
                    .checked_add(1)
                    .and_then(|n| n.checked_add(len))
                    .is_none_or(|n| n > MAX_MESSAGE_BYTES + 1)
            {
                return Err(Error::new(
                    "MESSAGE_TOO_LARGE",
                    "Engram owner frame exceeds the session bound",
                ));
            }
            let mut frame = Vec::new();
            frame.try_reserve_exact(len).map_err(|e| {
                Error::new(
                    "ENGRAM_ALLOC",
                    format!("cannot allocate Engram owner frame: {e}"),
                )
            })?;
            frame.resize(len, 0);
            input.read_exact(&mut frame).map_err(|e| {
                Error::new(
                    "WORKER_PROTOCOL",
                    format!("Engram owner frame ended before its declared length: {e}"),
                )
            })?;
            Ok(Some(Reply::EngramRowsRaw {
                attempt,
                frame_bytes,
                frame,
            }))
        }
        other => Ok(other.map(|(reply, _)| reply)),
    }
}
#[cfg(test)]
fn read_reply(input: &mut impl BufRead, raw_allowed: &AtomicBool) -> Result<Option<Reply>> {
    read_reply_with_engram(input, raw_allowed, &AtomicU64::new(0))
}
fn send_reply(output: &mut impl Write, reply: &Reply) -> Result<()> {
    write_frame(output, &encode(reply)?, "session reply")
}
fn hello(rank: u32) -> ServerHello {
    ServerHello {
        protocol: PROTOCOL,
        rank,
        arch: std::env::consts::ARCH.into(),
        os: std::env::consts::OS.into(),
        endian: if cfg!(target_endian = "little") {
            "little"
        } else {
            "big"
        }
        .into(),
        build: env!("CARGO_PKG_VERSION").into(),
        capabilities: vec![
            CAPABILITY.into(),
            mxfp4::CAPABILITY.into(),
            mxfp4::BF16_ACTIVATION_CAPABILITY.into(),
            CAPABILITY_QPN2.into(),
            CAPABILITY_QPN2_RAW.into(),
            CAPABILITY_EXEC_F64_RAW.into(),
            CAPABILITY_EXEC_FP16_RAW.into(),
            CAPABILITY_COMPLETION_F64_RAW.into(),
            CAPABILITY_COMPLETION_BATCH_F64_RAW.into(),
            CAPABILITY_COMPLETION_BATCH_SESSION_F64_RAW.into(),
            CAPABILITY_COMPLETION_BATCH_FP16_RAW.into(),
            CAPABILITY_ENGRAM_ROWS_RAW.into(),
            CAPABILITY_ENGRAM_ROW_IDS_VARINT.into(),
        ],
        max_message_bytes: MAX_MESSAGE_BYTES,
        pid: std::process::id(),
    }
}
fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}
fn refuse(code: &str, message: &str) -> Reply {
    Reply::Refused {
        error: Error::new(code, message),
    }
}
fn validate_resident(model: &moe::Model, order: &[u32], revision: &str) -> Result<()> {
    let n = model.hidden.checked_mul(model.intermediate);
    if model.experts.is_empty()
        || model.experts.len() > 1024
        || order.len() != model.experts.len()
        || order.iter().collect::<BTreeSet<_>>().len() != order.len()
        || model
            .experts
            .iter()
            .map(|e| e.id)
            .collect::<BTreeSet<_>>()
            .len()
            != model.experts.len()
        || model.hidden == 0
        || model.hidden > 65536
        || model.intermediate == 0
        || n.is_none()
        || revision.trim().is_empty()
        || revision.len() > 256
    {
        return Err(invalid(
            "resident artifact identity or dimensions outside the session contract",
        ));
    }
    let n = n.unwrap_or(0);
    if model.experts.iter().any(|e| {
        e.gate.len() != n
            || e.up.len() != n
            || e.down.len() != n
            || e.gate
                .iter()
                .chain(&e.up)
                .chain(&e.down)
                .any(|x| !x.is_finite())
    }) {
        return Err(invalid(
            "resident expert weights are malformed or nonfinite",
        ));
    }
    Ok(())
}
/// One LOAD_ARTIFACT request as the worker received it.
pub(crate) struct ArtifactLoad {
    artifact_sha256: String,
    model_revision: String,
    layer: u32,
    epoch: u64,
    resident_order: Vec<u32>,
    artifact: Vec<u8>,
}

fn read_artifact_body(input: &mut impl BufRead, declared: u64) -> Result<Vec<u8>> {
    read_raw_body(input, 0, declared, false)
}

fn read_raw_body(
    input: &mut impl BufRead,
    header_bytes: usize,
    declared: u64,
    allow_empty: bool,
) -> Result<Vec<u8>> {
    let len = usize::try_from(declared).map_err(|_| {
        Error::new(
            "MESSAGE_TOO_LARGE",
            "raw artifact length does not fit this host",
        )
    })?;
    if len == 0 && !allow_empty {
        return Err(invalid("raw artifact body must not be empty"));
    }
    if len > MAX_RAW_ARTIFACT_BYTES
        || header_bytes
            .checked_add(1)
            .and_then(|n| n.checked_add(len))
            .is_none_or(|total| total > MAX_MESSAGE_BYTES + 1)
    {
        return Err(Error::new(
            "MESSAGE_TOO_LARGE",
            "raw artifact header and body exceed the session bound",
        ));
    }
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len).map_err(|e| {
        Error::new(
            "ARTIFACT_ALLOC",
            format!("cannot allocate raw artifact body: {e}"),
        )
    })?;
    bytes.resize(len, 0);
    input.read_exact(&mut bytes).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            Error::new(
                "ARTIFACT_TRUNCATED",
                "raw artifact body ended before its declared length",
            )
        } else {
            Error::new("ARTIFACT_IO", format!("cannot read raw artifact body: {e}"))
        }
    })?;
    Ok(bytes)
}

const MAX_VARINT_BYTES_PER_ID: usize = 10;
const MAX_ENGRAM_ROW_ID_BODY_BYTES: usize = engram::MAX_LOOKUPS * MAX_VARINT_BYTES_PER_ID;

#[cfg(test)]
fn encode_engram_row_ids_varint(row_ids: &[u64]) -> Result<Vec<u8>> {
    encode_engram_row_ids_varint_with_json_digit_bytes(row_ids).map(|(body, _)| body)
}

fn encode_engram_row_ids_varint_with_json_digit_bytes(row_ids: &[u64]) -> Result<(Vec<u8>, usize)> {
    if row_ids.len() > engram::MAX_LOOKUPS {
        return Err(invalid("Engram row ID count exceeds the request limit"));
    }
    let mut body = Vec::new();
    let initial_capacity = row_ids
        .len()
        .checked_mul(3)
        .ok_or_else(|| Error::new("MESSAGE_TOO_LARGE", "Engram row ID body size overflow"))?;
    body.try_reserve_exact(initial_capacity)
        .map_err(|e| Error::new("ENGRAM_ALLOC", format!("cannot allocate row ID body: {e}")))?;
    let mut json_digit_bytes = 0_usize;
    for &row_id in row_ids {
        let mut decimal = row_id;
        let mut decimal_digits = 1_usize;
        while decimal >= 10 {
            decimal /= 10;
            decimal_digits += 1;
        }
        json_digit_bytes = json_digit_bytes
            .checked_add(decimal_digits)
            .ok_or_else(|| Error::new("MESSAGE_TOO_LARGE", "Engram JSON ID size overflow"))?;

        let mut value = row_id;
        let mut encoded = [0_u8; MAX_VARINT_BYTES_PER_ID];
        let mut len = 0;
        while value >= 0x80 {
            encoded[len] = (value as u8) | 0x80;
            len += 1;
            value >>= 7;
        }
        encoded[len] = value as u8;
        len += 1;
        if body
            .len()
            .checked_add(len)
            .is_none_or(|total| total > MAX_ENGRAM_ROW_ID_BODY_BYTES)
        {
            return Err(invalid("Engram row ID body exceeds the request limit"));
        }
        if body.capacity() - body.len() < len {
            body.try_reserve(len)
                .map_err(|e| Error::new("ENGRAM_ALLOC", format!("cannot grow row ID body: {e}")))?;
        }
        body.extend_from_slice(&encoded[..len]);
    }
    if body.len() > MAX_ENGRAM_ROW_ID_BODY_BYTES {
        return Err(invalid("Engram row ID body exceeds the request limit"));
    }
    Ok((body, json_digit_bytes))
}

fn decode_engram_row_ids_varint(bytes: &[u8], count: usize) -> Result<Vec<u64>> {
    if count > engram::MAX_LOOKUPS {
        return Err(Error::new(
            "MESSAGE_TOO_LARGE",
            "Engram row ID count exceeds the request limit",
        ));
    }
    let maximum_bytes = count
        .checked_mul(MAX_VARINT_BYTES_PER_ID)
        .ok_or_else(|| Error::new("MESSAGE_TOO_LARGE", "Engram row ID body size overflow"))?;
    if bytes.len() < count || bytes.len() > maximum_bytes {
        return Err(Error::new(
            "WORKER_PROTOCOL",
            "Engram row ID body length differs from its declared count",
        ));
    }
    let mut row_ids = Vec::new();
    row_ids.try_reserve_exact(count).map_err(|e| {
        Error::new(
            "ENGRAM_ALLOC",
            format!("cannot allocate decoded row IDs: {e}"),
        )
    })?;
    let mut offset = 0;
    for _ in 0..count {
        let start = offset;
        let mut value = 0_u64;
        let mut shift = 0_u32;
        loop {
            let byte = *bytes
                .get(offset)
                .ok_or_else(|| Error::new("WORKER_PROTOCOL", "truncated Engram row ID varint"))?;
            offset += 1;
            let payload = byte & 0x7f;
            if shift == 63 && (payload > 1 || byte & 0x80 != 0) {
                return Err(Error::new(
                    "WORKER_PROTOCOL",
                    "Engram row ID varint overflows u64",
                ));
            }
            value |= u64::from(payload) << shift;
            if byte & 0x80 == 0 {
                if offset - start > 1 && payload == 0 {
                    return Err(Error::new(
                        "WORKER_PROTOCOL",
                        "non-canonical Engram row ID varint",
                    ));
                }
                row_ids.push(value);
                break;
            }
            shift += 7;
            if shift > 63 {
                return Err(Error::new(
                    "WORKER_PROTOCOL",
                    "Engram row ID varint exceeds ten bytes",
                ));
            }
        }
    }
    if offset != bytes.len() {
        return Err(Error::new(
            "WORKER_PROTOCOL",
            "extra bytes after declared Engram row IDs",
        ));
    }
    Ok(row_ids)
}

fn read_engram_row_ids_body(
    input: &mut impl BufRead,
    header_bytes: usize,
    row_id_bytes: u64,
    row_id_count: u64,
) -> Result<Vec<u64>> {
    let count = usize::try_from(row_id_count)
        .map_err(|_| Error::new("MESSAGE_TOO_LARGE", "Engram row count does not fit host"))?;
    if count > engram::MAX_LOOKUPS {
        return Err(Error::new(
            "MESSAGE_TOO_LARGE",
            "Engram row ID count exceeds the request limit",
        ));
    }
    let byte_len = usize::try_from(row_id_bytes)
        .map_err(|_| Error::new("MESSAGE_TOO_LARGE", "Engram row bytes do not fit host"))?;
    let maximum_bytes = count
        .checked_mul(MAX_VARINT_BYTES_PER_ID)
        .ok_or_else(|| Error::new("MESSAGE_TOO_LARGE", "Engram row ID body size overflow"))?;
    if byte_len < count || byte_len > maximum_bytes || byte_len > MAX_ENGRAM_ROW_ID_BODY_BYTES {
        return Err(Error::new(
            "WORKER_PROTOCOL",
            "Engram row ID body length differs from its declared count",
        ));
    }
    let bytes = read_raw_body(input, header_bytes, row_id_bytes, true)?;
    decode_engram_row_ids_varint(&bytes, count)
}

fn encode_engram_rows_request(
    stamp: &moe::Stamp,
    table: &TableIdentity,
    world_size: u32,
    owner_rank: u32,
    row_ids: &[u64],
    attempt: u64,
    supports_varint: bool,
) -> Result<(Vec<u8>, Option<Vec<u8>>)> {
    if supports_varint {
        let (body, json_digit_bytes) = encode_engram_row_ids_varint_with_json_digit_bytes(row_ids)?;
        let row_id_count = u64::try_from(row_ids.len())
            .map_err(|_| Error::new("MESSAGE_TOO_LARGE", "Engram row count does not fit u64"))?;
        let row_id_bytes = u64::try_from(body.len())
            .map_err(|_| Error::new("MESSAGE_TOO_LARGE", "Engram row bytes do not fit u64"))?;
        let header = encode(&Request::FetchEngramRowsVarint {
            stamp: stamp.clone(),
            table: table.clone(),
            world_size,
            owner_rank,
            row_id_count,
            row_id_bytes,
            attempt,
        })?;
        let varint_len = header.len().checked_add(body.len()).ok_or_else(|| {
            Error::new("MESSAGE_TOO_LARGE", "Engram varint request size overflow")
        })?;
        let empty_json = encode(&Request::FetchEngramRows {
            stamp: stamp.clone(),
            table: table.clone(),
            world_size,
            owner_rank,
            row_ids: Vec::new(),
            attempt,
        })?;
        let comma_bytes = row_ids.len().saturating_sub(1);
        let json_len = empty_json
            .len()
            .checked_add(comma_bytes)
            .and_then(|base| base.checked_add(json_digit_bytes))
            .ok_or_else(|| Error::new("MESSAGE_TOO_LARGE", "Engram JSON request size overflow"))?;

        if varint_len < json_len && varint_len <= MAX_MESSAGE_BYTES {
            return Ok((header, Some(body)));
        }
        if json_len > MAX_MESSAGE_BYTES {
            return Err(Error::new(
                "MESSAGE_TOO_LARGE",
                "Engram row ID request exceeds the session bound",
            ));
        }
    }
    let header = encode(&Request::FetchEngramRows {
        stamp: stamp.clone(),
        table: table.clone(),
        world_size,
        owner_rank,
        row_ids: row_ids.to_vec(),
        attempt,
    })?;
    Ok((header, None))
}

fn read_raw_activation_inputs(
    input: &mut impl BufRead,
    header_bytes: usize,
    declared: u64,
    metadata: Vec<RawTokenInputMeta>,
) -> Result<Vec<TokenInput>> {
    if metadata.len() > 4096 || metadata.iter().any(|item| item.values > 65536) {
        return Err(invalid(
            "raw EXEC input metadata exceeds the token or row bound",
        ));
    }
    let body_len = usize::try_from(declared)
        .map_err(|_| invalid("raw EXEC byte count does not fit this host"))?;
    if body_len > MAX_RAW_ACTIVATION_BYTES
        || header_bytes
            .checked_add(1)
            .and_then(|n| n.checked_add(body_len))
            .is_none_or(|n| n > MAX_MESSAGE_BYTES)
    {
        return Err(Error::new(
            "MESSAGE_TOO_LARGE",
            "raw EXEC header and activation body exceed the frame bound",
        ));
    }
    let expected = metadata.iter().try_fold(0_usize, |total, item| {
        let values = usize::try_from(item.values).ok()?;
        total.checked_add(values.checked_mul(std::mem::size_of::<f64>())?)
    });
    if expected != Some(body_len) {
        return Err(invalid(
            "raw EXEC activation byte count differs from its token metadata",
        ));
    }
    let mut body = Vec::new();
    body.try_reserve_exact(body_len).map_err(|e| {
        Error::new(
            "EXEC_ALLOC",
            format!("cannot allocate raw activation body: {e}"),
        )
    })?;
    body.resize(body_len, 0);
    input.read_exact(&mut body).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            Error::new(
                "EXEC_TRUNCATED",
                "raw activation body ended before its declared length",
            )
        } else {
            Error::new("EXEC_IO", format!("cannot read raw activation body: {e}"))
        }
    })?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(metadata.len())
        .map_err(|e| Error::new("EXEC_ALLOC", format!("cannot allocate raw input rows: {e}")))?;
    let mut offset = 0_usize;
    for item in metadata {
        let values_len = usize::try_from(item.values)
            .map_err(|_| invalid("raw EXEC row length does not fit this host"))?;
        let row_bytes = values_len
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| invalid("raw EXEC row byte count overflow"))?;
        let end = offset
            .checked_add(row_bytes)
            .ok_or_else(|| invalid("raw EXEC body offset overflow"))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(values_len)
            .map_err(|e| Error::new("EXEC_ALLOC", format!("cannot allocate raw input row: {e}")))?;
        for bytes in body[offset..end].chunks_exact(std::mem::size_of::<f64>()) {
            values.push(f64::from_le_bytes(
                bytes
                    .try_into()
                    .map_err(|_| invalid("raw EXEC f64 value is not eight bytes"))?,
            ));
        }
        rows.push(TokenInput {
            token: item.token,
            value: values,
        });
        offset = end;
    }
    debug_assert_eq!(offset, body.len());
    Ok(rows)
}

fn read_raw_activation_inputs_fp16(
    input: &mut impl BufRead,
    header_bytes: usize,
    declared: u64,
    metadata: Vec<RawTokenInputMeta>,
) -> Result<Vec<TokenInput>> {
    if metadata.len() > 4096 || metadata.iter().any(|item| item.values > 65536) {
        return Err(invalid(
            "raw FP16 EXEC input metadata exceeds the token or row bound",
        ));
    }
    let body_len = usize::try_from(declared)
        .map_err(|_| invalid("raw FP16 EXEC byte count does not fit this host"))?;
    if body_len > MAX_RAW_ACTIVATION_BYTES
        || header_bytes
            .checked_add(1)
            .and_then(|n| n.checked_add(body_len))
            .is_none_or(|n| n > MAX_MESSAGE_BYTES)
    {
        return Err(Error::new(
            "MESSAGE_TOO_LARGE",
            "raw FP16 EXEC header and activation body exceed the frame bound",
        ));
    }
    let expected = metadata.iter().try_fold(0_usize, |total, item| {
        let values = usize::try_from(item.values).ok()?;
        total.checked_add(values.checked_mul(std::mem::size_of::<u16>())?)
    });
    if expected != Some(body_len) {
        return Err(invalid(
            "raw FP16 EXEC activation byte count differs from its token metadata",
        ));
    }
    let mut body = Vec::new();
    body.try_reserve_exact(body_len).map_err(|e| {
        Error::new(
            "EXEC_ALLOC",
            format!("cannot allocate raw FP16 activation body: {e}"),
        )
    })?;
    body.resize(body_len, 0);
    input.read_exact(&mut body).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            Error::new(
                "EXEC_TRUNCATED",
                "raw FP16 activation body ended before its declared length",
            )
        } else {
            Error::new(
                "EXEC_IO",
                format!("cannot read raw FP16 activation body: {e}"),
            )
        }
    })?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(metadata.len()).map_err(|e| {
        Error::new(
            "EXEC_ALLOC",
            format!("cannot allocate raw FP16 input rows: {e}"),
        )
    })?;
    let mut offset = 0_usize;
    for item in metadata {
        let values_len = usize::try_from(item.values)
            .map_err(|_| invalid("raw FP16 EXEC row length does not fit this host"))?;
        let row_bytes = values_len
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| invalid("raw FP16 EXEC row byte count overflow"))?;
        let end = offset
            .checked_add(row_bytes)
            .ok_or_else(|| invalid("raw FP16 EXEC body offset overflow"))?;
        let mut values = Vec::new();
        values.try_reserve_exact(values_len).map_err(|e| {
            Error::new(
                "EXEC_ALLOC",
                format!("cannot allocate decoded FP16 input row: {e}"),
            )
        })?;
        for bytes in body[offset..end].chunks_exact(std::mem::size_of::<u16>()) {
            let bits = u16::from_le_bytes(
                bytes
                    .try_into()
                    .map_err(|_| invalid("raw FP16 value is not two bytes"))?,
            );
            let value = crate::completion::decode_fp16(bits).map_err(|_| {
                invalid("raw FP16 activation input must contain only finite values")
            })?;
            values.push(value);
        }
        rows.push(TokenInput {
            token: item.token,
            value: values,
        });
        offset = end;
    }
    debug_assert_eq!(offset, body.len());
    Ok(rows)
}

/// Admit a compressed artifact: its bytes must hash to the announced SHA-256, decode to a
/// valid canonical stack of this session's rank, and match the LOAD identity.
fn compressed(rank: u32, request: &ArtifactLoad) -> Result<ResidentStack> {
    if sha256::hex(&request.artifact) != request.artifact_sha256 {
        return Err(Error::new(
            "ARTIFACT_DIGEST_MISMATCH",
            "artifact_sha256 differs from the SHA-256 of the received artifact",
        ));
    }
    let mut stack = resident_artifact::import(&request.artifact)?;
    if stack.is_empty()
        || stack.rank() != rank
        || stack.model_revision() != request.model_revision
        || stack.layer() != request.layer
        || stack.resident_order() != request.resident_order
    {
        return Err(invalid(
            "artifact rank, revision, layer or physical order differs from this LOAD",
        ));
    }
    if request.epoch < stack.epoch() {
        return Err(Error::new(
            "EPOCH_ROLLBACK",
            "ownership epoch is older than the artifact's",
        ));
    }
    stack.set_epoch(request.epoch);
    Ok(stack)
}

fn load_artifact(resident: &mut Option<Resident>, rank: u32, request: ArtifactLoad) -> Reply {
    if resident.is_some() {
        return refuse(
            "ARTIFACT_LOADED",
            "evict the loaded artifact before loading another",
        );
    }
    match compressed(rank, &request) {
        Ok(stack) => {
            let experts = stack.physical_experts().len();
            *resident = Some(Resident {
                digest: request.artifact_sha256.clone(),
                weights: Weights::Compressed(Box::new(stack)),
                model_revision: request.model_revision,
                layer: request.layer,
                epoch: request.epoch,
                resident_order: request.resident_order,
            });
            Reply::Loaded {
                artifact_sha256: request.artifact_sha256,
                epoch: request.epoch,
                experts,
                transferred: true,
            }
        }
        Err(error) => Reply::Refused { error },
    }
}

fn load_raw_artifact(
    resident: &mut Option<Resident>,
    rank: u32,
    artifact_sha256: String,
    epoch: u64,
    artifact: Vec<u8>,
) -> Reply {
    if resident.is_some() {
        return refuse(
            "ARTIFACT_LOADED",
            "evict the loaded artifact before loading another",
        );
    }
    if sha256::hex(&artifact) != artifact_sha256 {
        return refuse(
            "ARTIFACT_DIGEST_MISMATCH",
            "artifact_sha256 differs from the SHA-256 of the received artifact",
        );
    }
    let mut stack = match resident_artifact::import(&artifact) {
        Ok(stack) => stack,
        Err(error) => return Reply::Refused { error },
    };
    if stack.is_empty() || stack.rank() != rank {
        return refuse(
            "INVALID_INPUT",
            "artifact is empty or belongs to a different worker rank",
        );
    }
    if epoch < stack.epoch() {
        return refuse(
            "EPOCH_ROLLBACK",
            "ownership epoch is older than the artifact's",
        );
    }
    stack.set_epoch(epoch);
    let experts = stack.physical_experts().len();
    *resident = Some(Resident {
        digest: artifact_sha256.clone(),
        model_revision: stack.model_revision().into(),
        layer: stack.layer(),
        epoch,
        resident_order: stack.resident_order().to_vec(),
        weights: Weights::Compressed(Box::new(stack)),
    });
    Reply::Loaded {
        artifact_sha256,
        epoch,
        experts,
        transferred: true,
    }
}

fn load_engram_shard(
    resident: &mut Option<EngramShard>,
    table: TableIdentity,
    world_size: u32,
    shard_rank: u32,
    artifact_sha256: String,
    bytes: Vec<u8>,
) -> Reply {
    if shard_rank >= world_size {
        return refuse(
            "INVALID_INPUT",
            "Engram owner rank is outside the shard world",
        );
    }
    let computed = match engram::artifact_sha256(&table, world_size, shard_rank, &bytes) {
        Ok(digest) => digest,
        Err(error) => return refuse("INVALID_INPUT", error.0),
    };
    if computed != artifact_sha256 {
        return refuse(
            "ARTIFACT_DIGEST_MISMATCH",
            "Engram artifact digest differs from identity and shard bytes",
        );
    }
    if let Some(existing) = resident.as_ref() {
        if existing.sha256() == artifact_sha256
            && existing.table() == &table
            && existing.world_size() == world_size
            && existing.rank() == shard_rank
        {
            return Reply::EngramLoaded {
                artifact_sha256,
                table,
                transferred: false,
            };
        }
        return refuse(
            "ENGRAM_ARTIFACT_LOADED",
            "evict the Engram shard before loading a different one",
        );
    }
    let shard = match EngramShard::new(table.clone(), world_size, shard_rank, bytes) {
        Ok(shard) => shard,
        Err(error) => return refuse("INVALID_INPUT", error.0),
    };
    if shard.sha256() != artifact_sha256 {
        return refuse(
            "ARTIFACT_DIGEST_MISMATCH",
            "Engram shard digest changed during admission",
        );
    }
    *resident = Some(shard);
    Reply::EngramLoaded {
        artifact_sha256,
        table,
        transferred: true,
    }
}

#[allow(clippy::too_many_arguments)]
fn load(
    resident: &mut Option<Resident>,
    artifact_sha256: String,
    model_revision: String,
    layer: u32,
    epoch: u64,
    resident_order: Vec<u32>,
    model: Option<moe::Model>,
) -> Reply {
    let Some(model) = model else {
        let Some(r) = resident.as_mut().filter(|r| r.digest == artifact_sha256) else {
            return refuse("ARTIFACT_NOT_LOADED", "no loaded artifact has this SHA-256");
        };
        if model_revision != r.model_revision
            || layer != r.layer
            || resident_order != r.resident_order
        {
            return refuse(
                "INVALID_INPUT",
                "rebinding cannot change revision, layer or physical order",
            );
        }
        if epoch < r.epoch {
            return refuse(
                "EPOCH_ROLLBACK",
                "ownership epoch is older than the loaded one",
            );
        }
        if epoch == r.epoch {
            return refuse(
                "EPOCH_NOT_NEWER",
                "rebinding requires a newer ownership epoch",
            );
        }
        let experts = match &mut r.weights {
            Weights::Dense(model) => model.experts.len(),
            Weights::Compressed(stack) => {
                stack.set_epoch(epoch);
                stack.physical_experts().len()
            }
            Weights::Mxfp4(artifact) => artifact.rebind(epoch),
        };
        r.epoch = epoch;
        return Reply::Loaded {
            artifact_sha256,
            epoch,
            experts,
            transferred: false,
        };
    };
    if resident.is_some() {
        return refuse(
            "ARTIFACT_LOADED",
            "evict the loaded artifact before loading another",
        );
    }
    match artifact_digest(&model) {
        Ok(digest) if digest == artifact_sha256 => {}
        Ok(_) => {
            return refuse(
                "ARTIFACT_DIGEST_MISMATCH",
                "artifact_sha256 differs from the SHA-256 of the received model",
            )
        }
        Err(error) => return Reply::Refused { error },
    }
    if let Err(error) = validate_resident(&model, &resident_order, &model_revision) {
        return Reply::Refused { error };
    }
    let experts = model.experts.len();
    *resident = Some(Resident {
        digest: artifact_sha256.clone(),
        weights: Weights::Dense(model),
        model_revision,
        layer,
        epoch,
        resident_order,
    });
    Reply::Loaded {
        artifact_sha256,
        epoch,
        experts,
        transferred: true,
    }
}

/// The CPU budget of one compressed EXEC on one rank: slots x 3 planes x hidden x
/// intermediate multiply-adds, at most 1e8. The worker enforces it; the coordinator
/// checks the same bound before any rank is sent work.
fn within_work_budget(stack: &ResidentStack, slots: usize) -> bool {
    let (_, n13, k13, _, _) = stack.shape();
    (slots as u64)
        .checked_mul(n13 / 2 * k13 * 3)
        .is_some_and(|n| n <= 100_000_000)
}

fn over_budget() -> Error {
    invalid("compressed EXEC needs 1 slot or more within the rank CPU work budget")
}

/// Validate one EXEC against a compressed resident: slots against the stack's static
/// identity and physical experts, activations against the dispatch, and the same CPU
/// work and fault bounds as the dense worker.
fn compressed_job(
    stack: &ResidentStack,
    dispatch: routing::RankDispatch,
    inputs: Vec<TokenInput>,
    fault: Fault,
) -> Result<Job> {
    stack.bind_dispatch(&dispatch)?;
    if dispatch.slots.is_empty() || !within_work_budget(stack, dispatch.slots.len()) {
        return Err(over_budget());
    }
    check_fault(&fault)?;
    let inputs = token_inputs(&dispatch, inputs)?;
    Ok(Job {
        dispatch,
        fault,
        inputs,
    })
}

/// Serve one session on stdin/stdout until SHUTDOWN. A framing or protocol error is
/// answered once and closes the session; any other refusal leaves it usable.
///
/// Replies go to a duplicate of fd 1, not through `io::stdout()`: std's stdout is
/// line-buffered, so a binary frame would leave in two writes split after its last `\n`.
/// `write_frame` then hands each whole frame to one `write_all` (partial writes are
/// continued) and nothing is held across frames. If fd 1 cannot be duplicated (closed),
/// std's stdout is kept, with its behaviour for a closed descriptor.
pub fn session_main() -> Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    #[cfg(unix)]
    if let Ok(fd) = std::os::fd::AsFd::as_fd(&stdout).try_clone_to_owned() {
        return serve(&mut stdin.lock(), &mut std::fs::File::from(fd));
    }
    serve(&mut stdin.lock(), &mut stdout.lock())
}

pub fn serve(input: &mut impl BufRead, output: &mut impl Write) -> Result<()> {
    serve_impl(input, output, &mut None)
}

pub fn session_main_with_mxfp4_executor(executor: &mut dyn mxfp4::Executor) -> Result<()> {
    executor.producer().validate()?;
    if !matches!(
        executor.producer(),
        mxfp4::Producer::GpuR13ResidentFp32V1 { .. }
    ) {
        return Err(invalid(
            "native executor must identify its qualified GPU producer",
        ));
    }
    let stdin = io::stdin();
    let stdout = io::stdout();
    #[cfg(unix)]
    if let Ok(fd) = std::os::fd::AsFd::as_fd(&stdout).try_clone_to_owned() {
        return serve_impl(
            &mut stdin.lock(),
            &mut std::fs::File::from(fd),
            &mut Some(executor),
        );
    }
    serve_impl(&mut stdin.lock(), &mut stdout.lock(), &mut Some(executor))
}

pub fn serve_with_mxfp4_executor(
    input: &mut impl BufRead,
    output: &mut impl Write,
    executor: &mut dyn mxfp4::Executor,
) -> Result<()> {
    executor.producer().validate()?;
    if !matches!(
        executor.producer(),
        mxfp4::Producer::GpuR13ResidentFp32V1 { .. }
    ) {
        return Err(invalid(
            "native executor must identify its qualified GPU producer",
        ));
    }
    serve_impl(input, output, &mut Some(executor))
}
fn serve_impl(
    input: &mut impl BufRead,
    output: &mut impl Write,
    executor: &mut Option<&mut dyn mxfp4::Executor>,
) -> Result<()> {
    let mut bound: Option<u32> = None;
    let mut resident: Option<Resident> = None;
    let mut engram_resident: Option<EngramShard> = None;
    let mut last_attempt = 0_u64;
    let mut last_engram_attempt = 0_u64;
    let mut execs = 0_u64;
    let mut raw_completion_batch_latched = false;
    #[cfg(feature = "native-ggml-cpu")]
    let mut native_owners: Vec<crate::native_ggml_cpu::Resident> = Vec::new();
    #[cfg(feature = "native-ggml-cpu")]
    let mut native_worker: Option<crate::native_ggml_cpu::RankWorker> = None;
    loop {
        let mut raw_completions = false;
        let mut raw_completion_batch = false;
        let mut raw_completion_fp16_batch = false;
        let (request, header_bytes): (Request, usize) = match read_message_with_len(input) {
            Ok(Some(frame)) => frame,
            Ok(None) => {
                return Err(Error::new(
                    "SESSION_ABANDONED",
                    "stream closed without SHUTDOWN",
                ))
            }
            Err(error) => {
                // The framing can no longer be trusted: answer once, then close.
                let _ = send_reply(
                    output,
                    &Reply::Refused {
                        error: error.clone(),
                    },
                );
                return Err(error);
            }
        };
        if executor.is_some()
            && !matches!(
                request,
                Request::Hello { .. }
                    | Request::LoadMxfp4Raw { .. }
                    | Request::ExecPreweightedFp32Raw { .. }
                    | Request::ExecPreweightedBf16Raw { .. }
                    | Request::ExecPreweightedFp32 { .. }
                    | Request::Load { model: None, .. }
                    | Request::Evict { .. }
                    | Request::Shutdown
            )
        {
            let error = Error::new(
                "CAPABILITY_MISSING",
                "GPU worker only admits the explicit MXFP4 session path",
            );
            let _ = send_reply(
                output,
                &Reply::Refused {
                    error: error.clone(),
                },
            );
            return Err(error); // unknown body position: close, never use a CPU fallback.
        }
        let bf16_activations = matches!(&request, Request::ExecPreweightedBf16Raw { .. });
        let request = if bound.is_some() {
            match request {
                Request::ExecPreweightedFp32Raw {
                    artifact_sha256,
                    dispatch,
                    contract,
                    attempt,
                    inputs,
                    activation_bytes,
                    fault,
                }
                | Request::ExecPreweightedBf16Raw {
                    artifact_sha256,
                    dispatch,
                    contract,
                    attempt,
                    inputs,
                    activation_bytes,
                    fault,
                } => {
                    let inputs = match mxfp4::read_inputs(
                        input,
                        header_bytes,
                        activation_bytes,
                        inputs,
                        bf16_activations,
                    ) {
                        Ok(inputs) => inputs,
                        Err(error) => {
                            send_reply(
                                output,
                                &Reply::Refused {
                                    error: error.clone(),
                                },
                            )?;
                            return Err(error);
                        }
                    };
                    Request::ExecPreweightedFp32 {
                        artifact_sha256,
                        dispatch,
                        contract,
                        attempt,
                        inputs,
                        fault,
                    }
                }
                Request::FetchEngramRowsVarint {
                    stamp,
                    table,
                    world_size,
                    owner_rank,
                    row_id_count,
                    row_id_bytes,
                    attempt,
                } => {
                    let row_ids = match read_engram_row_ids_body(
                        input,
                        header_bytes,
                        row_id_bytes,
                        row_id_count,
                    ) {
                        Ok(row_ids) => row_ids,
                        Err(error) => {
                            send_reply(
                                output,
                                &Reply::Refused {
                                    error: error.clone(),
                                },
                            )?;
                            // A malformed raw body closes the stream; the attempt is never retried.
                            return Err(error);
                        }
                    };
                    Request::FetchEngramRows {
                        stamp,
                        table,
                        world_size,
                        owner_rank,
                        row_ids,
                        attempt,
                    }
                }
                Request::ExecRaw {
                    artifact_sha256,
                    dispatch,
                    attempt,
                    inputs,
                    activation_bytes,
                    fault,
                    raw_completions: raw,
                    raw_completion_batch: raw_batch,
                    raw_completion_fp16_batch: raw_fp16_batch,
                } => {
                    raw_completions = raw;
                    raw_completion_batch = raw && raw_completion_batch_latched;
                    raw_completion_fp16_batch = raw_fp16_batch;
                    let inputs = match read_raw_activation_inputs(
                        input,
                        header_bytes,
                        activation_bytes,
                        inputs,
                    ) {
                        Ok(inputs) => inputs,
                        Err(error) => {
                            send_reply(
                                output,
                                &Reply::Refused {
                                    error: error.clone(),
                                },
                            )?;
                            // The body boundary can no longer be trusted; this stream is
                            // closed and the started EXEC is never replayed.
                            return Err(error);
                        }
                    };
                    if (raw_batch || raw_fp16_batch) && !raw {
                        send_reply(
                            output,
                            &refuse(
                                "SESSION_PROTOCOL",
                                "raw completion batching requires raw completion mode",
                            ),
                        )?;
                        continue;
                    }
                    if raw && raw_batch {
                        raw_completion_batch_latched = true;
                        raw_completion_batch = true;
                    }
                    Request::Exec {
                        artifact_sha256,
                        dispatch,
                        attempt,
                        inputs,
                        fault,
                    }
                }
                Request::ExecRawFp16 {
                    artifact_sha256,
                    dispatch,
                    attempt,
                    inputs,
                    activation_bytes,
                    fault,
                    raw_completions: raw,
                    raw_completion_batch: raw_batch,
                    raw_completion_fp16_batch: raw_fp16_batch,
                } => {
                    raw_completions = raw;
                    raw_completion_batch = raw && raw_completion_batch_latched;
                    raw_completion_fp16_batch = raw_fp16_batch;
                    let inputs = match read_raw_activation_inputs_fp16(
                        input,
                        header_bytes,
                        activation_bytes,
                        inputs,
                    ) {
                        Ok(inputs) => inputs,
                        Err(error) => {
                            send_reply(
                                output,
                                &Reply::Refused {
                                    error: error.clone(),
                                },
                            )?;
                            // A malformed typed body closes the stream; its attempt is never replayed.
                            return Err(error);
                        }
                    };
                    if (raw_batch || raw_fp16_batch) && !raw {
                        send_reply(
                            output,
                            &refuse(
                                "SESSION_PROTOCOL",
                                "raw completion batching requires raw completion mode",
                            ),
                        )?;
                        continue;
                    }
                    if raw && raw_batch {
                        raw_completion_batch_latched = true;
                        raw_completion_batch = true;
                    }
                    Request::Exec {
                        artifact_sha256,
                        dispatch,
                        attempt,
                        inputs,
                        fault,
                    }
                }
                request => request,
            }
        } else {
            request
        };
        let Some(rank) = bound else {
            let error = match request {
                Request::Hello { protocol, rank } if protocol == PROTOCOL => {
                    bound = Some(rank);
                    let mut hello = hello(rank);
                    #[cfg(feature = "native-ggml-cpu")]
                    if executor.is_none() {
                        hello.capabilities.push(CAPABILITY_NATIVE_GGML_CPU.into());
                    }
                    if let Some(provider) = executor.as_ref() {
                        hello.capabilities = vec![
                            mxfp4::GPU_CAPABILITY.into(),
                            provider.producer().capability(),
                            mxfp4::BF16_ACTIVATION_CAPABILITY.into(),
                        ];
                    }
                    send_reply(output, &Reply::Hello { hello })?;
                    continue;
                }
                Request::Hello { .. } => {
                    Error::new("PROTOCOL_MISMATCH", "session protocol version differs")
                }
                _ => Error::new("SESSION_PROTOCOL", "HELLO must open the session"),
            };
            send_reply(
                output,
                &Reply::Refused {
                    error: error.clone(),
                },
            )?;
            return Err(error);
        };
        #[cfg(feature = "native-ggml-cpu")]
        if (!native_owners.is_empty() || native_worker.is_some())
            && !matches!(
                request,
                Request::LoadNativeGgmlRaw { .. }
                    | Request::SealNativeGgml
                    | Request::ExecNativeGgml { .. }
                    | Request::Shutdown
            )
        {
            let error = invalid("native GGML session cannot mix resident contracts");
            send_reply(
                output,
                &Reply::Refused {
                    error: error.clone(),
                },
            )?;
            return Err(error);
        }
        match request {
            #[cfg(feature = "native-ggml-cpu")]
            Request::LoadNativeGgmlRaw {
                binding,
                index,
                pack_bytes,
                artifact_bytes,
            } => {
                use crate::strata_native_layout::{NativeLayout, VerifiedNativeExpert};
                if resident.is_some()
                    || engram_resident.is_some()
                    || native_worker.is_some()
                    || binding.rank != rank
                    || native_owners.len() >= 4
                    || artifact_bytes > 4 * 1024 * 1024
                {
                    let error = invalid("native load state, rank or size");
                    send_reply(
                        output,
                        &Reply::Refused {
                            error: error.clone(),
                        },
                    )?;
                    return Err(error); // raw body remains unread
                }
                let bytes = match read_raw_body(input, header_bytes, artifact_bytes, false) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        send_reply(
                            output,
                            &Reply::Refused {
                                error: error.clone(),
                            },
                        )?;
                        return Err(error);
                    }
                };
                let loaded = (|| -> Result<crate::native_ggml_cpu::Resident> {
                    if native_owners.iter().any(|r| {
                        r.binding().local == binding.local || r.binding().expert == binding.expert
                    }) {
                        return Err(invalid("duplicate native rank expert or local"));
                    }
                    let layout = NativeLayout::parse_sc117_v3(&index, pack_bytes)?;
                    let span = layout.expert(binding.layer as usize, u64::from(binding.expert))?;
                    let blob = VerifiedNativeExpert {
                        layer: binding.layer as usize,
                        expert: u64::from(binding.expert),
                        span,
                        bytes,
                    };
                    crate::native_ggml_cpu::Resident::load(&layout, blob, binding.clone())
                })();
                match loaded {
                    Ok(owner) => {
                        native_owners.push(owner);
                        send_reply(output, &Reply::NativeGgmlLoaded { binding })?;
                    }
                    Err(error) => send_reply(output, &Reply::Refused { error })?,
                }
            }
            #[cfg(feature = "native-ggml-cpu")]
            Request::SealNativeGgml => {
                if native_worker.is_some() || native_owners.is_empty() {
                    send_reply(
                        output,
                        &refuse("NATIVE_STATE", "native owners missing or already sealed"),
                    )?;
                    continue;
                }
                let count = native_owners.len();
                match crate::native_ggml_cpu::RankWorker::new(std::mem::take(&mut native_owners)) {
                    Ok(worker) => {
                        native_worker = Some(worker);
                        send_reply(output, &Reply::NativeGgmlReady { experts: count })?;
                    }
                    Err(error) => {
                        send_reply(
                            output,
                            &Reply::Refused {
                                error: error.clone(),
                            },
                        )?;
                        return Err(error);
                    }
                }
            }
            #[cfg(feature = "native-ggml-cpu")]
            Request::ExecNativeGgml {
                dispatch,
                attempt,
                inputs,
            } => {
                let Some(worker) = native_worker.as_mut() else {
                    send_reply(
                        output,
                        &refuse("NATIVE_STATE", "native owners are not sealed"),
                    )?;
                    continue;
                };
                let stamp = dispatch.stamp.clone();
                let event = |event| Reply::Event {
                    event: WorkerEvent {
                        stamp: stamp.clone(),
                        rank,
                        attempt,
                        event,
                    },
                };
                send_reply(output, &event(EventKind::Started))?;
                match worker.execute(&dispatch, &inputs, attempt) {
                    Ok(result) => {
                        send_reply(output, &Reply::NativeGgmlResult { output: result })?;
                        execs += 1;
                    }
                    Err(error) => {
                        send_reply(
                            output,
                            &event(EventKind::Failed {
                                error: error.clone(),
                            }),
                        )?;
                        return Err(error); // after Started this session cannot be replayed
                    }
                }
            }
            Request::LoadMxfp4Raw {
                binding,
                artifact_bytes,
            } => {
                let bytes = match mxfp4::read_artifact(input, artifact_bytes) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        send_reply(
                            output,
                            &Reply::Refused {
                                error: error.clone(),
                            },
                        )?;
                        return Err(error);
                    }
                };
                let reply = mxfp4::load(&mut resident, rank, binding, bytes, executor);
                send_reply(output, &reply)?;
                if executor.as_ref().is_some_and(|p| p.fatal()) {
                    return Err(Error::new(
                        "GPU_WORKER_FATAL",
                        "native LOAD failed terminally; stop worker, never retry",
                    ));
                }
            }
            Request::ExecPreweightedFp32Raw { .. } | Request::ExecPreweightedBf16Raw { .. } => {
                return Err(invalid("raw MXFP4 EXEC was not decoded"))
            }
            Request::ExecPreweightedFp32 {
                artifact_sha256,
                dispatch,
                contract,
                attempt,
                inputs,
                fault,
            } => {
                let refusal = if attempt == 0 || attempt <= last_attempt {
                    Some(Error::new("STALE_ATTEMPT", "MXFP4 attempt is not newer"))
                } else if dispatch.rank != rank {
                    Some(invalid("foreign MXFP4 worker rank"))
                } else if resident
                    .as_ref()
                    .is_none_or(|r| r.digest != artifact_sha256)
                {
                    Some(Error::new(
                        "ARTIFACT_NOT_LOADED",
                        "no MXFP4 artifact has this SHA-256",
                    ))
                } else {
                    None
                };
                if let Some(error) = refusal {
                    send_reply(
                        output,
                        &Reply::Event {
                            event: WorkerEvent {
                                stamp: dispatch.stamp,
                                rank: dispatch.rank,
                                attempt,
                                event: EventKind::Rejected { error },
                            },
                        },
                    )?;
                    continue;
                }
                last_attempt = attempt;
                if mxfp4::execute(
                    resident.as_ref().unwrap(),
                    dispatch,
                    contract,
                    attempt,
                    inputs,
                    fault,
                    output,
                    executor
                        .as_mut()
                        .map(|p| &mut **p as &mut dyn mxfp4::Executor),
                )? {
                    execs += 1;
                }
                if executor.as_ref().is_some_and(|p| p.fatal()) {
                    return Err(Error::new(
                        "GPU_WORKER_FATAL",
                        "native EXEC failed terminally; stop worker, never retry",
                    ));
                }
            }
            Request::Hello { .. } => {
                let error = Error::new("SESSION_PROTOCOL", "duplicate HELLO");
                send_reply(
                    output,
                    &Reply::Refused {
                        error: error.clone(),
                    },
                )?;
                return Err(error);
            }
            Request::Load {
                artifact_sha256,
                model_revision,
                layer,
                epoch,
                resident_order,
                model,
            } => {
                let reply = load(
                    &mut resident,
                    artifact_sha256,
                    model_revision,
                    layer,
                    epoch,
                    resident_order,
                    model,
                );
                send_reply(output, &reply)?;
            }
            Request::LoadArtifact {
                artifact_sha256,
                model_revision,
                layer,
                epoch,
                resident_order,
                artifact,
            } => {
                let artifact = match base64::decode(&artifact) {
                    Ok(artifact) => artifact,
                    Err(error) => {
                        send_reply(output, &Reply::Refused { error })?;
                        continue;
                    }
                };
                let request = ArtifactLoad {
                    artifact_sha256,
                    model_revision,
                    layer,
                    epoch,
                    resident_order,
                    artifact,
                };
                send_reply(output, &load_artifact(&mut resident, rank, request))?;
            }
            Request::LoadArtifactRaw {
                artifact_sha256,
                epoch,
                artifact_bytes,
            } => {
                let artifact = match read_artifact_body(input, artifact_bytes) {
                    Ok(artifact) => artifact,
                    Err(error) => {
                        send_reply(
                            output,
                            &Reply::Refused {
                                error: error.clone(),
                            },
                        )?;
                        // A bad declared length or truncated payload leaves the stream
                        // boundary untrusted; never parse another request from it.
                        return Err(error);
                    }
                };
                send_reply(
                    output,
                    &load_raw_artifact(&mut resident, rank, artifact_sha256, epoch, artifact),
                )?;
            }
            Request::LoadEngramShardRaw {
                table,
                world_size,
                worker_rank,
                owner_rank,
                artifact_sha256,
                artifact_bytes,
            } => {
                let bytes = match read_raw_body(input, header_bytes, artifact_bytes, true) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        send_reply(
                            output,
                            &Reply::Refused {
                                error: error.clone(),
                            },
                        )?;
                        return Err(error);
                    }
                };
                if worker_rank != rank {
                    send_reply(
                        output,
                        &refuse("RANK_MISMATCH", "Engram load names a different worker rank"),
                    )?;
                    continue;
                }
                let reply = load_engram_shard(
                    &mut engram_resident,
                    table,
                    world_size,
                    owner_rank,
                    artifact_sha256,
                    bytes,
                );
                send_reply(output, &reply)?;
            }
            Request::FetchEngramRows {
                stamp,
                table,
                world_size,
                owner_rank,
                row_ids,
                attempt,
            } => {
                if attempt <= last_engram_attempt {
                    send_reply(
                        output,
                        &refuse(
                            "STALE_ATTEMPT",
                            "Engram row attempt is not newer than the prior lookup",
                        ),
                    )?;
                    continue;
                }
                last_engram_attempt = attempt;
                let Some(shard) = engram_resident.as_ref() else {
                    send_reply(
                        output,
                        &refuse("ENGRAM_NOT_LOADED", "no Engram shard is loaded"),
                    )?;
                    continue;
                };
                if world_size != shard.world_size()
                    || table != *shard.table()
                    || owner_rank != shard.rank()
                {
                    send_reply(
                        output,
                        &refuse(
                            "ENGRAM_IDENTITY_MISMATCH",
                            "row request differs from the loaded shard identity",
                        ),
                    )?;
                    continue;
                }
                let binding =
                    match engram::RequestBinding::new(&stamp, &table, world_size, &row_ids) {
                        Ok(binding) => binding,
                        Err(error) => {
                            send_reply(output, &refuse("INVALID_INPUT", error.0))?;
                            continue;
                        }
                    };
                let frame = match shard.encode_rows_frame(&stamp, &row_ids, &binding, owner_rank) {
                    Ok(frame) if frame.len() <= MAX_MESSAGE_BYTES => frame,
                    Ok(_) => {
                        send_reply(
                            output,
                            &refuse(
                                "MESSAGE_TOO_LARGE",
                                "Engram owner frame exceeds the session bound",
                            ),
                        )?;
                        continue;
                    }
                    Err(error) => {
                        send_reply(output, &refuse("INVALID_INPUT", error.0))?;
                        continue;
                    }
                };
                let header = encode(&Reply::EngramRowsRaw {
                    attempt,
                    frame_bytes: frame.len() as u64,
                    frame: Vec::new(),
                })?;
                output
                    .write_all(&header)
                    .and_then(|()| output.write_all(&frame))
                    .and_then(|()| output.flush())
                    .map_err(|e| {
                        Error::new("WORKER_IO", format!("write Engram owner frame: {e}"))
                    })?;
            }
            Request::FetchEngramRowsVarint { .. } => {
                let error = Error::new(
                    "SESSION_PROTOCOL",
                    "raw Engram row IDs were not decoded before request dispatch",
                );
                send_reply(
                    output,
                    &Reply::Refused {
                        error: error.clone(),
                    },
                )?;
                return Err(error);
            }
            Request::EvictEngram { artifact_sha256 } => {
                if engram_resident
                    .as_ref()
                    .is_some_and(|a| a.sha256() == artifact_sha256)
                {
                    engram_resident = None;
                    send_reply(output, &Reply::EngramEvicted { artifact_sha256 })?;
                } else {
                    send_reply(
                        output,
                        &refuse("ENGRAM_NOT_LOADED", "no Engram shard has this SHA-256"),
                    )?;
                }
            }
            Request::ExecRaw { .. } => {
                let error = Error::new(
                    "SESSION_PROTOCOL",
                    "raw EXEC body was not decoded before request dispatch",
                );
                send_reply(
                    output,
                    &Reply::Refused {
                        error: error.clone(),
                    },
                )?;
                return Err(error);
            }
            Request::ExecRawFp16 { .. } => {
                let error = Error::new(
                    "SESSION_PROTOCOL",
                    "raw FP16 EXEC body was not decoded before request dispatch",
                );
                send_reply(
                    output,
                    &Reply::Refused {
                        error: error.clone(),
                    },
                )?;
                return Err(error);
            }
            Request::Exec {
                artifact_sha256,
                dispatch,
                attempt,
                inputs,
                fault,
            } => {
                let Some(r) = resident.as_ref().filter(|r| r.digest == artifact_sha256) else {
                    send_reply(
                        output,
                        &refuse("ARTIFACT_NOT_LOADED", "no loaded artifact has this SHA-256"),
                    )?;
                    continue;
                };
                let stamp = dispatch.stamp.clone();
                let event_rank = dispatch.rank;
                let mut events = SessionEvents {
                    output,
                    raw_completions,
                    raw_completion_batch,
                    raw_completion_fp16_batch,
                };
                let refusal = if attempt <= last_attempt {
                    Some(Error::new(
                        "STALE_ATTEMPT",
                        "attempt is not newer than this session's last execution",
                    ))
                } else if dispatch.rank != rank || dispatch.resident_order != r.resident_order {
                    Some(invalid(
                        "dispatch rank or physical order differs from this session",
                    ))
                } else if stamp.epoch != r.epoch
                    || stamp.model_revision != r.model_revision
                    || stamp.layer != r.layer
                {
                    Some(Error::new(
                        "STALE_EXECUTION",
                        "dispatch identity differs from the loaded artifact epoch, revision or layer",
                    ))
                } else {
                    None
                };
                if let Some(error) = refusal {
                    events.send(&WorkerEvent {
                        stamp,
                        rank: event_rank,
                        attempt,
                        event: EventKind::Rejected { error },
                    })?;
                    continue;
                }
                last_attempt = attempt;
                let admitted = match &r.weights {
                    // An in-process copy of the resident weights; nothing crosses the stream.
                    Weights::Dense(model) => local_work(RankRequest {
                        dispatch,
                        attempt,
                        model: model.clone(),
                        inputs,
                        fault,
                    })
                    .map(|work| {
                        execs += 1;
                        execute(work, &stamp, event_rank, attempt, &mut events)
                    }),
                    Weights::Compressed(stack) => compressed_job(stack, dispatch, inputs, fault)
                        .map(|job| {
                            execs += 1;
                            execute_batch_with(
                                job,
                                &stamp,
                                event_rank,
                                attempt,
                                &mut events,
                                |dispatch, inputs| {
                                    dispatch
                                        .slots
                                        .iter()
                                        .map(|routed| {
                                            stack.expert_value(
                                                routed.physical_expert,
                                                &inputs[&routed.slot.token],
                                            )
                                        })
                                        .collect()
                                },
                            )
                        }),
                    Weights::Mxfp4(_) => Err(Error::new(
                        "ARITHMETIC_CONTRACT_MISMATCH",
                        "MXFP4 requires an explicit preweighted FP32 EXEC",
                    )),
                };
                match admitted {
                    Ok(streamed) => streamed?,
                    Err(error) => events.send(&WorkerEvent {
                        stamp,
                        rank: event_rank,
                        attempt,
                        event: EventKind::Rejected { error },
                    })?,
                }
            }
            Request::Evict { artifact_sha256 } => {
                if resident
                    .as_ref()
                    .is_some_and(|r| r.digest == artifact_sha256)
                {
                    if let Some(provider) = executor.as_deref_mut() {
                        provider.evict()?;
                    }
                    resident = None;
                    send_reply(output, &Reply::Evicted { artifact_sha256 })?;
                } else {
                    send_reply(
                        output,
                        &refuse("ARTIFACT_NOT_LOADED", "no loaded artifact has this SHA-256"),
                    )?;
                }
            }
            Request::Shutdown => {
                if let Some(provider) = executor.as_deref_mut() {
                    provider.evict()?; // close must finish before BYE can attest shutdown.
                }
                return send_reply(
                    output,
                    &Reply::Bye {
                        execs,
                        peak_rss_kib: peak_rss_kib(),
                    },
                );
            }
        }
    }
}

// -------------------------------------------------------------- coordinator side

/// How to start one rank's session worker: a local executable, or a transport such as
/// `ssh -T host /path/vf-rank-session`. The argv is the operator's, run as given.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    pub rank: u32,
    pub argv: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Traffic {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub frames_sent: u64,
    pub frames_received: u64,
    /// Bytes of the LOAD frames that carried weights.
    pub weight_bytes_sent: u64,
    pub loads_transferred: u64,
    pub rebinds: u64,
    pub execs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadReceipt {
    pub rank: u32,
    pub artifact_sha256: String,
    pub epoch: u64,
    pub transferred: bool,
    pub request_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramLoadReceipt {
    pub rank: u32,
    pub artifact_sha256: String,
    pub transferred: bool,
    pub request_bytes: u64,
}

/// Measurements for one owner-partitioned row batch. Payload bytes count the validated raw
/// owner-row bodies, excluding protocol/SSH framing and without implying a throughput result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngramBatchReceipt {
    pub rows: Vec<u8>,
    pub requested_ids: usize,
    pub unique_ids_transferred: usize,
    pub owner_payload_bytes: u64,
    pub owner_requests: usize,
    pub skipped_empty_owners: usize,
}

/// Cache-aware batch counters; byte counts exclude protocol framing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngramCachedBatchReceipt {
    pub rows: Vec<u8>,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub owner_payload_bytes: u64,
    pub owner_requests: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecReceipt {
    pub rank: u32,
    pub attempt: u64,
    pub state: ReceiptState,
    pub started: bool,
    pub expected_slots: usize,
    pub received_slots: usize,
    pub request_bytes: u64,
    pub error: Option<Error>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloseReceipt {
    pub rank: u32,
    pub pid: u32,
    pub bye_execs: Option<u64>,
    pub peak_rss_kib: Option<u64>,
    pub exit_code: Option<i32>,
    pub killed: bool,
    pub stderr: String,
    pub stderr_bytes: usize,
    pub error: Option<Error>,
}

enum Frame {
    Reply(Box<Reply>),
    Error(Error),
    Eof,
}

struct Counting<R> {
    inner: R,
    count: Arc<AtomicU64>,
}
impl<R: Read> Read for Counting<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buffer)?;
        self.count.fetch_add(n as u64, Ordering::Relaxed);
        Ok(n)
    }
}

fn closed() -> Error {
    Error::new("SESSION_CLOSED", "worker stream closed")
}

enum SessionInput {
    #[cfg(unix)]
    Socket(std::os::unix::net::UnixStream),
    #[cfg(not(unix))]
    Pipe(std::process::ChildStdin),
}

impl Write for SessionInput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        match self {
            #[cfg(unix)]
            Self::Socket(stream) => stream.write(bytes),
            #[cfg(not(unix))]
            Self::Pipe(stream) => stream.write(bytes),
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        match self {
            #[cfg(unix)]
            Self::Socket(stream) => stream.flush(),
            #[cfg(not(unix))]
            Self::Pipe(stream) => stream.flush(),
        }
    }
}

/// One rank session: a live worker process and its framed stream.
pub struct Session {
    pub rank: u32,
    pub hello: ServerHello,
    pub traffic: Traffic,
    child: Child,
    stdin: Option<SessionInput>,
    #[cfg(unix)]
    output_cancellation: Option<(
        std::os::unix::net::UnixStream,
        std::os::unix::net::UnixStream,
    )>,
    frames: mpsc::Receiver<Frame>,
    received: Arc<AtomicU64>,
    /// Set before the first EXEC that asks for raw completions; the reader refuses them
    /// until then.
    raw_completions: Arc<AtomicBool>,
    /// Set only when both peers negotiated compact rank-attempt completion frames.
    raw_completion_batch: Arc<AtomicBool>,
    /// Enables decoding FP16 reply frames after the first opt-in EXEC; `PendingExec`
    /// separately enforces that the current EXEC requested FP16 before collection.
    raw_completion_fp16_batch: Arc<AtomicBool>,
    raw_preweighted: Arc<Mutex<Option<mxfp4::Admission>>>,
    preweighted_owner: Arc<()>,
    preweighted_inflight: bool,
    mxfp4_binding: Option<(mxfp4::Binding, usize)>,
    mxfp4_producer: mxfp4::Producer,
    prefer_mxfp4_bf16_activations: bool,
    /// The v2 capability opts in once per worker session; failed writes do not latch it.
    raw_completion_batch_session_sent: bool,
    raw_engram_attempt: Arc<AtomicU64>,
    reader: Option<JoinHandle<()>>,
    stderr: Option<JoinHandle<(usize, Vec<u8>)>>,
    artifact: Option<(String, u64)>,
    engram_artifact: Option<String>,
    engram_owner_rank: Option<u32>,
    next_engram_attempt: u64,
    /// The coordinator's copy of what the worker holds, to check each batch cheaply.
    resident: Option<moe::Model>,
    usable: bool,
    reaped: bool,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("rank", &self.rank)
            .field("pid", &self.child.id())
            .field("usable", &self.usable)
            .field("artifact", &self.artifact)
            .finish_non_exhaustive()
    }
}

/// An EXEC already sent, whose events are still to be read.
pub struct PendingExec {
    dispatch: routing::RankDispatch,
    receipt: ExecReceipt,
    expect_raw_completion_batch: bool,
    expect_raw_completion_fp16_batch: bool,
    latch_raw_completion_batch_session: bool,
}

struct LoadTransfer<'a> {
    request: &'a Request,
    body: Option<&'a [u8]>,
    digest: String,
    epoch: u64,
    rebind: bool,
    model: Option<&'a moe::Model>,
}

/// Locate the row offset when a fetched batch is emitted in ascending miss-ID order.
fn engram_fetched_row_index(sorted_miss_ids: &[u64], row_id: u64) -> Option<usize> {
    sorted_miss_ids.binary_search(&row_id).ok()
}

impl Session {
    /// Transfer one digest-bound SC117 GGML expert. The expected digest must come
    /// from a trusted manifest; the worker independently verifies the exact bytes.
    #[cfg(feature = "native-ggml-cpu")]
    pub fn load_native_ggml(
        &mut self,
        binding: &crate::native_ggml_cpu::Binding,
        index: &str,
        pack_bytes: u64,
        bytes: &[u8],
        timeout: Duration,
    ) -> Result<()> {
        if !self
            .hello
            .capabilities
            .iter()
            .any(|c| c == CAPABILITY_NATIVE_GGML_CPU)
            || binding.rank != self.rank
            || bytes.len() > 4 * 1024 * 1024
        {
            return Err(invalid("native GGML capability, rank or size"));
        }
        let layout = crate::strata_native_layout::NativeLayout::parse_sc117_v3(index, pack_bytes)?;
        let span = layout.expert(binding.layer as usize, u64::from(binding.expert))?;
        if layout.index_sha256() != binding.index_sha256
            || span.bytes != bytes.len() as u64
            || sha256::hex(bytes) != binding.blob_sha256
        {
            return Err(invalid("native GGML index or payload digest"));
        }
        let request = Request::LoadNativeGgmlRaw {
            binding: binding.clone(),
            index: index.to_owned(),
            pack_bytes,
            artifact_bytes: bytes.len() as u64,
        };
        self.write_parts(&[&encode(&request)?, bytes], true)?;
        match self.reply(timeout)? {
            Reply::NativeGgmlLoaded { binding: returned } if returned == *binding => Ok(()),
            Reply::Refused { error } => Err(error),
            _ => {
                self.usable = false;
                Err(invalid("native GGML load reply identity"))
            }
        }
    }

    #[cfg(feature = "native-ggml-cpu")]
    pub fn seal_native_ggml(&mut self, expected_experts: usize, timeout: Duration) -> Result<()> {
        if !self
            .hello
            .capabilities
            .iter()
            .any(|c| c == CAPABILITY_NATIVE_GGML_CPU)
            || !(1..=4).contains(&expected_experts)
        {
            return Err(invalid("native GGML capability or expert count"));
        }
        self.send(&Request::SealNativeGgml)?;
        match self.reply(timeout)? {
            Reply::NativeGgmlReady { experts } if experts == expected_experts => Ok(()),
            Reply::Refused { error } => Err(error),
            _ => {
                self.usable = false;
                Err(invalid("native GGML seal reply count"))
            }
        }
    }

    /// Execute once and atomically admit this rank's compact unweighted rows.
    /// On any failure the coordinator must discard the entire batch and never
    /// replay an uncertain identity; this method is not a durable replay ledger.
    #[cfg(feature = "native-ggml-cpu")]
    fn exec_native_ggml(
        &mut self,
        dispatch: routing::RankDispatch,
        attempt: u64,
        inputs: Vec<(usize, Vec<f32>)>,
        collector: &mut moe::Collector,
        timeout: Duration,
    ) -> Result<usize> {
        if !self
            .hello
            .capabilities
            .iter()
            .any(|c| c == CAPABILITY_NATIVE_GGML_CPU)
            || dispatch.rank != self.rank
            || attempt == 0
        {
            return Err(invalid("native GGML capability, rank or attempt"));
        }
        let deadline = Instant::now() + timeout;
        self.send(&Request::ExecNativeGgml {
            dispatch: dispatch.clone(),
            attempt,
            inputs,
        })?;
        let started = self.reply_until(deadline)?;
        if !matches!(started, Reply::Event { event: WorkerEvent { ref stamp, rank, attempt: n, event: EventKind::Started } } if *stamp == dispatch.stamp && rank == self.rank && n == attempt)
        {
            self.usable = false;
            return Err(invalid("native GGML start reply identity"));
        }
        match self.reply_until(deadline)? {
            Reply::NativeGgmlResult { output }
                if output.stamp == dispatch.stamp
                    && output.rank == self.rank
                    && output.attempt == attempt =>
            {
                match crate::completion::accept_routed_output(attempt, output, collector) {
                    Ok(count) if count == dispatch.slots.len() => Ok(count),
                    Ok(_) => {
                        self.usable = false;
                        Err(invalid("native GGML result slot count"))
                    }
                    Err(error) => {
                        self.usable = false;
                        Err(error)
                    }
                }
            }
            Reply::Event {
                event:
                    WorkerEvent {
                        stamp,
                        rank,
                        attempt: n,
                        event: EventKind::Failed { error },
                    },
            } if stamp == dispatch.stamp && rank == self.rank && n == attempt => {
                self.usable = false;
                Err(error)
            }
            _ => {
                self.usable = false;
                Err(invalid("native GGML result reply identity"))
            }
        }
    }

    /// One native rank wave. On any error no output is returned and every
    /// participating session is invalidated. The caller must retain and consistently
    /// reuse the same local durable ledger across coordinator restarts. It is not R53.
    #[cfg(feature = "native-ggml-cpu")]
    pub fn exec_native_ggml_batch(
        sessions: &mut [Session],
        problem: &routing::RoutingProblem,
        inputs: &[Vec<f32>],
        ledger: &crate::native_ledger::NativeBatchLedger,
        timeout: Duration,
    ) -> Result<Vec<Vec<f64>>> {
        if timeout.is_zero()
            || timeout > Duration::from_secs(60)
            || inputs.len() != problem.routes.len()
            || inputs
                .iter()
                .any(|r| r.len() != 2560 || r.iter().any(|v| !v.is_finite()))
        {
            return Err(invalid("native batch input or timeout"));
        }
        let dispatch = routing::dispatch(problem)?;
        if dispatch.hidden != 2560 || dispatch.slots.is_empty() {
            return Err(invalid("native batch dimensions"));
        }
        let mut selected = Vec::new();
        for rank in dispatch.ranks.into_iter().filter(|r| !r.slots.is_empty()) {
            let matches: Vec<_> = sessions
                .iter()
                .enumerate()
                .filter(|(_, s)| s.rank == rank.rank)
                .collect();
            if matches.len() != 1
                || !matches[0].1.usable
                || !matches[0]
                    .1
                    .hello
                    .capabilities
                    .iter()
                    .any(|c| c == CAPABILITY_NATIVE_GGML_CPU)
            {
                return Err(invalid(
                    "native participating session missing, duplicate or unusable",
                ));
            }
            let tokens: BTreeSet<_> = rank.slots.iter().map(|r| r.slot.token).collect();
            let rows = tokens
                .into_iter()
                .map(|t| (t, inputs[t].clone()))
                .collect::<Vec<_>>();
            selected.push((matches[0].0, rank, rows));
        }
        let mut collector = moe::Collector::from_routing(problem)?;
        let attempt = next_attempt()?;
        claim_identity(&problem.stamp, false)?;
        ledger.reserve(&problem.stamp, attempt)?;
        let deadline = Instant::now() + timeout;
        for (index, dispatch, rows) in &selected {
            if let Err(error) = sessions[*index].exec_native_ggml(
                dispatch.clone(),
                attempt,
                rows.clone(),
                &mut collector,
                deadline.saturating_duration_since(Instant::now()),
            ) {
                for (i, _, _) in &selected {
                    sessions[*i].usable = false;
                }
                return Err(error);
            }
        }
        match collector.finish() {
            Ok(output) => Ok(output),
            Err(error) => {
                for (i, _, _) in &selected {
                    sessions[*i].usable = false;
                }
                Err(error)
            }
        }
    }
    /// Fetch only unique cache misses, then reconstruct caller order and duplicates. Cache
    /// additions occur only after all owner frames pass identity and coverage validation.
    pub fn fetch_engram_batch_cached(
        sessions: &mut [Session],
        stamp: &moe::Stamp,
        table: &TableIdentity,
        row_ids: &[u64],
        timeout: Duration,
        cache: &mut EngramRowCache,
    ) -> Result<EngramCachedBatchReceipt> {
        let world_size = u32::try_from(sessions.len())
            .map_err(|_| invalid("Engram worker count does not fit the protocol"))?;
        if world_size == 0 {
            return Err(invalid("Engram batch requires at least one worker"));
        }
        engram::RequestBinding::new(stamp, table, world_size, row_ids)
            .map_err(|error| invalid(error.0))?;
        // Cache lookup has no recency side effects, so a sorted unique ID vector is
        // sufficient here; the original caller order is restored during output assembly.
        let mut unique_ids = row_ids.to_vec();
        unique_ids.sort_unstable();
        unique_ids.dedup();
        let mut misses = Vec::new();
        cache.for_each_lookup(table, &unique_ids, |id, row| {
            if row.is_none() {
                misses.push(id);
            }
            Ok(())
        })?;
        // unique_ids is sorted, so misses and the owner response are already in index order.
        let cache_hits = unique_ids.len() - misses.len();
        let fetched = if misses.is_empty() {
            None
        } else {
            Some(Self::fetch_engram_batch_with_receipt(
                sessions, stamp, table, &misses, timeout,
            )?)
        };
        let (owner_payload_bytes, owner_requests) = fetched
            .as_ref()
            .map(|r| (r.owner_payload_bytes, r.owner_requests))
            .unwrap_or((0, 0));
        let output_len = row_ids
            .len()
            .checked_mul(264)
            .ok_or_else(|| invalid("Engram output byte count overflow"))?;
        let mut rows = Vec::new();
        rows.try_reserve_exact(output_len)
            .map_err(|_| invalid("Engram output allocation failed"))?;
        cache.for_each_lookup(table, row_ids, |id, cached_row| {
            if let Some(row) = cached_row {
                rows.extend_from_slice(row);
            } else if let Some(index) = engram_fetched_row_index(&misses, id) {
                let fetched_rows = fetched
                    .as_ref()
                    .ok_or_else(|| invalid("Engram misses have no validated response"))?;
                let start = index
                    .checked_mul(264)
                    .ok_or_else(|| invalid("Engram fetched row offset overflow"))?;
                let end = start
                    .checked_add(264)
                    .ok_or_else(|| invalid("Engram fetched row offset overflow"))?;
                let row = fetched_rows
                    .rows
                    .get(start..end)
                    .ok_or_else(|| invalid("Engram fetched response lost a validated row"))?;
                rows.extend_from_slice(row);
            } else {
                return Err(invalid("Engram cache reconstruction lost a validated row"));
            }
            Ok(())
        })?;
        // Materialize all warm hits before inserting misses: a bounded FIFO insertion can
        // evict one of those hits when the requested union exceeds cache capacity.
        if let Some(receipt) = fetched.as_ref() {
            cache
                .insert_many(table, &misses, &receipt.rows)
                .map_err(|error| invalid(error.0))?;
        }
        Ok(EngramCachedBatchReceipt {
            rows,
            cache_hits,
            cache_misses: misses.len(),
            owner_payload_bytes,
            owner_requests,
        })
    }

    /// Start the worker without the handshake (tests drive HELLO by hand).
    pub fn spawn(endpoint: &Endpoint) -> Result<Session> {
        let (program, args) = endpoint
            .argv
            .split_first()
            .ok_or_else(|| invalid("session endpoint has an empty argv"))?;
        // A private socket pair preserves the byte-stream stdin contract while allowing
        // deadline-bounded writes without a detached thread retaining an artifact.
        #[cfg(unix)]
        let (parent_input, child_input) = std::os::unix::net::UnixStream::pair()
            .map_err(|e| Error::new("WORKER_IO", format!("session stdin pair: {e}")))?;
        #[cfg(unix)]
        let child_stdin = Stdio::from(std::os::fd::OwnedFd::from(child_input));
        #[cfg(not(unix))]
        let child_stdin = Stdio::piped();
        #[cfg(unix)]
        let (parent_output, child_output) = std::os::unix::net::UnixStream::pair()
            .map_err(|e| Error::new("WORKER_IO", format!("session stdout pair: {e}")))?;
        #[cfg(unix)]
        let (parent_error, child_error) = std::os::unix::net::UnixStream::pair()
            .map_err(|e| Error::new("WORKER_IO", format!("session stderr pair: {e}")))?;
        #[cfg(unix)]
        let output_cancellation = Some((
            parent_output
                .try_clone()
                .map_err(|e| Error::new("WORKER_IO", e.to_string()))?,
            parent_error
                .try_clone()
                .map_err(|e| Error::new("WORKER_IO", e.to_string()))?,
        ));
        #[cfg(unix)]
        let child_stdout = Stdio::from(std::os::fd::OwnedFd::from(child_output));
        #[cfg(unix)]
        let child_stderr = Stdio::from(std::os::fd::OwnedFd::from(child_error));
        #[cfg(not(unix))]
        let (child_stdout, child_stderr) = (Stdio::piped(), Stdio::piped());
        let child = Command::new(program)
            .args(args)
            .stdin(child_stdin)
            .stdout(child_stdout)
            .stderr(child_stderr)
            .spawn()
            .map_err(|e| Error::new("WORKER_IO", format!("spawn session worker: {e}")))?;
        #[cfg(not(unix))]
        let mut child = child;
        #[cfg(unix)]
        let stdin = Some(SessionInput::Socket(parent_input));
        #[cfg(not(unix))]
        let stdin = child.stdin.take().map(SessionInput::Pipe);
        #[cfg(unix)]
        let (stdout, stderr) = (Some(parent_output), Some(parent_error));
        #[cfg(not(unix))]
        let (stdout, stderr) = (child.stdout.take(), child.stderr.take());
        let received = Arc::new(AtomicU64::new(0));
        // Backpressure all reply kinds, including control frames. The reader can
        // retain two queued frames plus the one currently being decoded/sent.
        let (sender, frames) = mpsc::sync_channel(2);
        let count = received.clone();
        let raw_completions = Arc::new(AtomicBool::new(false));
        let raw_allowed = raw_completions.clone();
        let raw_completion_batch = Arc::new(AtomicBool::new(false));
        let raw_batch_allowed = raw_completion_batch.clone();
        let raw_completion_fp16_batch = Arc::new(AtomicBool::new(false));
        let raw_fp16_batch_allowed = raw_completion_fp16_batch.clone();
        let raw_preweighted = Arc::new(Mutex::new(None));
        let preweighted_allowed = raw_preweighted.clone();
        let raw_engram_attempt = Arc::new(AtomicU64::new(0));
        let engram_allowed = raw_engram_attempt.clone();
        let reader = stdout.map(|stdout| {
            thread::spawn(move || {
                let mut stream = BufReader::new(Counting {
                    inner: stdout,
                    count,
                });
                loop {
                    let frame = match read_reply_with_preweighted_flags(
                        &mut stream,
                        &raw_allowed,
                        &raw_batch_allowed,
                        &raw_fp16_batch_allowed,
                        &engram_allowed,
                        &preweighted_allowed,
                    ) {
                        Ok(Some(reply)) => Frame::Reply(Box::new(reply)),
                        Ok(None) => Frame::Eof,
                        Err(error) => Frame::Error(error),
                    };
                    let last = !matches!(frame, Frame::Reply(_));
                    if sender.send(frame).is_err() || last {
                        break;
                    }
                }
            })
        });
        let stderr = stderr.map(|mut stderr| {
            thread::spawn(move || {
                let mut kept = Vec::new();
                let mut total = 0_usize;
                let mut buffer = [0; 8192];
                loop {
                    match stderr.read(&mut buffer) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            total = total.saturating_add(n);
                            let room = MAX_STDERR_KEPT.saturating_sub(kept.len());
                            kept.extend_from_slice(&buffer[..n.min(room)]);
                        }
                    }
                }
                (total, kept)
            })
        });
        Ok(Session {
            rank: endpoint.rank,
            hello: ServerHello::default(),
            traffic: Traffic::default(),
            child,
            stdin,
            #[cfg(unix)]
            output_cancellation,
            frames,
            received,
            raw_completions,
            raw_completion_batch,
            raw_completion_fp16_batch,
            raw_preweighted,
            preweighted_owner: Arc::new(()),
            preweighted_inflight: false,
            mxfp4_binding: None,
            mxfp4_producer: mxfp4::Producer::default(),
            prefer_mxfp4_bf16_activations: false,
            raw_completion_batch_session_sent: false,
            raw_engram_attempt,
            reader,
            stderr,
            artifact: None,
            engram_artifact: None,
            engram_owner_rank: None,
            next_engram_attempt: 0,
            resident: None,
            usable: true,
            reaped: false,
        })
    }

    /// Start the worker and complete HELLO: protocol, rank, capability and bounds must
    /// be exactly the coordinator's, or the session is closed.
    pub fn connect(endpoint: &Endpoint, timeout: Duration) -> Result<Session> {
        Self::connect_with_capability(endpoint, timeout, CAPABILITY)
    }
    fn connect_with_capability(
        endpoint: &Endpoint,
        timeout: Duration,
        required_capability: &str,
    ) -> Result<Session> {
        let mut session = Session::spawn(endpoint)?;
        session.send(&Request::Hello {
            protocol: PROTOCOL,
            rank: endpoint.rank,
        })?;
        match session.reply(timeout)? {
            Reply::Hello { hello }
                if hello.protocol == PROTOCOL
                    && hello.rank == endpoint.rank
                    && hello.capabilities.iter().any(|c| c == required_capability)
                    && hello.max_message_bytes == MAX_MESSAGE_BYTES =>
            {
                session.hello = hello;
                Ok(session)
            }
            Reply::Refused { error } => {
                session.usable = false;
                Err(error)
            }
            _ => {
                session.usable = false;
                Err(Error::new(
                    "SESSION_PROTOCOL",
                    "HELLO answer differs from this coordinator's protocol, rank or bounds",
                ))
            }
        }
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }
    pub fn is_usable(&self) -> bool {
        self.usable
    }
    /// The loaded artifact digest and its bound epoch, as the coordinator believes.
    pub fn artifact(&self) -> Option<(&str, u64)> {
        self.artifact.as_ref().map(|(d, e)| (d.as_str(), *e))
    }

    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        self.write_parts(&[bytes], true)
    }

    /// Write one logical request made of a JSON header and optional opaque body without
    /// copying the body into a second contiguous buffer.
    fn write_parts(&mut self, parts: &[&[u8]], enforce_bound: bool) -> Result<()> {
        // An unusable stream is never written again: the next frame read from it could be
        // the late answer to the request that failed, taken for this one's.
        if !self.usable {
            return Err(Error::new(
                "SESSION_UNUSABLE",
                "session must be reopened after a failure",
            ));
        }
        let length = parts
            .iter()
            .try_fold(0_usize, |total, part| total.checked_add(part.len()));
        let Some(length) = length.filter(|&length| !enforce_bound || length <= MAX_MESSAGE_BYTES)
        else {
            return Err(Error::new(
                "MESSAGE_TOO_LARGE",
                "session request exceeds 16 MiB",
            ));
        };
        let result = match self.stdin.as_mut() {
            None => Err(closed()),
            Some(stdin) => {
                let write = parts.iter().try_for_each(|part| stdin.write_all(part));
                write
                    .and_then(|()| stdin.flush())
                    .map_err(|e| Error::new("WORKER_IO", format!("write session request: {e}")))
            }
        };
        match result {
            Ok(()) => {
                self.traffic.bytes_sent += length as u64;
                self.traffic.frames_sent += 1;
                Ok(())
            }
            Err(error) => {
                self.usable = false;
                // A broken pipe is the peer closing the stream.
                Err(if error.message.contains("Broken pipe") {
                    closed()
                } else {
                    error
                })
            }
        }
    }
    /// Frame and send one request. An oversize frame is refused before any byte is sent.
    pub fn send(&mut self, request: &Request) -> Result<()> {
        let bytes = encode(request)?;
        self.write(&bytes)
    }
    /// Test hook: write raw bytes, bypassing framing.
    pub fn send_raw(&mut self, bytes: &[u8]) -> Result<()> {
        self.write_parts(&[bytes], false)
    }
    fn reply_until(&mut self, deadline: Instant) -> Result<Reply> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let frame = self.frames.recv_timeout(remaining);
        self.traffic.bytes_received = self.received.load(Ordering::Relaxed);
        match frame {
            Ok(Frame::Reply(reply)) => {
                self.traffic.frames_received += 1;
                Ok(*reply)
            }
            Ok(Frame::Error(error)) => {
                self.usable = false;
                Err(error)
            }
            Ok(Frame::Eof) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.usable = false;
                Err(closed())
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.usable = false;
                Err(Error::new(
                    "SESSION_TIMEOUT",
                    "no session reply before the deadline",
                ))
            }
        }
    }
    /// The next reply frame. Contact lost or a missed deadline makes the session unusable.
    pub fn reply(&mut self, timeout: Duration) -> Result<Reply> {
        self.reply_until(Instant::now() + timeout)
    }

    /// Research multipart LOAD with an explicit identity and capability negotiation.
    /// Any failure after transmission invalidates this session; caller reconciles UNKNOWN.
    pub fn load_compressed_multipart(
        &mut self,
        artifact: &CompressedArtifact,
        stamp: &moe::Stamp,
        attempt: u64,
        timeout: Duration,
    ) -> Result<LoadReceipt> {
        const CHUNK: usize = 4 * 1024 * 1024;
        let stack = &artifact.stack;
        if !self
            .hello
            .capabilities
            .iter()
            .any(|c| c == CAPABILITY_QPN2_MULTIPART)
        {
            return Err(Error::new(
                "CAPABILITY_MISSING",
                "multipart not negotiated; nothing sent",
            ));
        }
        if timeout.is_zero()
            || self.preweighted_inflight
            || self.artifact.is_some()
            || stamp.run_id.is_empty()
            || stamp.plan_id.is_empty()
            || attempt == 0
            || stamp.model_revision != stack.model_revision()
            || stamp.layer != stack.layer()
            || stamp.epoch != stack.epoch()
            || self.rank != stack.rank()
            || artifact.bytes.is_empty()
            || artifact.bytes.len() > 128 * 1024 * 1024 + 8 * 1024 * 1024 + 8
        {
            return Err(invalid(
                "multipart identity, size or state mismatch; nothing sent",
            ));
        }
        #[cfg(unix)]
        if let Some(SessionInput::Socket(socket)) = self.stdin.as_ref() {
            socket
                .set_write_timeout(Some(timeout))
                .map_err(|e| Error::new("WORKER_IO", e.to_string()))?;
        }
        let before = self.traffic.bytes_sent;
        let digest = artifact.sha256.clone();
        let result = (|| -> Result<LoadReceipt> {
            let begin = serde_json::json!({"op":"begin_multipart", "identity": {
                "run_id":stamp.run_id,"model_revision":stamp.model_revision,
                "plan_id":stamp.plan_id,"layer":stamp.layer,"epoch":stamp.epoch,
                "rank":self.rank,"attempt":attempt}, "artifact_bytes":artifact.bytes.len(),
                "artifact_sha256":digest});
            self.write(&encode(&begin)?)?;
            match self.reply(timeout)? {
                Reply::MultipartReady { offset: 0 } => {}
                _ => return Err(invalid("invalid multipart ready")),
            }
            for (index, chunk) in artifact.bytes.chunks(CHUNK).enumerate() {
                let offset = index * CHUNK;
                let header =
                    serde_json::json!({"op":"multipart_chunk","offset":offset,"bytes":chunk.len()});
                self.write_parts(&[&encode(&header)?, chunk], true)?;
                match self.reply(timeout)? {
                    Reply::ChunkAccepted { offset: accepted }
                        if accepted == (offset + chunk.len()) as u64 => {}
                    _ => return Err(invalid("invalid multipart acknowledgement")),
                }
            }
            self.write(&encode(&serde_json::json!({"op":"multipart_finish"}))?)?;
            match self.reply(timeout)? {
                Reply::Loaded {
                    artifact_sha256,
                    epoch,
                    transferred,
                    ..
                } if artifact_sha256 == digest && epoch == stamp.epoch && transferred => {}
                _ => return Err(invalid("invalid multipart loaded reply")),
            }
            let request_bytes = self.traffic.bytes_sent - before;
            self.traffic.loads_transferred += 1;
            self.traffic.weight_bytes_sent += request_bytes;
            self.artifact = Some((digest.clone(), stamp.epoch));
            Ok(LoadReceipt {
                rank: self.rank,
                artifact_sha256: digest,
                epoch: stamp.epoch,
                transferred: true,
                request_bytes,
            })
        })();
        if result.is_err() {
            self.usable = false;
        }
        result
    }

    /// Send the resident experts once; if this exact artifact is already loaded, rebind
    /// it to a newer epoch without resending a single weight.
    pub fn load(
        &mut self,
        model: &moe::Model,
        model_revision: &str,
        layer: u32,
        epoch: u64,
        resident_order: &[u32],
        timeout: Duration,
    ) -> Result<LoadReceipt> {
        let digest = artifact_digest(model)?;
        let rebind = self.artifact.as_ref().is_some_and(|(d, _)| *d == digest);
        let request = Request::Load {
            artifact_sha256: digest.clone(),
            model_revision: model_revision.into(),
            layer,
            epoch,
            resident_order: resident_order.to_vec(),
            model: (!rebind).then(|| model.clone()),
        };
        self.send_load(
            LoadTransfer {
                request: &request,
                body: None,
                digest,
                epoch,
                rebind,
                model: Some(model),
            },
            timeout,
        )
    }

    /// Send a compressed artifact's bytes once; if it is already loaded, rebind it to a
    /// newer epoch with a weightless LOAD.
    pub fn load_compressed(
        &mut self,
        artifact: &CompressedArtifact,
        epoch: u64,
        timeout: Duration,
    ) -> Result<LoadReceipt> {
        if !self.hello.capabilities.iter().any(|c| c == CAPABILITY_QPN2) {
            return Err(Error::new(
                "CAPABILITY_MISSING",
                "worker does not advertise cpu_qpn2_resident_v1; nothing was sent",
            ));
        }
        let stack = &artifact.stack;
        let digest = artifact.sha256.clone();
        let rebind = self.artifact.as_ref().is_some_and(|(d, _)| *d == digest);
        if rebind {
            let request = Request::Load {
                artifact_sha256: digest.clone(),
                model_revision: stack.model_revision().into(),
                layer: stack.layer(),
                epoch,
                resident_order: stack.resident_order().to_vec(),
                model: None,
            };
            return self.send_load(
                LoadTransfer {
                    request: &request,
                    body: None,
                    digest,
                    epoch,
                    rebind: true,
                    model: None,
                },
                timeout,
            );
        }
        if artifact.bytes.is_empty() {
            return Err(Error::new(
                "ARTIFACT_BYTES_RELEASED",
                "artifact transfer bytes were discarded; a fresh artifact is required",
            ));
        }
        if artifact.bytes.len() > MAX_RAW_ARTIFACT_BYTES {
            return Err(Error::new(
                "MESSAGE_TOO_LARGE",
                "artifact exceeds the raw LOAD_ARTIFACT body limit",
            ));
        }
        if self
            .hello
            .capabilities
            .iter()
            .any(|capability| capability == CAPABILITY_QPN2_RAW)
        {
            let artifact_bytes = u64::try_from(artifact.bytes.len()).map_err(|_| {
                Error::new(
                    "MESSAGE_TOO_LARGE",
                    "artifact length does not fit the protocol",
                )
            })?;
            let request = Request::LoadArtifactRaw {
                artifact_sha256: digest.clone(),
                epoch,
                artifact_bytes,
            };
            self.send_load(
                LoadTransfer {
                    request: &request,
                    body: Some(&artifact.bytes),
                    digest,
                    epoch,
                    rebind: false,
                    model: None,
                },
                timeout,
            )
        } else {
            if artifact.bytes.len() > MAX_ARTIFACT_BYTES {
                return Err(Error::new(
                    "MESSAGE_TOO_LARGE",
                    "artifact exceeds what one base64 LOAD_ARTIFACT frame can carry",
                ));
            }
            let request = Request::LoadArtifact {
                artifact_sha256: digest.clone(),
                model_revision: stack.model_revision().into(),
                layer: stack.layer(),
                epoch,
                resident_order: stack.resident_order().to_vec(),
                artifact: base64::encode(&artifact.bytes),
            };
            self.send_load(
                LoadTransfer {
                    request: &request,
                    body: None,
                    digest,
                    epoch,
                    rebind: false,
                    model: None,
                },
                timeout,
            )
        }
    }

    fn send_load(&mut self, transfer: LoadTransfer<'_>, timeout: Duration) -> Result<LoadReceipt> {
        if self.preweighted_inflight {
            return Err(invalid("an MXFP4 EXEC is still pending"));
        }
        let LoadTransfer {
            request,
            body,
            digest,
            epoch,
            rebind,
            model,
        } = transfer;
        let bytes = encode(request)?;
        let request_bytes = bytes
            .len()
            .checked_add(body.map_or(0, <[u8]>::len))
            .ok_or_else(|| Error::new("MESSAGE_TOO_LARGE", "LOAD byte count overflow"))?;
        let request_bytes_u64 = u64::try_from(request_bytes)
            .map_err(|_| Error::new("MESSAGE_TOO_LARGE", "LOAD byte count does not fit u64"))?;
        match body {
            Some(body) => self.write_parts(&[&bytes, body], true)?,
            None => self.write(&bytes)?,
        }
        match self.reply(timeout)? {
            Reply::Loaded {
                artifact_sha256,
                epoch: bound,
                transferred,
                ..
            } if artifact_sha256 == digest && bound == epoch && transferred != rebind => {
                if transferred {
                    self.traffic.loads_transferred += 1;
                    self.traffic.weight_bytes_sent += request_bytes_u64;
                    self.resident = model.cloned();
                    self.mxfp4_binding = None;
                } else {
                    self.traffic.rebinds += 1;
                }
                self.artifact = Some((digest.clone(), epoch));
                Ok(LoadReceipt {
                    rank: self.rank,
                    artifact_sha256: digest,
                    epoch,
                    transferred,
                    request_bytes: request_bytes_u64,
                })
            }
            Reply::Refused { error } => {
                if body.is_some() {
                    // A capability-advertising peer that refuses a raw transfer is not
                    // trusted to have consumed the complete body; never reuse or replay it.
                    self.usable = false;
                }
                Err(error)
            }
            _ => {
                self.usable = false;
                Err(Error::new("SESSION_PROTOCOL", "unexpected reply to LOAD"))
            }
        }
    }

    /// Load one immutable Engram shard into its own worker slot, alongside any MoE resident.
    /// A repeated identical call is local and transfers zero bytes.
    pub fn load_engram_shard(
        &mut self,
        shard: &EngramShard,
        timeout: Duration,
    ) -> Result<EngramLoadReceipt> {
        self.load_engram_shard_for_owner(shard, shard.rank(), timeout)
    }

    /// Load a shard whose dense owner index is independent from the worker's global rank.
    pub fn load_engram_shard_for_owner(
        &mut self,
        shard: &EngramShard,
        owner_rank: u32,
        timeout: Duration,
    ) -> Result<EngramLoadReceipt> {
        if !self
            .hello
            .capabilities
            .iter()
            .any(|c| c == CAPABILITY_ENGRAM_ROWS_RAW)
        {
            return Err(Error::new(
                "CAPABILITY_MISSING",
                "worker does not advertise raw Engram owner rows; nothing was sent",
            ));
        }
        if shard.rank() != owner_rank || owner_rank >= shard.world_size() {
            return Err(invalid(
                "Engram shard rank differs from the assigned owner index",
            ));
        }
        let digest = shard.sha256().to_owned();
        if self.engram_artifact.as_deref() == Some(&digest) {
            if self.engram_owner_rank != Some(owner_rank) {
                return Err(invalid(
                    "loaded Engram shard is assigned to another owner index",
                ));
            }
            return Ok(EngramLoadReceipt {
                rank: self.rank,
                artifact_sha256: digest,
                transferred: false,
                request_bytes: 0,
            });
        }
        if self.engram_artifact.is_some() {
            self.evict_engram(timeout)?;
        }
        let artifact_bytes = u64::try_from(shard.bytes().len()).map_err(|_| {
            Error::new(
                "MESSAGE_TOO_LARGE",
                "Engram shard length does not fit protocol",
            )
        })?;
        let request = Request::LoadEngramShardRaw {
            table: shard.table().clone(),
            world_size: shard.world_size(),
            worker_rank: self.rank,
            owner_rank,
            artifact_sha256: digest.clone(),
            artifact_bytes,
        };
        let header = encode(&request)?;
        let total = header
            .len()
            .checked_add(shard.bytes().len())
            .filter(|&n| n <= MAX_MESSAGE_BYTES)
            .ok_or_else(|| {
                Error::new(
                    "MESSAGE_TOO_LARGE",
                    "Engram shard header and bytes exceed the session bound",
                )
            })?;
        self.write_parts(&[&header, shard.bytes()], true)?;
        match self.reply(timeout)? {
            Reply::EngramLoaded {
                artifact_sha256,
                table,
                transferred,
            } if artifact_sha256 == digest && table == *shard.table() && transferred => {
                self.engram_artifact = Some(digest.clone());
                self.engram_owner_rank = Some(owner_rank);
                Ok(EngramLoadReceipt {
                    rank: self.rank,
                    artifact_sha256: digest,
                    transferred: true,
                    request_bytes: total as u64,
                })
            }
            Reply::EngramLoaded {
                artifact_sha256,
                table,
                transferred: false,
            } if artifact_sha256 == digest && table == *shard.table() => {
                self.engram_artifact = Some(digest.clone());
                self.engram_owner_rank = Some(owner_rank);
                Ok(EngramLoadReceipt {
                    rank: self.rank,
                    artifact_sha256: digest,
                    transferred: false,
                    request_bytes: total as u64,
                })
            }
            Reply::Refused { error } => {
                self.usable = false;
                Err(error)
            }
            _ => {
                self.usable = false;
                Err(Error::new(
                    "SESSION_PROTOCOL",
                    "unexpected reply to Engram LOAD",
                ))
            }
        }
    }

    /// Exchange one bounded row-ID list with this worker. Any lost reply makes the stream
    /// unusable (the lookup outcome is UNKNOWN); this method never retries automatically.
    pub fn fetch_engram_rows(
        &mut self,
        stamp: &moe::Stamp,
        table: &TableIdentity,
        world_size: u32,
        row_ids: &[u64],
        timeout: Duration,
    ) -> Result<OwnerRows> {
        self.fetch_engram_rows_for_owner(stamp, table, world_size, self.rank, row_ids, timeout)
    }

    /// Fetch owner rows when worker rank and the shard's dense owner index differ.
    pub fn fetch_engram_rows_for_owner(
        &mut self,
        stamp: &moe::Stamp,
        table: &TableIdentity,
        world_size: u32,
        owner_rank: u32,
        row_ids: &[u64],
        timeout: Duration,
    ) -> Result<OwnerRows> {
        if self.engram_artifact.is_none() {
            return Err(invalid("no Engram shard is loaded in this session"));
        }
        if self.engram_owner_rank != Some(owner_rank) {
            return Err(invalid(
                "requested Engram owner index differs from loaded shard",
            ));
        }
        let binding = engram::RequestBinding::new(stamp, table, world_size, row_ids)
            .map_err(|e| invalid(e.0))?;
        self.fetch_engram_rows_bound(
            stamp, table, world_size, owner_rank, row_ids, &binding, timeout,
        )
        .map(engram::VerifiedOwnerFrame::into_owner_rows)
    }

    #[allow(clippy::too_many_arguments)]
    fn fetch_engram_rows_bound(
        &mut self,
        stamp: &moe::Stamp,
        table: &TableIdentity,
        world_size: u32,
        owner_rank: u32,
        row_ids: &[u64],
        binding: &engram::RequestBinding,
        timeout: Duration,
    ) -> Result<engram::VerifiedOwnerFrame> {
        if self.engram_artifact.is_none() {
            return Err(invalid("no Engram shard is loaded in this session"));
        }
        let attempt = self
            .next_engram_attempt
            .checked_add(1)
            .ok_or_else(|| invalid("Engram row attempt counter overflow"))?;
        self.next_engram_attempt = attempt;
        let supports_varint = self
            .hello
            .capabilities
            .iter()
            .any(|capability| capability == CAPABILITY_ENGRAM_ROW_IDS_VARINT);
        let (header, body) = encode_engram_rows_request(
            stamp,
            table,
            world_size,
            owner_rank,
            row_ids,
            attempt,
            supports_varint,
        )?;
        self.raw_engram_attempt.store(attempt, Ordering::SeqCst);
        let write = match body.as_deref() {
            Some(body) => self.write_parts(&[&header, body], true),
            None => self.write(&header),
        };
        if let Err(error) = write {
            self.raw_engram_attempt.store(0, Ordering::SeqCst);
            return Err(error);
        }
        let reply = self.reply(timeout);
        self.raw_engram_attempt.store(0, Ordering::SeqCst);
        match reply? {
            Reply::EngramRowsRaw {
                attempt: returned,
                frame_bytes,
                frame,
            } if returned == attempt && frame_bytes == frame.len() as u64 => {
                engram::decode_owner_frame_verified_owned(frame, binding, owner_rank).map_err(|e| {
                    self.usable = false;
                    Error::new("WORKER_PROTOCOL", e.0)
                })
            }
            Reply::Refused { error } => Err(error),
            _ => {
                self.usable = false;
                Err(Error::new(
                    "SESSION_PROTOCOL",
                    "unexpected reply to Engram row request",
                ))
            }
        }
    }

    pub fn evict_engram(&mut self, timeout: Duration) -> Result<()> {
        let Some(digest) = self.engram_artifact.clone() else {
            return Ok(());
        };
        self.send(&Request::EvictEngram {
            artifact_sha256: digest.clone(),
        })?;
        match self.reply(timeout)? {
            Reply::EngramEvicted { artifact_sha256 } if artifact_sha256 == digest => {
                self.engram_artifact = None;
                self.engram_owner_rank = None;
                Ok(())
            }
            Reply::Refused { error } => Err(error),
            _ => {
                self.usable = false;
                Err(Error::new(
                    "SESSION_PROTOCOL",
                    "unexpected reply to Engram EVICT",
                ))
            }
        }
    }

    /// Request owner rows concurrently from one complete rank set and reconstruct them in
    /// original token/hash order. The full ordered-ID binding is built once for the batch.
    /// Any rank failure returns no partial table and its session records UNKNOWN when applicable.
    pub fn fetch_engram_batch(
        sessions: &mut [Session],
        stamp: &moe::Stamp,
        table: &TableIdentity,
        row_ids: &[u64],
        timeout: Duration,
    ) -> Result<Vec<u8>> {
        Self::fetch_engram_batch_for_owners(sessions, stamp, table, row_ids, timeout)
    }

    /// Dispatch owner-filtered rows in dense owner order, independent of global worker ranks.
    /// Empty owners receive no request at all; nonempty owners receive unique owner-local IDs.
    pub fn fetch_engram_batch_for_owners(
        sessions: &mut [Session],
        stamp: &moe::Stamp,
        table: &TableIdentity,
        row_ids: &[u64],
        timeout: Duration,
    ) -> Result<Vec<u8>> {
        Self::fetch_engram_batch_with_receipt(sessions, stamp, table, row_ids, timeout)
            .map(|receipt| receipt.rows)
    }

    /// As `fetch_engram_batch_for_owners`, with validated data-payload counters for ablation
    /// and runtime telemetry. Empty owners are counted as skipped, never as successful calls.
    pub fn fetch_engram_batch_with_receipt(
        sessions: &mut [Session],
        stamp: &moe::Stamp,
        table: &TableIdentity,
        row_ids: &[u64],
        timeout: Duration,
    ) -> Result<EngramBatchReceipt> {
        let world_size = u32::try_from(sessions.len())
            .map_err(|_| invalid("Engram worker count does not fit the protocol"))?;
        if world_size == 0 {
            return Err(invalid("Engram batch requires at least one worker"));
        }
        let mut seen_ranks = BTreeSet::new();
        for session in sessions.iter() {
            if !seen_ranks.insert(session.rank) {
                return Err(invalid("Engram sessions contain a duplicate worker rank"));
            }
        }
        for (owner_rank, session) in sessions.iter().enumerate() {
            if session.engram_owner_rank != Some(owner_rank as u32) {
                return Err(invalid(
                    "Engram sessions are not loaded in dense owner order",
                ));
            }
        }
        // Validate the complete public request before deriving smaller owner-local requests.
        engram::RequestBinding::new(stamp, table, world_size, row_ids)
            .map_err(|error| invalid(error.0))?;
        let partitions = engram::partition_unique_row_ids(table, world_size, row_ids)
            .map_err(|error| invalid(error.0))?;
        let bindings: Vec<_> = partitions
            .iter()
            .map(|ids| {
                if ids.is_empty() {
                    Ok(None)
                } else {
                    engram::RequestBinding::new(stamp, table, world_size, ids)
                        .map(Some)
                        .map_err(|error| invalid(error.0))
                }
            })
            .collect::<Result<_>>()?;
        let owner_requests = partitions.iter().filter(|ids| !ids.is_empty()).count();
        let unique_ids_transferred = partitions.iter().map(Vec::len).sum::<usize>();
        let owners = std::thread::scope(|scope| {
            let handles: Vec<_> = sessions
                .iter_mut()
                .enumerate()
                .filter_map(|(owner_rank, session)| {
                    let ids = &partitions[owner_rank];
                    let binding = bindings[owner_rank].as_ref()?;
                    Some((
                        owner_rank,
                        scope.spawn(move || {
                            session.fetch_engram_rows_bound(
                                stamp,
                                table,
                                world_size,
                                owner_rank as u32,
                                ids,
                                binding,
                                timeout,
                            )
                        }),
                    ))
                })
                .collect();
            handles
                .into_iter()
                .map(|(owner_rank, handle)| {
                    let owner = handle
                        .join()
                        .map_err(|_| Error::new("WORKER_PANIC", "Engram row worker panicked"))??;
                    Ok((owner_rank, owner))
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let owner_payload_bytes = owners
            .iter()
            .try_fold(0u64, |total, (_, owner)| {
                total.checked_add(owner.payload_len() as u64)
            })
            .ok_or_else(|| invalid("Engram owner payload byte count overflow"))?;
        let owner_rows: Vec<_> = owners.into_iter().map(|(_, owner)| owner).collect();
        let rows = engram::reconstruct_partitioned_rows_owned_verified(
            stamp,
            table,
            world_size,
            row_ids,
            &owner_rows,
        )
        .map_err(|error| Error::new("WORKER_PROTOCOL", error.0))?;
        Ok(EngramBatchReceipt {
            rows,
            requested_ids: row_ids.len(),
            unique_ids_transferred,
            owner_payload_bytes,
            owner_requests,
            skipped_empty_owners: world_size as usize - owner_requests,
        })
    }

    /// Drop the loaded artifact on the worker.
    pub fn evict(&mut self, timeout: Duration) -> Result<()> {
        if self.preweighted_inflight {
            return Err(invalid("an MXFP4 EXEC is still pending"));
        }
        let Some((digest, _)) = self.artifact.clone() else {
            return Ok(());
        };
        self.send(&Request::Evict {
            artifact_sha256: digest.clone(),
        })?;
        match self.reply(timeout)? {
            Reply::Evicted { artifact_sha256 } if artifact_sha256 == digest => {
                self.artifact = None;
                self.resident = None;
                self.mxfp4_binding = None;
                Ok(())
            }
            Reply::Refused { error } => Err(error),
            _ => {
                self.usable = false;
                Err(Error::new("SESSION_PROTOCOL", "unexpected reply to EVICT"))
            }
        }
    }

    /// Send one EXEC. Errors here are local refusals: nothing was sent.
    pub fn start_exec(
        &mut self,
        dispatch: &routing::RankDispatch,
        attempt: u64,
        inputs: Vec<TokenInput>,
        fault: Fault,
    ) -> Result<PendingExec> {
        let exec = self.encode_exec(dispatch, attempt, inputs, fault, false)?;
        Ok(self.send_exec(exec))
    }

    /// One EXEC encoded, not yet sent. Errors here are local refusals.
    fn raw_completions_supported(&self) -> bool {
        self.hello
            .capabilities
            .iter()
            .any(|capability| capability == CAPABILITY_COMPLETION_F64_RAW)
    }

    fn raw_completion_batch_supported(&self) -> bool {
        self.raw_completions_supported()
            && self.hello.capabilities.iter().any(|capability| {
                capability == CAPABILITY_COMPLETION_BATCH_F64_RAW
                    || capability == CAPABILITY_COMPLETION_BATCH_SESSION_F64_RAW
            })
    }

    fn raw_completion_batch_session_supported(&self) -> bool {
        self.raw_completions_supported()
            && self
                .hello
                .capabilities
                .iter()
                .any(|capability| capability == CAPABILITY_COMPLETION_BATCH_SESSION_F64_RAW)
    }

    fn raw_completion_fp16_batch_supported(&self) -> bool {
        self.hello
            .capabilities
            .iter()
            .any(|capability| capability == CAPABILITY_COMPLETION_BATCH_FP16_RAW)
    }

    fn encode_exec<I: Into<RankInputs>>(
        &self,
        dispatch: &routing::RankDispatch,
        attempt: u64,
        inputs: I,
        fault: Fault,
        completion_fp16: bool,
    ) -> Result<(PendingExec, Vec<u8>, Option<Vec<u8>>)> {
        if self.mxfp4_binding.is_some() {
            return Err(Error::new(
                "ARITHMETIC_CONTRACT_MISMATCH",
                "MXFP4 requires an explicit preweighted FP32 EXEC",
            ));
        }
        if !self.usable {
            return Err(Error::new(
                "SESSION_UNUSABLE",
                "session must be reopened after a failure",
            ));
        }
        let Some((digest, _)) = self.artifact.clone() else {
            return Err(invalid("no artifact is loaded in this session"));
        };
        if completion_fp16 && !self.raw_completion_fp16_batch_supported() {
            return Err(Error::new(
                "SESSION_CAPABILITY",
                "rank worker does not advertise raw FP16 completion batches",
            ));
        }
        let (inputs, fp16_inputs) = match inputs.into() {
            RankInputs::F64(inputs) => (inputs, None),
            RankInputs::Fp16(inputs) => (Vec::new(), Some(inputs)),
        };
        if fp16_inputs.is_some()
            && !self
                .hello
                .capabilities
                .iter()
                .any(|capability| capability == CAPABILITY_EXEC_FP16_RAW)
        {
            return Err(Error::new(
                "CAPABILITY_MISSING",
                "rank worker does not advertise raw FP16 activation input support",
            ));
        }
        let raw_supported = if fp16_inputs.is_some() {
            true
        } else {
            self.hello
                .capabilities
                .iter()
                .any(|capability| capability == CAPABILITY_EXEC_F64_RAW)
                && inputs.len() <= 4096
                && inputs.iter().all(|input| input.value.len() <= 65536)
        };
        if completion_fp16 && !raw_supported {
            return Err(Error::new(
                "SESSION_CAPABILITY",
                "rank worker cannot accept the bounded raw EXEC required for FP16 replies",
            ));
        }
        let mut expect_raw_completion_batch = false;
        let mut expect_raw_completion_fp16_batch = false;
        let mut latch_raw_completion_batch_session = false;
        let raw = if let Some(fp16_inputs) = fp16_inputs {
            if fp16_inputs.len() > 4096
                || fp16_inputs
                    .iter()
                    .any(|input| input.value_bits.len() > 65536)
            {
                return Err(invalid(
                    "raw FP16 EXEC inputs exceed the token or row bound",
                ));
            }
            let element_count = fp16_inputs.iter().try_fold(0_usize, |total, input| {
                total.checked_add(input.value_bits.len())
            });
            let body_len = element_count.and_then(|n| n.checked_mul(std::mem::size_of::<u16>()));
            let Some(body_len) = body_len.filter(|&n| n <= MAX_RAW_ACTIVATION_BYTES) else {
                return Err(Error::new(
                    "MESSAGE_TOO_LARGE",
                    "raw FP16 EXEC activation body exceeds the session bound",
                ));
            };
            let mut metadata = Vec::new();
            metadata.try_reserve_exact(fp16_inputs.len()).map_err(|e| {
                Error::new(
                    "EXEC_ALLOC",
                    format!("cannot allocate raw FP16 input metadata: {e}"),
                )
            })?;
            for input in &fp16_inputs {
                metadata.push(RawTokenInputMeta {
                    token: input.token,
                    values: u64::try_from(input.value_bits.len())
                        .map_err(|_| invalid("raw FP16 row length does not fit u64"))?,
                });
            }
            let activation_bytes = u64::try_from(body_len)
                .map_err(|_| invalid("raw FP16 activation byte count does not fit u64"))?;
            let batch_session = self.raw_completion_batch_session_supported();
            let batch_raw = !completion_fp16
                && if batch_session {
                    !self.raw_completion_batch_session_sent
                } else {
                    self.raw_completion_batch_supported()
                };
            let header = encode(&Request::ExecRawFp16 {
                artifact_sha256: digest.clone(),
                dispatch: dispatch.clone(),
                attempt,
                inputs: metadata,
                activation_bytes,
                fault: fault.clone(),
                raw_completions: completion_fp16 || self.raw_completions_supported(),
                raw_completion_batch: batch_raw,
                raw_completion_fp16_batch: completion_fp16,
            })?;
            if header
                .len()
                .checked_add(body_len)
                .is_none_or(|n| n > MAX_MESSAGE_BYTES)
            {
                return Err(Error::new(
                    "MESSAGE_TOO_LARGE",
                    "raw FP16 EXEC header and activation body exceed the session bound",
                ));
            }
            let mut body = Vec::new();
            body.try_reserve_exact(body_len).map_err(|e| {
                Error::new(
                    "EXEC_ALLOC",
                    format!("cannot allocate raw FP16 activation body: {e}"),
                )
            })?;
            for input in &fp16_inputs {
                for &bits in &input.value_bits {
                    body.extend_from_slice(&bits.to_le_bytes());
                }
            }
            debug_assert_eq!(body.len(), body_len);
            if completion_fp16 || self.raw_completions_supported() {
                self.raw_completions.store(true, Ordering::SeqCst);
            }
            if batch_raw {
                expect_raw_completion_batch = true;
                latch_raw_completion_batch_session = batch_session;
            }
            expect_raw_completion_fp16_batch = completion_fp16;
            Some((header, body))
        } else if raw_supported {
            let element_count = inputs
                .iter()
                .try_fold(0_usize, |total, input| total.checked_add(input.value.len()));
            let body_len = element_count.and_then(|n| n.checked_mul(std::mem::size_of::<f64>()));
            if let Some(body_len) = body_len.filter(|&n| n <= MAX_RAW_ACTIVATION_BYTES) {
                let mut metadata = Vec::new();
                metadata.try_reserve_exact(inputs.len()).map_err(|e| {
                    Error::new(
                        "EXEC_ALLOC",
                        format!("cannot allocate raw input metadata: {e}"),
                    )
                })?;
                for input in &inputs {
                    metadata.push(RawTokenInputMeta {
                        token: input.token,
                        values: u64::try_from(input.value.len())
                            .map_err(|_| invalid("raw EXEC input row length does not fit u64"))?,
                    });
                }
                let activation_bytes = u64::try_from(body_len)
                    .map_err(|_| invalid("raw EXEC activation body length does not fit u64"))?;
                let batch_session = self.raw_completion_batch_session_supported();
                let batch_raw = !completion_fp16
                    && if batch_session {
                        !self.raw_completion_batch_session_sent
                    } else {
                        self.raw_completion_batch_supported()
                    };
                let header = encode(&Request::ExecRaw {
                    artifact_sha256: digest.clone(),
                    dispatch: dispatch.clone(),
                    attempt,
                    inputs: metadata,
                    activation_bytes,
                    fault: fault.clone(),
                    raw_completions: completion_fp16 || self.raw_completions_supported(),
                    raw_completion_batch: batch_raw,
                    raw_completion_fp16_batch: completion_fp16,
                });
                match header {
                    Ok(header)
                        if header
                            .len()
                            .checked_add(body_len)
                            .is_some_and(|n| n <= MAX_MESSAGE_BYTES) =>
                    {
                        let mut body = Vec::new();
                        body.try_reserve_exact(body_len).map_err(|e| {
                            Error::new(
                                "EXEC_ALLOC",
                                format!("cannot allocate raw activation body: {e}"),
                            )
                        })?;
                        for input in &inputs {
                            for value in &input.value {
                                body.extend_from_slice(&value.to_le_bytes());
                            }
                        }
                        debug_assert_eq!(body.len(), body_len);
                        if completion_fp16 || self.raw_completions_supported() {
                            // Admitted before the request is written, so before any reply.
                            self.raw_completions.store(true, Ordering::SeqCst);
                        }
                        if batch_raw {
                            expect_raw_completion_batch = true;
                            latch_raw_completion_batch_session = batch_session;
                        }
                        expect_raw_completion_fp16_batch = completion_fp16;
                        Some((header, body))
                    }
                    Ok(_) if completion_fp16 => {
                        return Err(Error::new(
                            "MESSAGE_TOO_LARGE",
                            "raw FP16 EXEC header and activation body exceed the session bound",
                        ))
                    }
                    Ok(_) => None,
                    Err(error) if error.code == "MESSAGE_TOO_LARGE" && completion_fp16 => {
                        return Err(error)
                    }
                    Err(error) if error.code == "MESSAGE_TOO_LARGE" => None,
                    Err(error) => return Err(error),
                }
            } else if completion_fp16 {
                return Err(Error::new(
                    "MESSAGE_TOO_LARGE",
                    "raw FP16 EXEC activation body exceeds the session bound",
                ));
            } else {
                None
            }
        } else if completion_fp16 {
            return Err(Error::new(
                "SESSION_CAPABILITY",
                "raw FP16 completion batches require raw EXEC support",
            ));
        } else {
            None
        };
        let (bytes, body) = match raw {
            Some((header, body)) => (header, Some(body)),
            None => (
                encode(&Request::Exec {
                    artifact_sha256: digest,
                    dispatch: dispatch.clone(),
                    attempt,
                    inputs,
                    fault,
                })?,
                None,
            ),
        };
        let request_bytes = bytes
            .len()
            .checked_add(body.as_ref().map_or(0, Vec::len))
            .ok_or_else(|| Error::new("MESSAGE_TOO_LARGE", "EXEC byte count overflow"))?;
        let receipt = ExecReceipt {
            rank: dispatch.rank,
            attempt,
            // Admission alone cannot prove whether calculation began.
            state: ReceiptState::Unknown,
            started: false,
            expected_slots: dispatch.slots.len(),
            received_slots: 0,
            request_bytes: u64::try_from(request_bytes)
                .map_err(|_| Error::new("MESSAGE_TOO_LARGE", "EXEC byte count does not fit u64"))?,
            error: None,
        };
        let pending = PendingExec {
            dispatch: dispatch.clone(),
            receipt,
            expect_raw_completion_batch,
            expect_raw_completion_fp16_batch,
            latch_raw_completion_batch_session,
        };
        Ok((pending, bytes, body))
    }

    /// Send an encoded EXEC; a failed write is its receipt's error, as any lost contact.
    fn send_exec(
        &mut self,
        (mut pending, header, body): (PendingExec, Vec<u8>, Option<Vec<u8>>),
    ) -> PendingExec {
        if pending.expect_raw_completion_fp16_batch {
            self.raw_completion_fp16_batch.store(true, Ordering::SeqCst);
        }
        if pending.expect_raw_completion_batch {
            // Admit the matching reply before the worker can write it, but only after
            // identity validation has succeeded and this EXEC is about to be sent.
            self.raw_completion_batch.store(true, Ordering::SeqCst);
        }
        let write = match body.as_deref() {
            Some(body) => self.write_parts(&[&header, body], true),
            None => self.write(&header),
        };
        if let Err(error) = write {
            pending.receipt.error = Some(error);
        } else {
            if pending.latch_raw_completion_batch_session {
                self.raw_completion_batch_session_sent = true;
            }
            self.traffic.execs += 1;
        }
        pending
    }

    /// Read one EXEC's events to its end. Completions go to `collector`, or are only
    /// checked and dropped when `None` (draining after another rank failed).
    pub fn finish_exec(
        &mut self,
        pending: PendingExec,
        mut collector: Option<&mut moe::Collector>,
        deadline: Instant,
    ) -> ExecReceipt {
        let PendingExec {
            dispatch,
            mut receipt,
            expect_raw_completion_fp16_batch,
            ..
        } = pending;
        if receipt.error.is_some() {
            return receipt;
        }
        let mut seen = BTreeSet::new();
        loop {
            let reply = match self.reply_until(deadline) {
                Ok(reply) => reply,
                Err(error) => {
                    receipt.state = ReceiptState::Unknown;
                    receipt.error = Some(error);
                    return receipt;
                }
            };
            let step: Result<bool> = match reply {
                Reply::Refused { error } if !receipt.started => {
                    // EXEC refusals are protocol-level: recoverable worker rejections use
                    // Event::Rejected, while raw-body refusals close the worker stream.
                    self.usable = false;
                    receipt.state = ReceiptState::NotStarted;
                    receipt.error = Some(error);
                    return receipt;
                }
                Reply::Event { event } => {
                    if event.attempt != receipt.attempt {
                        Err(Error::new(
                            "STALE_ATTEMPT",
                            "worker event belongs to another attempt",
                        ))
                    } else if event.stamp != dispatch.stamp || event.rank != dispatch.rank {
                        Err(Error::new(
                            "INVALID_COMPLETION",
                            "worker event has stale stamp or foreign rank",
                        ))
                    } else {
                        match event.event {
                            EventKind::Started if !receipt.started => {
                                receipt.started = true;
                                receipt.state = ReceiptState::Started;
                                Ok(false)
                            }
                            EventKind::Rejected { error } if !receipt.started => {
                                receipt.state = ReceiptState::NotStarted;
                                receipt.error = Some(error);
                                return receipt;
                            }
                            // The worker reported the failure and is back at its loop.
                            EventKind::Failed { error } if receipt.started => {
                                receipt.error = Some(error);
                                return receipt;
                            }
                            EventKind::Completion { completion } if receipt.started => {
                                if expect_raw_completion_fp16_batch {
                                    Err(Error::new(
                                        "WORKER_PROTOCOL",
                                        "f64 completion event arrived for an FP16 EXEC",
                                    ))
                                } else if completion.rank != dispatch.rank {
                                    Err(Error::new(
                                        "INVALID_COMPLETION",
                                        "completion has foreign rank",
                                    ))
                                } else if let Some(collector) = collector.as_deref_mut() {
                                    collector.accept(completion).map(|()| {
                                        receipt.received_slots += 1;
                                        false
                                    })
                                } else if seen.insert((completion.token, completion.slot)) {
                                    receipt.received_slots += 1;
                                    Ok(false)
                                } else {
                                    Err(Error::new(
                                        "INVALID_COMPLETION",
                                        "duplicate completion while draining",
                                    ))
                                }
                            }
                            EventKind::Finished { slots } if receipt.started => {
                                if slots != receipt.expected_slots
                                    || receipt.received_slots != slots
                                {
                                    Err(Error::new(
                                        "BATCH_INCOMPLETE",
                                        "rank finished with missing results",
                                    ))
                                } else {
                                    receipt.state = ReceiptState::Completed;
                                    Ok(true)
                                }
                            }
                            _ => Err(Error::new(
                                "WORKER_PROTOCOL",
                                "invalid worker event lifecycle",
                            )),
                        }
                    }
                }
                Reply::CompletionBatch {
                    stamp,
                    rank,
                    attempt,
                    completions,
                } => {
                    if expect_raw_completion_fp16_batch {
                        Err(Error::new(
                            "WORKER_PROTOCOL",
                            "f64 completion batch arrived for an FP16 EXEC",
                        ))
                    } else if attempt != receipt.attempt
                        || stamp != dispatch.stamp
                        || rank != dispatch.rank
                    {
                        Err(Error::new(
                            "INVALID_COMPLETION",
                            "completion batch has a stale stamp, attempt or foreign rank",
                        ))
                    } else if !receipt.started || completions.is_empty() {
                        Err(Error::new(
                            "WORKER_PROTOCOL",
                            "completion batch arrived outside an active rank execution",
                        ))
                    } else {
                        let mut failure = None;
                        for completion in completions {
                            if completion.rank != dispatch.rank {
                                failure = Some(Error::new(
                                    "INVALID_COMPLETION",
                                    "completion batch contains a foreign rank",
                                ));
                                break;
                            }
                            if let Some(collector) = collector.as_deref_mut() {
                                if let Err(error) = collector.accept(completion) {
                                    failure = Some(error);
                                    break;
                                }
                                receipt.received_slots += 1;
                            } else if seen.insert((completion.token, completion.slot)) {
                                receipt.received_slots += 1;
                            } else {
                                failure = Some(Error::new(
                                    "INVALID_COMPLETION",
                                    "duplicate completion while draining",
                                ));
                                break;
                            }
                        }
                        match failure {
                            Some(error) => Err(error),
                            None => Ok(false),
                        }
                    }
                }
                Reply::CompletionBatchFp16 {
                    stamp,
                    completion_stamp,
                    rank,
                    attempt,
                    hidden,
                    completions,
                    fp16_bits,
                } => {
                    if !expect_raw_completion_fp16_batch {
                        Err(Error::new(
                            "WORKER_PROTOCOL",
                            "FP16 completion batch arrived without this EXEC's opt-in",
                        ))
                    } else if attempt != receipt.attempt
                        || stamp != dispatch.stamp
                        || completion_stamp != dispatch.stamp
                        || rank != dispatch.rank
                    {
                        Err(Error::new(
                            "INVALID_COMPLETION",
                            "FP16 batch has a stale stamp, attempt or foreign rank",
                        ))
                    } else if !receipt.started || completions.is_empty() {
                        Err(Error::new(
                            "WORKER_PROTOCOL",
                            "FP16 completion batch arrived outside an active rank execution",
                        ))
                    } else if let Some(collector) = collector.as_deref_mut() {
                        // The bounded frame reader already checked exact payload shape and
                        // every finite half encoding. Keep the row metadata borrowed here
                        // rather than building another per-slot identity vector.
                        collector
                            .accept_fp16_batch_with(
                                &completion_stamp,
                                rank,
                                completions.len(),
                                |index| {
                                    let completion = &completions[index];
                                    moe::Fp16SlotIdentity {
                                        token: completion.token,
                                        slot: completion.slot,
                                        expert: completion.expert,
                                        rank,
                                        local: completion.local,
                                    }
                                },
                                hidden,
                                fp16_bits,
                            )
                            .map(|count| {
                                receipt.received_slots += count;
                                false
                            })
                    } else {
                        let mut failure = None;
                        for completion in completions {
                            let expected = dispatch.slots.iter().find(|slot| {
                                slot.slot.token == completion.token
                                    && slot.slot.slot == completion.slot
                            });
                            if expected.is_none_or(|slot| {
                                slot.slot.rank != rank
                                    || slot.slot.expert != completion.expert
                                    || slot.slot.local != completion.local
                            }) || !seen.insert((completion.token, completion.slot))
                            {
                                failure = Some(Error::new(
                                    "INVALID_COMPLETION",
                                    "duplicate or unexpected FP16 slot while draining",
                                ));
                                break;
                            }
                            receipt.received_slots += 1;
                        }
                        match failure {
                            Some(error) => Err(error),
                            None => Ok(false),
                        }
                    }
                }
                _ => Err(Error::new("SESSION_PROTOCOL", "unexpected reply to EXEC")),
            };
            match step {
                Ok(true) => return receipt,
                Ok(false) => {}
                Err(error) => {
                    // The worker may still be streaming: the position is no longer trusted.
                    self.usable = false;
                    receipt.state = ReceiptState::Unknown;
                    receipt.error = Some(error);
                    return receipt;
                }
            }
        }
    }

    /// One EXEC, end to end. `Err` only for a local refusal before anything was sent.
    pub fn exec(
        &mut self,
        dispatch: &routing::RankDispatch,
        attempt: u64,
        inputs: Vec<TokenInput>,
        fault: Fault,
        collector: &mut moe::Collector,
        timeout: Duration,
    ) -> Result<ExecReceipt> {
        let deadline = Instant::now() + timeout;
        let pending = self.start_exec(dispatch, attempt, inputs, fault)?;
        let receipt = self.finish_exec(pending, Some(collector), deadline);
        if receipt.error.is_some() {
            collector.expire();
        }
        Ok(receipt)
    }

    fn disconnect_reader_queue(&mut self) {
        let (_, disconnected) = mpsc::channel();
        drop(std::mem::replace(&mut self.frames, disconnected));
    }

    fn cancel_output_readers(&mut self) {
        #[cfg(unix)]
        if let Some((stdout, stderr)) = self.output_cancellation.take() {
            // shutdown affects the reader's cloned descriptor too, even if a
            // descendant still owns the opposite endpoint. No descendant is killed.
            let _ = stdout.shutdown(std::net::Shutdown::Both);
            let _ = stderr.shutdown(std::net::Shutdown::Both);
        }
    }

    fn reap(&mut self, deadline: Instant) -> (Option<ExitStatus>, bool) {
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    self.reaped = true;
                    return (Some(status), false);
                }
                Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(2)),
                _ => break,
            }
        }
        let _ = self.child.kill();
        let status = self.child.wait().ok();
        self.reaped = true;
        (status, true)
    }
    fn finish_close(
        mut self,
        bye: Option<(u64, Option<u64>)>,
        error: Option<Error>,
        deadline: Instant,
    ) -> CloseReceipt {
        // No more replies will be consumed. Disconnect a reader blocked on the
        // bounded queue before waiting for it; otherwise close would wait on itself.
        self.disconnect_reader_queue();
        drop(self.stdin.take());
        let (status, killed) = self.reap(deadline);
        // A process the worker left behind can hold its output pipes open: wait for the
        // readers until the deadline (with a grace for the pipes of a process just
        // killed). Preserve SESSION_PIPES diagnostics, then on Unix cancel local
        // output endpoints so inherited descriptors cannot retain reader threads.
        let until = deadline.max(Instant::now() + PIPE_GRACE);
        let pending = |s: &Session| {
            s.reader.as_ref().is_some_and(|h| !h.is_finished())
                || s.stderr.as_ref().is_some_and(|h| !h.is_finished())
        };
        while pending(&self) && Instant::now() < until {
            thread::sleep(Duration::from_millis(2));
        }
        let mut error = error;
        if pending(&self) {
            error = error.or(Some(Error::new(
                "SESSION_PIPES",
                "a process left by the worker kept its output open past the close deadline",
            )));
        }
        self.cancel_output_readers();
        if let Some(reader) = self.reader.take().filter(|h| cfg!(unix) || h.is_finished()) {
            let _ = reader.join();
        }
        let (stderr_bytes, kept) = self
            .stderr
            .take()
            .filter(|h| cfg!(unix) || h.is_finished())
            .and_then(|h| h.join().ok())
            .unwrap_or_default();
        CloseReceipt {
            rank: self.rank,
            pid: self.child.id(),
            bye_execs: bye.map(|(execs, _)| execs),
            peak_rss_kib: bye.and_then(|(_, rss)| rss),
            exit_code: status.and_then(|s| s.code()),
            killed,
            stderr: String::from_utf8_lossy(&kept).into_owned(),
            stderr_bytes,
            error,
        }
    }
    /// SHUTDOWN, confirmed by BYE, then reap; a worker that does not exit by the deadline
    /// is killed. An unusable session is only closed and reaped.
    pub fn close(mut self, timeout: Duration) -> CloseReceipt {
        let deadline = Instant::now() + timeout;
        let mut bye = None;
        let mut error = None;
        if self.usable {
            match self
                .send(&Request::Shutdown)
                .and_then(|()| self.reply_until(deadline))
            {
                Ok(Reply::Bye {
                    execs,
                    peak_rss_kib,
                }) => bye = Some((execs, peak_rss_kib)),
                Ok(_) => {
                    error = Some(Error::new(
                        "SESSION_PROTOCOL",
                        "unexpected reply to SHUTDOWN",
                    ))
                }
                Err(e) => error = Some(e),
            }
        }
        self.finish_close(bye, error, deadline)
    }
    /// Test hook: close the stream without SHUTDOWN, then reap.
    pub fn close_input(self, timeout: Duration) -> CloseReceipt {
        let deadline = Instant::now() + timeout;
        self.finish_close(None, None, deadline)
    }
}

impl Drop for Session {
    /// No worker outlives its session: an unclosed one is killed and reaped.
    fn drop(&mut self) {
        self.disconnect_reader_queue();
        self.cancel_output_readers();
        if !self.reaped {
            drop(self.stdin.take());
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        #[cfg(unix)]
        {
            if let Some(reader) = self.reader.take() {
                let _ = reader.join();
            }
            if let Some(stderr) = self.stderr.take() {
                let _ = stderr.join();
            }
        }
    }
}

fn same_model(a: &moe::Model, b: &moe::Model) -> bool {
    a.hidden == b.hidden
        && a.intermediate == b.intermediate
        && a.experts.len() == b.experts.len()
        && a.experts
            .iter()
            .zip(&b.experts)
            .all(|(x, y)| x.id == y.id && x.gate == y.gate && x.up == y.up && x.down == y.down)
}

/// The rank's resident experts in its declared physical order.
pub fn resident_model(p: &moe::Problem, rank: &routing::RankDispatch) -> Result<moe::Model> {
    let resident: BTreeMap<_, _> = p
        .registry
        .copies
        .iter()
        .filter(|r| r.rank == rank.rank)
        .map(|r| (r.local, r.expert))
        .collect();
    let order: BTreeSet<_> = rank.resident_order.iter().copied().collect();
    if order.len() != rank.resident_order.len() || order != resident.keys().copied().collect() {
        return Err(invalid(
            "physical order does not cover this rank's residents",
        ));
    }
    let experts = rank
        .resident_order
        .iter()
        .map(|local| {
            let id = resident[local];
            p.model
                .experts
                .iter()
                .find(|e| e.id == id)
                .cloned()
                .ok_or_else(|| invalid("resident expert missing from model"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(moe::Model {
        hidden: p.model.hidden,
        intermediate: p.model.intermediate,
        experts,
    })
}

fn rank_dispatch(p: &moe::Problem, rank: u32) -> Result<routing::RankDispatch> {
    routing::dispatch(&routing::from_dense(p))?
        .ranks
        .into_iter()
        .find(|r| r.rank == rank)
        .ok_or_else(|| invalid("endpoint rank owns no resident experts"))
}

/// Make every session hold exactly this problem's resident experts at its epoch:
/// nothing when already bound, a weightless rebind for a newer epoch, else a transfer.
pub fn bind(sessions: &mut [Session], p: &moe::Problem, timeout: Duration) -> Result<()> {
    for session in sessions.iter_mut() {
        let d = rank_dispatch(p, session.rank)?;
        let model = resident_model(p, &d)?;
        let digest = artifact_digest(&model)?;
        match session.artifact.clone() {
            Some((loaded, epoch)) if loaded == digest && epoch == p.stamp.epoch => continue,
            Some((loaded, _)) if loaded != digest => session.evict(timeout)?,
            _ => {}
        }
        session.load(
            &model,
            &p.stamp.model_revision,
            p.stamp.layer,
            p.stamp.epoch,
            &d.resident_order,
            timeout,
        )?;
    }
    Ok(())
}

/// Connect one session per endpoint and load each rank's residents for `p`.
pub fn prepare(
    endpoints: &[Endpoint],
    p: &moe::Problem,
    timeout: Duration,
) -> Result<Vec<Session>> {
    moe::validate(p)?;
    if endpoints.is_empty()
        || endpoints.len() > 64
        || endpoints
            .iter()
            .map(|e| e.rank)
            .collect::<BTreeSet<_>>()
            .len()
            != endpoints.len()
    {
        return Err(invalid("session endpoints must name 1..64 distinct ranks"));
    }
    let mut sessions = Vec::new();
    for endpoint in endpoints {
        sessions.push(Session::connect(endpoint, timeout)?);
    }
    bind(&mut sessions, p, timeout)?;
    Ok(sessions)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchReport {
    pub ok: bool,
    pub output: Option<Vec<Vec<f64>>>,
    pub attempt: u64,
    pub receipts: Vec<ExecReceipt>,
    pub error: Option<Error>,
}

/// One batch over bound sessions: every participating rank receives its EXEC before any
/// result is read, so ranks compute concurrently. All results or none; never retried.
pub fn run_batch(
    sessions: &mut [Session],
    p: &moe::Problem,
    faults: &BTreeMap<u32, Fault>,
    recovery: bool,
    timeout: Duration,
) -> Result<BatchReport> {
    moe::validate(p)?;
    let ranks: Vec<_> = routing::dispatch(&routing::from_dense(p))?
        .ranks
        .into_iter()
        .filter(|r| !r.slots.is_empty())
        .collect();
    if faults
        .keys()
        .any(|rank| !ranks.iter().any(|r| r.rank == *rank))
    {
        return Err(invalid("fault refers to a nonparticipating rank"));
    }
    // Fault bounds too are checked before any identity is claimed or work is sent,
    // not left to the faulted rank's worker once the other ranks have computed.
    for fault in faults.values() {
        check_fault(fault)?;
    }
    let mut plan = Vec::new();
    for rank in ranks {
        let index = sessions
            .iter()
            .position(|s| s.rank == rank.rank)
            .ok_or_else(|| invalid("no session serves a participating rank"))?;
        let session = &sessions[index];
        if !session.usable {
            return Err(Error::new(
                "SESSION_UNUSABLE",
                "session must be reopened after a failure",
            ));
        }
        // The validated rank payload of the single-request path; its weights stay here.
        // Compared value by value with what was loaded: re-hashing the weights of every
        // rank on every batch cost about 7 ms per batch on the A52 fixture.
        let request = rank_request(p, &rank, 1, Fault::None)?;
        let bound = session
            .artifact
            .as_ref()
            .is_some_and(|(_, epoch)| *epoch == p.stamp.epoch)
            && session
                .resident
                .as_ref()
                .is_some_and(|loaded| same_model(loaded, &request.model));
        if !bound {
            return Err(Error::new(
                "STALE_EXECUTION",
                "session does not hold this batch's resident artifact and epoch; bind first",
            ));
        }
        let fault = faults.get(&rank.rank).cloned().unwrap_or_default();
        plan.push((index, rank, RankInputs::F64(request.inputs), fault));
    }
    let collector = moe::Collector::new(p)?;
    exec_all(
        sessions, plan, &p.stamp, collector, recovery, timeout, false,
    )
}

#[derive(Debug, Clone)]
struct Fp16TokenInput {
    token: usize,
    value_bits: Vec<u16>,
}

#[derive(Debug, Clone)]
enum RankInputs {
    F64(Vec<TokenInput>),
    Fp16(Vec<Fp16TokenInput>),
}

impl From<Vec<TokenInput>> for RankInputs {
    fn from(inputs: Vec<TokenInput>) -> Self {
        Self::F64(inputs)
    }
}

/// One planned rank EXEC: session index, rank dispatch, compacted inputs, fault.
type RankPlan = (usize, routing::RankDispatch, RankInputs, Fault);

struct ExecutedBatch<T> {
    output: Option<T>,
    attempt: u64,
    receipts: Vec<ExecReceipt>,
    error: Option<Error>,
}

/// Send every planned EXEC before reading any result, so ranks compute concurrently, then
/// collect. All results or none; never retried. Every EXEC is encoded before the identity
/// is claimed or any is sent: a local refusal (a frame over 16 MiB) refuses the batch
/// whole, not one rank once the others were sent their work.
fn exec_all(
    sessions: &mut [Session],
    plan: Vec<RankPlan>,
    stamp: &moe::Stamp,
    collector: moe::Collector,
    recovery: bool,
    timeout: Duration,
    completion_fp16: bool,
) -> Result<BatchReport> {
    let result = exec_all_with_finish(
        sessions,
        plan,
        stamp,
        collector,
        recovery,
        timeout,
        completion_fp16,
        moe::Collector::finish,
    )?;
    Ok(BatchReport {
        ok: result.error.is_none(),
        output: result.output,
        attempt: result.attempt,
        receipts: result.receipts,
        error: result.error,
    })
}

#[allow(clippy::too_many_arguments)]
fn exec_all_with_finish<T>(
    sessions: &mut [Session],
    plan: Vec<RankPlan>,
    stamp: &moe::Stamp,
    mut collector: moe::Collector,
    recovery: bool,
    timeout: Duration,
    completion_fp16: bool,
    finish: fn(moe::Collector) -> Result<T>,
) -> Result<ExecutedBatch<T>> {
    let attempt = next_attempt()?;
    let mut encoded = Vec::new();
    for (index, rank, inputs, fault) in plan {
        encoded.push((
            index,
            sessions[index].encode_exec(&rank, attempt, inputs, fault, completion_fp16)?,
        ));
    }
    claim_identity(stamp, recovery)?;
    let deadline = Instant::now() + timeout;
    let pending: Vec<_> = encoded
        .into_iter()
        .map(|(index, exec)| (index, sessions[index].send_exec(exec)))
        .collect();
    let mut error = None;
    let mut receipts = Vec::new();
    for (index, exec) in pending {
        let target = if error.is_none() {
            Some(&mut collector)
        } else {
            None
        };
        let receipt = sessions[index].finish_exec(exec, target, deadline);
        if error.is_none() {
            error.clone_from(&receipt.error);
        }
        receipts.push(receipt);
    }
    let output = if error.is_none() {
        match finish(collector) {
            Ok(output) => Some(output),
            Err(e) => {
                error = Some(e);
                None
            }
        }
    } else {
        collector.expire();
        None
    };
    Ok(ExecutedBatch {
        output,
        attempt,
        receipts,
        error,
    })
}

// ------------------------------------------------------------------- series runs

/// A run of `batches` microbatches derived from one base problem over persistent
/// sessions, each checked against the dense oracle. The caller of `vf session-run`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeriesInput {
    pub problem: moe::Problem,
    pub workers: Vec<Endpoint>,
    pub batches: u64,
    pub timeout_ms: u64,
    /// Renew the ownership epoch every this many batches (0: never).
    #[serde(default)]
    pub epoch_every: u64,
    /// Leading batches left out of the latency statistics.
    #[serde(default)]
    pub warmup: u64,
    /// Largest accepted |output - oracle| over every value of every batch.
    #[serde(default)]
    pub tolerance_abs: f64,
    /// Faults injected into given batches (negative tests over real transports).
    #[serde(default)]
    pub faults: Vec<SeriesFault>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeriesFault {
    pub batch: u64,
    pub rank: u32,
    pub fault: Fault,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub n: usize,
    pub min_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub max_us: f64,
    pub mean_us: f64,
}
impl LatencyStats {
    pub fn of(samples: &[f64]) -> Option<LatencyStats> {
        if samples.is_empty() {
            return None;
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        // Nearest-rank percentiles.
        let rank =
            |q: f64| sorted[((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len()) - 1];
        Some(LatencyStats {
            n: sorted.len(),
            min_us: sorted[0],
            p50_us: rank(0.50),
            p95_us: rank(0.95),
            max_us: sorted[sorted.len() - 1],
            mean_us: sorted.iter().sum::<f64>() / sorted.len() as f64,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankTraffic {
    pub rank: u32,
    pub traffic: Traffic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchFailure {
    pub batch: u64,
    pub error: Option<Error>,
    pub receipts: Vec<ExecReceipt>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeriesReport {
    pub ok: bool,
    pub protocol: u32,
    pub hellos: Vec<ServerHello>,
    pub batches_requested: u64,
    pub batches_ok: u64,
    pub exact_batches: u64,
    pub max_abs_diff: f64,
    pub tolerance_abs: f64,
    pub setup_ms: f64,
    pub batch_latency: Option<LatencyStats>,
    pub rebind_latency: Option<LatencyStats>,
    pub exec_request_bytes: u64,
    pub exec_requests: u64,
    /// Per-rank application bytes as snapshotted before the sessions are closed: the
    /// SHUTDOWN request and BYE reply are not counted. Not SSH or NIC wire bytes.
    pub traffic: Vec<RankTraffic>,
    pub failure: Option<BatchFailure>,
    pub close: Vec<CloseReceipt>,
    pub error: Option<Error>,
    /// Outputs of the batches the input asked to record (compressed series only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recorded: Vec<RecordedBatch>,
}

/// Batch `b` of a series: its own batch number, deterministic new activations, and the
/// ownership epoch renewed every `epoch_every` batches. Weights and routes are kept.
pub fn series_batch(base: &moe::Problem, b: u64, epoch_every: u64) -> moe::Problem {
    let mut p = base.clone();
    p.stamp = series_stamp(&base.stamp, b, epoch_every);
    if epoch_every > 0 {
        p.registry.epoch = p.stamp.epoch;
    }
    scale_inputs(&mut p.batch.inputs, b);
    p
}

/// The identity of batch `b`: numbered `batch + b`, at epoch `epoch + b / epoch_every`.
fn series_stamp(base: &moe::Stamp, b: u64, epoch_every: u64) -> moe::Stamp {
    let mut stamp = base.clone();
    stamp.batch = base.batch.saturating_add(b);
    if epoch_every > 0 {
        stamp.epoch = base.epoch.saturating_add(b / epoch_every);
    }
    stamp
}

/// The deterministic activations of batch `b`, derived from the base rows.
fn scale_inputs(rows: &mut [Vec<f64>], b: u64) {
    for (t, row) in rows.iter_mut().enumerate() {
        for (i, x) in row.iter_mut().enumerate() {
            *x = scaled(*x, b, t, i);
        }
    }
}

/// Base value `x` of token `t`, column `i`, as batch `b` sees it: a factor 1 + s/64 with
/// s = (7b + 3t + i) mod 11, so the factors of one value repeat every 11 batches.
fn scaled(x: f64, b: u64, t: usize, i: usize) -> f64 {
    let step = (b as usize).wrapping_mul(7).wrapping_add(t * 3 + i) % 11;
    x * (1.0 + step as f64 / 64.0)
}

fn micros(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

/// What a series needs from its kind of resident: the batches, binding and the oracle.
trait Series {
    type Batch;
    fn batch(&self, b: u64) -> Self::Batch;
    fn epoch(batch: &Self::Batch) -> u64;
    fn bind(&self, sessions: &mut [Session], batch: &Self::Batch, timeout: Duration) -> Result<()>;
    fn run(
        &self,
        sessions: &mut [Session],
        batch: &Self::Batch,
        faults: &BTreeMap<u32, Fault>,
        timeout: Duration,
    ) -> Result<BatchReport>;
    fn reference(&self, batch: &Self::Batch) -> Result<Vec<Vec<f64>>>;
}

/// The bounds every series obeys, dense or compressed.
struct SeriesParams<'a> {
    batches: u64,
    timeout_ms: u64,
    warmup: u64,
    tolerance_abs: f64,
    faults: &'a [SeriesFault],
    record: &'a [u64],
}
impl SeriesParams<'_> {
    fn check(&self) -> Result<Duration> {
        if self.batches == 0
            || self.batches > 100_000
            || self.timeout_ms == 0
            || self.timeout_ms > 60_000
            || self.warmup >= self.batches
            || !self.tolerance_abs.is_finite()
            || self.tolerance_abs < 0.0
            || self.faults.len() > 64
        {
            return Err(invalid(
                "series needs 1..100000 batches, a 1..60000 ms timeout, warmup below batches and a finite tolerance",
            ));
        }
        if self.record.len() > 16 || self.record.iter().any(|&b| b >= self.batches) {
            return Err(invalid("a series records at most 16 of its own batches"));
        }
        // A fault past the last batch would never be injected, and a second one for the
        // same batch and rank would silently replace the first.
        let mut scheduled = BTreeSet::new();
        if self
            .faults
            .iter()
            .any(|f| f.batch >= self.batches || !scheduled.insert((f.batch, f.rank)))
        {
            return Err(invalid(
                "a series fault names one of its own batches, at most once per rank",
            ));
        }
        // Each batch claims one coordinator identity: a series the guard cannot finish
        // is refused here, before any worker starts, not at the batch that hits the limit.
        if self.batches > attempts_left()? as u64 {
            return Err(invalid(
                "series has more batches than identities this coordinator can still attempt (coordinator attempt limit)",
            ));
        }
        Ok(Duration::from_millis(self.timeout_ms))
    }
}

/// A series' faults must target ranks with slots in its routing, which every batch
/// shares, and stay within the bounded fault contract: refused before any worker, not
/// at the faulted batch once the earlier batches ran.
fn check_series_faults(p: &routing::RoutingProblem, faults: &[SeriesFault]) -> Result<()> {
    let participating: BTreeSet<u32> = routing::dispatch(p)?
        .ranks
        .into_iter()
        .filter(|r| !r.slots.is_empty())
        .map(|r| r.rank)
        .collect();
    for f in faults {
        if !participating.contains(&f.rank) {
            return Err(invalid("fault refers to a nonparticipating rank"));
        }
        check_fault(&f.fault)?;
    }
    Ok(())
}

/// The dense series' endpoints must serve every rank its routing sends work to and name
/// only ranks that hold residents, as the compressed series requires one endpoint per
/// artifact rank: refused before any weight is sent, not at batch 0.
fn check_series_endpoints(p: &routing::RoutingProblem, endpoints: &[Endpoint]) -> Result<()> {
    let ranks = routing::dispatch(p)?.ranks;
    let served: BTreeSet<u32> = endpoints.iter().map(|e| e.rank).collect();
    if ranks
        .iter()
        .any(|r| !r.slots.is_empty() && !served.contains(&r.rank))
        || served
            .iter()
            .any(|rank| !ranks.iter().any(|r| r.rank == *rank))
    {
        return Err(invalid(
            "series endpoints must serve every rank the routing sends work to, and only resident ranks",
        ));
    }
    Ok(())
}

/// Each rank's first LOAD carries its resident model as JSON in one frame: a rank whose
/// frame would exceed the bound is refused for the whole dense series here, before any
/// rank is sent its weights. The request is the one `bind` sends, encoded and dropped.
fn check_series_loads(p: &moe::Problem, endpoints: &[Endpoint]) -> Result<()> {
    for endpoint in endpoints {
        let d = rank_dispatch(p, endpoint.rank)?;
        let model = resident_model(p, &d)?;
        encode(&Request::Load {
            artifact_sha256: artifact_digest(&model)?,
            model_revision: p.stamp.model_revision.clone(),
            layer: p.stamp.layer,
            epoch: p.stamp.epoch,
            resident_order: d.resident_order,
            model: Some(model),
        })?;
    }
    Ok(())
}

/// Each batch sends every rank it routes work to one EXEC frame with that rank's token
/// activations, which the encoder refuses past 16 MiB: a dense rank whose frame was too
/// large failed only at that batch, once every rank was loaded. The frames are encoded
/// here instead, before any worker. Activations repeat every 11 batches and batch and
/// epoch numbers only widen, so the last batch of each phase stands for its phase; each
/// faulted batch is also encoded on its rank with its fault. The attempt, assigned at run
/// time, is encoded at its widest (`u64::MAX`), and the digest has its fixed 64 hex digits.
/// A compressed rank's EXEC is bounded far below 16 MiB by its work budget.
fn check_series_execs(input: &SeriesInput) -> Result<()> {
    let base = &input.problem;
    let last = input.batches.saturating_sub(1);
    let phases = (0..input.batches.min(11)).map(|phase| (phase + (last - phase) / 11 * 11, None));
    let faulted = input.faults.iter().map(|f| (f.batch, Some(f)));
    let ranks: Vec<_> = routing::dispatch(&routing::from_dense(base))?
        .ranks
        .into_iter()
        .filter(|r| !r.slots.is_empty())
        .collect();
    for (b, fault) in phases.chain(faulted) {
        for rank in ranks
            .iter()
            .filter(|r| fault.is_none_or(|f: &SeriesFault| f.rank == r.rank))
        {
            let mut dispatch = rank.clone();
            dispatch.stamp = series_stamp(&base.stamp, b, input.epoch_every);
            let tokens: BTreeSet<_> = rank.slots.iter().map(|s| s.slot.token).collect();
            let inputs = tokens
                .into_iter()
                .map(|token| TokenInput {
                    token,
                    value: base.batch.inputs[token]
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| scaled(x, b, token, i))
                        .collect(),
                })
                .collect();
            encode(&Request::Exec {
                artifact_sha256: "0".repeat(64),
                dispatch,
                attempt: u64::MAX,
                inputs,
                fault: fault.map_or(Fault::None, |f| f.fault.clone()),
            })?;
        }
    }
    Ok(())
}

/// A series verifies computed batches against its oracle: a routing with no token would
/// run every batch with no work and report it exact.
fn check_work_present(routes: &[Vec<moe::Choice>]) -> Result<()> {
    if routes.is_empty() {
        return Err(invalid("a series needs at least one routed token"));
    }
    Ok(())
}

/// Each batch derives its activations from the finite base rows (`scale_inputs`). A base
/// value one of its factors takes past `f64::MAX` is refused here, before any worker, not
/// as a nonfinite input at the batch that derives it. The factors repeat every 11 batches
/// (`batches` is at most 100 000, so `7b` never wraps): the first 11 cover them all.
fn check_series_rows(rows: &[Vec<f64>], batches: u64) -> Result<()> {
    for b in 0..batches.min(11) {
        for (t, row) in rows.iter().enumerate() {
            if row
                .iter()
                .enumerate()
                .any(|(i, &x)| !scaled(x, b, t, i).is_finite())
            {
                return Err(invalid(
                    "series activations must stay finite in every derived batch",
                ));
            }
        }
    }
    Ok(())
}

/// Batch `b` of a series is numbered `stamp.batch + b` and, with `epoch_every > 0`, runs
/// at epoch `stamp.epoch + b / epoch_every`. Neither may overflow: two batches would
/// share one identity, or epoch renewals would silently stop at `u64::MAX`.
fn check_numbering(stamp: &moe::Stamp, batches: u64, epoch_every: u64) -> Result<()> {
    let last = batches.saturating_sub(1);
    let renewals = last.checked_div(epoch_every).unwrap_or(0);
    if stamp.batch.checked_add(last).is_none() || stamp.epoch.checked_add(renewals).is_none() {
        return Err(invalid("series batch or epoch numbers would overflow u64"));
    }
    Ok(())
}

/// One batch's output kept for an external oracle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedBatch {
    pub batch: u64,
    pub output: Vec<Vec<f64>>,
}

/// Drive bound sessions through the series; the first failed batch ends it (no retry),
/// and every session is closed.
fn drive<S: Series>(
    series: &S,
    params: &SeriesParams<'_>,
    mut sessions: Vec<Session>,
    setup_ms: f64,
    first_epoch: u64,
    timeout: Duration,
) -> Result<SeriesReport> {
    let mut latencies = Vec::new();
    let mut rebinds = Vec::new();
    let mut batches_ok = 0;
    let mut exact_batches = 0;
    let mut max_abs_diff = 0.0_f64;
    let mut exec_request_bytes = 0;
    let mut exec_requests = 0;
    let mut failure = None;
    let mut error = None;
    let mut recorded = Vec::new();
    let mut epoch = first_epoch;
    for b in 0..params.batches {
        let batch = series.batch(b);
        if S::epoch(&batch) != epoch {
            let t = Instant::now();
            if let Err(e) = series.bind(&mut sessions, &batch, timeout) {
                failure = Some(BatchFailure {
                    batch: b,
                    error: Some(e),
                    receipts: vec![],
                });
                break;
            }
            rebinds.push(micros(t.elapsed()));
            epoch = S::epoch(&batch);
        }
        let faults: BTreeMap<u32, Fault> = params
            .faults
            .iter()
            .filter(|f| f.batch == b)
            .map(|f| (f.rank, f.fault.clone()))
            .collect();
        let t = Instant::now();
        let report = match series.run(&mut sessions, &batch, &faults, timeout) {
            Ok(report) => report,
            Err(e) => {
                failure = Some(BatchFailure {
                    batch: b,
                    error: Some(e),
                    receipts: vec![],
                });
                break;
            }
        };
        let elapsed = micros(t.elapsed());
        exec_requests += report.receipts.len() as u64;
        exec_request_bytes += report.receipts.iter().map(|r| r.request_bytes).sum::<u64>();
        let Some(output) = report.output.filter(|_| report.ok) else {
            failure = Some(BatchFailure {
                batch: b,
                error: report.error,
                receipts: report.receipts,
            });
            break;
        };
        let reference = series.reference(&batch)?;
        let diff = output
            .iter()
            .flatten()
            .zip(reference.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        if diff == 0.0 {
            exact_batches += 1;
        }
        max_abs_diff = max_abs_diff.max(diff);
        if b >= params.warmup {
            latencies.push(elapsed);
        }
        if params.record.contains(&b) {
            recorded.push(RecordedBatch { batch: b, output });
        }
        batches_ok += 1;
    }
    let traffic = sessions
        .iter()
        .map(|s| RankTraffic {
            rank: s.rank,
            traffic: s.traffic.clone(),
        })
        .collect();
    let hellos = sessions.iter().map(|s| s.hello.clone()).collect();
    let close: Vec<_> = sessions.into_iter().map(|s| s.close(timeout)).collect();
    if failure.is_none() && max_abs_diff > params.tolerance_abs {
        error = Some(Error::new(
            "TOLERANCE_EXCEEDED",
            "a session output differs from the dense oracle beyond tolerance_abs",
        ));
    }
    if failure.is_none()
        && error.is_none()
        && close
            .iter()
            .any(|c| c.exit_code != Some(0) || c.bye_execs.is_none() || c.error.is_some())
    {
        error = Some(Error::new(
            "SESSION_CLOSE",
            "a session worker did not confirm SHUTDOWN and exit cleanly",
        ));
    }
    Ok(SeriesReport {
        ok: failure.is_none() && error.is_none(),
        protocol: PROTOCOL,
        hellos,
        batches_requested: params.batches,
        batches_ok,
        exact_batches,
        max_abs_diff,
        tolerance_abs: params.tolerance_abs,
        setup_ms,
        batch_latency: LatencyStats::of(&latencies),
        rebind_latency: LatencyStats::of(&rebinds),
        exec_request_bytes,
        exec_requests,
        traffic,
        failure,
        close,
        error,
        recorded,
    })
}

struct DenseSeries<'a> {
    input: &'a SeriesInput,
}
impl Series for DenseSeries<'_> {
    type Batch = moe::Problem;
    fn batch(&self, b: u64) -> moe::Problem {
        series_batch(&self.input.problem, b, self.input.epoch_every)
    }
    fn epoch(p: &moe::Problem) -> u64 {
        p.stamp.epoch
    }
    fn bind(&self, sessions: &mut [Session], p: &moe::Problem, timeout: Duration) -> Result<()> {
        bind(sessions, p, timeout)
    }
    fn run(
        &self,
        sessions: &mut [Session],
        p: &moe::Problem,
        faults: &BTreeMap<u32, Fault>,
        timeout: Duration,
    ) -> Result<BatchReport> {
        run_batch(sessions, p, faults, false, timeout)
    }
    fn reference(&self, p: &moe::Problem) -> Result<Vec<Vec<f64>>> {
        moe::reference(p)
    }
}

/// Run a series; the first failed batch ends it (no retry), and every session is closed.
pub fn run_series(input: &SeriesInput) -> Result<SeriesReport> {
    let params = SeriesParams {
        batches: input.batches,
        timeout_ms: input.timeout_ms,
        warmup: input.warmup,
        tolerance_abs: input.tolerance_abs,
        faults: &input.faults,
        record: &[],
    };
    let timeout = params.check()?;
    check_numbering(&input.problem.stamp, input.batches, input.epoch_every)?;
    check_work_present(&input.problem.batch.routes)?;
    moe::validate(&input.problem)?;
    check_series_rows(&input.problem.batch.inputs, input.batches)?;
    let routing = routing::from_dense(&input.problem);
    check_series_faults(&routing, &input.faults)?;
    check_series_endpoints(&routing, &input.workers)?;
    check_series_loads(&input.problem, &input.workers)?;
    check_series_execs(input)?;
    let started = Instant::now();
    let sessions = prepare(&input.workers, &input.problem, timeout)?;
    let setup_ms = started.elapsed().as_secs_f64() * 1e3;
    drive(
        &DenseSeries { input },
        &params,
        sessions,
        setup_ms,
        input.problem.stamp.epoch,
        timeout,
    )
}

// ------------------------------------------------- compressed residents (A53)

/// One rank's compressed resident as the coordinator holds it: the exact artifact bytes
/// it sends once, their SHA-256, and the validated stack they decode to. Built only by
/// [`CompressedArtifact::from_bytes`], so the three always agree: the coordinator checks
/// the stack (identity, replicas, oracle) while the worker computes with the bytes.
/// The caller may explicitly discard the transfer bytes once loading succeeds;
/// the retained digest and validated stack are then still used for batch checks.
///
/// ```
/// use voltafabric::{resident::ResidentStack, session::CompressedArtifact};
/// fn parts(a: &CompressedArtifact) -> (&str, &[u8], &ResidentStack) {
///     (a.sha256(), a.bytes(), a.stack())
/// }
/// ```
///
/// No field can be reached directly, so the struct cannot be assembled from parts that
/// disagree. Each block below must fail to compile:
///
/// ```compile_fail
/// fn field(a: &voltafabric::session::CompressedArtifact) -> &String {
///     &a.sha256
/// }
/// ```
///
/// ```compile_fail
/// fn field(a: &voltafabric::session::CompressedArtifact) -> &Vec<u8> {
///     &a.bytes
/// }
/// ```
///
/// ```compile_fail
/// fn field(a: &voltafabric::session::CompressedArtifact) -> &voltafabric::resident::ResidentStack {
///     &a.stack
/// }
/// ```
#[derive(Debug, Clone)]
pub struct CompressedArtifact {
    sha256: String,
    bytes: Vec<u8>,
    stack: ResidentStack,
}
impl CompressedArtifact {
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Irreversibly discard the serialized transfer copy after all owners have
    /// acknowledged it. The validated stack and digest remain available for
    /// routing and batch checks. A fresh transfer needs a fresh artifact; this
    /// operation never reconstructs bytes or replays an unknown request.
    pub fn discard_transfer_bytes(&mut self) {
        self.bytes = Vec::new();
    }
    pub fn stack(&self) -> &ResidentStack {
        &self.stack
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<CompressedArtifact> {
        let stack = resident_artifact::import(&bytes)?;
        if stack.is_empty() {
            return Err(invalid("a session artifact holds at least one expert"));
        }
        Ok(CompressedArtifact {
            sha256: sha256::hex(&bytes),
            bytes,
            stack,
        })
    }
}

/// Each artifact must be exactly what the routing problem says its rank holds, and every
/// rank that holds experts needs its artifact. Returns the artifacts by rank.
fn by_rank(
    p: &routing::RoutingProblem,
    artifacts: &[CompressedArtifact],
) -> Result<BTreeMap<u32, usize>> {
    routing::validate(p)?;
    let mut ranks = BTreeMap::new();
    for (index, a) in artifacts.iter().enumerate() {
        let stack = &a.stack;
        let replicas: BTreeMap<u32, u32> = p
            .registry
            .copies
            .iter()
            .filter(|r| r.rank == stack.rank())
            .map(|r| (r.local, r.expert))
            .collect();
        let held: BTreeMap<u32, u32> = stack
            .physical_experts()
            .iter()
            .map(|e| (e.local(), e.id()))
            .collect();
        if ranks.insert(stack.rank(), index).is_some()
            || p.resident_order.get(&stack.rank()).map(Vec::as_slice)
                != Some(stack.resident_order())
            || replicas != held
            || stack.model_revision() != p.stamp.model_revision
            || stack.layer() != p.stamp.layer
            || stack.epoch() > p.stamp.epoch
            || stack.shape().2 != p.hidden as u64
        {
            return Err(invalid(
                "artifact differs from its rank's registry, order, identity, epoch or hidden size",
            ));
        }
    }
    if p.resident_order
        .iter()
        .any(|(rank, order)| !order.is_empty() && !ranks.contains_key(rank))
    {
        return Err(invalid("a rank holding experts has no artifact"));
    }
    // Every rank carries its own copy of a replicated expert: the copies must be one set
    // of weights, or the output would depend on which rank the routing picks.
    let mut first: BTreeMap<u32, (usize, usize)> = BTreeMap::new();
    for (index, a) in artifacts.iter().enumerate() {
        for (physical, e) in a.stack.physical_experts().iter().enumerate() {
            let &mut (i, p) = first.entry(e.id()).or_insert((index, physical));
            if !artifacts[i].stack.same_expert(p, &a.stack, physical) {
                return Err(invalid("replicas of one expert differ between artifacts"));
            }
        }
    }
    Ok(ranks)
}

/// Make every session hold its rank's artifact at the problem's epoch: nothing when
/// already bound, a weightless rebind for a newer epoch, else the one transfer.
pub fn bind_compressed(
    sessions: &mut [Session],
    artifacts: &[CompressedArtifact],
    p: &routing::RoutingProblem,
    timeout: Duration,
) -> Result<()> {
    let ranks = by_rank(p, artifacts)?;
    // A mixed fleet is refused whole: no rank is sent its artifact before every worker
    // has advertised that it can load one.
    if sessions
        .iter()
        .any(|s| !s.hello.capabilities.iter().any(|c| c == CAPABILITY_QPN2))
    {
        return Err(Error::new(
            "CAPABILITY_MISSING",
            "worker does not advertise cpu_qpn2_resident_v1; nothing was sent",
        ));
    }
    // Preflight every rank before eviction or LOAD so a missing transfer copy
    // cannot leave an otherwise valid fleet partially rebound.
    for session in sessions.iter() {
        let artifact = ranks
            .get(&session.rank)
            .map(|&i| &artifacts[i])
            .ok_or_else(|| invalid("endpoint rank has no artifact"))?;
        let already_loaded = session
            .artifact
            .as_ref()
            .is_some_and(|(digest, _)| *digest == artifact.sha256);
        if !already_loaded && artifact.bytes.is_empty() {
            return Err(Error::new(
                "ARTIFACT_BYTES_RELEASED",
                "artifact transfer bytes were discarded; a fresh artifact is required",
            ));
        }
    }
    for session in sessions.iter_mut() {
        let artifact = ranks
            .get(&session.rank)
            .map(|&i| &artifacts[i])
            .ok_or_else(|| invalid("endpoint rank has no artifact"))?;
        match session.artifact.clone() {
            Some((loaded, epoch)) if loaded == artifact.sha256 && epoch == p.stamp.epoch => {
                continue
            }
            Some((loaded, _)) if loaded != artifact.sha256 => session.evict(timeout)?,
            _ => {}
        }
        session.load_compressed(artifact, p.stamp.epoch, timeout)?;
    }
    Ok(())
}

/// Connect one session per endpoint and send each rank its compressed artifact.
pub fn prepare_compressed(
    endpoints: &[Endpoint],
    artifacts: &[CompressedArtifact],
    p: &routing::RoutingProblem,
    timeout: Duration,
) -> Result<Vec<Session>> {
    let ranks = by_rank(p, artifacts)?;
    if endpoints.len() != ranks.len()
        || endpoints.iter().map(|e| e.rank).collect::<BTreeSet<_>>()
            != ranks.keys().copied().collect()
    {
        return Err(invalid(
            "compressed sessions need exactly one endpoint per artifact rank",
        ));
    }
    let mut sessions = Vec::new();
    for endpoint in endpoints {
        sessions.push(Session::connect(endpoint, timeout)?);
    }
    bind_compressed(&mut sessions, artifacts, p, timeout)?;
    Ok(sessions)
}

/// Prepare a fresh multipart-only fleet. All rank/artifact identities and capabilities
/// are checked before the first byte of weights is sent. Any error invalidates every
/// session and the caller must reconcile the attempt, never replay it.
pub fn prepare_compressed_multipart(
    endpoints: &[Endpoint],
    artifacts: &[CompressedArtifact],
    p: &routing::RoutingProblem,
    attempt: u64,
    timeout: Duration,
) -> Result<Vec<Session>> {
    let ranks = by_rank(p, artifacts)?;
    if attempt == 0
        || timeout.is_zero()
        || endpoints.len() != ranks.len()
        || endpoints.iter().map(|e| e.rank).collect::<BTreeSet<_>>()
            != ranks.keys().copied().collect()
        || artifacts
            .iter()
            .any(|a| a.stack.epoch() != p.stamp.epoch || a.bytes.is_empty())
    {
        return Err(invalid(
            "multipart fleet identity, coverage or bounds mismatch",
        ));
    }
    let mut sessions = Vec::new();
    for endpoint in endpoints {
        sessions.push(Session::connect(endpoint, timeout)?);
    }
    if sessions.iter().any(|s| {
        !s.hello
            .capabilities
            .iter()
            .any(|c| c == CAPABILITY_QPN2_MULTIPART)
    }) {
        return Err(Error::new(
            "CAPABILITY_MISSING",
            "multipart fleet preflight failed; nothing sent",
        ));
    }
    for index in 0..sessions.len() {
        let artifact = &artifacts[*ranks
            .get(&sessions[index].rank)
            .ok_or_else(|| invalid("missing rank"))?];
        if let Err(error) =
            sessions[index].load_compressed_multipart(artifact, &p.stamp, attempt, timeout)
        {
            for s in &mut sessions {
                s.usable = false;
            }
            return Err(error);
        }
    }
    Ok(sessions)
}

/// Every participating rank's slots within the work budget its worker enforces.
fn check_work(
    p: &routing::RoutingProblem,
    artifacts: &[CompressedArtifact],
    ranks_of: &BTreeMap<u32, usize>,
) -> Result<()> {
    let over = routing::dispatch(p)?.ranks.iter().any(|rank| {
        ranks_of
            .get(&rank.rank)
            .is_some_and(|&i| !within_work_budget(&artifacts[i].stack, rank.slots.len()))
    });
    if over {
        return Err(over_budget());
    }
    Ok(())
}

/// One finite activation row of the routing's hidden size per routed token.
fn check_rows(p: &routing::RoutingProblem, inputs: &[Vec<f64>]) -> Result<()> {
    if inputs.len() != p.routes.len()
        || inputs
            .iter()
            .any(|row| row.len() != p.hidden || row.iter().any(|x| !x.is_finite()))
    {
        return Err(invalid(
            "one finite activation row of the hidden size per routed token",
        ));
    }
    Ok(())
}

fn check_fp16_rows(p: &routing::RoutingProblem, inputs: &[Vec<u16>]) -> Result<()> {
    if inputs.len() != p.routes.len() || inputs.iter().any(|row| row.len() != p.hidden) {
        return Err(invalid(
            "one binary16 activation row of the hidden size per routed token",
        ));
    }
    for &bits in inputs.iter().flatten() {
        crate::completion::decode_fp16(bits)
            .map_err(|_| invalid("FP16 activation input must contain only finite values"))?;
    }
    Ok(())
}

/// One batch over sessions bound to compressed artifacts. Only routing metadata and
/// activations leave the coordinator; the collector applies each coefficient once.
pub fn run_compressed_batch(
    sessions: &mut [Session],
    artifacts: &[CompressedArtifact],
    p: &routing::RoutingProblem,
    inputs: &[Vec<f64>],
    faults: &BTreeMap<u32, Fault>,
    recovery: bool,
    timeout: Duration,
) -> Result<BatchReport> {
    run_compressed_batch_with_format(
        sessions,
        artifacts,
        p,
        BatchInputs::F64(inputs),
        faults,
        timeout,
        BatchExecutionOptions {
            recovery,
            completion_fp16: false,
        },
    )
}

/// Complete unweighted expert rows, ordered by token and routing slot. This
/// uses the same admission, identity claim and rank draining as normal batches.
/// The consumer must apply the routing coefficients once; no recovery is attempted.
#[derive(Debug, Clone)]
pub struct UnweightedBatchReport {
    pub ok: bool,
    pub output: Option<Vec<Vec<Vec<f64>>>>,
    pub attempt: u64,
    pub receipts: Vec<ExecReceipt>,
    pub error: Option<Error>,
}

pub fn run_compressed_batch_unweighted(
    sessions: &mut [Session],
    artifacts: &[CompressedArtifact],
    p: &routing::RoutingProblem,
    inputs: &[Vec<f64>],
    faults: &BTreeMap<u32, Fault>,
    timeout: Duration,
) -> Result<UnweightedBatchReport> {
    let ranks_of = by_rank(p, artifacts)?;
    check_work(p, artifacts, &ranks_of)?;
    let digests = ranks_of
        .iter()
        .map(|(&rank, &index)| (rank, artifacts[index].sha256.clone()))
        .collect();
    let result = run_prevalidated_batch_with_finish(
        sessions,
        &digests,
        p,
        BatchInputs::F64(inputs),
        faults,
        timeout,
        BatchExecutionOptions {
            recovery: false,
            completion_fp16: false,
        },
        moe::Collector::finish_unweighted_slots,
    )?;
    Ok(UnweightedBatchReport {
        ok: result.error.is_none(),
        output: result.output,
        attempt: result.attempt,
        receipts: result.receipts,
        error: result.error,
    })
}

/// One compressed-resident batch using the explicitly negotiated FP16 response format.
/// The worker rounds each unweighted completion to finite IEEE binary16 before transport;
/// the collector applies the unchanged route coefficients after decode.
pub fn run_compressed_batch_fp16(
    sessions: &mut [Session],
    artifacts: &[CompressedArtifact],
    p: &routing::RoutingProblem,
    inputs: &[Vec<f64>],
    faults: &BTreeMap<u32, Fault>,
    recovery: bool,
    timeout: Duration,
) -> Result<BatchReport> {
    run_compressed_batch_with_format(
        sessions,
        artifacts,
        p,
        BatchInputs::F64(inputs),
        faults,
        timeout,
        BatchExecutionOptions {
            recovery,
            completion_fp16: true,
        },
    )
}

/// The response precision requested after an explicitly typed FP16 activation input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionFormat {
    F64,
    Fp16,
}

/// One compressed-resident batch using caller-owned finite IEEE binary16 activation bits.
/// Input rows are never converted from f64 here; each participating rank must advertise
/// `cpu_exec_fp16_raw_v1`. The worker decodes each finite half exactly before CPU execution.
pub fn run_compressed_batch_with_fp16_inputs(
    sessions: &mut [Session],
    artifacts: &[CompressedArtifact],
    p: &routing::RoutingProblem,
    inputs_fp16: &[Vec<u16>],
    faults: &BTreeMap<u32, Fault>,
    completion_format: CompletionFormat,
    timeout: Duration,
) -> Result<BatchReport> {
    // Keep the candidate protocol path fail-closed and never replay an ambiguous EXEC.
    run_compressed_batch_with_format(
        sessions,
        artifacts,
        p,
        BatchInputs::Fp16(inputs_fp16),
        faults,
        timeout,
        BatchExecutionOptions {
            recovery: false,
            completion_fp16: completion_format == CompletionFormat::Fp16,
        },
    )
}

#[derive(Debug, Clone, Copy)]
enum BatchInputs<'a> {
    F64(&'a [Vec<f64>]),
    Fp16(&'a [Vec<u16>]),
}

#[derive(Debug, Clone, Copy)]
struct BatchExecutionOptions {
    recovery: bool,
    completion_fp16: bool,
}

fn run_compressed_batch_with_format(
    sessions: &mut [Session],
    artifacts: &[CompressedArtifact],
    p: &routing::RoutingProblem,
    inputs: BatchInputs<'_>,
    faults: &BTreeMap<u32, Fault>,
    timeout: Duration,
    options: BatchExecutionOptions,
) -> Result<BatchReport> {
    let ranks_of = by_rank(p, artifacts)?;
    check_work(p, artifacts, &ranks_of)?;
    let digests = ranks_of
        .iter()
        .map(|(&rank, &index)| (rank, artifacts[index].sha256.clone()))
        .collect();
    run_prevalidated_batch(sessions, &digests, p, inputs, faults, timeout, options)
}

fn run_prevalidated_batch(
    sessions: &mut [Session],
    digests: &BTreeMap<u32, String>,
    p: &routing::RoutingProblem,
    inputs: BatchInputs<'_>,
    faults: &BTreeMap<u32, Fault>,
    timeout: Duration,
    options: BatchExecutionOptions,
) -> Result<BatchReport> {
    let result = run_prevalidated_batch_with_finish(
        sessions,
        digests,
        p,
        inputs,
        faults,
        timeout,
        options,
        moe::Collector::finish,
    )?;
    Ok(BatchReport {
        ok: result.error.is_none(),
        output: result.output,
        attempt: result.attempt,
        receipts: result.receipts,
        error: result.error,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_prevalidated_batch_with_finish<T>(
    sessions: &mut [Session],
    digests: &BTreeMap<u32, String>,
    p: &routing::RoutingProblem,
    inputs: BatchInputs<'_>,
    faults: &BTreeMap<u32, Fault>,
    timeout: Duration,
    options: BatchExecutionOptions,
    finish: fn(moe::Collector) -> Result<T>,
) -> Result<ExecutedBatch<T>> {
    match inputs {
        BatchInputs::F64(rows) => check_rows(p, rows)?,
        BatchInputs::Fp16(rows) => check_fp16_rows(p, rows)?,
    }
    let collector = moe::Collector::from_routing(p)?;
    let ranks: Vec<_> = routing::dispatch(p)?
        .ranks
        .into_iter()
        .filter(|r| !r.slots.is_empty())
        .collect();
    if faults
        .keys()
        .any(|rank| !ranks.iter().any(|r| r.rank == *rank))
    {
        return Err(invalid("fault refers to a nonparticipating rank"));
    }
    // Fault bounds too are checked before any identity is claimed or work is sent,
    // not left to the faulted rank's worker once the other ranks have computed.
    for fault in faults.values() {
        check_fault(fault)?;
    }
    let mut plan = Vec::new();
    for rank in ranks {
        let index = sessions
            .iter()
            .position(|s| s.rank == rank.rank)
            .ok_or_else(|| invalid("no session serves a participating rank"))?;
        let session = &sessions[index];
        if !session.usable {
            return Err(Error::new(
                "SESSION_UNUSABLE",
                "session must be reopened after a failure",
            ));
        }
        let digest = digests
            .get(&rank.rank)
            .ok_or_else(|| invalid("missing rank digest"))?;
        if session.artifact.as_ref() != Some(&(digest.clone(), p.stamp.epoch)) {
            return Err(Error::new(
                "STALE_EXECUTION",
                "session does not hold this batch's resident artifact and epoch; bind first",
            ));
        }
        let tokens: BTreeSet<_> = rank.slots.iter().map(|s| s.slot.token).collect();
        let rank_inputs = if let BatchInputs::Fp16(fp16_inputs) = inputs {
            if !session
                .hello
                .capabilities
                .iter()
                .any(|capability| capability == CAPABILITY_EXEC_FP16_RAW)
            {
                return Err(Error::new(
                    "CAPABILITY_MISSING",
                    format!(
                        "rank {} does not advertise raw FP16 activation input support; no EXEC was sent",
                        rank.rank
                    ),
                ));
            }
            RankInputs::Fp16(
                tokens
                    .into_iter()
                    .map(|token| Fp16TokenInput {
                        token,
                        value_bits: fp16_inputs[token].clone(),
                    })
                    .collect(),
            )
        } else {
            RankInputs::F64(
                tokens
                    .into_iter()
                    .map(|token| TokenInput {
                        token,
                        value: match inputs {
                            BatchInputs::F64(rows) => rows[token].clone(),
                            BatchInputs::Fp16(_) => unreachable!("matched above"),
                        },
                    })
                    .collect(),
            )
        };
        let fault = faults.get(&rank.rank).cloned().unwrap_or_default();
        plan.push((index, rank, rank_inputs, fault));
    }
    exec_all_with_finish(
        sessions,
        plan,
        &p.stamp,
        collector,
        options.recovery,
        timeout,
        options.completion_fp16,
        finish,
    )
}

/// Immutable admission evidence for an already-loaded, fixed-epoch fleet.
/// It retains routing metadata and hashes, not expert weight planes. Construct it
/// only after successful multipart loading; it cannot load, rebind or recover.
pub struct CompactCompressedFleet {
    baseline: routing::RoutingProblem,
    digests: BTreeMap<u32, String>,
    per_slot_work: BTreeMap<u32, u64>,
}
impl CompactCompressedFleet {
    pub fn from_loaded(
        sessions: &[Session],
        artifacts: &[CompressedArtifact],
        p: &routing::RoutingProblem,
    ) -> Result<Self> {
        let ranks = by_rank(p, artifacts)?;
        if sessions.len() != ranks.len()
            || sessions.iter().map(|s| s.rank).collect::<BTreeSet<_>>()
                != ranks.keys().copied().collect()
        {
            return Err(invalid("compact fleet endpoint coverage mismatch"));
        }
        let mut digests = BTreeMap::new();
        let mut per_slot_work = BTreeMap::new();
        for s in sessions {
            let a = &artifacts[ranks[&s.rank]];
            if !s.usable || s.artifact.as_ref() != Some(&(a.sha256.clone(), p.stamp.epoch)) {
                return Err(Error::new(
                    "STALE_EXECUTION",
                    "compact fleet requires every resident acknowledged at the current epoch",
                ));
            }
            let (_, n13, k13, _, _) = a.stack.shape();
            let cost = (n13 / 2)
                .checked_mul(k13)
                .and_then(|v| v.checked_mul(3))
                .ok_or_else(over_budget)?;
            digests.insert(s.rank, a.sha256.clone());
            per_slot_work.insert(s.rank, cost);
        }
        let mut baseline = p.clone();
        baseline.routes.clear();
        Ok(Self {
            baseline,
            digests,
            per_slot_work,
        })
    }
    /// Execute one typed-FP16 batch without retaining coordinator weight copies.
    /// Static identity, registry and owner preference are frozen; only routes,
    /// activations and batch number may change. Unknown work is never replayed.
    pub fn run_fp16(
        &self,
        sessions: &mut [Session],
        p: &routing::RoutingProblem,
        inputs_fp16: &[Vec<u16>],
        timeout: Duration,
    ) -> Result<BatchReport> {
        routing::validate(p)?;
        let mut stamp = p.stamp.clone();
        stamp.batch = self.baseline.stamp.batch;
        if stamp != self.baseline.stamp
            || p.hidden != self.baseline.hidden
            || p.experts != self.baseline.experts
            || p.registry.epoch != self.baseline.registry.epoch
            || p.registry.copies != self.baseline.registry.copies
            || p.resident_order != self.baseline.resident_order
            || p.preferred_owners != self.baseline.preferred_owners
        {
            return Err(invalid("compact fleet identity or registry changed"));
        }
        for rank in routing::dispatch(p)?.ranks {
            let cost = *self
                .per_slot_work
                .get(&rank.rank)
                .ok_or_else(|| invalid("missing compact rank"))?;
            if (rank.slots.len() as u64)
                .checked_mul(cost)
                .is_none_or(|n| n > 100_000_000)
            {
                return Err(over_budget());
            }
        }
        run_prevalidated_batch(
            sessions,
            &self.digests,
            p,
            BatchInputs::Fp16(inputs_fp16),
            &BTreeMap::new(),
            timeout,
            BatchExecutionOptions {
                recovery: false,
                completion_fp16: true,
            },
        )
    }
}

/// The dense f64 oracle of compressed batches: held experts decoded once by the
/// row-major path (`ResidentStack::dense_expert`), each slot's value from
/// `moe::expert_value`, coefficients applied in route order as `moe::reference` does.
pub struct DenseOracle {
    dims: moe::Model,
    experts: BTreeMap<(u32, u32), moe::Expert>,
}
impl DenseOracle {
    /// Every expert the artifacts hold, decoded (8 bytes per weight).
    pub fn new(artifacts: &[CompressedArtifact]) -> Result<DenseOracle> {
        Self::decode(artifacts, |_| true)
    }

    /// Only the (rank, expert) pairs that `p` dispatches to, decoded: the oracle of every
    /// batch routed as `p` is, whose memory follows the routing rather than every expert
    /// held. A batch routed to another pair is refused by `output`.
    pub fn for_routing(
        artifacts: &[CompressedArtifact],
        p: &routing::RoutingProblem,
    ) -> Result<DenseOracle> {
        let routed: BTreeSet<(u32, u32)> = routing::dispatch(p)?
            .slots
            .iter()
            .map(|s| (s.rank, s.expert))
            .collect();
        Self::decode(artifacts, |pair| routed.contains(&pair))
    }

    /// Number of dense experts this oracle holds.
    pub fn experts(&self) -> usize {
        self.experts.len()
    }

    fn decode(
        artifacts: &[CompressedArtifact],
        wanted: impl Fn((u32, u32)) -> bool,
    ) -> Result<DenseOracle> {
        let mut experts = BTreeMap::new();
        let mut dims = None;
        let mut ranks = BTreeSet::new();
        for a in artifacts {
            // A second artifact of a rank would replace the first's experts by order.
            if !ranks.insert(a.stack.rank()) {
                return Err(invalid("an oracle holds one artifact per rank"));
            }
            let (_, n13, k13, _, _) = a.stack.shape();
            let shape = (k13 as usize, (n13 / 2) as usize);
            if dims.is_some_and(|d| d != shape) {
                return Err(invalid("artifacts disagree on hidden or intermediate size"));
            }
            dims = Some(shape);
            for (physical, e) in a.stack.physical_experts().iter().enumerate() {
                if wanted((a.stack.rank(), e.id())) {
                    experts.insert((a.stack.rank(), e.id()), a.stack.dense_expert(physical)?);
                }
            }
        }
        let (hidden, intermediate) = dims.ok_or_else(|| invalid("no artifact"))?;
        Ok(DenseOracle {
            dims: moe::Model {
                hidden,
                intermediate,
                experts: vec![],
            },
            experts,
        })
    }
    pub fn output(
        &self,
        p: &routing::RoutingProblem,
        inputs: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>> {
        if inputs.len() != p.routes.len()
            || inputs
                .iter()
                .any(|row| row.len() != self.dims.hidden || row.iter().any(|x| !x.is_finite()))
        {
            return Err(invalid(
                "one finite activation row of the hidden size per routed token",
            ));
        }
        let mut slots = routing::dispatch(p)?.slots;
        slots.sort_by_key(|s| (s.token, s.slot));
        let mut out = vec![vec![0.0; self.dims.hidden]; inputs.len()];
        for s in slots {
            let expert = self
                .experts
                .get(&(s.rank, s.expert))
                .ok_or_else(|| invalid("oracle lacks a routed expert"))?;
            let value = moe::expert_value(&self.dims, expert, &inputs[s.token])?;
            for (y, v) in out[s.token].iter_mut().zip(value) {
                *y = moe::finite(*y + s.weight * v)?;
            }
        }
        Ok(out)
    }
}

/// A series over compressed residents (`vf session-run-compressed`): the artifacts are
/// read from files by the coordinator and sent once per session.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompressedSeriesInput {
    pub artifacts: Vec<std::path::PathBuf>,
    pub routing: routing::RoutingProblem,
    #[serde(default)]
    pub inputs: Vec<Vec<f64>>,
    /// Optional caller-owned IEEE binary16 values represented as numeric bit patterns
    /// (`0x3c00` is 1.0). This selects the raw FP16 EXEC input path; it is never inferred
    /// by converting `inputs`, including when the original producer used BF16.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inputs_fp16_bits: Option<Vec<Vec<u16>>>,
    pub workers: Vec<Endpoint>,
    pub batches: u64,
    pub timeout_ms: u64,
    #[serde(default)]
    pub epoch_every: u64,
    #[serde(default)]
    pub warmup: u64,
    #[serde(default)]
    pub tolerance_abs: f64,
    /// Enable opt-in finite-only binary16 response frames. This rounds each unweighted
    /// expert result before route weighting and therefore requires a caller-chosen,
    /// evidence-based tolerance; the default remains the exact f64 reference path.
    #[serde(default)]
    pub fp16_completions: bool,
    #[serde(default)]
    pub faults: Vec<SeriesFault>,
    /// Batches whose outputs the report returns for an external oracle (at most 16).
    #[serde(default)]
    pub record: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactSummary {
    pub rank: u32,
    pub sha256: String,
    pub file_bytes: u64,
    pub payload_bytes: u64,
    pub experts: Vec<u32>,
    pub shape: (usize, u64, u64, u64, u64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedSeriesReport {
    pub artifacts: Vec<ArtifactSummary>,
    pub oracle: String,
    /// Dense experts the oracle decoded: the (rank, expert) pairs the routing uses.
    #[serde(default)]
    pub oracle_experts: usize,
    #[serde(flatten)]
    pub series: SeriesReport,
}

struct CompressedSeries<'a> {
    input: &'a CompressedSeriesInput,
    artifacts: Vec<CompressedArtifact>,
    oracle: DenseOracle,
    base_inputs: Vec<Vec<f64>>,
}

struct CompressedSeriesBatch {
    routing: routing::RoutingProblem,
    inputs: Vec<Vec<f64>>,
    inputs_fp16_bits: Option<Vec<Vec<u16>>>,
}
impl Series for CompressedSeries<'_> {
    type Batch = CompressedSeriesBatch;
    fn batch(&self, b: u64) -> Self::Batch {
        let mut routing = self.input.routing.clone();
        routing.stamp.batch = routing.stamp.batch.saturating_add(b);
        if self.input.epoch_every > 0 {
            routing.stamp.epoch = routing
                .stamp
                .epoch
                .saturating_add(b / self.input.epoch_every);
            routing.registry.epoch = routing.stamp.epoch;
        }
        let inputs_fp16_bits = self.input.inputs_fp16_bits.clone();
        let mut inputs = self.base_inputs.clone();
        // Raw FP16 inputs already belong to the caller. Reusing them verbatim across a
        // series keeps those bits unchanged; generated scaling remains on the f64 mode.
        if inputs_fp16_bits.is_none() {
            scale_inputs(&mut inputs, b);
        }
        CompressedSeriesBatch {
            routing,
            inputs,
            inputs_fp16_bits,
        }
    }
    fn epoch(batch: &Self::Batch) -> u64 {
        batch.routing.stamp.epoch
    }
    fn bind(&self, sessions: &mut [Session], batch: &Self::Batch, timeout: Duration) -> Result<()> {
        bind_compressed(sessions, &self.artifacts, &batch.routing, timeout)
    }
    fn run(
        &self,
        sessions: &mut [Session],
        batch: &Self::Batch,
        faults: &BTreeMap<u32, Fault>,
        timeout: Duration,
    ) -> Result<BatchReport> {
        if let Some(inputs_fp16_bits) = &batch.inputs_fp16_bits {
            run_compressed_batch_with_fp16_inputs(
                sessions,
                &self.artifacts,
                &batch.routing,
                inputs_fp16_bits,
                faults,
                if self.input.fp16_completions {
                    CompletionFormat::Fp16
                } else {
                    CompletionFormat::F64
                },
                timeout,
            )
        } else if self.input.fp16_completions {
            run_compressed_batch_fp16(
                sessions,
                &self.artifacts,
                &batch.routing,
                &batch.inputs,
                faults,
                false,
                timeout,
            )
        } else {
            run_compressed_batch(
                sessions,
                &self.artifacts,
                &batch.routing,
                &batch.inputs,
                faults,
                false,
                timeout,
            )
        }
    }
    fn reference(&self, batch: &Self::Batch) -> Result<Vec<Vec<f64>>> {
        self.oracle.output(&batch.routing, &batch.inputs)
    }
}

/// Select exactly one compressed-series activation representation. Binary16 rows are
/// decoded only for the independent f64 oracle; the original u16 bits continue to the
/// worker wire unchanged. The input mode is selected before any worker process is started.
fn compressed_series_inputs(input: &CompressedSeriesInput) -> Result<Vec<Vec<f64>>> {
    match (&input.inputs, &input.inputs_fp16_bits) {
        (rows, None) if !rows.is_empty() => Ok(rows.clone()),
        (rows, Some(bits)) if rows.is_empty() => bits
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&value| {
                        crate::completion::decode_fp16(value).map_err(|_| {
                            invalid("FP16 activation input must contain only finite values")
                        })
                    })
                    .collect()
            })
            .collect(),
        _ => Err(invalid(
            "provide either f64 inputs or inputs_fp16_bits, but not both",
        )),
    }
}

/// Legacy base64 bound: the encoded artifact and JSON fields fit one message frame.
pub const MAX_ARTIFACT_BYTES: usize = (MAX_MESSAGE_BYTES - 64 * 1024) / 4 * 3;
/// Raw body bound leaves room for the separately framed, bounded LOAD header.
pub const MAX_RAW_ARTIFACT_BYTES: usize = MAX_MESSAGE_BYTES - 64 * 1024;

/// Read one artifact file, refusing anything the raw LOAD_ARTIFACT body could not carry.
/// The file must be regular, opened non-blocking and checked again once open, as the
/// checkpoint importer requires: a FIFO or device path never blocks or feeds the reader.
fn read_artifact(path: &std::path::Path) -> Result<CompressedArtifact> {
    let io = |e: std::io::Error| Error::new("ARTIFACT_IO", format!("cannot read artifact: {e}"));
    let regular = || invalid("artifact path must be a regular file");
    if !std::fs::metadata(path).map_err(io)?.is_file() {
        return Err(regular());
    }
    let mut bytes = Vec::new();
    let still_regular = crate::checkpoint::open_nonblocking(path)
        .and_then(|file| {
            if !file.metadata()?.is_file() {
                return Ok(false);
            }
            file.take(MAX_RAW_ARTIFACT_BYTES as u64 + 1)
                .read_to_end(&mut bytes)?;
            Ok(true)
        })
        .map_err(io)?;
    if !still_regular {
        return Err(regular());
    }
    if bytes.len() > MAX_RAW_ARTIFACT_BYTES {
        return Err(Error::new(
            "MESSAGE_TOO_LARGE",
            "artifact exceeds the raw LOAD_ARTIFACT body limit",
        ));
    }
    CompressedArtifact::from_bytes(bytes)
}

/// Run a compressed series; as `run_series`, but the weights are QPN2 artifacts.
pub fn run_compressed_series(input: &CompressedSeriesInput) -> Result<CompressedSeriesReport> {
    let params = SeriesParams {
        batches: input.batches,
        timeout_ms: input.timeout_ms,
        warmup: input.warmup,
        tolerance_abs: input.tolerance_abs,
        faults: &input.faults,
        record: &input.record,
    };
    let timeout = params.check()?;
    check_numbering(&input.routing.stamp, input.batches, input.epoch_every)?;
    check_work_present(&input.routing.routes)?;
    if input.artifacts.is_empty() || input.artifacts.len() > 64 {
        return Err(invalid("a compressed series needs 1..64 artifacts"));
    }
    let artifacts = input
        .artifacts
        .iter()
        .map(|path| read_artifact(path))
        .collect::<Result<Vec<_>>>()?;
    let ranks_of = by_rank(&input.routing, &artifacts)?;
    // Refuse malformed activation input before any worker starts or weight is sent.
    // Only f64 inputs are generated/scaled by batch; raw FP16 bits are caller-owned and
    // replayed unchanged under each new batch identity.
    let base_inputs = compressed_series_inputs(input)?;
    check_rows(&input.routing, &base_inputs)?;
    if input.inputs_fp16_bits.is_none() {
        check_series_rows(&base_inputs, input.batches)?;
    }
    check_work(&input.routing, &artifacts, &ranks_of)?;
    check_series_faults(&input.routing, &input.faults)?;
    // Every batch shares this routing: its collection bound is checked once, here,
    // rather than at batch 0 once every artifact has been sent.
    moe::Collector::from_routing(&input.routing)?;
    let summaries = artifacts
        .iter()
        .map(|a| ArtifactSummary {
            rank: a.stack.rank(),
            sha256: a.sha256.clone(),
            file_bytes: a.bytes.len() as u64,
            payload_bytes: a.stack.payload_bytes(),
            experts: a.stack.physical_experts().iter().map(|e| e.id()).collect(),
            shape: a.stack.shape(),
        })
        .collect();
    let oracle = DenseOracle::for_routing(&artifacts, &input.routing)?;
    let oracle_experts = oracle.experts();
    let started = Instant::now();
    let sessions = prepare_compressed(&input.workers, &artifacts, &input.routing, timeout)?;
    let setup_ms = started.elapsed().as_secs_f64() * 1e3;
    let series = CompressedSeries {
        input,
        artifacts,
        oracle,
        base_inputs,
    };
    let report = drive(
        &series,
        &params,
        sessions,
        setup_ms,
        input.routing.stamp.epoch,
        timeout,
    )?;
    Ok(CompressedSeriesReport {
        artifacts: summaries,
        oracle: "dense f64: QPN2 unpack, linear NVFP4 decode, moe::expert_value".into(),
        oracle_experts,
        series: report,
    })
}

#[cfg(test)]
mod a55_raw_activation_tests {
    use super::*;

    #[test]
    fn raw_activation_decoder_preserves_f64_bits() {
        let expected = [
            0.0,
            -0.0,
            f64::from_bits(1),
            f64::from_bits(0x000f_ffff_ffff_ffff),
            f64::MAX,
        ];
        let mut body = Vec::new();
        for value in expected {
            body.extend_from_slice(&value.to_le_bytes());
        }
        let mut input = io::Cursor::new(body.clone());
        let rows = read_raw_activation_inputs(
            &mut input,
            128,
            body.len() as u64,
            vec![RawTokenInputMeta {
                token: 7,
                values: expected.len() as u64,
            }],
        )
        .unwrap();
        assert_eq!(rows[0].token, 7);
        assert_eq!(
            rows[0]
                .value
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn raw_activation_decoder_checks_sizes_before_reading() {
        let meta = vec![RawTokenInputMeta {
            token: 0,
            values: 1,
        }];
        let mut input = io::Cursor::new(Vec::<u8>::new());
        let error =
            read_raw_activation_inputs(&mut input, MAX_MESSAGE_BYTES, 8, meta.clone()).unwrap_err();
        assert_eq!(error.code, "MESSAGE_TOO_LARGE");

        let error = read_raw_activation_inputs(
            &mut input,
            128,
            MAX_RAW_ACTIVATION_BYTES as u64 + 1,
            meta.clone(),
        )
        .unwrap_err();
        assert_eq!(error.code, "MESSAGE_TOO_LARGE");

        let error = read_raw_activation_inputs(&mut input, 128, 0, meta).unwrap_err();
        assert_eq!(error.code, "INVALID_INPUT");
    }
}

#[cfg(test)]
mod a169_raw_fp16_activation_tests {
    use super::*;

    fn metadata(values: u64) -> Vec<RawTokenInputMeta> {
        vec![RawTokenInputMeta { token: 4, values }]
    }

    #[test]
    fn fp16_activation_decoder_preserves_signed_zero_subnormal_and_finite_edges() {
        let bits = [0x8000_u16, 0x0001, 0x3c00, 0x7bff, 0xfbff];
        let expected = [-0.0_f64, 2.0_f64.powi(-24), 1.0, 65504.0, -65504.0];
        let body: Vec<u8> = bits.iter().flat_map(|bits| bits.to_le_bytes()).collect();
        let mut input = io::Cursor::new(body.clone());
        let rows = read_raw_activation_inputs_fp16(
            &mut input,
            128,
            body.len() as u64,
            metadata(bits.len() as u64),
        )
        .unwrap();
        assert_eq!(rows[0].token, 4);
        assert_eq!(
            rows[0]
                .value
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn fp16_activation_decoder_rejects_bad_lengths_nonfinite_and_oversized_frames() {
        let mut empty = io::Cursor::new(Vec::<u8>::new());
        let mismatch =
            read_raw_activation_inputs_fp16(&mut empty, 128, 3, metadata(2)).unwrap_err();
        assert_eq!(mismatch.code, "INVALID_INPUT");

        let mut truncated = io::Cursor::new(vec![0, 0, 0]);
        let error =
            read_raw_activation_inputs_fp16(&mut truncated, 128, 4, metadata(2)).unwrap_err();
        assert_eq!(error.code, "EXEC_TRUNCATED");

        for bits in [0x7c00_u16, 0xfc00, 0x7e00] {
            let mut nonfinite = io::Cursor::new(bits.to_le_bytes().to_vec());
            let error =
                read_raw_activation_inputs_fp16(&mut nonfinite, 128, 2, metadata(1)).unwrap_err();
            assert_eq!(error.code, "INVALID_INPUT");
        }

        let size = read_raw_activation_inputs_fp16(&mut empty, MAX_MESSAGE_BYTES, 2, metadata(1))
            .unwrap_err();
        assert_eq!(size.code, "MESSAGE_TOO_LARGE");
    }
}

#[cfg(test)]
mod a56_raw_completion_tests {
    use super::*;

    fn event(value: Vec<f64>) -> WorkerEvent {
        let stamp = moe::Stamp {
            run_id: "run".into(),
            model_revision: "rev".into(),
            plan_id: "plan".into(),
            layer: 3,
            batch: 7,
            epoch: 4,
        };
        WorkerEvent {
            stamp: stamp.clone(),
            rank: 2,
            attempt: 9,
            event: EventKind::Completion {
                completion: moe::Completion {
                    stamp,
                    token: 1,
                    slot: 0,
                    expert: 5,
                    rank: 2,
                    local: 0,
                    value,
                },
            },
        }
    }

    fn frame(event: &WorkerEvent) -> Vec<u8> {
        let EventKind::Completion { completion } = &event.event else {
            unreachable!()
        };
        encode_raw_completion(event, completion).unwrap()
    }

    #[test]
    fn raw_completion_round_trips_bits_and_identity() {
        let values = vec![-0.0, f64::from_bits(1), f64::MAX, -1.5e-300, 3.25];
        let sent = event(values.clone());
        let mut wire = frame(&sent);
        wire.extend_from_slice(b"{\"reply\":\"evicted\",\"artifact_sha256\":\"x\"}\n");
        let mut input = io::Cursor::new(wire);
        let allowed = AtomicBool::new(true);
        let Some(Reply::Event { event }) = read_reply(&mut input, &allowed).unwrap() else {
            panic!("not an event")
        };
        assert_eq!(encode(&event).unwrap(), encode(&sent).unwrap());
        let EventKind::Completion { completion } = &event.event else {
            panic!("not a completion")
        };
        let bits: Vec<u64> = completion.value.iter().map(|x| x.to_bits()).collect();
        assert_eq!(bits, values.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        // The body boundary is exact: the next frame still parses.
        assert!(matches!(
            read_reply(&mut input, &allowed).unwrap(),
            Some(Reply::Evicted { .. })
        ));
    }

    #[test]
    fn raw_completion_is_refused_unsolicited_truncated_nonfinite_or_oversized() {
        let wire = frame(&event(vec![1.0, 2.0]));
        let unsolicited = read_reply(&mut io::Cursor::new(&wire), &AtomicBool::new(false));
        assert_eq!(unsolicited.unwrap_err().code, "WORKER_PROTOCOL");

        let allowed = AtomicBool::new(true);
        let truncated = read_reply(&mut io::Cursor::new(&wire[..wire.len() - 1]), &allowed);
        assert_eq!(truncated.unwrap_err().code, "WORKER_PROTOCOL");

        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let wire = frame(&event(vec![1.0, bad]));
            let error = read_reply(&mut io::Cursor::new(&wire), &allowed).unwrap_err();
            assert_eq!(error.code, "WORKER_PROTOCOL");
        }

        // A declared length past the frame bound is refused before allocating or reading.
        let mut header = encode(&Reply::CompletionRaw {
            stamp: event(vec![]).stamp,
            rank: 2,
            attempt: 9,
            completion: RawCompletionMeta {
                stamp: event(vec![]).stamp,
                token: 1,
                slot: 0,
                expert: 5,
                rank: 2,
                local: 0,
                values: u64::MAX,
            },
        })
        .unwrap();
        let error = read_reply(&mut io::Cursor::new(&header), &allowed).unwrap_err();
        assert_eq!(error.code, "MESSAGE_TOO_LARGE");
        header.truncate(header.len() - 1);
        let values = (MAX_RAW_ACTIVATION_BYTES / 8 + 1).to_string();
        let header = String::from_utf8(header)
            .unwrap()
            .replace(&u64::MAX.to_string(), &values);
        let error = read_reply(&mut io::Cursor::new(format!("{header}\n")), &allowed).unwrap_err();
        assert_eq!(error.code, "MESSAGE_TOO_LARGE");
    }

    #[test]
    fn raw_completion_combined_header_and_body_bound_is_checked_before_reading() {
        // The body (two values, sixteen bytes) is small; only header + newline + body passes
        // the frame bound, so the reader must refuse without touching the input.
        let meta = || RawCompletionMeta {
            stamp: event(vec![]).stamp,
            token: 1,
            slot: 0,
            expert: 5,
            rank: 2,
            local: 0,
            values: 2,
        };
        let body: Vec<u8> = [1.0f64, -0.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let mut input = io::Cursor::new(body.clone());
        let stamp = event(vec![]).stamp;
        let error = read_raw_completion(
            &mut input,
            MAX_MESSAGE_BYTES - 16,
            stamp.clone(),
            2,
            9,
            meta(),
        )
        .unwrap_err();
        assert_eq!(error.code, "MESSAGE_TOO_LARGE");
        assert_eq!(input.position(), 0);
        let mut input = io::Cursor::new(body);
        let reply =
            read_raw_completion(&mut input, MAX_MESSAGE_BYTES - 17, stamp, 2, 9, meta()).unwrap();
        assert!(matches!(reply, Reply::Event { .. }));
        assert_eq!(input.position(), 16);
    }

    #[test]
    fn completion_frame_is_never_longer_than_json_and_keeps_bits() {
        let width = 2048;
        let zeros = vec![0.0; width];
        let signed: Vec<f64> = (0..width)
            .map(|i| if i % 2 == 0 { -0.0 } else { 0.0 })
            .collect();
        let simple: Vec<f64> = (0..width)
            .map(|i| [0.0, -0.0, 1.0, -0.5, 2.25, -3.0, 0.125][i % 7])
            .collect();
        let dense: Vec<f64> = (0..width)
            .map(|i| ((i as f64) * 0.7193).sin() * 1.234_567e-3)
            .collect();
        let allowed = AtomicBool::new(true);
        for (values, want_raw) in [
            (zeros, false),
            (signed, false),
            (simple, false),
            (dense, true),
        ] {
            let sent = event(values.clone());
            let EventKind::Completion { completion } = &sent.event else {
                unreachable!()
            };
            let json = encode(&EventReply::Event { event: &sent }).unwrap();
            let raw = encode_raw_completion(&sent, completion).unwrap();
            let chosen = smaller_completion_frame(&sent, completion).unwrap();
            assert_eq!(chosen.len(), json.len().min(raw.len()));
            assert_eq!(chosen == raw, want_raw);
            if want_raw {
                assert!(raw.len() * 2 < json.len());
            }
            let Some(Reply::Event { event }) =
                read_reply(&mut io::Cursor::new(chosen), &allowed).unwrap()
            else {
                panic!("not an event")
            };
            let EventKind::Completion { completion } = &event.event else {
                panic!("not a completion")
            };
            let bits: Vec<u64> = completion.value.iter().map(|x| x.to_bits()).collect();
            assert_eq!(bits, values.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        }
    }

    #[test]
    fn maximum_valid_hidden_row_fits_the_completion_frame_bound() {
        // Routing admits hidden <= 65536. Even the longest finite f64 representation
        // at that width must fit the per-event message bound in both wire encodings.
        let values = vec![f64::MAX; 65_536];
        let sent = event(values);
        let EventKind::Completion { completion } = &sent.event else {
            unreachable!()
        };
        let json = encode(&EventReply::Event { event: &sent }).unwrap();
        let raw = encode_raw_completion(&sent, completion).unwrap();
        assert!(json.len() <= MAX_MESSAGE_BYTES);
        assert!(raw.len() <= MAX_MESSAGE_BYTES);

        let chosen = smaller_completion_frame(&sent, completion).unwrap();
        assert!(chosen.len() <= MAX_MESSAGE_BYTES);
        let Some(Reply::Event { event }) =
            read_reply(&mut io::Cursor::new(chosen), &AtomicBool::new(true)).unwrap()
        else {
            panic!("not a completion")
        };
        let EventKind::Completion { completion } = event.event else {
            panic!("not a completion")
        };
        assert_eq!(completion.value.len(), 65_536);
        assert!(completion.value.iter().all(|&value| value == f64::MAX));
    }

    #[test]
    fn json_completion_frames_remain_accepted_without_negotiation() {
        let sent = event(vec![0.5, -2.0]);
        let wire = encode(&EventReply::Event { event: &sent }).unwrap();
        let reply = read_reply(&mut io::Cursor::new(wire), &AtomicBool::new(false)).unwrap();
        assert!(matches!(reply, Some(Reply::Event { .. })));
    }

    fn batch_header(
        stamp: moe::Stamp,
        hidden: u64,
        values: u64,
        completions: Vec<RawBatchCompletionMeta>,
    ) -> Vec<u8> {
        encode(&Reply::CompletionBatchRaw {
            completion_stamp: stamp.clone(),
            stamp,
            rank: 2,
            attempt: 9,
            hidden,
            values,
            completions,
        })
        .unwrap()
    }

    #[test]
    fn raw_completion_batch_requires_negotiation_and_preserves_f64_bits() {
        let stamp = event(vec![]).stamp;
        let metadata = vec![
            RawBatchCompletionMeta {
                token: 1,
                slot: 0,
                expert: 5,
                local: 7,
            },
            RawBatchCompletionMeta {
                token: 4,
                slot: 1,
                expert: 8,
                local: 3,
            },
        ];
        let expected = [f64::MAX, -0.0, f64::from_bits(1), -1.5e-300];
        let mut wire = batch_header(stamp, 2, expected.len() as u64, metadata);
        for value in expected {
            wire.extend_from_slice(&value.to_le_bytes());
        }
        let allowed = AtomicBool::new(true);
        let batch_allowed = AtomicBool::new(true);
        let Some(Reply::CompletionBatch {
            rank,
            attempt,
            completions,
            ..
        }) = read_reply_with_flags(
            &mut io::Cursor::new(wire.clone()),
            &allowed,
            &batch_allowed,
            &AtomicU64::new(0),
        )
        .unwrap()
        else {
            panic!("not a decoded completion batch")
        };
        assert_eq!((rank, attempt), (2, 9));
        assert_eq!(completions.len(), 2);
        assert_eq!(
            completions[0]
                .value
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            expected[..2]
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            completions[1]
                .value
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            expected[2..]
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(completions[0].local, 7);
        assert_eq!(completions[1].local, 3);

        let error = read_reply_with_flags(
            &mut io::Cursor::new(wire),
            &allowed,
            &AtomicBool::new(false),
            &AtomicU64::new(0),
        )
        .unwrap_err();
        assert_eq!(error.code, "WORKER_PROTOCOL");
        assert!(error.message.contains("unsolicited raw completion batch"));
    }

    #[test]
    fn raw_completion_batch_rejects_bad_shapes_truncation_and_nonfinite_values() {
        let stamp = event(vec![]).stamp;
        let metadata = vec![RawBatchCompletionMeta {
            token: 1,
            slot: 0,
            expert: 5,
            local: 7,
        }];
        let allowed = AtomicBool::new(true);
        let batch_allowed = AtomicBool::new(true);

        let bad_shape = batch_header(stamp.clone(), 2, 1, metadata.clone());
        let error = read_reply_with_flags(
            &mut io::Cursor::new(bad_shape),
            &allowed,
            &batch_allowed,
            &AtomicU64::new(0),
        )
        .unwrap_err();
        assert_eq!(error.code, "WORKER_PROTOCOL");
        assert!(error.message.contains("value count"));

        let mut truncated = batch_header(stamp.clone(), 2, 2, metadata.clone());
        truncated.extend_from_slice(&1.0f64.to_le_bytes());
        let error = read_reply_with_flags(
            &mut io::Cursor::new(truncated),
            &allowed,
            &batch_allowed,
            &AtomicU64::new(0),
        )
        .unwrap_err();
        assert_eq!(error.code, "WORKER_PROTOCOL");
        assert!(error.message.contains("ended before"));

        let mut nonfinite = batch_header(stamp.clone(), 2, 2, metadata.clone());
        nonfinite.extend_from_slice(&1.0f64.to_le_bytes());
        nonfinite.extend_from_slice(&f64::NAN.to_le_bytes());
        let error = read_reply_with_flags(
            &mut io::Cursor::new(nonfinite),
            &allowed,
            &batch_allowed,
            &AtomicU64::new(0),
        )
        .unwrap_err();
        assert_eq!(error.code, "WORKER_PROTOCOL");
        assert!(error.message.contains("non-finite"));

        let oversized = batch_header(stamp, 65_537, 65_537, metadata);
        let error = read_reply_with_flags(
            &mut io::Cursor::new(oversized),
            &allowed,
            &batch_allowed,
            &AtomicU64::new(0),
        )
        .unwrap_err();
        assert_eq!(error.code, "WORKER_PROTOCOL");
        assert!(error.message.contains("hidden width"));
    }

    #[test]
    fn raw_completion_batch_writer_chunks_at_the_session_limit() {
        let stamp = event(vec![]).stamp;
        let rows = (0..33)
            .map(|token| moe::Completion {
                stamp: stamp.clone(),
                token,
                slot: 0,
                expert: 5,
                rank: 2,
                local: 0,
                value: vec![f64::MAX; 65_536],
            })
            .collect();
        let mut wire = Vec::new();
        write_raw_completion_batch(
            &mut wire,
            WorkerCompletions {
                stamp,
                rank: 2,
                attempt: 9,
                rows,
            },
        )
        .unwrap();

        let mut remaining = wire.as_slice();
        let mut frames = 0;
        let mut rows = 0;
        while !remaining.is_empty() {
            let newline = remaining.iter().position(|byte| *byte == b'\n').unwrap();
            let reply: Reply = serde_json::from_slice(&remaining[..newline]).unwrap();
            let Reply::CompletionBatchRaw {
                hidden,
                values,
                completions,
                ..
            } = reply
            else {
                panic!("not a raw completion batch")
            };
            let body_len = values as usize * 8;
            let frame_len = newline + 1 + body_len;
            assert!(frame_len <= MAX_MESSAGE_BYTES);
            assert_eq!(hidden, 65_536);
            assert_eq!(values as usize, completions.len() * hidden as usize);
            rows += completions.len();
            frames += 1;
            remaining = &remaining[frame_len..];
        }
        assert_eq!(rows, 33);
        assert_eq!(frames, 2);
    }

    #[test]
    fn raw_completion_batch_writer_respects_a_small_frame_bound() {
        let stamp = event(vec![]).stamp;
        let rows = (0..32)
            .map(|token| moe::Completion {
                stamp: stamp.clone(),
                token,
                slot: 0,
                expert: 5,
                rank: 2,
                local: 0,
                value: vec![token as f64, -(token as f64)],
            })
            .collect();
        let mut wire = Vec::new();
        write_raw_completion_batch_bounded(
            &mut wire,
            WorkerCompletions {
                stamp,
                rank: 2,
                attempt: 9,
                rows,
            },
            1024,
        )
        .unwrap();

        let mut remaining = wire.as_slice();
        let mut frames = 0;
        let mut tokens = Vec::new();
        while !remaining.is_empty() {
            let newline = remaining.iter().position(|byte| *byte == b'\n').unwrap();
            let reply: Reply = serde_json::from_slice(&remaining[..newline]).unwrap();
            let Reply::CompletionBatchRaw {
                values,
                completions,
                ..
            } = reply
            else {
                panic!("not a raw completion batch")
            };
            let body_len = values as usize * 8;
            let frame_len = newline + 1 + body_len;
            assert!(frame_len <= 1024);
            let frame_values = remaining[newline + 1..frame_len]
                .chunks_exact(8)
                .map(|bytes| f64::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(values as usize, completions.len() * 2);
            for (index, completion) in completions.iter().enumerate() {
                assert_eq!(completion.token, frame_values[index * 2] as usize);
                assert_eq!(frame_values[index * 2 + 1], -(completion.token as f64));
                tokens.push(completion.token);
            }
            frames += 1;
            remaining = &remaining[frame_len..];
        }
        assert_eq!(tokens, (0..32).collect::<Vec<_>>());
        assert!(frames > 1);
    }
}

#[cfg(test)]
mod a73_engram_reconstruction_tests {
    use super::engram_fetched_row_index;

    #[test]
    fn sorted_miss_index_maps_reordered_and_duplicate_requests() {
        let misses = [1, 2, 5, 6, 7];
        let requested = [7, 0, 6, 2, 7, 1, 5];
        let actual: Vec<Option<usize>> = requested
            .iter()
            .map(|&id| engram_fetched_row_index(&misses, id))
            .collect();
        assert_eq!(
            actual,
            [Some(4), None, Some(3), Some(1), Some(4), Some(0), Some(2)]
        );
    }
}

#[cfg(test)]
mod a165_fp16_session_frame_tests {
    use super::*;

    fn stamp() -> moe::Stamp {
        moe::Stamp {
            run_id: "a165".into(),
            model_revision: "rev-a165".into(),
            plan_id: "plan-a165".into(),
            layer: 3,
            batch: 8,
            epoch: 5,
        }
    }

    fn row(token: usize, value: Vec<f64>) -> moe::Completion {
        moe::Completion {
            stamp: stamp(),
            token,
            slot: 0,
            expert: 17,
            rank: 2,
            local: 9,
            value,
        }
    }

    fn header(hidden: u64, values: u64, rows: usize) -> Vec<u8> {
        encode(&Reply::CompletionBatchFp16Raw {
            stamp: stamp(),
            completion_stamp: stamp(),
            rank: 2,
            attempt: 11,
            hidden,
            values,
            completions: (0..rows)
                .map(|token| RawBatchCompletionMeta {
                    token,
                    slot: 0,
                    expert: 17,
                    local: 9,
                })
                .collect(),
        })
        .unwrap()
    }

    #[test]
    fn fp16_batch_roundtrips_little_endian_bits_and_requires_opt_in() {
        let values = vec![-0.0, f64::from_bits(1), 1.0, -2.5, 65504.0];
        let sent = WorkerCompletions {
            stamp: stamp(),
            rank: 2,
            attempt: 11,
            rows: vec![row(0, values.clone())],
        };
        let mut wire = Vec::new();
        write_raw_completion_fp16_batch(&mut wire, sent).unwrap();
        let mut input = io::Cursor::new(wire.clone());
        let Some(Reply::CompletionBatchFp16 {
            stamp: outer,
            completion_stamp,
            rank,
            attempt,
            hidden,
            completions,
            fp16_bits,
        }) = read_reply_with_fp16_flags(
            &mut input,
            &AtomicBool::new(true),
            &AtomicBool::new(false),
            &AtomicBool::new(true),
            &AtomicU64::new(0),
        )
        .unwrap()
        else {
            panic!("not a decoded FP16 batch")
        };
        assert_eq!(outer, stamp());
        assert_eq!(completion_stamp, stamp());
        assert_eq!((rank, attempt, hidden), (2, 11, values.len()));
        assert_eq!(completions.len(), 1);
        assert_eq!(completions[0].token, 0);
        assert_eq!(
            fp16_bits,
            values
                .iter()
                .map(|&value| crate::completion::encode_fp16(value).unwrap())
                .collect::<Vec<_>>()
        );
        assert_eq!(input.position() as usize, wire.len());
        let error = read_reply_with_fp16_flags(
            &mut io::Cursor::new(wire),
            &AtomicBool::new(true),
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            &AtomicU64::new(0),
        )
        .unwrap_err();
        assert_eq!(error.code, "WORKER_PROTOCOL");
        assert!(error.message.contains("unsolicited raw FP16"));
    }

    #[test]
    fn fp16_writer_preflights_all_rows_before_emitting_any_frame() {
        let mut wire = Vec::new();
        let error = write_raw_completion_fp16_batch(
            &mut wire,
            WorkerCompletions {
                stamp: stamp(),
                rank: 2,
                attempt: 11,
                rows: vec![row(0, vec![1.0]), row(1, vec![65520.0])],
            },
        )
        .unwrap_err();
        assert_eq!(error.code, "FP16_OVERFLOW");
        assert!(wire.is_empty());
    }

    #[test]
    fn fp16_writer_frame_bound_counts_the_header_newline() {
        let batch = || WorkerCompletions {
            stamp: stamp(),
            rank: 2,
            attempt: 11,
            rows: vec![row(0, vec![1.0])],
        };
        let mut complete_frame = Vec::new();
        write_raw_completion_fp16_batch_bounded(&mut complete_frame, batch(), MAX_MESSAGE_BYTES)
            .unwrap();
        let newline = complete_frame
            .iter()
            .position(|&byte| byte == b'\n')
            .unwrap();
        assert_eq!(
            newline + 1 + std::mem::size_of::<u16>(),
            complete_frame.len()
        );

        let mut one_byte_short = Vec::new();
        let error = write_raw_completion_fp16_batch_bounded(
            &mut one_byte_short,
            batch(),
            complete_frame.len() - 1,
        )
        .unwrap_err();
        assert_eq!(error.code, "MESSAGE_TOO_LARGE");
        assert!(one_byte_short.is_empty());

        let mut exact_limit = Vec::new();
        write_raw_completion_fp16_batch_bounded(&mut exact_limit, batch(), complete_frame.len())
            .unwrap();
        assert_eq!(exact_limit.len(), complete_frame.len());
    }

    #[test]
    fn fp16_reader_rejects_bad_shape_truncation_nonfinite_and_oversize() {
        let flags = || {
            (
                AtomicBool::new(true),
                AtomicBool::new(false),
                AtomicBool::new(true),
                AtomicU64::new(0),
            )
        };

        let (raw, batch, fp16, engram) = flags();
        let error = read_reply_with_fp16_flags(
            &mut io::Cursor::new(header(2, 1, 1)),
            &raw,
            &batch,
            &fp16,
            &engram,
        )
        .unwrap_err();
        assert_eq!(error.code, "WORKER_PROTOCOL");
        assert!(error.message.contains("value count"));

        let (raw, batch, fp16, engram) = flags();
        let mut truncated = header(2, 2, 1);
        truncated.extend_from_slice(&0x3c00_u16.to_le_bytes());
        let error = read_reply_with_fp16_flags(
            &mut io::Cursor::new(truncated),
            &raw,
            &batch,
            &fp16,
            &engram,
        )
        .unwrap_err();
        assert_eq!(error.code, "WORKER_PROTOCOL");
        assert!(error.message.contains("ended before"));

        let (raw, batch, fp16, engram) = flags();
        let mut nonfinite = header(1, 1, 1);
        nonfinite.extend_from_slice(&0x7c00_u16.to_le_bytes());
        let error = read_reply_with_fp16_flags(
            &mut io::Cursor::new(nonfinite),
            &raw,
            &batch,
            &fp16,
            &engram,
        )
        .unwrap_err();
        assert_eq!(error.code, "WORKER_PROTOCOL");
        assert!(error.message.contains("non-finite"));

        let (raw, batch, fp16, engram) = flags();
        let mut oversized = encode(&Reply::CompletionBatchFp16Raw {
            stamp: stamp(),
            completion_stamp: stamp(),
            rank: 2,
            attempt: 11,
            hidden: 65_536,
            values: 257 * 65_536,
            completions: (0..257)
                .map(|token| RawBatchCompletionMeta {
                    token,
                    slot: 0,
                    expert: 17,
                    local: 9,
                })
                .collect(),
        })
        .unwrap();
        let error = read_reply_with_fp16_flags(
            &mut io::Cursor::new(std::mem::take(&mut oversized)),
            &raw,
            &batch,
            &fp16,
            &engram,
        )
        .unwrap_err();
        assert_eq!(error.code, "MESSAGE_TOO_LARGE");
        assert!(error.message.contains("value-count cap"));
    }

    #[test]
    fn fp16_writer_chunks_bounded_frames_and_sends_fewer_bytes_than_f64() {
        let rows: Vec<_> = (0..32)
            .map(|token| row(token, vec![token as f64 / 8.0, -(token as f64) / 16.0]))
            .collect();
        let batch = WorkerCompletions {
            stamp: stamp(),
            rank: 2,
            attempt: 11,
            rows: rows.clone(),
        };
        let mut f64_wire = Vec::new();
        write_raw_completion_batch(
            &mut f64_wire,
            WorkerCompletions {
                stamp: stamp(),
                rank: 2,
                attempt: 11,
                rows: rows.clone(),
            },
        )
        .unwrap();
        let mut fp16_wire = Vec::new();
        write_raw_completion_fp16_batch(
            &mut fp16_wire,
            WorkerCompletions {
                stamp: stamp(),
                rank: 2,
                attempt: 11,
                rows: rows.clone(),
            },
        )
        .unwrap();
        assert!(fp16_wire.len() < f64_wire.len());

        let mut chunked_wire = Vec::new();
        write_raw_completion_fp16_batch_bounded(&mut chunked_wire, batch, 1024).unwrap();

        let mut input = io::Cursor::new(chunked_wire.clone());
        let mut decoded = 0;
        let mut frames = 0;
        while (input.position() as usize) < chunked_wire.len() {
            let Some(Reply::CompletionBatchFp16 { completions, .. }) = read_reply_with_fp16_flags(
                &mut input,
                &AtomicBool::new(true),
                &AtomicBool::new(false),
                &AtomicBool::new(true),
                &AtomicU64::new(0),
            )
            .unwrap() else {
                panic!("not a decoded FP16 batch")
            };
            decoded += completions.len();
            frames += 1;
        }
        assert_eq!(decoded, rows.len());
        assert!(frames > 1);
    }
}

#[cfg(test)]
mod a64_engram_reply_tests {
    use super::*;

    fn fixture() -> (moe::Stamp, TableIdentity, Vec<u8>, Vec<u8>) {
        let stamp = moe::Stamp {
            run_id: "a64-wire-test".into(),
            model_revision: "rev".into(),
            plan_id: "p".into(),
            layer: 2,
            batch: 5,
            epoch: 3,
        };
        let table = TableIdentity {
            model_revision: "rev".into(),
            table_sha256: "11".repeat(32),
            tokenizer_map_sha256: "22".repeat(32),
            layer: 2,
            epoch: 3,
            num_embeddings: 1,
        };
        let shard_bytes = (0..264).map(|i| (i % 251) as u8).collect::<Vec<_>>();
        let shard = EngramShard::new(table.clone(), 1, 0, shard_bytes.clone()).unwrap();
        let rows = shard.rows(&stamp, &[0]).unwrap();
        let binding = engram::RequestBinding::new(&stamp, &table, 1, &[0]).unwrap();
        let owner_frame = engram::encode_owner_frame(&rows, &binding, 0).unwrap();
        (stamp, table, shard_bytes, owner_frame)
    }

    fn wire(attempt: u64, owner_frame: &[u8]) -> Vec<u8> {
        let mut bytes = encode(&Reply::EngramRowsRaw {
            attempt,
            frame_bytes: owner_frame.len() as u64,
            frame: Vec::new(),
        })
        .unwrap();
        bytes.extend_from_slice(owner_frame);
        bytes
    }

    #[test]
    fn owner_reply_reads_exact_raw_body_and_binds_attempt_and_request() {
        let (stamp, table, _, owner_frame) = fixture();
        let allowed = AtomicU64::new(17);
        let reply = read_reply_with_engram(
            &mut io::Cursor::new(wire(17, &owner_frame)),
            &AtomicBool::new(false),
            &allowed,
        )
        .unwrap();
        let Some(Reply::EngramRowsRaw {
            frame, frame_bytes, ..
        }) = reply
        else {
            panic!("not an Engram reply")
        };
        assert_eq!(frame_bytes as usize, frame.len());
        let binding = engram::RequestBinding::new(&stamp, &table, 1, &[0]).unwrap();
        let decoded = engram::decode_owner_frame(&frame, &binding, 0).unwrap();
        assert_eq!(decoded.payload.len(), 264);
    }

    #[test]
    fn owner_reply_refuses_unsolicited_stale_truncated_and_oversized_frames() {
        let (_, _, _, owner_frame) = fixture();
        let unsolicited = read_reply_with_engram(
            &mut io::Cursor::new(wire(17, &owner_frame)),
            &AtomicBool::new(false),
            &AtomicU64::new(0),
        );
        assert_eq!(unsolicited.unwrap_err().code, "WORKER_PROTOCOL");

        let stale = read_reply_with_engram(
            &mut io::Cursor::new(wire(17, &owner_frame)),
            &AtomicBool::new(false),
            &AtomicU64::new(18),
        );
        assert_eq!(stale.unwrap_err().code, "WORKER_PROTOCOL");

        let mut truncated = wire(17, &owner_frame);
        truncated.pop();
        let truncated = read_reply_with_engram(
            &mut io::Cursor::new(truncated),
            &AtomicBool::new(false),
            &AtomicU64::new(17),
        );
        assert_eq!(truncated.unwrap_err().code, "WORKER_PROTOCOL");

        let oversized = encode(&Reply::EngramRowsRaw {
            attempt: 17,
            frame_bytes: u64::MAX,
            frame: Vec::new(),
        })
        .unwrap();
        let oversized = read_reply_with_engram(
            &mut io::Cursor::new(oversized),
            &AtomicBool::new(false),
            &AtomicU64::new(17),
        );
        assert_eq!(oversized.unwrap_err().code, "MESSAGE_TOO_LARGE");
    }
}

#[cfg(test)]
mod a78_engram_row_ids_tests {
    use super::{
        decode_engram_row_ids_varint, encode_engram_row_ids_varint, encode_engram_rows_request,
        read_engram_row_ids_body, Request,
    };
    use crate::engram::{self, TableIdentity};
    use crate::moe::Stamp;
    use std::io::Cursor;

    const PROFILE_ROW_BYTES: usize = 264;

    fn identity() -> (Stamp, TableIdentity) {
        let table = TableIdentity {
            model_revision: "a78-test-v1".into(),
            table_sha256: "11".repeat(32),
            tokenizer_map_sha256: "22".repeat(32),
            layer: 3,
            epoch: 9,
            num_embeddings: 256,
        };
        let stamp = Stamp {
            run_id: "a78-test".into(),
            model_revision: table.model_revision.clone(),
            plan_id: "wire-test".into(),
            layer: table.layer,
            batch: 4,
            epoch: table.epoch,
        };
        (stamp, table)
    }

    fn take_reply(bytes: &[u8], offset: &mut usize) -> super::Reply {
        let end = bytes[*offset..]
            .iter()
            .position(|byte| *byte == b'\n')
            .unwrap()
            + *offset
            + 1;
        let reply = serde_json::from_slice(&bytes[*offset..end]).unwrap();
        *offset = end;
        reply
    }

    #[test]
    fn row_id_varints_round_trip_u64_boundaries_and_order() {
        let ids = [0, 1, 127, 128, 16_383, 16_384, u32::MAX as u64, u64::MAX];
        let encoded = encode_engram_row_ids_varint(&ids).unwrap();
        assert_eq!(
            decode_engram_row_ids_varint(&encoded, ids.len()).unwrap(),
            ids
        );
    }

    #[test]
    fn row_id_varints_refuse_truncation_overflow_overlong_and_count_mismatch() {
        assert!(decode_engram_row_ids_varint(&[0x80], 1).is_err());
        assert!(decode_engram_row_ids_varint(&[0x80, 0x00], 1).is_err());
        assert!(decode_engram_row_ids_varint(
            &[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x02],
            1
        )
        .is_err());
        assert!(decode_engram_row_ids_varint(&[0x01, 0x02], 1).is_err());
        assert!(decode_engram_row_ids_varint(&[0x01], 2).is_err());
    }

    #[test]
    fn row_id_varint_reader_accepts_exact_limit_and_rejects_one_over_before_read() {
        let ids = vec![u64::MAX; engram::MAX_LOOKUPS];
        let body = encode_engram_row_ids_varint(&ids).unwrap();
        assert_eq!(body.len(), super::MAX_ENGRAM_ROW_ID_BODY_BYTES);
        assert_eq!(decode_engram_row_ids_varint(&body, ids.len()).unwrap(), ids);

        let too_many = vec![0_u64; engram::MAX_LOOKUPS + 1];
        assert!(encode_engram_row_ids_varint(&too_many).is_err());
        assert!(decode_engram_row_ids_varint(&[0], engram::MAX_LOOKUPS + 1).is_err());
        let mut unread = Cursor::new(vec![0_u8]);
        assert!(read_engram_row_ids_body(
            &mut unread,
            64,
            (engram::MAX_LOOKUPS + 1) as u64,
            (engram::MAX_LOOKUPS + 1) as u64,
        )
        .is_err());
        assert_eq!(unread.position(), 0, "reject count before reading the body");
    }

    #[test]
    fn request_builder_negotiates_raw_varints_and_retains_legacy_json() {
        let (stamp, table) = identity();
        let row_ids = [0_u64, 127, 128, 255];
        assert!(super::hello(0)
            .capabilities
            .iter()
            .any(|capability| capability == super::CAPABILITY_ENGRAM_ROW_IDS_VARINT));
        let (legacy, no_body) =
            encode_engram_rows_request(&stamp, &table, 1, 0, &row_ids, 1, false).unwrap();
        assert!(no_body.is_none());
        match serde_json::from_slice::<Request>(&legacy).unwrap() {
            Request::FetchEngramRows {
                row_ids: actual, ..
            } => assert_eq!(actual, row_ids),
            other => panic!("unexpected legacy request: {other:?}"),
        }

        let varint_row_ids: Vec<u64> = (0..256).collect();
        let (header, body) =
            encode_engram_rows_request(&stamp, &table, 1, 0, &varint_row_ids, 2, true).unwrap();
        let body = body.unwrap();
        match serde_json::from_slice::<Request>(&header).unwrap() {
            Request::FetchEngramRowsVarint {
                row_id_count,
                row_id_bytes,
                ..
            } => {
                assert_eq!(row_id_count, varint_row_ids.len() as u64);
                assert_eq!(row_id_bytes, body.len() as u64);
                assert_eq!(
                    decode_engram_row_ids_varint(&body, varint_row_ids.len()).unwrap(),
                    varint_row_ids
                );
            }
            other => panic!("unexpected varint request: {other:?}"),
        }
    }

    #[test]
    fn negotiated_row_id_format_never_sends_a_larger_request() {
        let (stamp, table) = identity();
        let actual_varint_len = |row_ids: &[u64], attempt| {
            let body = encode_engram_row_ids_varint(row_ids).unwrap();
            let header = super::encode(&Request::FetchEngramRowsVarint {
                stamp: stamp.clone(),
                table: table.clone(),
                world_size: 2,
                owner_rank: 1,
                row_id_count: row_ids.len() as u64,
                row_id_bytes: body.len() as u64,
                attempt,
            })
            .unwrap();
            header.len() + body.len()
        };
        for row_ids in [vec![0_u64], vec![0, 1], vec![0, 127, 128, 255]] {
            let (legacy, _) =
                encode_engram_rows_request(&stamp, &table, 2, 1, &row_ids, 1, false).unwrap();
            let (selected, body) =
                encode_engram_rows_request(&stamp, &table, 2, 1, &row_ids, 1, true).unwrap();
            let selected_bytes = selected.len() + body.as_ref().map_or(0, Vec::len);
            assert!(
                selected_bytes <= legacy.len(),
                "negotiated request grew from {} to {} bytes for {} IDs",
                legacy.len(),
                selected_bytes,
                row_ids.len()
            );
            assert_eq!(
                selected_bytes,
                legacy.len().min(actual_varint_len(&row_ids, 1)),
                "selector must choose the exact shorter JSON or varint representation"
            );
            if let Some(body) = body {
                match serde_json::from_slice::<Request>(&selected).unwrap() {
                    Request::FetchEngramRowsVarint {
                        row_id_count,
                        row_id_bytes,
                        ..
                    } => {
                        assert_eq!(row_id_count as usize, row_ids.len());
                        assert_eq!(row_id_bytes as usize, body.len());
                        assert_eq!(
                            decode_engram_row_ids_varint(&body, row_ids.len()).unwrap(),
                            row_ids
                        );
                    }
                    other => panic!("expected negotiated varint request: {other:?}"),
                }
            } else {
                match serde_json::from_slice::<Request>(&selected).unwrap() {
                    Request::FetchEngramRows {
                        row_ids: actual, ..
                    } => {
                        assert_eq!(actual, row_ids)
                    }
                    other => panic!("expected legacy JSON request for small batch: {other:?}"),
                }
            }
        }

        let large_ids: Vec<u64> = (0..256).collect();
        let (legacy, _) =
            encode_engram_rows_request(&stamp, &table, 2, 1, &large_ids, 2, false).unwrap();
        let (selected, body) =
            encode_engram_rows_request(&stamp, &table, 2, 1, &large_ids, 2, true).unwrap();
        let body = body.expect("large batches should retain the compact varint path");
        assert!(selected.len() + body.len() < legacy.len());
        assert_eq!(
            decode_engram_row_ids_varint(&body, large_ids.len()).unwrap(),
            large_ids
        );

        let largest_ids = vec![u64::MAX; 256];
        let (legacy, _) =
            encode_engram_rows_request(&stamp, &table, 2, 1, &largest_ids, 3, false).unwrap();
        let (selected, body) =
            encode_engram_rows_request(&stamp, &table, 2, 1, &largest_ids, 3, true).unwrap();
        let body = body.expect("large decimal IDs should use the shorter varint form");
        assert_eq!(
            selected.len() + body.len(),
            legacy.len().min(actual_varint_len(&largest_ids, 3))
        );
        assert_eq!(
            decode_engram_row_ids_varint(&body, largest_ids.len()).unwrap(),
            largest_ids
        );
    }

    #[test]
    fn raw_row_id_reader_checks_declared_bounds_and_truncated_bodies() {
        let mut oversized = Cursor::new(vec![0x01]);
        assert!(read_engram_row_ids_body(
            &mut oversized,
            64,
            (super::MAX_ENGRAM_ROW_ID_BODY_BYTES as u64) + 1,
            1,
        )
        .is_err());
        assert_eq!(
            oversized.position(),
            0,
            "reject the length before reading a body"
        );

        let mut truncated = Cursor::new(vec![0x80]);
        assert!(read_engram_row_ids_body(&mut truncated, 64, 2, 1).is_err());
    }

    #[test]
    fn worker_decodes_raw_ids_before_dispatch_and_closes_on_truncated_body() {
        let (stamp, table) = identity();
        let hello = super::encode(&Request::Hello {
            protocol: super::PROTOCOL,
            rank: 0,
        })
        .unwrap();
        let row_ids: Vec<u64> = (0..256).collect();
        let (header, body) =
            encode_engram_rows_request(&stamp, &table, 1, 0, &row_ids, 1, true).unwrap();
        let mut input = hello.clone();
        input.extend_from_slice(&header);
        input.extend_from_slice(body.as_deref().unwrap());
        let mut output = Vec::new();
        let error = super::serve(&mut Cursor::new(input), &mut output).unwrap_err();
        assert_eq!(error.code, "SESSION_ABANDONED");
        let mut offset = 0;
        assert!(matches!(
            take_reply(&output, &mut offset),
            super::Reply::Hello { .. }
        ));
        match take_reply(&output, &mut offset) {
            super::Reply::Refused { error } => assert_eq!(error.code, "ENGRAM_NOT_LOADED"),
            other => panic!("unexpected reply after decoded varints: {other:?}"),
        }
        assert_eq!(offset, output.len());

        let truncated_header = super::encode(&Request::FetchEngramRowsVarint {
            stamp,
            table,
            world_size: 1,
            owner_rank: 0,
            row_id_count: 1,
            row_id_bytes: 2,
            attempt: 2,
        })
        .unwrap();
        let mut truncated_input = hello;
        truncated_input.extend_from_slice(&truncated_header);
        truncated_input.push(0x80);
        let mut truncated_output = Vec::new();
        let error =
            super::serve(&mut Cursor::new(truncated_input), &mut truncated_output).unwrap_err();
        assert_eq!(error.code, "ARTIFACT_TRUNCATED");
        let mut offset = 0;
        assert!(matches!(
            take_reply(&truncated_output, &mut offset),
            super::Reply::Hello { .. }
        ));
        match take_reply(&truncated_output, &mut offset) {
            super::Reply::Refused { error } => assert_eq!(error.code, "ARTIFACT_TRUNCATED"),
            other => panic!("unexpected truncated-body reply: {other:?}"),
        }
        assert_eq!(offset, truncated_output.len());
    }

    fn profile_ids(requested: usize, duplicate_percent: usize) -> Vec<u64> {
        let unique = requested * (100 - duplicate_percent) / 100;
        (0..requested)
            .map(|index| {
                let source = if index < unique {
                    index
                } else {
                    index % unique
                };
                ((source * 257 + 19) % 65_536) as u64
            })
            .collect()
    }

    fn profile_json_request(
        stamp: &Stamp,
        table: &TableIdentity,
        world: u32,
        partitions: &[Vec<u64>],
    ) -> (Vec<Vec<u64>>, usize) {
        let mut rows = Vec::new();
        let mut bytes = 0;
        for (rank, ids) in partitions.iter().enumerate() {
            if ids.is_empty() {
                continue;
            }
            let (header, body) =
                encode_engram_rows_request(stamp, table, world, rank as u32, ids, 1, false)
                    .unwrap();
            assert!(body.is_none());
            bytes += header.len();
            match serde_json::from_slice::<Request>(&header).unwrap() {
                Request::FetchEngramRows { row_ids, .. } => rows.push(row_ids),
                other => panic!("unexpected JSON profile request: {other:?}"),
            }
        }
        (rows, bytes)
    }

    fn profile_varint_request(
        stamp: &Stamp,
        table: &TableIdentity,
        world: u32,
        partitions: &[Vec<u64>],
    ) -> (Vec<Vec<u64>>, usize) {
        let mut rows = Vec::new();
        let mut bytes = 0;
        for (rank, ids) in partitions.iter().enumerate() {
            if ids.is_empty() {
                continue;
            }
            let (header, body) =
                encode_engram_rows_request(stamp, table, world, rank as u32, ids, 1, true).unwrap();
            let body = body.unwrap();
            bytes += header.len() + body.len();
            let Request::FetchEngramRowsVarint {
                row_id_count,
                row_id_bytes,
                ..
            } = serde_json::from_slice::<Request>(&header).unwrap()
            else {
                panic!("unexpected varint profile request")
            };
            let mut body_reader = Cursor::new(body);
            rows.push(
                read_engram_row_ids_body(
                    &mut body_reader,
                    header.len() - 1,
                    row_id_bytes,
                    row_id_count,
                )
                .unwrap(),
            );
        }
        (rows, bytes)
    }

    fn profile_median(values: &mut [u128]) -> u128 {
        values.sort_unstable();
        values[values.len() / 2]
    }

    #[test]
    #[ignore = "explicit A78 production request-codec profile"]
    fn production_varint_request_path_profile() {
        for requested in [1_024, 16_384, 60_000] {
            for duplicate_percent in [0, 25, 50] {
                for world in [2, 16] {
                    let (stamp, mut table) = identity();
                    table.num_embeddings = 65_536;
                    let stamp = Stamp {
                        model_revision: table.model_revision.clone(),
                        ..stamp
                    };
                    let ids = profile_ids(requested, duplicate_percent);
                    let partitions = engram::partition_unique_row_ids(&table, world, &ids).unwrap();
                    let expected_count: usize = partitions.iter().map(Vec::len).sum();
                    let expected_payload_bytes = expected_count * PROFILE_ROW_BYTES;
                    let mut baseline_ns = Vec::with_capacity(7);
                    let mut varint_ns = Vec::with_capacity(7);
                    let mut baseline_bytes = 0;
                    let mut varint_bytes = 0;
                    for iteration in 0..9 {
                        let (baseline_rows, varint_rows, baseline_elapsed, varint_elapsed) =
                            if iteration % 2 == 0 {
                                let start = std::time::Instant::now();
                                let (rows, bytes) =
                                    profile_json_request(&stamp, &table, world, &partitions);
                                let baseline_elapsed = start.elapsed().as_nanos();
                                let start = std::time::Instant::now();
                                let (varint_rows, _varint_bytes) =
                                    profile_varint_request(&stamp, &table, world, &partitions);
                                let varint_elapsed = start.elapsed().as_nanos();
                                baseline_bytes = bytes;
                                (rows, varint_rows, baseline_elapsed, varint_elapsed)
                            } else {
                                let start = std::time::Instant::now();
                                let (varint_rows, bytes) =
                                    profile_varint_request(&stamp, &table, world, &partitions);
                                let varint_elapsed = start.elapsed().as_nanos();
                                let start = std::time::Instant::now();
                                let (rows, _baseline_bytes) =
                                    profile_json_request(&stamp, &table, world, &partitions);
                                let baseline_elapsed = start.elapsed().as_nanos();
                                varint_bytes = bytes;
                                (rows, varint_rows, baseline_elapsed, varint_elapsed)
                            };
                        assert_eq!(baseline_rows, partitions);
                        assert_eq!(varint_rows, partitions);
                        if iteration >= 2 {
                            baseline_ns.push(baseline_elapsed);
                            varint_ns.push(varint_elapsed);
                        }
                    }
                    let baseline_median = profile_median(&mut baseline_ns);
                    let varint_median = profile_median(&mut varint_ns);
                    let cpu_reduction = (baseline_median as f64 - varint_median as f64) * 100.0
                        / baseline_median as f64;
                    let request_reduction =
                        (baseline_bytes - varint_bytes) as f64 * 100.0 / baseline_bytes as f64;
                    let total_reduction = (baseline_bytes - varint_bytes) as f64 * 100.0
                        / (baseline_bytes + expected_payload_bytes) as f64;
                    println!(
                        "A78_PRODUCTION {{\"requested_ids\":{requested},\"duplicate_percent\":{duplicate_percent},\"world_size\":{world},\"unique_ids\":{expected_count},\"baseline_request_bytes\":{baseline_bytes},\"varint_request_bytes\":{varint_bytes},\"request_bytes_reduction_percent\":{request_reduction:.4},\"payload_bytes\":{expected_payload_bytes},\"request_plus_payload_reduction_percent\":{total_reduction:.4},\"baseline_encode_decode_ns\":{baseline_median},\"varint_encode_decode_ns\":{varint_median},\"cpu_reduction_percent\":{cpu_reduction:.4},\"exact_ids_match\":true}}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod a155_completion_batch_latch_tests {
    use super::*;

    fn dispatch() -> routing::RankDispatch {
        let stamp = moe::Stamp {
            run_id: "a155-batch-latch".into(),
            model_revision: "test-revision".into(),
            plan_id: "test-plan".into(),
            layer: 0,
            batch: 0,
            epoch: 1,
        };
        routing::RankDispatch {
            stamp,
            rank: 2,
            hidden: 2,
            tokens: 1,
            resident_order: vec![7],
            slots: vec![routing::RankSlot {
                slot: moe::Slot {
                    token: 0,
                    slot: 0,
                    expert: 3,
                    rank: 2,
                    local: 7,
                    weight: 1.0,
                },
                physical_expert: 0,
            }],
        }
    }

    fn batch_flag(header: &[u8]) -> bool {
        match serde_json::from_slice::<Request>(header).unwrap() {
            Request::ExecRaw {
                raw_completion_batch,
                ..
            } => raw_completion_batch,
            other => panic!("expected raw EXEC, got {other:?}"),
        }
    }

    fn prepared_session(capabilities: &[&str]) -> Session {
        let endpoint = Endpoint {
            rank: 2,
            argv: vec!["/bin/cat".into()],
        };
        let mut session = Session::spawn(&endpoint).unwrap();
        session.hello.capabilities = capabilities.iter().map(|c| (*c).into()).collect();
        session.artifact = Some(("0".repeat(64), 1));
        session
    }

    fn stop(mut session: Session) {
        let _ = session.child.kill();
        let _ = session.child.wait();
        if let Some(reader) = session.reader.take() {
            let _ = reader.join();
        }
        if let Some(stderr) = session.stderr.take() {
            let _ = stderr.join();
        }
    }

    #[test]
    fn session_batch_capability_sends_one_opt_in_per_worker_session() {
        let mut session = prepared_session(&[
            CAPABILITY_EXEC_F64_RAW,
            CAPABILITY_COMPLETION_F64_RAW,
            CAPABILITY_COMPLETION_BATCH_F64_RAW,
            CAPABILITY_COMPLETION_BATCH_SESSION_F64_RAW,
        ]);
        let inputs = vec![TokenInput {
            token: 0,
            value: vec![0.25, -0.5],
        }];

        let first = session
            .encode_exec(&dispatch(), 1, inputs.clone(), Fault::None, false)
            .unwrap();
        let first_header_bytes = first.1.len();
        assert!(batch_flag(&first.1));
        assert!(session.send_exec(first).receipt.error.is_none());

        let second = session
            .encode_exec(&dispatch(), 2, inputs, Fault::None, false)
            .unwrap();
        assert!(
            !batch_flag(&second.1),
            "a session-negotiated worker must not receive the batch opt-in on every EXEC"
        );
        assert_eq!(
            first_header_bytes,
            second.1.len() + 28,
            "omitting the repeated JSON opt-in must save exactly 28 request-header bytes"
        );

        stop(session);
    }

    #[test]
    fn a154_per_exec_capability_keeps_sending_the_opt_in_each_time() {
        let session = prepared_session(&[
            CAPABILITY_EXEC_F64_RAW,
            CAPABILITY_COMPLETION_F64_RAW,
            CAPABILITY_COMPLETION_BATCH_F64_RAW,
        ]);
        let inputs = vec![TokenInput {
            token: 0,
            value: vec![0.25, -0.5],
        }];
        let first = session
            .encode_exec(&dispatch(), 1, inputs.clone(), Fault::None, false)
            .unwrap();
        let second = session
            .encode_exec(&dispatch(), 2, inputs, Fault::None, false)
            .unwrap();
        assert!(batch_flag(&first.1));
        assert!(batch_flag(&second.1));
        stop(session);
    }

    #[test]
    fn fp16_completion_encoding_requires_and_sets_the_explicit_capability() {
        let old_peer = prepared_session(&[CAPABILITY_EXEC_F64_RAW]);
        let error = match old_peer.encode_exec(
            &dispatch(),
            1,
            vec![TokenInput {
                token: 0,
                value: vec![0.25, -0.5],
            }],
            Fault::None,
            true,
        ) {
            Ok(_) => panic!("an old peer cannot accept FP16 completion mode"),
            Err(error) => error,
        };
        assert_eq!(error.code, "SESSION_CAPABILITY");
        assert_eq!(old_peer.traffic.execs, 0);
        stop(old_peer);

        let peer = prepared_session(&[
            CAPABILITY_EXEC_F64_RAW,
            CAPABILITY_COMPLETION_BATCH_FP16_RAW,
        ]);
        let encoded = peer
            .encode_exec(
                &dispatch(),
                1,
                vec![TokenInput {
                    token: 0,
                    value: vec![0.25, -0.5],
                }],
                Fault::None,
                true,
            )
            .unwrap();
        match serde_json::from_slice::<Request>(&encoded.1).unwrap() {
            Request::ExecRaw {
                raw_completions,
                raw_completion_batch,
                raw_completion_fp16_batch,
                ..
            } => {
                assert!(raw_completions);
                assert!(!raw_completion_batch);
                assert!(raw_completion_fp16_batch);
            }
            other => panic!("expected raw EXEC, got {other:?}"),
        }
        stop(peer);
    }
}

#[cfg(test)]
mod a166_fp16_mode_tests {
    use super::*;

    fn dispatch() -> routing::RankDispatch {
        let stamp = moe::Stamp {
            run_id: "a166-fp16-mode".into(),
            model_revision: "test-revision".into(),
            plan_id: "test-plan".into(),
            layer: 0,
            batch: 0,
            epoch: 1,
        };
        routing::RankDispatch {
            stamp,
            rank: 2,
            hidden: 2,
            tokens: 1,
            resident_order: vec![7],
            slots: vec![routing::RankSlot {
                slot: moe::Slot {
                    token: 0,
                    slot: 0,
                    expert: 3,
                    rank: 2,
                    local: 7,
                    weight: 1.0,
                },
                physical_expert: 0,
            }],
        }
    }

    fn prepared_session(capabilities: &[&str]) -> Session {
        let endpoint = Endpoint {
            rank: 2,
            argv: vec!["/bin/cat".into()],
        };
        let mut session = Session::spawn(&endpoint).unwrap();
        session.hello.capabilities = capabilities.iter().map(|c| (*c).into()).collect();
        session.artifact = Some(("0".repeat(64), 1));
        session
    }

    fn stop(mut session: Session) {
        let _ = session.child.kill();
        let _ = session.child.wait();
        if let Some(reader) = session.reader.take() {
            let _ = reader.join();
        }
        if let Some(stderr) = session.stderr.take() {
            let _ = stderr.join();
        }
    }

    #[test]
    fn exec_refused_reply_invalidates_closed_worker_session() {
        let dispatch = dispatch();
        let mut session = prepared_session(&[CAPABILITY_EXEC_F64_RAW]);
        let attempt = 19;
        let refusal = encode(&Reply::Refused {
            error: Error::new("INVALID_INPUT", "malformed raw EXEC body; worker closed"),
        })
        .unwrap();
        session.send_raw(&refusal).unwrap();
        // /bin/cat echoes the refusal and exits on EOF, modeling serve()'s refusal-then-close path.
        drop(session.stdin.take());

        let receipt = session.finish_exec(
            PendingExec {
                dispatch: dispatch.clone(),
                receipt: ExecReceipt {
                    rank: dispatch.rank,
                    attempt,
                    state: ReceiptState::Unknown,
                    started: false,
                    expected_slots: dispatch.slots.len(),
                    received_slots: 0,
                    request_bytes: 0,
                    error: None,
                },
                expect_raw_completion_batch: false,
                expect_raw_completion_fp16_batch: false,
                latch_raw_completion_batch_session: false,
            },
            None,
            Instant::now() + Duration::from_secs(2),
        );

        assert_eq!(receipt.state, ReceiptState::NotStarted);
        assert_eq!(receipt.error.as_ref().unwrap().code, "INVALID_INPUT");
        assert!(!session.is_usable());
        let next_request = session.send(&Request::Shutdown).unwrap_err();
        assert_eq!(next_request.code, "SESSION_UNUSABLE");
        stop(session);
    }

    #[test]
    fn fp16_exec_rejects_legacy_f64_completion_events() {
        let dispatch = dispatch();
        let routing = routing::RoutingProblem {
            hidden: dispatch.hidden,
            experts: vec![3],
            routes: vec![vec![moe::Choice {
                expert: 3,
                weight: 1.0,
            }]],
            registry: moe::Registry {
                epoch: dispatch.stamp.epoch,
                copies: vec![moe::Replica {
                    expert: 3,
                    rank: 2,
                    local: 7,
                }],
            },
            stamp: dispatch.stamp.clone(),
            preferred_owners: BTreeMap::new(),
            resident_order: BTreeMap::from([(2, vec![7])]),
        };
        let mut collector = moe::Collector::from_routing(&routing).unwrap();
        let mut session = prepared_session(&[
            CAPABILITY_EXEC_F64_RAW,
            CAPABILITY_COMPLETION_BATCH_FP16_RAW,
        ]);
        session.raw_completions.store(true, Ordering::SeqCst);
        session
            .raw_completion_fp16_batch
            .store(true, Ordering::SeqCst);

        let attempt = 17;
        let events = [
            EventKind::Started,
            EventKind::Completion {
                completion: moe::Completion {
                    stamp: dispatch.stamp.clone(),
                    token: 0,
                    slot: 0,
                    expert: 3,
                    rank: 2,
                    local: 7,
                    value: vec![1.0, -2.0],
                },
            },
            EventKind::Finished { slots: 1 },
        ];
        let mut wire = Vec::new();
        for event in events {
            wire.extend(
                encode(&Reply::Event {
                    event: WorkerEvent {
                        stamp: dispatch.stamp.clone(),
                        rank: dispatch.rank,
                        attempt,
                        event,
                    },
                })
                .unwrap(),
            );
        }
        // `/bin/cat` in prepared_session acts as an adversarial peer that emits a valid
        // f64 event sequence even though this pending EXEC explicitly requested FP16.
        session.send_raw(&wire).unwrap();

        let pending = PendingExec {
            dispatch: dispatch.clone(),
            receipt: ExecReceipt {
                rank: dispatch.rank,
                attempt,
                state: ReceiptState::Unknown,
                started: false,
                expected_slots: dispatch.slots.len(),
                received_slots: 0,
                request_bytes: 0,
                error: None,
            },
            expect_raw_completion_batch: false,
            expect_raw_completion_fp16_batch: true,
            latch_raw_completion_batch_session: false,
        };
        let receipt = session.finish_exec(
            pending,
            Some(&mut collector),
            Instant::now() + Duration::from_secs(2),
        );

        assert_eq!(receipt.state, ReceiptState::Unknown);
        assert!(receipt.started);
        assert_eq!(receipt.received_slots, 0);
        assert_eq!(receipt.error.as_ref().unwrap().code, "WORKER_PROTOCOL");
        assert!(!session.is_usable());
        assert_eq!(collector.finish().unwrap_err().code, "BATCH_INCOMPLETE");
        stop(session);
    }

    #[test]
    fn fp16_exec_rejects_legacy_f64_completion_batches() {
        let dispatch = dispatch();
        let routing = routing::RoutingProblem {
            hidden: dispatch.hidden,
            experts: vec![3],
            routes: vec![vec![moe::Choice {
                expert: 3,
                weight: 1.0,
            }]],
            registry: moe::Registry {
                epoch: dispatch.stamp.epoch,
                copies: vec![moe::Replica {
                    expert: 3,
                    rank: 2,
                    local: 7,
                }],
            },
            stamp: dispatch.stamp.clone(),
            preferred_owners: BTreeMap::new(),
            resident_order: BTreeMap::from([(2, vec![7])]),
        };
        let mut collector = moe::Collector::from_routing(&routing).unwrap();
        let mut session = prepared_session(&[
            CAPABILITY_EXEC_F64_RAW,
            CAPABILITY_COMPLETION_BATCH_FP16_RAW,
            CAPABILITY_COMPLETION_BATCH_F64_RAW,
        ]);
        session.raw_completions.store(true, Ordering::SeqCst);
        session.raw_completion_batch.store(true, Ordering::SeqCst);
        session
            .raw_completion_fp16_batch
            .store(true, Ordering::SeqCst);

        let attempt = 18;
        let mut wire = Vec::new();
        wire.extend(
            encode(&Reply::Event {
                event: WorkerEvent {
                    stamp: dispatch.stamp.clone(),
                    rank: dispatch.rank,
                    attempt,
                    event: EventKind::Started,
                },
            })
            .unwrap(),
        );
        write_raw_completion_batch(
            &mut wire,
            WorkerCompletions {
                stamp: dispatch.stamp.clone(),
                rank: dispatch.rank,
                attempt,
                rows: vec![moe::Completion {
                    stamp: dispatch.stamp.clone(),
                    token: 0,
                    slot: 0,
                    expert: 3,
                    rank: 2,
                    local: 7,
                    value: vec![1.0, -2.0],
                }],
            },
        )
        .unwrap();
        wire.extend(
            encode(&Reply::Event {
                event: WorkerEvent {
                    stamp: dispatch.stamp.clone(),
                    rank: dispatch.rank,
                    attempt,
                    event: EventKind::Finished { slots: 1 },
                },
            })
            .unwrap(),
        );
        // A prior f64 batch EXEC can leave this session parser latch enabled. The current
        // EXEC requested only FP16, so the f64 batch remains a protocol error.
        session.send_raw(&wire).unwrap();

        let pending = PendingExec {
            dispatch: dispatch.clone(),
            receipt: ExecReceipt {
                rank: dispatch.rank,
                attempt,
                state: ReceiptState::Unknown,
                started: false,
                expected_slots: dispatch.slots.len(),
                received_slots: 0,
                request_bytes: 0,
                error: None,
            },
            expect_raw_completion_batch: false,
            expect_raw_completion_fp16_batch: true,
            latch_raw_completion_batch_session: false,
        };
        let receipt = session.finish_exec(
            pending,
            Some(&mut collector),
            Instant::now() + Duration::from_secs(2),
        );

        assert_eq!(receipt.state, ReceiptState::Unknown);
        assert_eq!(receipt.received_slots, 0);
        assert_eq!(receipt.error.as_ref().unwrap().code, "WORKER_PROTOCOL");
        assert!(!session.is_usable());
        assert_eq!(collector.finish().unwrap_err().code, "BATCH_INCOMPLETE");
        stop(session);
    }
}
