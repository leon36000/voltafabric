//! Private-pipe interprocess dense CPU oracle. This is not a CUDA backend.
use crate::{invalid, moe, routing, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

pub const MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Fault {
    #[default]
    None,
    Delay {
        milliseconds: u64,
    },
    Reverse,
    Duplicate,
    Missing,
    StaleEpoch,
    OldAttempt {
        attempt: u64,
    },
    RejectBeforeCompute,
    DieBeforeStart,
    DieAfterStart,
    DieAfterCompute,
    StderrFlood {
        bytes: usize,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessProblem {
    pub problem: moe::Problem,
    #[serde(default)]
    pub faults: BTreeMap<u32, Fault>,
    pub timeout_ms: u64,
    #[serde(default)]
    pub recovery: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenInput {
    pub token: usize,
    pub value: Vec<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankRequest {
    pub dispatch: routing::RankDispatch,
    pub attempt: u64,
    pub model: moe::Model,
    pub inputs: Vec<TokenInput>,
    #[serde(default)]
    pub fault: Fault,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReceiptState {
    NotStarted,
    Started,
    Unknown,
    Completed,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub stamp: moe::Stamp,
    pub rank: u32,
    pub attempt: u64,
    pub state: ReceiptState,
    pub started: bool,
    pub pid: Option<u32>,
    pub reaped: bool,
    pub reaped_pids: Vec<u32>,
    pub children_spawned: usize,
    pub children_reaped: usize,
    pub expected_slots: usize,
    pub received_slots: usize,
    pub resident_experts: usize,
    pub local_inputs: usize,
    pub request_bytes: usize,
    pub stderr_bytes: usize,
    pub exit_code: Option<i32>,
    pub error: Option<Error>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessReport {
    pub ok: bool,
    pub output: Option<Vec<Vec<f64>>>,
    pub receipts: Vec<Receipt>,
    pub attempt: u64,
    pub error: Option<Error>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EventKind {
    Started,
    Rejected { error: Error },
    Failed { error: Error },
    Completion { completion: moe::Completion },
    Finished { slots: usize },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerEvent {
    pub stamp: moe::Stamp,
    pub rank: u32,
    pub attempt: u64,
    pub event: EventKind,
}

use std::collections::{BTreeSet, HashSet};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, LazyLock, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

static NEXT_ATTEMPT: AtomicU64 = AtomicU64::new(1);
// Coordinator-local replay guard, not a durable/distributed execution guarantee.
static ATTEMPTED: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(|| Mutex::new(HashSet::new()));
/// Identities the guard keeps per coordinator process; a new one beyond them is refused.
const MAX_ATTEMPTED: usize = 65536;

fn io_error(context: &str, error: impl std::fmt::Display) -> Error {
    Error::new("WORKER_IO", format!("{context}: {error}"))
}
struct BoundedBuffer {
    bytes: Vec<u8>,
    overflow: bool,
}
impl Write for BoundedBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.bytes.len().saturating_add(bytes.len()) > MAX_MESSAGE_BYTES {
            self.overflow = true;
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "JSON message exceeds 16 MiB",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub(crate) fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut output = BoundedBuffer {
        bytes: Vec::new(),
        overflow: false,
    };
    if let Err(error) = serde_json::to_writer(&mut output, value) {
        return Err(if output.overflow {
            Error::new(
                "MESSAGE_TOO_LARGE",
                "private-pipe JSON message exceeds 16 MiB",
            )
        } else {
            Error::new("WORKER_PROTOCOL", error.to_string())
        });
    }
    output.bytes.push(b'\n');
    Ok(output.bytes)
}
pub(crate) fn read_message<T: serde::de::DeserializeOwned>(
    reader: &mut impl BufRead,
) -> Result<Option<T>> {
    Ok(read_message_with_len(reader)?.map(|(message, _)| message))
}
/// Read one bounded JSON frame and return its payload length (without the newline).
/// Raw-body protocol variants use this to keep header plus body inside the frame bound.
pub(crate) fn read_message_with_len<T: serde::de::DeserializeOwned>(
    reader: &mut impl BufRead,
) -> Result<Option<(T, usize)>> {
    let mut bytes = Vec::new();
    loop {
        let buffered = reader.fill_buf().map_err(|e| io_error("read pipe", e))?;
        if buffered.is_empty() {
            return if bytes.is_empty() {
                Ok(None)
            } else {
                Err(Error::new("WORKER_PROTOCOL", "unterminated JSON frame"))
            };
        }
        let newline = buffered.iter().position(|&b| b == b'\n');
        let count = newline.map_or(buffered.len(), |n| n + 1);
        if bytes.len().saturating_add(count) > MAX_MESSAGE_BYTES + 1 {
            return Err(Error::new(
                "MESSAGE_TOO_LARGE",
                "private-pipe JSON message exceeds 16 MiB",
            ));
        }
        bytes.extend_from_slice(&buffered[..count]);
        reader.consume(count);
        if newline.is_some() {
            bytes.pop();
            if bytes.len() > MAX_MESSAGE_BYTES {
                return Err(Error::new(
                    "MESSAGE_TOO_LARGE",
                    "private-pipe JSON message exceeds 16 MiB",
                ));
            }
            let length = bytes.len();
            let message = serde_json::from_slice(&bytes)
                .map_err(|e| Error::new("WORKER_PROTOCOL", e.to_string()))?;
            return Ok(Some((message, length)));
        }
    }
}

/// Build a private rank payload in explicitly declared physical order.
/// Unused resident experts are allowed; nonresident experts and unrelated inputs are absent.
pub fn rank_request(
    p: &moe::Problem,
    rank: &routing::RankDispatch,
    attempt: u64,
    fault: Fault,
) -> Result<RankRequest> {
    let expected: BTreeMap<_, _> = moe::dispatch(p)?
        .into_iter()
        .filter(|s| s.rank == rank.rank)
        .map(|s| ((s.token, s.slot), s))
        .collect();
    if attempt == 0
        || rank.stamp != p.stamp
        || rank.hidden != p.model.hidden
        || rank.tokens != p.batch.inputs.len()
        || expected.len() != rank.slots.len()
        || rank
            .slots
            .iter()
            .any(|s| expected.get(&(s.slot.token, s.slot.slot)) != Some(&s.slot))
        || rank
            .slots
            .iter()
            .map(|s| (s.slot.token, s.slot.slot))
            .collect::<BTreeSet<_>>()
            .len()
            != rank.slots.len()
    {
        return Err(invalid("rank dispatch does not match the dense oracle"));
    }
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
    if rank.slots.iter().any(|routed| {
        rank.resident_order.get(routed.physical_expert) != Some(&routed.slot.local)
            || experts
                .get(routed.physical_expert)
                .is_none_or(|e| e.id != routed.slot.expert)
    }) {
        return Err(invalid("rank slot does not match its physical resident"));
    }
    let tokens: BTreeSet<_> = rank.slots.iter().map(|s| s.slot.token).collect();
    Ok(RankRequest {
        dispatch: rank.clone(),
        attempt,
        model: moe::Model {
            hidden: p.model.hidden,
            intermediate: p.model.intermediate,
            experts,
        },
        inputs: tokens
            .into_iter()
            .map(|token| TokenInput {
                token,
                value: p.batch.inputs[token].clone(),
            })
            .collect(),
        fault,
    })
}
pub(crate) fn check_fault(fault: &Fault) -> Result<()> {
    if matches!(fault, Fault::Delay { milliseconds } if *milliseconds > 60_000)
        || matches!(fault, Fault::StderrFlood { bytes } if *bytes > MAX_MESSAGE_BYTES)
    {
        return Err(invalid("fault injection outside bounded CPU test contract"));
    }
    Ok(())
}
/// The compacted activations of one rank dispatch: finite rows of the dispatch hidden
/// size, one per token, covering exactly the dispatched tokens.
pub(crate) fn token_inputs(
    d: &routing::RankDispatch,
    rows: Vec<TokenInput>,
) -> Result<BTreeMap<usize, Vec<f64>>> {
    // `inputs` keeps the name scripts/rust_mutations.py (R08) anchors on.
    let mut inputs = BTreeMap::new();
    for input in rows {
        if input.token >= d.tokens
            || input.value.len() != d.hidden
            || input.value.iter().any(|x| !x.is_finite())
            || inputs.insert(input.token, input.value).is_some()
        {
            return Err(invalid("invalid or duplicate compacted input"));
        }
    }
    let tokens: BTreeSet<_> = d.slots.iter().map(|s| s.slot.token).collect();
    if tokens != inputs.keys().copied().collect() {
        return Err(invalid(
            "local inputs must cover exactly the dispatched tokens",
        ));
    }
    Ok(inputs)
}
pub(crate) struct LocalWork {
    dispatch: routing::RankDispatch,
    fault: Fault,
    problem: moe::Problem,
    experts: Vec<moe::Expert>,
    inputs: BTreeMap<usize, Vec<f64>>,
}
pub(crate) fn local_work(request: RankRequest) -> Result<LocalWork> {
    let d = &request.dispatch;
    if request.attempt == 0
        || d.slots.is_empty()
        || d.slots.len() > 262144
        || d.tokens > 4096
        || d.hidden != request.model.hidden
        || d.resident_order.len() != request.model.experts.len()
        || d.resident_order
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != d.resident_order.len()
    {
        return Err(invalid("invalid rank-local dimensions or physical order"));
    }
    let n = request
        .model
        .hidden
        .checked_mul(request.model.intermediate)
        .and_then(|n| n.checked_mul(d.slots.len()))
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| invalid("rank CPU work overflow"))?;
    if n > 100_000_000 {
        return Err(invalid("rank CPU work budget exceeded"));
    }
    check_fault(&request.fault)?;
    let inputs = token_inputs(d, request.inputs)?;
    let mut slots = BTreeSet::new();
    for routed in &d.slots {
        let slot = &routed.slot;
        if slot.rank != d.rank
            || slot.token >= d.tokens
            || slot.slot >= 64
            || !slot.weight.is_finite()
            || slot.weight < 0.0
            || !slots.insert((slot.token, slot.slot))
            || d.resident_order.get(routed.physical_expert) != Some(&slot.local)
            || request
                .model
                .experts
                .get(routed.physical_expert)
                .is_none_or(|e| e.id != slot.expert)
        {
            return Err(invalid("rank slot has invalid identity or physical expert"));
        }
    }
    let first = &d.slots[0].slot;
    let registry = moe::Registry {
        epoch: d.stamp.epoch,
        copies: request
            .model
            .experts
            .iter()
            .zip(&d.resident_order)
            .map(|(expert, &local)| moe::Replica {
                expert: expert.id,
                rank: d.rank,
                local,
            })
            .collect(),
    };
    let mut problem = moe::Problem {
        model: request.model,
        batch: moe::Batch {
            inputs: vec![inputs[&first.token].clone()],
            routes: vec![vec![moe::Choice {
                expert: first.expert,
                weight: first.weight,
            }]],
        },
        registry,
        stamp: d.stamp.clone(),
        preferred_owners: BTreeMap::new(),
    };
    moe::validate(&problem)?;
    // Validate residents once; subsequent compute_slot validation sees only the
    // active expert, so CPU validation remains linear in dispatched work.
    let experts = std::mem::take(&mut problem.model.experts);
    problem
        .model
        .experts
        .push(experts[d.slots[0].physical_expert].clone());
    problem
        .registry
        .copies
        .retain(|replica| replica.expert == first.expert);
    Ok(LocalWork {
        dispatch: request.dispatch,
        fault: request.fault,
        problem,
        experts,
        inputs,
    })
}
/// Where a worker's events go. The single-request worker writes bare events; a session
/// wraps them in replies and may batch validated rows when the peer supports it.
pub(crate) trait EventSink {
    fn send(&mut self, event: &WorkerEvent) -> Result<()>;

    /// Send validated completion rows from one rank execution. The default preserves the
    /// legacy one-event-per-slot stream; session transports may negotiate a compact batch.
    fn send_completions(&mut self, batch: WorkerCompletions) -> Result<()> {
        for completion in batch.rows {
            self.send(&WorkerEvent {
                stamp: batch.stamp.clone(),
                rank: batch.rank,
                attempt: batch.attempt,
                event: EventKind::Completion { completion },
            })?;
        }
        Ok(())
    }
}

pub(crate) struct WorkerCompletions {
    pub(crate) stamp: moe::Stamp,
    pub(crate) rank: u32,
    pub(crate) attempt: u64,
    pub(crate) rows: Vec<moe::Completion>,
}
/// Bare WorkerEvent frames, as the single-request worker writes them.
pub(crate) struct PlainEvents<W: Write>(pub W);
impl<W: Write> EventSink for PlainEvents<W> {
    fn send(&mut self, event: &WorkerEvent) -> Result<()> {
        write_frame(&mut self.0, &encode(event)?, "worker event")
    }
}
pub(crate) fn write_frame(output: &mut impl Write, bytes: &[u8], what: &str) -> Result<()> {
    output
        .write_all(bytes)
        .map_err(|e| io_error(&format!("write {what}"), e))?;
    output
        .flush()
        .map_err(|e| io_error(&format!("flush {what}"), e))
}
fn emit(
    output: &mut impl EventSink,
    stamp: &moe::Stamp,
    rank: u32,
    attempt: u64,
    event: EventKind,
) -> Result<()> {
    output.send(&WorkerEvent {
        stamp: stamp.clone(),
        rank,
        attempt,
        event,
    })
}

/// One bounded request on stdin, newline-delimited events on stdout; stderr is separate.
pub fn worker_main() -> Result<()> {
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let request: RankRequest =
        read_message(&mut input)?.ok_or_else(|| invalid("missing rank request"))?;
    let stamp = request.dispatch.stamp.clone();
    let rank = request.dispatch.rank;
    let attempt = request.attempt;
    let stdout = io::stdout();
    let mut output = PlainEvents(stdout.lock());
    if !input
        .fill_buf()
        .map_err(|e| io_error("request EOF", e))?
        .is_empty()
    {
        emit(
            &mut output,
            &stamp,
            rank,
            attempt,
            EventKind::Rejected {
                error: invalid("worker accepts one request only"),
            },
        )?;
        return Ok(());
    }
    let work = match local_work(request) {
        Ok(work) => work,
        Err(error) => {
            emit(
                &mut output,
                &stamp,
                rank,
                attempt,
                EventKind::Rejected { error },
            )?;
            return Ok(());
        }
    };
    execute(work, &stamp, rank, attempt, &mut output)
}

/// Compute one validated rank request and stream its events: Started, one unweighted
/// Completion per slot, then Finished (or Rejected/Failed). Injected faults are honoured
/// here, including the process exits that model a crash before or after STARTED.
pub(crate) fn execute(
    work: LocalWork,
    stamp: &moe::Stamp,
    rank: u32,
    attempt: u64,
    output: &mut impl EventSink,
) -> Result<()> {
    let LocalWork {
        dispatch,
        fault,
        mut problem,
        experts,
        inputs,
    } = work;
    let job = Job {
        dispatch,
        fault,
        inputs,
    };
    execute_with(job, stamp, rank, attempt, output, |routed, input| {
        let original = &routed.slot;
        if problem.model.experts[0].id != original.expert {
            problem.model.experts[0] = experts[routed.physical_expert].clone();
            problem.registry.copies[0] = moe::Replica {
                expert: original.expert,
                rank: original.rank,
                local: original.local,
            };
        }
        problem.batch.inputs[0] = input.to_vec();
        problem.batch.routes[0][0] = moe::Choice {
            expert: original.expert,
            weight: original.weight,
        };
        let local_slot = moe::Slot {
            token: 0,
            slot: 0,
            ..original.clone()
        };
        moe::compute_slot(&problem, &local_slot).map(|c| c.value)
    })
}

/// A validated rank dispatch with its compacted activations and injected fault.
pub(crate) struct Job {
    pub(crate) dispatch: routing::RankDispatch,
    pub(crate) fault: Fault,
    pub(crate) inputs: BTreeMap<usize, Vec<f64>>,
}

/// The event lifecycle of one execution, whatever computes a slot's unweighted value:
/// the dense oracle of the single-request worker or a session's compressed residents.
pub(crate) fn execute_with(
    mut job: Job,
    stamp: &moe::Stamp,
    rank: u32,
    attempt: u64,
    output: &mut impl EventSink,
    mut compute: impl FnMut(&routing::RankSlot, &[f64]) -> Result<Vec<f64>>,
) -> Result<()> {
    if matches!(job.fault, Fault::RejectBeforeCompute) {
        emit(
            output,
            stamp,
            rank,
            attempt,
            EventKind::Rejected {
                error: Error::new("WORKER_REJECTED", "injected rejection before computation"),
            },
        )?;
        return Ok(());
    }
    if matches!(job.fault, Fault::DieBeforeStart) {
        std::process::exit(71);
    }
    emit(output, stamp, rank, attempt, EventKind::Started)?;
    if matches!(job.fault, Fault::DieAfterStart) {
        std::process::exit(72);
    }
    if let Fault::Delay { milliseconds } = job.fault {
        thread::sleep(Duration::from_millis(milliseconds));
    }
    if let Fault::StderrFlood { bytes } = job.fault {
        let stderr = io::stderr();
        let mut stderr = stderr.lock();
        let buffer = [b'x'; 8192];
        let mut remaining = bytes;
        while remaining != 0 {
            let count = remaining.min(buffer.len());
            stderr
                .write_all(&buffer[..count])
                .map_err(|e| io_error("stderr fault", e))?;
            remaining -= count;
        }
        stderr.flush().map_err(|e| io_error("stderr flush", e))?;
    }
    if matches!(job.fault, Fault::Reverse) {
        job.dispatch.slots.reverse();
    }
    let count = job.dispatch.slots.len();
    for (index, routed) in job.dispatch.slots.iter().enumerate() {
        let original = &routed.slot;
        let value = match compute(routed, &job.inputs[&original.token]) {
            Ok(value) => value,
            Err(error) => {
                emit(output, stamp, rank, attempt, EventKind::Failed { error })?;
                return Ok(());
            }
        };
        let mut completion = moe::Completion {
            stamp: job.dispatch.stamp.clone(),
            token: original.token,
            slot: original.slot,
            expert: original.expert,
            rank: original.rank,
            local: original.local,
            value,
        };
        if matches!(job.fault, Fault::DieAfterCompute)
            || (matches!(job.fault, Fault::Missing) && index + 1 == count)
        {
            continue;
        }
        if matches!(job.fault, Fault::StaleEpoch) {
            completion.stamp.epoch = if completion.stamp.epoch == 0 {
                1
            } else {
                completion.stamp.epoch - 1
            };
        }
        let event_attempt = if let Fault::OldAttempt {
            attempt: old_attempt,
        } = &job.fault
        {
            *old_attempt
        } else {
            attempt
        };
        if matches!(job.fault, Fault::Duplicate) && index == 0 {
            emit(
                output,
                stamp,
                rank,
                event_attempt,
                EventKind::Completion {
                    completion: completion.clone(),
                },
            )?;
        }
        emit(
            output,
            stamp,
            rank,
            event_attempt,
            EventKind::Completion { completion },
        )?;
    }
    if matches!(job.fault, Fault::DieAfterCompute) {
        std::process::exit(73);
    }
    emit(
        output,
        stamp,
        rank,
        attempt,
        EventKind::Finished { slots: count },
    )
}

/// Execute a rank as one compute batch. The callback sees the complete dispatch and
/// compact inputs once, and must return one unweighted hidden row for every slot in
/// dispatch order. All rows are validated before the first completion is published;
/// a compute or shape failure therefore cannot expose a partial batch. Pipe failures
/// during publication remain potentially partial and are handled as UNKNOWN upstream.
pub(crate) fn execute_batch_with(
    mut job: Job,
    stamp: &moe::Stamp,
    rank: u32,
    attempt: u64,
    output: &mut impl EventSink,
    compute: impl FnOnce(&routing::RankDispatch, &BTreeMap<usize, Vec<f64>>) -> Result<Vec<Vec<f64>>>,
) -> Result<()> {
    let output_values = job.dispatch.slots.len().checked_mul(job.dispatch.hidden);
    if job.dispatch.slots.is_empty()
        || job.dispatch.hidden == 0
        || output_values.is_none_or(|n| n > crate::completion::MAX_OUTPUT_VALUES)
    {
        emit(
            output,
            stamp,
            rank,
            attempt,
            EventKind::Rejected {
                error: Error::new(
                    "BATCH_OUTPUT_TOO_LARGE",
                    "batch output exceeds the collector's 16777216-value bound",
                ),
            },
        )?;
        return Ok(());
    }
    if matches!(job.fault, Fault::RejectBeforeCompute) {
        emit(
            output,
            stamp,
            rank,
            attempt,
            EventKind::Rejected {
                error: Error::new("WORKER_REJECTED", "injected rejection before computation"),
            },
        )?;
        return Ok(());
    }
    if matches!(job.fault, Fault::DieBeforeStart) {
        std::process::exit(71);
    }
    emit(output, stamp, rank, attempt, EventKind::Started)?;
    if matches!(job.fault, Fault::DieAfterStart) {
        std::process::exit(72);
    }
    if let Fault::Delay { milliseconds } = job.fault {
        thread::sleep(Duration::from_millis(milliseconds));
    }
    if let Fault::StderrFlood { bytes } = job.fault {
        let stderr = io::stderr();
        let mut stderr = stderr.lock();
        let buffer = [b'x'; 8192];
        let mut remaining = bytes;
        while remaining != 0 {
            let count = remaining.min(buffer.len());
            stderr
                .write_all(&buffer[..count])
                .map_err(|e| io_error("stderr fault", e))?;
            remaining -= count;
        }
        stderr.flush().map_err(|e| io_error("stderr flush", e))?;
    }
    if matches!(job.fault, Fault::Reverse) {
        job.dispatch.slots.reverse();
    }

    let count = job.dispatch.slots.len();
    let values = match compute(&job.dispatch, &job.inputs) {
        Ok(values) => values,
        Err(error) => {
            emit(output, stamp, rank, attempt, EventKind::Failed { error })?;
            return Ok(());
        }
    };
    let invalid_result = if values.len() != count {
        Some("batch result row count does not match dispatched slot count")
    } else if values
        .iter()
        .any(|row| row.len() != job.dispatch.hidden || row.iter().any(|x| !x.is_finite()))
    {
        Some("batch result rows must be finite and match hidden size")
    } else {
        None
    };
    if let Some(message) = invalid_result {
        emit(
            output,
            stamp,
            rank,
            attempt,
            EventKind::Failed {
                error: invalid(message),
            },
        )?;
        return Ok(());
    }
    if matches!(job.fault, Fault::DieAfterCompute) {
        std::process::exit(73);
    }

    let mut completions = Vec::new();
    completions
        .try_reserve_exact(count.saturating_add(usize::from(matches!(job.fault, Fault::Duplicate))))
        .map_err(|e| {
            Error::new(
                "EXEC_ALLOC",
                format!("cannot allocate completion batch: {e}"),
            )
        })?;
    let completion_attempt = if let Fault::OldAttempt {
        attempt: old_attempt,
    } = &job.fault
    {
        *old_attempt
    } else {
        attempt
    };
    for (index, (routed, value)) in job.dispatch.slots.iter().zip(values).enumerate() {
        if matches!(job.fault, Fault::Missing) && index + 1 == count {
            continue;
        }
        let original = &routed.slot;
        let mut completion = moe::Completion {
            stamp: job.dispatch.stamp.clone(),
            token: original.token,
            slot: original.slot,
            expert: original.expert,
            rank: original.rank,
            local: original.local,
            value,
        };
        if matches!(job.fault, Fault::StaleEpoch) {
            completion.stamp.epoch = if completion.stamp.epoch == 0 {
                1
            } else {
                completion.stamp.epoch - 1
            };
        }
        if matches!(job.fault, Fault::Duplicate) && index == 0 {
            completions.push(completion.clone());
        }
        completions.push(completion);
    }
    if let Err(error) = output.send_completions(WorkerCompletions {
        stamp: stamp.clone(),
        rank,
        attempt: completion_attempt,
        rows: completions,
    }) {
        // FP16 representability is preflighted before its writer emits any completion
        // frame. A finite but out-of-range result is therefore a known batch failure,
        // unlike a transport error that can leave a partial response on the stream.
        if error.code == "FP16_OVERFLOW" {
            emit(output, stamp, rank, attempt, EventKind::Failed { error })?;
            return Ok(());
        }
        return Err(error);
    }
    emit(
        output,
        stamp,
        rank,
        attempt,
        EventKind::Finished { slots: count },
    )
}

enum PipeMessage {
    Event(usize, Box<WorkerEvent>),
    Error(usize, Error),
    Eof(usize),
}
struct Worker {
    index: usize,
    child: Child,
    status: Option<ExitStatus>,
    input: JoinHandle<()>,
    output: JoinHandle<()>,
    stderr: JoinHandle<usize>,
}
fn spawn_worker(
    exe: &Path,
    index: usize,
    request: Vec<u8>,
    sender: mpsc::SyncSender<PipeMessage>,
) -> Result<Worker> {
    let mut child = Command::new(exe)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| io_error("spawn rank worker", e))?;
    let mut input = child.stdin.take().expect("Command configured piped stdin");
    let output = child
        .stdout
        .take()
        .expect("Command configured piped stdout");
    let mut stderr = child
        .stderr
        .take()
        .expect("Command configured piped stderr");
    let writes = sender.clone();
    let input = thread::spawn(move || {
        if let Err(error) = input.write_all(&request) {
            let _ = writes.send(PipeMessage::Error(
                index,
                io_error("send rank request", error),
            ));
        }
        // Closing stdin terminates the single-request frame.
    });
    let output = thread::spawn(move || {
        let mut output = BufReader::new(output);
        loop {
            match read_message(&mut output) {
                Ok(Some(event)) => {
                    if sender
                        .send(PipeMessage::Event(index, Box::new(event)))
                        .is_err()
                    {
                        break;
                    }
                }
                Ok(None) => {
                    let _ = sender.send(PipeMessage::Eof(index));
                    break;
                }
                Err(error) => {
                    let _ = sender.send(PipeMessage::Error(index, error));
                    break;
                }
            }
        }
    });
    let stderr = thread::spawn(move || {
        let mut buffer = [0; 8192];
        let mut count = 0_usize;
        loop {
            match stderr.read(&mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(n) => count = count.saturating_add(n),
            }
        }
        count
    });
    Ok(Worker {
        index,
        child,
        status: None,
        input,
        output,
        stderr,
    })
}
fn check_identity(event: &WorkerEvent, receipt: &Receipt) -> Result<()> {
    if event.attempt != receipt.attempt {
        return Err(Error::new(
            "STALE_ATTEMPT",
            "worker event belongs to another attempt",
        ));
    }
    if event.stamp != receipt.stamp || event.rank != receipt.rank {
        return Err(Error::new(
            "INVALID_COMPLETION",
            "worker event has stale stamp or foreign rank",
        ));
    }
    Ok(())
}

/// A fresh coordinator attempt number: strictly increasing within this process and
/// never below the wall clock in nanoseconds, so a restarted coordinator does not
/// reuse a recent attempt. Not a durable or distributed sequence.
pub(crate) fn next_attempt() -> Result<u64> {
    let clock = u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| invalid("coordinator clock precedes Unix epoch"))?
            .as_nanos(),
    )
    .map_err(|_| invalid("attempt clock exceeds u64"))?;
    let previous = NEXT_ATTEMPT
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
            n.checked_add(1).map(|next| next.max(clock))
        })
        .map_err(|_| invalid("attempt counter exhausted"))?;
    Ok(clock.max(previous + 1))
}

fn attempted() -> Result<MutexGuard<'static, HashSet<String>>> {
    ATTEMPTED
        .lock()
        .map_err(|_| Error::new("COORDINATOR_FAILED", "attempt guard poisoned"))
}

/// How many new identities this coordinator can still attempt (one per batch).
pub(crate) fn attempts_left() -> Result<usize> {
    Ok(MAX_ATTEMPTED.saturating_sub(attempted()?.len()))
}

/// Refuse a second execution of the same full stamp unless recovery is explicit.
pub(crate) fn claim_identity(stamp: &moe::Stamp, recovery: bool) -> Result<()> {
    let key = serde_json::to_string(stamp).map_err(|e| invalid(e.to_string()))?;
    let mut seen = attempted()?;
    if (!recovery && seen.contains(&key)) || (seen.len() >= MAX_ATTEMPTED && !seen.contains(&key)) {
        return Err(invalid(
            "identity already attempted; explicit recovery required, or coordinator attempt limit",
        ));
    }
    seen.insert(key);
    Ok(())
}

/// Run one real child wave. Never automatically replays a request.
/// recovery=true authorizes a fresh coordinator attempt; it does not imply exactly-once
/// execution, a durable journal, or a recovery guarantee across coordinator restarts.
pub fn run(p: &ProcessProblem, worker_exe: &Path) -> Result<ProcessReport> {
    moe::validate(&p.problem)?;
    if p.timeout_ms == 0 || p.timeout_ms > 60_000 {
        return Err(invalid("worker timeout must be within 1..60000 ms"));
    }
    let dispatch = routing::dispatch(&routing::from_dense(&p.problem))?;
    let ranks: Vec<_> = dispatch
        .ranks
        .into_iter()
        .filter(|r| !r.slots.is_empty())
        .collect();
    if p.faults
        .keys()
        .any(|rank| !ranks.iter().any(|r| r.rank == *rank))
    {
        return Err(invalid("fault refers to a nonparticipating rank"));
    }
    // Fault bounds too are checked before any identity is claimed or work is sent,
    // not left to the faulted rank's worker once the other ranks have computed.
    for fault in p.faults.values() {
        check_fault(fault)?;
    }
    let attempt = next_attempt()?;
    let mut requests = Vec::new();
    let mut receipts = Vec::new();
    // Finish all validation/serialization before admitting the first child.
    for rank in ranks {
        let request = rank_request(
            &p.problem,
            &rank,
            attempt,
            p.faults.get(&rank.rank).cloned().unwrap_or_default(),
        )?;
        let bytes = encode(&request)?;
        receipts.push(Receipt {
            stamp: rank.stamp.clone(),
            rank: rank.rank,
            attempt,
            state: ReceiptState::NotStarted,
            started: false,
            pid: None,
            reaped: false,
            reaped_pids: Vec::new(),
            children_spawned: 0,
            children_reaped: 0,
            expected_slots: rank.slots.len(),
            received_slots: 0,
            resident_experts: request.model.experts.len(),
            local_inputs: request.inputs.len(),
            request_bytes: bytes.len() - 1,
            stderr_bytes: 0,
            exit_code: None,
            error: None,
        });
        requests.push(bytes);
    }
    claim_identity(&p.problem.stamp, p.recovery)?;
    let mut collector = moe::Collector::new(&p.problem)?;
    let (sender, receiver) = mpsc::sync_channel(128);
    let mut workers = Vec::new();
    let mut error = None;
    let mut eof = vec![false; receipts.len()];
    let mut finished = vec![false; receipts.len()];
    let deadline = Instant::now() + Duration::from_millis(p.timeout_ms);
    for (index, request) in requests.into_iter().enumerate() {
        match spawn_worker(worker_exe, index, request, sender.clone()) {
            Ok(worker) => {
                receipts[index].pid = Some(worker.child.id());
                // Admission alone cannot prove whether calculation began.
                receipts[index].state = ReceiptState::Unknown;
                workers.push(worker);
            }
            Err(e) => {
                receipts[index].error = Some(e.clone());
                error = Some(e);
                break;
            }
        }
    }
    drop(sender);
    while error.is_none() && !workers.is_empty() {
        for worker in &mut workers {
            if worker.status.is_none() {
                match worker.child.try_wait() {
                    Ok(status) => worker.status = status,
                    Err(e) => {
                        error = Some(io_error("poll worker", e));
                        break;
                    }
                }
            }
        }
        if error.is_some() {
            break;
        }
        if workers.iter().all(|w| w.status.is_some() && eof[w.index]) {
            for worker in &workers {
                let index = worker.index;
                if !worker.status.as_ref().is_some_and(ExitStatus::success)
                    || !finished[index]
                    || receipts[index].received_slots != receipts[index].expected_slots
                {
                    error = Some(Error::new(
                        "WORKER_INCOMPLETE",
                        "worker died or omitted completion/finish",
                    ));
                    break;
                }
            }
            break;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            error = Some(Error::new(
                "WORKER_TIMEOUT",
                "worker wave exceeded deadline",
            ));
            break;
        }
        match receiver.recv_timeout(remaining.min(Duration::from_millis(5))) {
            Ok(PipeMessage::Eof(index)) => {
                eof[index] = true;
                if !finished[index] {
                    error = Some(Error::new(
                        "WORKER_INCOMPLETE",
                        "worker closed pipe without a complete finish",
                    ));
                }
            }
            Ok(PipeMessage::Error(index, e)) => {
                receipts[index].error = Some(e.clone());
                error = Some(e);
            }
            Ok(PipeMessage::Event(index, event)) => {
                let receipt = &mut receipts[index];
                if let Err(e) = check_identity(&event, receipt) {
                    receipt.error = Some(e.clone());
                    error = Some(e);
                    continue;
                }
                let result = match event.event {
                    EventKind::Started if !receipt.started && !finished[index] => {
                        receipt.started = true;
                        receipt.state = ReceiptState::Started;
                        Ok(())
                    }
                    EventKind::Rejected { error: e } if !receipt.started => {
                        receipt.state = ReceiptState::NotStarted;
                        Err(e)
                    }
                    EventKind::Failed { error: e } if receipt.started => Err(e),
                    EventKind::Completion { completion } if receipt.started && !finished[index] => {
                        if completion.rank != receipt.rank {
                            Err(Error::new(
                                "INVALID_COMPLETION",
                                "completion has foreign rank",
                            ))
                        } else {
                            collector
                                .accept(completion)
                                .map(|()| receipt.received_slots += 1)
                        }
                    }
                    EventKind::Finished { slots } if receipt.started && !finished[index] => {
                        if slots != receipt.expected_slots || receipt.received_slots != slots {
                            Err(Error::new(
                                "BATCH_INCOMPLETE",
                                "rank finished with missing results",
                            ))
                        } else {
                            finished[index] = true;
                            Ok(())
                        }
                    }
                    _ => Err(Error::new(
                        "WORKER_PROTOCOL",
                        "invalid worker event lifecycle",
                    )),
                };
                if let Err(e) = result {
                    receipt.error = Some(e.clone());
                    error = Some(e);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // Readers are finished. Poll wait/reap on the next iteration.
                if !workers.iter().all(|w| eof[w.index]) {
                    error = Some(Error::new(
                        "WORKER_INCOMPLETE",
                        "worker pipes closed without EOF",
                    ));
                } else {
                    thread::sleep(Duration::from_millis(1));
                }
            }
        }
    }
    // On any failure, terminate the whole admitted wave, including successful peers.
    // Child::try_wait and Child::wait both reap; no child is abandoned.
    if error.is_some() {
        collector.expire();
        for worker in &mut workers {
            if worker.status.is_none() {
                let _ = worker.child.kill();
            }
        }
    }
    // Dropping the receiver releases readers blocked on the bounded event channel.
    drop(receiver);
    for mut worker in workers {
        let receipt = &mut receipts[worker.index];
        let waited = match worker.status.take() {
            Some(status) => Ok(status),
            None => worker.child.wait(),
        };
        match waited {
            Ok(status) => {
                receipt.reaped = true;
                receipt.reaped_pids.push(worker.child.id());
                receipt.exit_code = status.code();
                if status.success()
                    && finished[worker.index]
                    && receipt.received_slots == receipt.expected_slots
                {
                    receipt.state = ReceiptState::Completed;
                } else if receipt.state != ReceiptState::NotStarted {
                    receipt.state = ReceiptState::Unknown;
                }
            }
            Err(e) => {
                if error.is_none() {
                    error = Some(io_error("reap worker", e));
                }
                receipt.state = ReceiptState::Unknown;
            }
        }
        let input_panicked = worker.input.join().is_err();
        let output_panicked = worker.output.join().is_err();
        if (input_panicked || output_panicked) && error.is_none() {
            error = Some(Error::new("WORKER_IO", "pipe thread panicked"));
        }
        match worker.stderr.join() {
            Ok(count) => receipt.stderr_bytes = count,
            Err(_) => {
                if error.is_none() {
                    error = Some(Error::new("WORKER_IO", "stderr thread panicked"));
                }
            }
        }
    }
    let spawned = receipts.iter().filter(|r| r.pid.is_some()).count();
    let reaped = receipts.iter().filter(|r| r.reaped).count();
    for receipt in &mut receipts {
        receipt.children_spawned = spawned;
        receipt.children_reaped = reaped;
    }
    let output = if error.is_none() {
        match collector.finish() {
            Ok(output) => Some(output),
            Err(e) => {
                error = Some(e);
                None
            }
        }
    } else {
        None
    };
    Ok(ProcessReport {
        ok: error.is_none(),
        output,
        receipts,
        attempt,
        error,
    })
}

#[cfg(test)]
mod batch_tests {
    use super::*;

    #[derive(Default)]
    struct Events(Vec<WorkerEvent>);

    impl EventSink for Events {
        fn send(&mut self, event: &WorkerEvent) -> Result<()> {
            self.0.push(event.clone());
            Ok(())
        }
    }

    fn stamp() -> moe::Stamp {
        moe::Stamp {
            run_id: "batch-test".into(),
            model_revision: "rev-1".into(),
            plan_id: "plan-1".into(),
            layer: 0,
            batch: 4,
            epoch: 7,
        }
    }

    fn job(fault: Fault) -> Job {
        let dispatch_stamp = stamp();
        Job {
            dispatch: routing::RankDispatch {
                stamp: dispatch_stamp,
                rank: 3,
                hidden: 2,
                tokens: 1,
                resident_order: vec![10, 20],
                slots: vec![
                    routing::RankSlot {
                        slot: moe::Slot {
                            token: 0,
                            slot: 0,
                            expert: 10,
                            rank: 3,
                            local: 0,
                            weight: 0.25,
                        },
                        physical_expert: 0,
                    },
                    routing::RankSlot {
                        slot: moe::Slot {
                            token: 0,
                            slot: 1,
                            expert: 20,
                            rank: 3,
                            local: 1,
                            weight: 0.75,
                        },
                        physical_expert: 1,
                    },
                ],
            },
            fault,
            inputs: BTreeMap::from([(0, vec![2.0, -3.0])]),
        }
    }

    fn event_kinds(events: &Events) -> Vec<&'static str> {
        events
            .0
            .iter()
            .map(|event| match &event.event {
                EventKind::Started => "started",
                EventKind::Rejected { .. } => "rejected",
                EventKind::Failed { .. } => "failed",
                EventKind::Completion { .. } => "completion",
                EventKind::Finished { .. } => "finished",
            })
            .collect()
    }

    #[test]
    fn batch_callback_runs_once_and_publishes_unweighted_rows_in_slot_order() {
        let mut events = Events::default();
        let mut calls = 0;
        execute_batch_with(
            job(Fault::None),
            &stamp(),
            3,
            12,
            &mut events,
            |dispatch, inputs| {
                calls += 1;
                assert_eq!(dispatch.slots.len(), 2);
                assert_eq!(inputs[&0], [2.0, -3.0]);
                Ok(dispatch
                    .slots
                    .iter()
                    .map(|slot| vec![slot.slot.expert as f64, slot.slot.weight])
                    .collect())
            },
        )
        .unwrap();

        assert_eq!(calls, 1);
        assert_eq!(
            event_kinds(&events),
            ["started", "completion", "completion", "finished"]
        );
        let values: Vec<_> = events
            .0
            .iter()
            .filter_map(|event| match &event.event {
                EventKind::Completion { completion } => {
                    Some((completion.expert, completion.value.clone()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(values, [(10, vec![10.0, 0.25]), (20, vec![20.0, 0.75])]);
    }

    type BatchCompute = Box<
        dyn FnOnce(&routing::RankDispatch, &BTreeMap<usize, Vec<f64>>) -> Result<Vec<Vec<f64>>>,
    >;

    #[test]
    fn invalid_or_failed_batch_computation_publishes_no_partial_completions() {
        let cases: Vec<(&str, BatchCompute)> = vec![
            (
                "error",
                Box::new(|_, _| Err(Error::new("TEST_COMPUTE", "failed"))),
            ),
            ("row_count", Box::new(|_, _| Ok(vec![vec![1.0, 2.0]]))),
            (
                "row_shape",
                Box::new(|_, _| Ok(vec![vec![1.0, 2.0], vec![3.0]])),
            ),
            (
                "nonfinite",
                Box::new(|_, _| Ok(vec![vec![1.0, 2.0], vec![3.0, f64::NAN]])),
            ),
        ];
        for (name, compute) in cases {
            let mut events = Events::default();
            execute_batch_with(job(Fault::None), &stamp(), 3, 12, &mut events, compute).unwrap();
            assert_eq!(event_kinds(&events), ["started", "failed"], "case {name}");
        }
    }

    #[test]
    fn rejection_does_not_invoke_batch_callback_or_start_execution() {
        let mut events = Events::default();
        let mut calls = 0;
        execute_batch_with(
            job(Fault::RejectBeforeCompute),
            &stamp(),
            3,
            12,
            &mut events,
            |_, _| {
                calls += 1;
                Ok(vec![vec![0.0; 2]; 2])
            },
        )
        .unwrap();
        assert_eq!(calls, 0);
        assert_eq!(event_kinds(&events), ["rejected"]);
    }

    #[test]
    fn oversized_or_overflowing_batch_is_rejected_before_callback_and_start() {
        for hidden in [crate::completion::MAX_OUTPUT_VALUES / 2 + 1, usize::MAX] {
            let mut job = job(Fault::None);
            job.dispatch.hidden = hidden;
            let mut events = Events::default();
            let mut calls = 0;
            execute_batch_with(job, &stamp(), 3, 12, &mut events, |_, _| {
                calls += 1;
                Ok(Vec::new())
            })
            .unwrap();
            assert_eq!(calls, 0);
            assert_eq!(event_kinds(&events), ["rejected"]);
            let EventKind::Rejected { error } = &events.0[0].event else {
                panic!("oversized batch was not rejected")
            };
            assert_eq!(error.code, "BATCH_OUTPUT_TOO_LARGE");
        }
    }

    #[test]
    fn reversed_slots_keep_identity_and_missing_fault_withholds_a_whole_result() {
        let mut events = Events::default();
        execute_batch_with(
            job(Fault::Reverse),
            &stamp(),
            3,
            12,
            &mut events,
            |dispatch, _| {
                Ok(dispatch
                    .slots
                    .iter()
                    .map(|slot| vec![slot.slot.expert as f64, 0.0])
                    .collect())
            },
        )
        .unwrap();
        let experts: Vec<_> = events
            .0
            .iter()
            .filter_map(|event| match &event.event {
                EventKind::Completion { completion } => Some(completion.expert),
                _ => None,
            })
            .collect();
        assert_eq!(experts, [20, 10]);

        let mut events = Events::default();
        execute_batch_with(
            job(Fault::Missing),
            &stamp(),
            3,
            12,
            &mut events,
            |dispatch, _| Ok(vec![vec![1.0, 2.0]; dispatch.slots.len()]),
        )
        .unwrap();
        assert_eq!(event_kinds(&events), ["started", "completion", "finished"]);
    }

    #[test]
    fn duplicate_stale_epoch_and_old_attempt_faults_preserve_their_distinct_identities() {
        let mut events = Events::default();
        execute_batch_with(
            job(Fault::Duplicate),
            &stamp(),
            3,
            12,
            &mut events,
            |dispatch, _| Ok(vec![vec![1.0, 2.0]; dispatch.slots.len()]),
        )
        .unwrap();
        assert_eq!(
            event_kinds(&events),
            [
                "started",
                "completion",
                "completion",
                "completion",
                "finished"
            ]
        );
        let duplicate_ids: Vec<_> = events
            .0
            .iter()
            .filter_map(|event| match &event.event {
                EventKind::Completion { completion } => {
                    Some((completion.token, completion.slot, completion.expert))
                }
                _ => None,
            })
            .collect();
        assert_eq!(duplicate_ids, [(0, 0, 10), (0, 0, 10), (0, 1, 20)]);

        let mut events = Events::default();
        execute_batch_with(
            job(Fault::StaleEpoch),
            &stamp(),
            3,
            12,
            &mut events,
            |dispatch, _| Ok(vec![vec![1.0, 2.0]; dispatch.slots.len()]),
        )
        .unwrap();
        let epochs: Vec<_> = events
            .0
            .iter()
            .filter_map(|event| match &event.event {
                EventKind::Completion { completion } => Some(completion.stamp.epoch),
                _ => None,
            })
            .collect();
        assert_eq!(epochs, [6, 6]);

        let mut events = Events::default();
        execute_batch_with(
            job(Fault::OldAttempt { attempt: 11 }),
            &stamp(),
            3,
            12,
            &mut events,
            |dispatch, _| Ok(vec![vec![1.0, 2.0]; dispatch.slots.len()]),
        )
        .unwrap();
        let attempts: Vec<_> = events
            .0
            .iter()
            .filter_map(|event| match &event.event {
                EventKind::Completion { .. } => Some(event.attempt),
                _ => None,
            })
            .collect();
        assert_eq!(attempts, [11, 11]);
    }
}
