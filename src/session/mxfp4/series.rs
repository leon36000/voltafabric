//! Bounded CLI client for the real session API, including authenticated SSH endpoints.
use super::*;

/// Aggregate compressed owners retained by this CPU reference client, before any
/// worker is spawned. This is not a GPU service/model capacity limit.
const MAX_CLIENT_ARTIFACT_BYTES: usize = 128 * 1024 * 1024;

pub(super) fn client_artifact_budget_error() -> Error {
    Error::new(
        "ARTIFACT_BUDGET",
        "MXFP4 CPU series exceeds 128 MiB of compressed client owners",
    )
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerSource {
    pub endpoint: Endpoint,
    pub binding: Binding,
    pub artifact_path: String,
    #[serde(default)]
    pub producer: Producer,
    /// Wire-only preference; old workers retain raw32. Default stays raw32 until
    /// a workload/transport-specific byte and latency comparison justifies it.
    #[serde(default)]
    pub prefer_bf16_activations: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Batch {
    pub problem: routing::RoutingProblem,
    /// Nonzero identity for this exact routed execution; never inferred from batch order.
    pub attempt: u64,
    pub inputs: Vec<TokenInput>,
    #[serde(default)]
    pub faults: BTreeMap<u32, Fault>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeriesInput {
    pub workers: Vec<WorkerSource>,
    pub batches: Vec<Batch>,
    pub timeout_ms: u64,
    /// Independent connect/LOAD operations in flight. Default 1 preserves serial startup.
    /// At most four sessions start together; all started sessions are joined and closed
    /// on any failure, before another wave or any EXEC is admitted.
    #[serde(default = "serial_startup")]
    pub startup_parallelism: usize,
}

fn serial_startup() -> usize {
    1
}

#[derive(Debug, Serialize)]
pub struct BatchReport {
    pub stamp: moe::Stamp,
    pub attempt: u64,
    pub contract: arithmetic::Contract,
    pub output_bits: Option<Vec<Vec<u32>>>,
    pub receipts: Vec<ExecReceipt>,
    pub elapsed_micros: u128,
    pub error: Option<Error>,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub ok: bool,
    pub scope: &'static str,
    pub loads: Vec<LoadReceipt>,
    pub batches: Vec<BatchReport>,
    /// HELLO+LOAD+EXEC traffic before SHUTDOWN; includes headers, excludes SSH framing.
    pub data_phase_traffic: Vec<(u32, Traffic)>,
    pub closes: Vec<CloseReceipt>,
    pub error: Option<Error>,
}

fn compact(batch: &Batch, d: &routing::RankDispatch) -> Vec<TokenInput> {
    let needed: BTreeSet<_> = d.slots.iter().map(|r| r.slot.token).collect();
    batch
        .inputs
        .iter()
        .filter(|row| needed.contains(&row.token))
        .cloned()
        .collect()
}

fn validate_batch_attempts(attempts: impl IntoIterator<Item = u64>) -> Result<()> {
    let mut seen = BTreeSet::new();
    for attempt in attempts {
        if attempt == 0 || !seen.insert(attempt) {
            return Err(invalid(
                "MXFP4 series requires unique nonzero execution attempts",
            ));
        }
    }
    Ok(())
}

/// Validate all route identities against declared rank bindings before opening
/// even one artifact path. This avoids loading a partial expert set when a route
/// already names a rank with no resident owner.
fn preflight_bindings(input: &SeriesInput) -> Result<BTreeSet<u32>> {
    let mut sources = BTreeMap::new();
    for worker in &input.workers {
        worker.producer.validate()?;
        if worker.endpoint.rank != worker.binding.rank
            || worker.endpoint.argv.is_empty()
            || sources.insert(worker.endpoint.rank, worker).is_some()
        {
            return Err(invalid(
                "MXFP4 series has duplicate/foreign/empty worker endpoints",
            ));
        }
    }
    let mut identities = BTreeSet::new();
    let mut participants = BTreeSet::new();
    for batch in &input.batches {
        arithmetic::Collector::new(&batch.problem)?;
        if !identities.insert(encode(&batch.problem.stamp)?) {
            return Err(invalid("MXFP4 series repeats a full execution identity"));
        }
        let mut tokens = BTreeSet::new();
        for row in &batch.inputs {
            if row.token >= batch.problem.routes.len()
                || row.value_bits.len() != batch.problem.hidden
                || !tokens.insert(row.token)
            {
                return Err(invalid(
                    "MXFP4 series global inputs have duplicate/foreign tokens or shapes",
                ));
            }
            for &bits in &row.value_bits {
                if deepseek_v41::cast_bf16_rne_reference(f32::from_bits(bits))?.to_bits() != bits {
                    return Err(invalid("MXFP4 series input must already be exactly BF16"));
                }
            }
        }
        if tokens.len() != batch.problem.routes.len() {
            return Err(invalid("MXFP4 series has missing inputs"));
        }
        let dispatch = routing::dispatch(&batch.problem)?;
        for rank in dispatch.ranks.iter().filter(|rank| !rank.slots.is_empty()) {
            let source = sources
                .get(&rank.rank)
                .ok_or_else(|| invalid("missing MXFP4 rank endpoint"))?;
            super::validate_binding(&source.binding, rank.hidden, rank)?;
            checked_inputs(rank, compact(batch, rank))?;
            participants.insert(rank.rank);
        }
        for (&rank, fault) in &batch.faults {
            if !dispatch
                .ranks
                .iter()
                .any(|entry| entry.rank == rank && !entry.slots.is_empty())
            {
                return Err(invalid("MXFP4 fault names a nonparticipating rank"));
            }
            check_fault(fault)?;
        }
    }
    if participants != sources.keys().copied().collect() {
        return Err(invalid(
            "MXFP4 series endpoints must match its participating ranks",
        ));
    }
    Ok(participants)
}

/// After all route bindings pass and files are loaded, verify that every payload
/// matches the metadata that admitted it. Still no Session has been connected.
fn preflight_artifacts(
    input: &SeriesInput,
    artifacts: &BTreeMap<u32, Artifact>,
    participants: &BTreeSet<u32>,
) -> Result<()> {
    if participants != &artifacts.keys().copied().collect() {
        return Err(invalid(
            "MXFP4 series artifacts must match its participating ranks",
        ));
    }
    for batch in &input.batches {
        let dispatch = routing::dispatch(&batch.problem)?;
        for rank in dispatch.ranks.iter().filter(|rank| !rank.slots.is_empty()) {
            let artifact = artifacts
                .get(&rank.rank)
                .ok_or_else(|| invalid("missing MXFP4 rank artifact"))?;
            artifact.validate_dispatch(rank)?;
        }
    }
    Ok(())
}

type StartupResult = (u32, Option<Session>, Result<LoadReceipt>);

fn connect_load(
    source: &WorkerSource,
    artifacts: &BTreeMap<u32, Artifact>,
    timeout: Duration,
) -> (Option<Session>, Result<LoadReceipt>) {
    match Session::connect_mxfp4(&source.endpoint, timeout, &source.producer) {
        Ok(mut worker) => {
            if source.prefer_bf16_activations {
                worker.prefer_mxfp4_bf16_activations();
            }
            let receipt = worker.load_mxfp4_as(
                &artifacts[&source.endpoint.rank],
                source.binding.epoch,
                timeout,
                &source.producer,
            );
            // Keep a connected session even on LOAD failure so the caller records close.
            (Some(worker), receipt)
        }
        Err(error) => (None, Err(error)),
    }
}

fn connect_wave(
    wave: &[WorkerSource],
    artifacts: &BTreeMap<u32, Artifact>,
    timeout: Duration,
) -> Vec<StartupResult> {
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(wave.len());
        let mut spawn_error = None;
        for source in wave {
            let rank = source.endpoint.rank;
            match thread::Builder::new()
                .name(format!("mxfp4-start-{rank}"))
                .spawn_scoped(scope, move || connect_load(source, artifacts, timeout))
            {
                Ok(handle) => handles.push((rank, handle)),
                Err(error) => {
                    spawn_error = Some((
                        rank,
                        None,
                        Err(Error::new(
                            "WORKER_STARTUP",
                            format!("startup thread: {error}"),
                        )),
                    ));
                    break;
                }
            }
        }
        // Join every started task, including tasks after the first failed rank. No
        // detached startup may retain an artifact or publish work after this wave.
        let mut results = Vec::with_capacity(wave.len());
        for (rank, handle) in handles {
            let (worker, receipt) = handle.join().unwrap_or_else(|_| {
                (
                    None,
                    Err(Error::new("WORKER_STARTUP", "startup thread panicked")),
                )
            });
            results.push((rank, worker, receipt));
        }
        if let Some(error) = spawn_error {
            results.push(error);
        }
        results
    })
}

/// There is no automatic retry. One full execution identity is claimed once per
/// coordinator process using the existing guard, with no distributed exactly-once claim.
pub fn run_series(input: SeriesInput) -> Result<Report> {
    if input.workers.is_empty()
        || input.workers.len() > 64
        || input.batches.is_empty()
        || input.batches.len() > 64
        || !(1..=60000).contains(&input.timeout_ms)
        || !(1..=4).contains(&input.startup_parallelism)
    {
        return Err(invalid(
            "MXFP4 series requires 1..64 workers/batches, 1..60000 ms and 1..4 startup parallelism",
        ));
    }
    if attempts_left()? < input.batches.len() {
        return Err(invalid("MXFP4 series exceeds attempt guard budget"));
    }
    validate_batch_attempts(input.batches.iter().map(|batch| batch.attempt))?;
    let participants = preflight_bindings(&input)?;
    let mut artifacts: BTreeMap<u32, Artifact> = BTreeMap::new();
    let mut admitted_paths: BTreeMap<&str, u32> = BTreeMap::new();
    let mut artifact_bytes = 0_usize;
    for worker in &input.workers {
        worker.producer.validate()?;
        if worker.endpoint.rank != worker.binding.rank
            || artifacts.contains_key(&worker.endpoint.rank)
            || worker.endpoint.argv.is_empty()
        {
            return Err(invalid(
                "MXFP4 series has duplicate/foreign/empty worker endpoints",
            ));
        }
        // One exact path denotes one immutable snapshot within this series. Do not
        // share by caller-supplied digest: a distinct corrupt path must still fail.
        let artifact = if let Some(rank) = admitted_paths.get(worker.artifact_path.as_str()) {
            artifacts[rank].sharing_binding(worker.binding.clone())?
        } else {
            let remaining_bytes = MAX_CLIENT_ARTIFACT_BYTES
                .checked_sub(artifact_bytes)
                .ok_or_else(client_artifact_budget_error)?;
            // In particular, do not block opening another path after exhaustion.
            if remaining_bytes == 0 {
                return Err(client_artifact_budget_error());
            }
            let file = std::fs::File::open(&worker.artifact_path)
                .map_err(|e| invalid(format!("MXFP4 artifact file: {e}")))?;
            let artifact = Artifact::from_client_bounded_reader(
                file,
                worker.binding.clone(),
                remaining_bytes,
            )?;
            artifact_bytes = artifact_bytes
                .checked_add(artifact.bytes().len())
                .filter(|&n| n <= MAX_CLIENT_ARTIFACT_BYTES)
                .ok_or_else(client_artifact_budget_error)?;
            admitted_paths.insert(worker.artifact_path.as_str(), worker.endpoint.rank);
            artifact
        };
        artifacts.insert(worker.endpoint.rank, artifact);
    }
    preflight_artifacts(&input, &artifacts, &participants)?;
    let timeout = Duration::from_millis(input.timeout_ms);
    let mut sessions = BTreeMap::new();
    let mut report = Report {
        ok: false,
        scope: if input
            .workers
            .iter()
            .any(|w| matches!(w.producer, Producer::GpuR13ResidentFp32V1 { .. }))
        {
            "EXPLICIT_NATIVE_GPU_MXFP4_SESSION_COMPONENT"
        } else {
            "CPU_MXFP4_R9_SESSION_REFERENCE_ONLY"
        },
        loads: Vec::new(),
        batches: Vec::new(),
        data_phase_traffic: Vec::new(),
        closes: Vec::new(),
        error: None,
    };
    let execution: Result<()> = (|| {
        for wave in input.workers.chunks(input.startup_parallelism) {
            let started = if input.startup_parallelism == 1 {
                let source = &wave[0];
                let (worker, receipt) = connect_load(source, &artifacts, timeout);
                vec![(source.endpoint.rank, worker, receipt)]
            } else {
                connect_wave(wave, &artifacts, timeout)
            };
            let mut failure = None;
            for (rank, worker, receipt) in started {
                if let Some(worker) = worker {
                    sessions.insert(rank, worker);
                }
                match receipt {
                    Ok(receipt) => report.loads.push(receipt),
                    Err(error) if failure.is_none() => failure = Some(error),
                    Err(_) => (),
                }
            }
            if let Some(error) = failure {
                return Err(error);
            }
        }
        // LOAD has acknowledged ownership in every worker. Later EXECs use only
        // session bindings, so the client need not retain compressed payload copies.
        drop(artifacts);
        for batch in &input.batches {
            claim_identity(&batch.problem.stamp, false)?;
            let start = Instant::now();
            let deadline = start + timeout;
            let mut collector = arithmetic::Collector::new(&batch.problem)?;
            let mut result = BatchReport {
                stamp: batch.problem.stamp.clone(),
                attempt: batch.attempt,
                contract: collector.contract().clone(),
                output_bits: None,
                receipts: Vec::new(),
                elapsed_micros: 0,
                error: None,
            };
            let mut pending = Vec::new();
            for rank in routing::dispatch(&batch.problem)?
                .ranks
                .iter()
                .filter(|r| !r.slots.is_empty())
            {
                let worker = sessions.get_mut(&rank.rank).unwrap();
                let fault = batch.faults.get(&rank.rank).cloned().unwrap_or(Fault::None);
                match worker.start_preweighted_fp32_until(
                    &batch.problem,
                    &collector,
                    batch.attempt,
                    compact(batch, rank),
                    fault,
                    deadline,
                ) {
                    Ok(work) => pending.push((rank.rank, work)),
                    Err(error) => {
                        result.error = Some(error);
                        break;
                    }
                }
            }
            for (rank, work) in pending {
                let receipt = sessions.get_mut(&rank).unwrap().finish_preweighted_fp32(
                    work,
                    &mut collector,
                    deadline,
                );
                if result.error.is_none() {
                    result.error = receipt.error.clone();
                }
                result.receipts.push(receipt);
            }
            if result.error.is_none() {
                match collector.finish() {
                    Ok(output) => {
                        result.output_bits = Some(
                            output
                                .into_iter()
                                .map(|row| row.into_iter().map(f32::to_bits).collect())
                                .collect(),
                        )
                    }
                    Err(error) => result.error = Some(error),
                }
            } else {
                collector.expire();
            }
            result.elapsed_micros = start.elapsed().as_micros();
            let error = result.error.clone();
            report.batches.push(result);
            if let Some(error) = error {
                return Err(error);
            }
        }
        Ok(())
    })();
    report.error = execution.err();
    for (rank, worker) in sessions {
        report
            .data_phase_traffic
            .push((rank, worker.traffic.clone()));
        let close = worker.close(timeout);
        if report.error.is_none() && (close.error.is_some() || close.exit_code != Some(0)) {
            report.error = Some(Error::new(
                "SESSION_CLOSE",
                "MXFP4 worker failed its terminal close",
            ));
        }
        report.closes.push(close);
    }
    report.ok = report.error.is_none();
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::validate_batch_attempts;

    #[test]
    fn mxfp4_batch_attempts_are_explicit_nonzero_and_unique() {
        assert!(validate_batch_attempts([41, 42, 9001]).is_ok());
        assert!(validate_batch_attempts([]).is_ok());
        assert!(validate_batch_attempts([0]).is_err());
        assert!(validate_batch_attempts([41, 41]).is_err());
    }
}
