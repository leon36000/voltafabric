//! Bounded research coordinator for retaining compressed-owner sessions across batches.
//! Not a production protocol: a process failure invalidates the session and is never replayed.
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{self, BufRead, Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::time::Duration;
use voltafabric::session::{self, CompletionFormat, CompressedArtifact, CompressedSeriesInput};

const MAX_LINE: u64 = 16 * 1024 * 1024;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    batch: u64,
    routes: Value,
    inputs_fp16_bits: Vec<Vec<u16>>,
}
fn line<R: BufRead>(input: &mut R) -> io::Result<Option<Vec<u8>>> {
    let mut data = Vec::new();
    input.take(MAX_LINE + 1).read_until(b'\n', &mut data)?;
    if data.is_empty() {
        return Ok(None);
    }
    if data.len() as u64 > MAX_LINE || data.last() != Some(&b'\n') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid or oversized frame",
        ));
    }
    Ok(Some(data))
}
fn emit(value: &Value) -> io::Result<()> {
    let mut o = io::stdout().lock();
    serde_json::to_writer(&mut o, value)?;
    o.write_all(b"\n")?;
    o.flush()
}
fn main() {
    if let Err(e) = run() {
        let _ = emit(&json!({"ok":false,"error":e}));
        std::process::exit(1);
    }
}
fn run() -> Result<(), String> {
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let first = line(&mut input)
        .map_err(|e| e.to_string())?
        .ok_or("missing init")?;
    let init: CompressedSeriesInput = serde_json::from_slice(&first).map_err(|e| e.to_string())?;
    if init.artifacts.is_empty()
        || init.artifacts.len() > 64
        || init.timeout_ms == 0
        || init.timeout_ms > 60_000
    {
        return Err("invalid artifact count or timeout".into());
    }
    let mut artifacts = Vec::new();
    for path in &init.artifacts {
        // Read only regular, bounded files; refuse symlink targets that change after open only
        // insofar as normal filesystem semantics permit. This is a private CPU fixture.
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(path)
            .map_err(|e| e.to_string())?;
        if !file.metadata().map_err(|e| e.to_string())?.is_file() {
            return Err("artifact not regular".into());
        }
        let mut data = Vec::new();
        file.take(session::MAX_RAW_ARTIFACT_BYTES as u64 + 1)
            .read_to_end(&mut data)
            .map_err(|e| e.to_string())?;
        if data.len() > session::MAX_RAW_ARTIFACT_BYTES {
            return Err("artifact too large".into());
        }
        artifacts.push(CompressedArtifact::from_bytes(data).map_err(|e| e.to_string())?);
    }
    let timeout = Duration::from_millis(init.timeout_ms);
    let mut sessions =
        session::prepare_compressed(&init.workers, &artifacts, &init.routing, timeout)
            .map_err(|e| e.to_string())?;
    emit(&json!({"ok":true,"state":"ready"})).map_err(|e| e.to_string())?;
    let mut next = init.routing.stamp.batch;
    loop {
        let Some(bytes) = line(&mut input).map_err(|e| e.to_string())? else {
            break;
        };
        let v: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if v == json!({"op":"shutdown"}) {
            break;
        }
        let req: Request = serde_json::from_value(v).map_err(|e| e.to_string())?;
        if req.batch != next {
            return Err("batch identity not next; session consumed".into());
        }
        next = next.checked_add(1).ok_or("batch overflow")?;
        let mut routing = init.routing.clone();
        routing.stamp.batch = req.batch;
        routing.routes = serde_json::from_value(req.routes).map_err(|e| e.to_string())?;
        // No recovery or replay on unknown; any failure terminates the process.
        let batch = session::run_compressed_batch_with_fp16_inputs(
            &mut sessions,
            &artifacts,
            &routing,
            &req.inputs_fp16_bits,
            &BTreeMap::new(),
            CompletionFormat::Fp16,
            timeout,
        )
        .map_err(|e| e.to_string())?;
        if !batch.ok || batch.output.is_none() {
            return Err("batch failed; no partial output".into());
        }
        emit(&json!({"ok":true,"batch":req.batch,"result":batch})).map_err(|e| e.to_string())?;
    }
    let close: Vec<_> = sessions.into_iter().map(|s| s.close(timeout)).collect();
    emit(&json!({"ok":true,"state":"closed","close":close})).map_err(|e| e.to_string())?;
    Ok(())
}
