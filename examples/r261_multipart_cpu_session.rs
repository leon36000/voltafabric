//! Bounded research coordinator for retaining compressed-owner sessions across batches.
//! Not a production protocol: a process failure invalidates the session and is never replayed.
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{self, BufRead, Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::time::Duration;
use voltafabric::session::{self, CompressedArtifact, CompressedSeriesInput};

const MAX_LINE: u64 = 16 * 1024 * 1024;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    batch: u64,
    routes: Value,
    inputs_fp16_bits: Vec<Vec<u16>>,
    request_id: Option<String>,
}
fn valid_id(id: &str) -> bool {
    id.len() == 32
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
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
    let mut error_id = None;
    if let Err(e) = run(&mut error_id) {
        let mut error = json!({"ok":false,"error":e});
        if let Some(id) = error_id {
            error["request_id"] = json!(id);
        }
        let _ = emit(&error);
        std::process::exit(1);
    }
}
fn run(error_id: &mut Option<String>) -> Result<(), String> {
    let stdin = io::stdin();
    let mut input = stdin.lock();
    let first = line(&mut input)
        .map_err(|e| e.to_string())?
        .ok_or("missing init")?;
    let mut first_value: Value = serde_json::from_slice(&first).map_err(|e| e.to_string())?;
    let init_id = if first_value.get("init").is_some() {
        let object = first_value.as_object_mut().ok_or("invalid init envelope")?;
        if object.len() != 2 {
            return Err("invalid init envelope".into());
        }
        let id = object
            .remove("request_id")
            .and_then(|v| v.as_str().map(str::to_owned))
            .ok_or("missing request id")?;
        if !valid_id(&id) {
            return Err("invalid request id".into());
        }
        first_value = object.remove("init").ok_or("missing init")?;
        Some(id)
    } else {
        None
    };
    *error_id = init_id.clone();
    let init: CompressedSeriesInput =
        serde_json::from_value(first_value).map_err(|e| e.to_string())?;
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
        file.take((128 * 1024 * 1024 + 8 * 1024 * 1024 + 8) as u64 + 1)
            .read_to_end(&mut data)
            .map_err(|e| e.to_string())?;
        if data.len() > 128 * 1024 * 1024 + 8 * 1024 * 1024 + 8 {
            return Err("artifact too large".into());
        }
        artifacts.push(CompressedArtifact::from_bytes(data).map_err(|e| e.to_string())?);
    }
    let timeout = Duration::from_millis(init.timeout_ms);
    let mut sessions =
        session::prepare_compressed_multipart(&init.workers, &artifacts, &init.routing, 1, timeout)
            .map_err(|e| e.to_string())?;
    // Freeze the validated identity/registry before releasing all coordinator
    // weight copies. This coordinator never rebinds or replays a consumed session.
    let compact =
        session::CompactCompressedFleet::from_loaded(&sessions, &artifacts, &init.routing)
            .map_err(|e| e.to_string())?;
    drop(artifacts);
    let mut ready = json!({"ok":true,"state":"ready"});
    if let Some(id) = &init_id {
        ready["request_id"] = json!(id);
    }
    emit(&ready).map_err(|e| e.to_string())?;
    *error_id = None;
    let mut next = init.routing.stamp.batch;
    let mut close_id = None;
    loop {
        let Some(bytes) = line(&mut input).map_err(|e| e.to_string())? else {
            break;
        };
        *error_id = None;
        let v: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if v.get("op") == Some(&json!("shutdown")) {
            let obj = v.as_object().ok_or("invalid shutdown")?;
            if init_id.is_some() {
                let id = obj
                    .get("request_id")
                    .and_then(Value::as_str)
                    .ok_or("missing shutdown request id")?;
                if obj.len() != 2 || !valid_id(id) {
                    return Err("invalid shutdown request id".into());
                }
                close_id = Some(id.to_owned());
                *error_id = close_id.clone();
            } else if obj.len() != 1 {
                return Err("invalid legacy shutdown".into());
            }
            break;
        }
        let req: Request = serde_json::from_value(v).map_err(|e| e.to_string())?;
        if init_id.is_some() != req.request_id.is_some()
            || req.request_id.as_deref().is_some_and(|id| !valid_id(id))
        {
            return Err("invalid batch request id".into());
        }
        *error_id = req.request_id.clone();
        if req.batch != next {
            return Err("batch identity not next; session consumed".into());
        }
        next = next.checked_add(1).ok_or("batch overflow")?;
        let mut routing = init.routing.clone();
        routing.stamp.batch = req.batch;
        routing.routes = serde_json::from_value(req.routes).map_err(|e| e.to_string())?;
        // No recovery or replay on unknown; any failure terminates the process.
        let batch = compact
            .run_fp16(&mut sessions, &routing, &req.inputs_fp16_bits, timeout)
            .map_err(|e| e.to_string())?;
        if !batch.ok || batch.output.is_none() {
            return Err("batch failed; no partial output".into());
        }
        let mut reply = json!({"ok":true,"batch":req.batch,"result":batch});
        if let Some(id) = req.request_id {
            reply["request_id"] = json!(id);
        }
        emit(&reply).map_err(|e| e.to_string())?;
        *error_id = None;
    }
    let close: Vec<_> = sessions.into_iter().map(|s| s.close(timeout)).collect();
    let mut reply = json!({"ok":true,"state":"closed","close":close});
    if let Some(id) = close_id {
        reply["request_id"] = json!(id);
    }
    emit(&reply).map_err(|e| e.to_string())?;
    Ok(())
}
