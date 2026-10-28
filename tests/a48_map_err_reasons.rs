//! A48 map_err sweep: each refusal below is translated by a `.map_err` whose error
//! path no test reached, or reached without checking the translated reason. Every
//! test pins the code and the message, so a replaced or dropped translation fails.
#![cfg(target_os = "linux")]
mod common;

use serde_json::{json, Value};
use std::{
    fs,
    io::{self, Cursor, Read, Seek, SeekFrom, Write},
    os::fd::OwnedFd,
    os::unix::{fs::PermissionsExt, net::UnixStream},
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::atomic::{AtomicU64, Ordering},
};
use voltafabric::{checkpoint, process::*, routing, Error};

static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
fn temp_dir() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "vf-a48-{}-{}",
        std::process::id(),
        NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&path).unwrap();
    path
}
fn checkpoint_request() -> checkpoint::Request {
    let bytes = fs::read(manifest().join("fixtures/resident_import_qwen_expert0.json")).unwrap();
    let mut request: checkpoint::Request = serde_json::from_slice(&bytes).unwrap();
    request.path = manifest().join(&request.path);
    request
}
fn assert_reason(error: &Error, code: &str, message: &str) {
    assert_eq!(error.code, code, "{error}");
    assert_eq!(error.message, message, "{error}");
}

/// A checkpoint source that reports `len` bytes and fails every seek after the
/// first `seeks`, the way a file truncated or faulted during import would.
struct Faulty {
    inner: Cursor<Vec<u8>>,
    len: u64,
    seeks: usize,
}
impl Faulty {
    fn new(bytes: Vec<u8>) -> Self {
        let len = bytes.len() as u64;
        Self {
            inner: Cursor::new(bytes),
            len,
            seeks: usize::MAX,
        }
    }
}
impl Read for Faulty {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buffer)
    }
}
impl Seek for Faulty {
    fn seek(&mut self, to: SeekFrom) -> io::Result<u64> {
        if self.seeks == 0 {
            return Err(io::Error::other("injected seek fault"));
        }
        self.seeks -= 1;
        match to {
            SeekFrom::End(0) => Ok(self.len),
            _ => self.inner.seek(to),
        }
    }
}
fn read_error(mut source: Faulty) -> Error {
    checkpoint::read(&mut source, &checkpoint_request().spec).unwrap_err()
}

#[test]
fn checkpoint_seek_faults_are_checkpoint_io() {
    // Seek to the end for the size, back to the start for the header, then one
    // seek per selected tensor: fail each of the three kinds in turn.
    let bytes = fs::read(checkpoint_request().path).unwrap();
    for seeks in 0..3 {
        let mut source = Faulty::new(bytes.clone());
        source.seeks = seeks;
        assert_reason(&read_error(source), "CHECKPOINT_IO", "injected seek fault");
    }
}

#[test]
fn checkpoint_shorter_than_its_length_prefix_is_checkpoint_io() {
    let error =
        checkpoint::read(&mut Cursor::new(vec![0; 3]), &checkpoint_request().spec).unwrap_err();
    assert_reason(&error, "CHECKPOINT_IO", "failed to fill whole buffer");
}

#[test]
fn checkpoint_truncated_after_its_size_was_read_is_checkpoint_io() {
    // The size check passed on the reported length; the header, then a selected
    // tensor, are cut short underneath it.
    let bytes = fs::read(checkpoint_request().path).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    for keep in [8 + n / 2, bytes.len() - 1] {
        let mut source = Faulty::new(bytes[..keep].to_vec());
        source.len = bytes.len() as u64;
        assert_reason(
            &read_error(source),
            "CHECKPOINT_IO",
            "failed to fill whole buffer",
        );
    }
}

fn header_only(header: Value, data: usize) -> Vec<u8> {
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(vec![0; data]);
    bytes
}

#[test]
fn safetensors_metadata_rejection_is_a_header_error() {
    // Unique, bounded names pass the first parse; SafeTensors then refuses the
    // tensor itself. The F32 scalar case is why the later `F32 scalar length`
    // check can never fire: an F32 scalar is always exactly four bytes here.
    for (tensor, data) in [
        (json!({"dtype":"U8","shape":[2],"data_offsets":[0,3]}), 3),
        (json!({"dtype":"F32","shape":[],"data_offsets":[0,8]}), 8),
        (json!({"dtype":"U8","shape":[2],"data_offsets":[1,3]}), 3),
    ] {
        let bytes = header_only(json!({ "t": tensor }), data);
        let error =
            checkpoint::read(&mut Cursor::new(bytes), &checkpoint_request().spec).unwrap_err();
        assert_eq!(error.code, "CHECKPOINT_HEADER", "{error}");
        assert!(!error.message.is_empty());
    }
}

#[test]
fn missing_checkpoint_path_is_checkpoint_io() {
    let mut request = checkpoint_request();
    request.path = manifest().join("fixtures/a48-missing.safetensors");
    let error = checkpoint::load(&request).unwrap_err();
    assert_reason(
        &error,
        "CHECKPOINT_IO",
        "No such file or directory (os error 2)",
    );
}

#[test]
fn unreadable_checkpoint_is_checkpoint_io() {
    // stat needs no read permission, so only the open refuses.
    let dir = temp_dir();
    let mut request = checkpoint_request();
    let path = dir.join("unreadable.safetensors");
    fs::copy(&request.path, &path).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o000)).unwrap();
    let readable = fs::File::open(&path).is_ok();
    request.path = path.clone();
    let result = checkpoint::load(&request);
    fs::remove_file(&path).unwrap();
    fs::remove_dir(&dir).unwrap();
    assert!(!readable, "test needs a user without CAP_DAC_OVERRIDE");
    assert_reason(
        &result.unwrap_err(),
        "CHECKPOINT_IO",
        "Permission denied (os error 13)",
    );
}

fn vf(args: &[&std::ffi::OsStr]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_vf"))
        .args(args)
        .output()
        .unwrap()
}
fn cli_error(output: &Output) -> Error {
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let value: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["ok"], false);
    serde_json::from_value(value["error"].clone()).unwrap()
}

#[test]
fn unreadable_cli_input_is_input_io() {
    let missing = manifest().join("fixtures/a48-missing.json");
    let error = cli_error(&vf(&["layout".as_ref(), missing.as_ref()]));
    assert_reason(&error, "INPUT_IO", "No such file or directory (os error 2)");
    // A directory opens but cannot be read.
    let error = cli_error(&vf(&["layout".as_ref(), manifest().as_ref()]));
    assert_reason(&error, "INPUT_IO", "Is a directory (os error 21)");
}

#[test]
fn malformed_resident_export_request_is_json_input() {
    // The outer input decodes; only the source request is malformed.
    let dir = temp_dir();
    let input = dir.join("input.json");
    for kind in ["checkpoint", "resident"] {
        fs::write(
            &input,
            serde_json::to_vec(&json!({
                "source": {"kind": kind, "request": {}},
                "output": dir.join("out.safetensors"),
            }))
            .unwrap(),
        )
        .unwrap();
        let error = cli_error(&vf(&["resident-export".as_ref(), input.as_ref()]));
        assert_eq!(error.code, "JSON_INPUT", "{error}");
        assert!(error.message.starts_with("missing field `"), "{error}");
    }
    fs::remove_file(&input).unwrap();
    fs::remove_dir(&dir).unwrap();
}

#[test]
fn closed_output_pipe_reports_the_broken_pipe() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_vf"))
        .current_dir(manifest())
        .args([
            "resident-import",
            "fixtures/resident_import_qwen_expert0.json",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdout.take());
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stderr),
        "JSON output failed: Broken pipe (os error 32)\n"
    );
}

#[test]
fn export_write_failure_never_publishes() {
    // A file-size limit makes the staged write fail like a full disk. SIGXFSZ is
    // ignored so the write returns EFBIG instead of killing the process.
    let dir = temp_dir();
    let input = dir.join("input.json");
    let output = dir.join("out.safetensors");
    let request = fs::read(manifest().join("fixtures/resident_import_qwen_expert0.json")).unwrap();
    let mut request: Value = serde_json::from_slice(&request).unwrap();
    request["path"] = json!(manifest().join(request["path"].as_str().unwrap()));
    fs::write(
        &input,
        serde_json::to_vec(&json!({
            "source": {"kind": "checkpoint", "request": request},
            "output": output,
        }))
        .unwrap(),
    )
    .unwrap();
    let result = Command::new("sh")
        .args(["-c", "trap '' XFSZ && ulimit -f 1 && exec \"$@\"", "sh"])
        .arg(env!("CARGO_BIN_EXE_vf"))
        .arg("resident-export")
        .arg(&input)
        .output()
        .unwrap();
    let left: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    fs::remove_file(&input).unwrap();
    fs::remove_dir(&dir).unwrap();
    let error = cli_error(&result);
    assert_eq!(error.code, "ARTIFACT_SERIALIZE", "{error}");
    assert!(
        error
            .message
            .starts_with("SafeTensors serialization failed: ")
            && error.message.contains("File too large (os error 27)"),
        "{error}"
    );
    assert_eq!(left, vec![std::ffi::OsString::from("input.json")]);
}

fn worker() -> Command {
    Command::new(env!("CARGO_BIN_EXE_vf-rank-worker"))
}
fn request_line() -> Vec<u8> {
    let p = common::problem();
    let dispatch = routing::dispatch(&routing::from_dense(&p)).unwrap();
    let request = rank_request(&p, &dispatch.ranks[0], 1, Fault::None).unwrap();
    let mut line = serde_json::to_vec(&request).unwrap();
    line.push(b'\n');
    line
}
fn worker_failed(output: &Output, stderr: &str) {
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");
    assert_eq!(String::from_utf8_lossy(&output.stderr), stderr);
}
fn piped_worker(mut command: Command, input: &[u8]) -> Output {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn worker_request_read_failure_is_worker_io() {
    let output = worker()
        .stdin(fs::File::open(manifest()).unwrap())
        .output()
        .unwrap();
    worker_failed(
        &output,
        "WORKER_IO: read pipe: Is a directory (os error 21)\n",
    );
}

#[test]
fn malformed_worker_request_is_worker_protocol() {
    let output = piped_worker(worker(), b"{\n");
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.starts_with("WORKER_PROTOCOL: EOF while parsing an object"),
        "{stderr}"
    );
}

#[test]
fn worker_request_eof_check_failure_is_worker_io() {
    // The whole request is already queued; the empty nonblocking socket then
    // fails the one-request check instead of reporting end of input.
    let (mut ours, theirs) = UnixStream::pair().unwrap();
    ours.write_all(&request_line()).unwrap();
    theirs.set_nonblocking(true).unwrap();
    let output = worker().stdin(OwnedFd::from(theirs)).output().unwrap();
    drop(ours);
    worker_failed(
        &output,
        "WORKER_IO: request EOF: Resource temporarily unavailable (os error 11)\n",
    );
}

#[test]
fn worker_event_write_failure_is_worker_io() {
    let full = fs::OpenOptions::new()
        .write(true)
        .open("/dev/full")
        .unwrap();
    let mut command = worker();
    command.stdout(full);
    let mut child = command
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&request_line())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    worker_failed(
        &output,
        "WORKER_IO: write worker event: No space left on device (os error 28)\n",
    );
}

#[test]
fn missing_worker_executable_names_the_spawn_failure() {
    let mut problem = common::problem();
    problem.stamp.run_id = "a48-missing-worker".into();
    let p = ProcessProblem {
        problem,
        faults: Default::default(),
        timeout_ms: 5000,
        recovery: false,
    };
    let report = run(&p, Path::new("/definitely-missing-vf-rank-worker")).unwrap();
    assert!(!report.ok);
    assert_reason(
        report.error.as_ref().unwrap(),
        "WORKER_IO",
        "spawn rank worker: No such file or directory (os error 2)",
    );
}
