//! A53: persistent sessions holding compressed QPN2 residents. LOAD_ARTIFACT carries the
//! bytes of a `QPN2_RESIDENT_V1` SafeTensors artifact once, named by their SHA-256; each
//! EXEC computes on the compressed planes, never on a decoded weight plane, and must equal
//! the dense f64 oracle on the dequantized weights bit for bit.
#![cfg(target_os = "linux")]
use safetensors::tensor::{serialize, Dtype, SafeTensors, TensorView};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use voltafabric::moe::{self, Choice, Registry, Replica, Stamp};
use voltafabric::process::{self, Fault, ReceiptState, TokenInput};
use voltafabric::resident::{self, ResidentExpert, ResidentInput, ResidentStack};
use voltafabric::session::{
    self, CompressedArtifact, CompressedSeriesInput, CompressedSeriesReport, Endpoint, Reply,
    Request, SeriesFault, Session,
};
use voltafabric::{base64, checkpoint, quant, resident_artifact, routing, sha256, Error};

const T: Duration = Duration::from_secs(20);
const HIDDEN: u64 = 128;
const INTER: u64 = 64;
const REVISION: &str = "rev-a53";
const LAYER: u32 = 3;
const EPOCH: u64 = 4;

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "vf-a53-{}-{}",
        std::process::id(),
        NEXT_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}
struct ProjectTempDir(PathBuf);
impl Drop for ProjectTempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn project_temp_dir() -> ProjectTempDir {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(".work");
    std::fs::create_dir_all(&root).unwrap();
    loop {
        let path = root.join(format!(
            "a173-raw-series-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        match std::fs::create_dir(&path) {
            Ok(()) => return ProjectTempDir(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => panic!("cannot create project-local test directory: {error}"),
        }
    }
}
fn endpoint(rank: u32) -> Endpoint {
    Endpoint {
        rank,
        argv: vec![env!("CARGO_BIN_EXE_vf-rank-session").into()],
    }
}
fn pc1_endpoint(rank: u32, worker: &str) -> Endpoint {
    assert!(
        Path::new(worker).is_absolute(),
        "PC1 worker path must be absolute"
    );
    assert!(
        !worker.chars().any(char::is_whitespace),
        "PC1 worker path must not contain whitespace"
    );
    Endpoint {
        rank,
        argv: vec![
            "ssh".into(),
            "-T".into(),
            "-oBatchMode=yes".into(),
            "-oStrictHostKeyChecking=yes".into(),
            "-oConnectTimeout=8".into(),
            "pc1".into(),
            worker.into(),
        ],
    }
}
fn assert_reason(error: &Error, code: &str, message: &str) {
    assert_eq!(
        (error.code.as_str(), error.message.as_str()),
        (code, message)
    );
}
fn refused(reply: Reply) -> Error {
    match reply {
        Reply::Refused { error } => error,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

// ------------------------------------------------------------------ fixtures

/// Deterministic NVFP4 plane: arbitrary codes, unsigned finite E4M3 scales in 0.25..2.
fn plane(rows: u64, columns: u64, seed: u64, global: f64) -> quant::PackedRows {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let n = (rows * columns) as usize;
    quant::PackedRows {
        rows,
        columns,
        codes: (0..n / 2).map(|_| next() as u8).collect(),
        scales: (0..n / 16).map(|_| 0x28 + (next() % 25) as u8).collect(),
        global_dequant: global,
    }
}
fn expert(id: u32, local: u32) -> ResidentExpert {
    let seed = u64::from(id) * 3;
    ResidentExpert {
        id,
        local,
        gate: plane(INTER, HIDDEN, seed, 0.125),
        up: plane(INTER, HIDDEN, seed + 1, 0.0625),
        down: plane(HIDDEN, INTER, seed + 2, 0.25),
    }
}
/// Rank 2 holds experts 10 (local 7) and 99 (local 3) in physical order [7, 3];
/// rank 9 holds 25 (local 0) and 42 (local 5) in order [5, 0].
fn registry(epoch: u64) -> Registry {
    Registry {
        epoch,
        copies: vec![
            Replica {
                expert: 10,
                rank: 2,
                local: 7,
            },
            Replica {
                expert: 99,
                rank: 2,
                local: 3,
            },
            Replica {
                expert: 25,
                rank: 9,
                local: 0,
            },
            Replica {
                expert: 42,
                rank: 9,
                local: 5,
            },
        ],
    }
}
fn order(rank: u32) -> Vec<u32> {
    if rank == 2 {
        vec![7, 3]
    } else {
        vec![5, 0]
    }
}
fn input(rank: u32) -> ResidentInput {
    let experts = registry(EPOCH)
        .copies
        .into_iter()
        .filter(|r| r.rank == rank)
        .map(|r| expert(r.expert, r.local))
        .collect();
    ResidentInput {
        model_revision: REVISION.into(),
        layer: LAYER,
        rank,
        registry: registry(EPOCH),
        resident_order: order(rank),
        experts,
    }
}
fn stack(rank: u32) -> ResidentStack {
    resident::pack(&input(rank)).unwrap()
}
fn artifact(rank: u32) -> CompressedArtifact {
    CompressedArtifact::from_bytes(resident_artifact::encode(&stack(rank)).unwrap()).unwrap()
}
fn routing_problem(run_id: &str) -> routing::RoutingProblem {
    let ids = [10_u32, 25, 42, 99];
    routing::RoutingProblem {
        hidden: HIDDEN as usize,
        experts: ids.to_vec(),
        routes: (0..6)
            .map(|t| {
                vec![
                    Choice {
                        expert: ids[t % 4],
                        weight: 0.75 - t as f64 / 32.0,
                    },
                    Choice {
                        expert: ids[(t + 1 + t / 4) % 4],
                        weight: 0.25 + t as f64 / 64.0,
                    },
                ]
            })
            .collect(),
        registry: registry(EPOCH),
        stamp: Stamp {
            run_id: run_id.into(),
            model_revision: REVISION.into(),
            plan_id: "plan-a53".into(),
            layer: LAYER,
            batch: 0,
            epoch: EPOCH,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(2, order(2)), (9, order(9))]),
    }
}
fn activations(tokens: usize, hidden: usize, seed: usize) -> Vec<Vec<f64>> {
    (0..tokens)
        .map(|t| {
            (0..hidden)
                .map(|i| (((t * 131 + i * 17 + seed) % 97) as f64 - 48.0) / 64.0)
                .collect()
        })
        .collect()
}
fn write_artifacts(dir: &Path) -> Vec<PathBuf> {
    [2, 9]
        .into_iter()
        .map(|rank| {
            let path = dir.join(format!("rank{rank}.safetensors"));
            std::fs::write(&path, resident_artifact::encode(&stack(rank)).unwrap()).unwrap();
            path
        })
        .collect()
}
fn series(run_id: &str, dir: &Path, batches: u64) -> CompressedSeriesInput {
    CompressedSeriesInput {
        artifacts: write_artifacts(dir),
        routing: routing_problem(run_id),
        inputs: activations(6, HIDDEN as usize, 5),
        inputs_fp16_bits: None,
        workers: vec![endpoint(2), endpoint(9)],
        batches,
        timeout_ms: 20_000,
        epoch_every: 50,
        warmup: 0,
        tolerance_abs: 0.0,
        fp16_completions: false,
        faults: vec![],
        record: vec![],
    }
}
fn real_stack() -> ResidentStack {
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/resident_import_qwen_expert0.json");
    let mut request: checkpoint::Request =
        serde_json::from_slice(&std::fs::read(fixture).unwrap()).unwrap();
    request.path = Path::new(env!("CARGO_MANIFEST_DIR")).join(&request.path);
    checkpoint::load(&request).unwrap().stack
}
fn real_qwen_checkpoint_series(dir: &Path) -> CompressedSeriesInput {
    let stack = real_stack();
    assert_eq!(stack.physical_experts().len(), 1);
    let physical = stack.physical_experts()[0];
    let rank = stack.rank();
    let epoch = stack.epoch();
    let revision = stack.model_revision().to_owned();
    let layer = stack.layer();
    let resident_order = stack.resident_order().to_vec();
    let (_, _, hidden, _, _) = stack.shape();
    let path = dir.join("qwen36-expert0-resident.qpn2.safetensors");
    std::fs::write(&path, resident_artifact::encode(&stack).unwrap()).unwrap();
    CompressedSeriesInput {
        artifacts: vec![path],
        routing: routing::RoutingProblem {
            hidden: hidden as usize,
            experts: vec![physical.id()],
            routes: (0..6)
                .map(|_| {
                    vec![Choice {
                        expert: physical.id(),
                        weight: 1.0,
                    }]
                })
                .collect(),
            registry: Registry {
                epoch,
                copies: vec![Replica {
                    expert: physical.id(),
                    rank,
                    local: physical.local(),
                }],
            },
            stamp: Stamp {
                run_id: "a53-real-qwen36-checkpoint".into(),
                model_revision: revision,
                plan_id: "plan-real-qwen36-expert0".into(),
                layer,
                batch: 0,
                epoch,
            },
            preferred_owners: BTreeMap::new(),
            resident_order: BTreeMap::from([(rank, resident_order)]),
        },
        inputs: activations(6, hidden as usize, 3),
        inputs_fp16_bits: None,
        workers: vec![endpoint(rank)],
        batches: 3,
        timeout_ms: 20_000,
        epoch_every: 2,
        warmup: 0,
        tolerance_abs: 0.0,
        fp16_completions: false,
        faults: vec![],
        record: vec![0, 1, 2],
    }
}
/// Dense oracle from the row-major source planes, independent of the QPN2 layout.
fn dense_from_rows(e: &ResidentExpert) -> moe::Expert {
    let layout = |p: &quant::PackedRows| {
        let w = quant::dequantize(p).unwrap();
        let rows = p.rows as usize;
        let mut out = vec![0.0; rows * p.columns as usize];
        for (r, values) in w.iter().enumerate() {
            for (c, &v) in values.iter().enumerate() {
                out[c * rows + r] = v;
            }
        }
        out
    };
    moe::Expert {
        id: e.id,
        gate: layout(&e.gate),
        up: layout(&e.up),
        down: layout(&e.down),
    }
}
fn dims(hidden: u64, intermediate: u64) -> moe::Model {
    moe::Model {
        hidden: hidden as usize,
        intermediate: intermediate as usize,
        experts: vec![],
    }
}
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

// ------------------------------------------------------------------ base64

#[test]
fn base64_round_trips_every_byte_and_length() {
    let all: Vec<u8> = (0..=255).collect();
    for len in 0..all.len() {
        let text = base64::encode(&all[..len]);
        assert_eq!(text.len(), len.div_ceil(3) * 4);
        assert_eq!(base64::decode(&text).unwrap(), &all[..len]);
    }
}

#[test]
fn base64_refuses_every_noncanonical_text() {
    for text in [
        "Zg=", "Zh==", "Zm9=", "Z===", "====", "Zg==Zg==", "Zg=A", "Z@==", " Zg=", "Zm9v\n",
        "Zm9v====",
    ] {
        assert_reason(
            &base64::decode(text).unwrap_err(),
            "ARTIFACT_ENCODING",
            "artifact is not canonical padded base64",
        );
    }
}

fn raw_artifact_wire(artifact: &CompressedArtifact, declared: u64, body: &[u8]) -> Vec<u8> {
    let mut wire = serde_json::to_vec(&Request::Hello {
        protocol: session::PROTOCOL,
        rank: artifact.stack().rank(),
    })
    .unwrap();
    wire.push(b'\n');
    let header = serde_json::json!({
        "op": "load_artifact_raw",
        "artifact_sha256": artifact.sha256(),
        "epoch": artifact.stack().epoch(),
        "artifact_bytes": declared,
    });
    wire.extend_from_slice(&serde_json::to_vec(&header).unwrap());
    wire.push(b'\n');
    wire.extend_from_slice(body);
    wire
}

fn raw_exec_request(
    artifact: &CompressedArtifact,
    problem: &routing::RoutingProblem,
) -> (Request, Vec<u8>, usize) {
    let dispatch = routing::dispatch(problem)
        .unwrap()
        .ranks
        .into_iter()
        .find(|rank| rank.rank == artifact.stack().rank())
        .unwrap();
    let slot_count = dispatch.slots.len();
    let tokens: std::collections::BTreeSet<_> =
        dispatch.slots.iter().map(|slot| slot.slot.token).collect();
    let rows = activations(problem.routes.len(), problem.hidden, 5);
    let mut inputs = Vec::new();
    let mut body = Vec::new();
    for token in tokens {
        inputs.push(session::RawTokenInputMeta {
            token,
            values: rows[token].len() as u64,
        });
        for value in &rows[token] {
            body.extend_from_slice(&value.to_le_bytes());
        }
    }
    let request = Request::ExecRaw {
        artifact_sha256: artifact.sha256().into(),
        dispatch,
        attempt: 1,
        inputs,
        activation_bytes: body.len() as u64,
        fault: Fault::None,
        raw_completions: false,
        raw_completion_batch: false,
        raw_completion_fp16_batch: false,
    };
    (request, body, slot_count)
}

fn raw_fp16_exec_request(
    artifact: &CompressedArtifact,
    problem: &routing::RoutingProblem,
) -> Request {
    let (request, _, _) = raw_exec_request(artifact, problem);
    let Request::ExecRaw {
        artifact_sha256,
        dispatch,
        attempt,
        inputs,
        fault,
        ..
    } = request
    else {
        unreachable!("raw_exec_request returns ExecRaw")
    };
    let activation_bytes = inputs
        .iter()
        .map(|input| input.values * std::mem::size_of::<u16>() as u64)
        .sum();
    Request::ExecRawFp16 {
        artifact_sha256,
        dispatch,
        attempt,
        inputs,
        activation_bytes,
        fault,
        raw_completions: false,
        raw_completion_batch: false,
        raw_completion_fp16_batch: false,
    }
}

fn raw_exec_wire(artifact: &CompressedArtifact, request: &Request, body: &[u8]) -> Vec<u8> {
    let mut wire = raw_artifact_wire(artifact, artifact.bytes().len() as u64, artifact.bytes());
    wire.extend_from_slice(&serde_json::to_vec(request).unwrap());
    wire.push(b'\n');
    wire.extend_from_slice(body);
    wire
}

fn replies(output: &[u8]) -> Vec<Reply> {
    output
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect()
}

#[test]
fn raw_artifact_frame_consumes_exact_bytes_including_newlines() {
    let artifact = artifact(2);
    assert!(artifact.bytes().contains(&b'\n'));
    let mut wire = raw_artifact_wire(&artifact, artifact.bytes().len() as u64, artifact.bytes());
    wire.extend_from_slice(&serde_json::to_vec(&Request::Shutdown).unwrap());
    wire.push(b'\n');
    let mut input = std::io::Cursor::new(wire);
    let mut output = Vec::new();
    session::serve(&mut input, &mut output).unwrap();
    let replies = replies(&output);
    assert!(matches!(replies[0], Reply::Hello { .. }));
    assert!(matches!(
        replies[1],
        Reply::Loaded {
            epoch: EPOCH,
            experts: 2,
            transferred: true,
            ..
        }
    ));
    assert!(matches!(replies[2], Reply::Bye { execs: 0, .. }));
}

#[test]
fn raw_artifact_frame_rejects_truncation_oversize_zero_and_bad_digest() {
    let artifact = artifact(2);

    let mut truncated = std::io::Cursor::new(raw_artifact_wire(
        &artifact,
        artifact.bytes().len() as u64,
        &artifact.bytes()[..artifact.bytes().len() - 1],
    ));
    let mut output = Vec::new();
    let error = session::serve(&mut truncated, &mut output).unwrap_err();
    assert_eq!(error.code, "ARTIFACT_TRUNCATED");
    assert!(
        matches!(replies(&output).last(), Some(Reply::Refused { error }) if error.code == "ARTIFACT_TRUNCATED")
    );

    for (declared, expected) in [
        (0, "INVALID_INPUT"),
        (
            session::MAX_RAW_ARTIFACT_BYTES as u64 + 1,
            "MESSAGE_TOO_LARGE",
        ),
    ] {
        let mut wire = raw_artifact_wire(&artifact, declared, &[]);
        let mut input = std::io::Cursor::new(&mut wire);
        let mut output = Vec::new();
        let error = session::serve(&mut input, &mut output).unwrap_err();
        assert_eq!(error.code, expected);
        assert!(
            matches!(replies(&output).last(), Some(Reply::Refused { error }) if error.code == expected)
        );
    }

    let mut wrong_digest =
        raw_artifact_wire(&artifact, artifact.bytes().len() as u64, artifact.bytes());
    let digest_start = wrong_digest
        .windows(64)
        .position(|window| window == artifact.sha256().as_bytes())
        .unwrap();
    wrong_digest[digest_start] = if wrong_digest[digest_start] == b'0' {
        b'1'
    } else {
        b'0'
    };
    let mut input = std::io::Cursor::new(wrong_digest);
    let mut output = Vec::new();
    let error = session::serve(&mut input, &mut output).unwrap_err();
    assert_eq!(error.code, "SESSION_ABANDONED");
    assert!(
        matches!(replies(&output).last(), Some(Reply::Refused { error }) if error.code == "ARTIFACT_DIGEST_MISMATCH")
    );
    assert!(!replies(&output)
        .iter()
        .any(|reply| matches!(reply, Reply::Loaded { .. })));
}

#[test]
fn raw_exec_frame_decodes_little_endian_values_and_completes() {
    let artifact = artifact(2);
    let p = routing_problem("a55-raw-exec-wire");
    let (request, body, expected_slots) = raw_exec_request(&artifact, &p);
    let mut wire = raw_exec_wire(&artifact, &request, &body);
    wire.extend_from_slice(&serde_json::to_vec(&Request::Shutdown).unwrap());
    wire.push(b'\n');
    let mut input = std::io::Cursor::new(wire);
    let mut output = Vec::new();
    session::serve(&mut input, &mut output).unwrap();
    let replies = replies(&output);
    assert!(replies.iter().any(|reply| matches!(
        reply,
        Reply::Event { event }
            if matches!(&event.event, process::EventKind::Finished { slots } if *slots == expected_slots)
    )));
    assert!(replies
        .iter()
        .any(|reply| matches!(reply, Reply::Bye { execs: 1, .. })));
}

#[test]
fn raw_exec_frame_refuses_truncation_oversize_and_length_mismatch() {
    let artifact = artifact(2);
    let p = routing_problem("a55-raw-exec-refusal");
    let (request, body, _) = raw_exec_request(&artifact, &p);

    let mut truncated = raw_exec_wire(&artifact, &request, &body[..body.len() - 1]);
    let mut input = std::io::Cursor::new(&mut truncated);
    let mut output = Vec::new();
    let error = session::serve(&mut input, &mut output).unwrap_err();
    assert_eq!(error.code, "EXEC_TRUNCATED");
    assert!(
        matches!(replies(&output).last(), Some(Reply::Refused { error }) if error.code == "EXEC_TRUNCATED")
    );
    assert!(!replies(&output)
        .iter()
        .any(|reply| matches!(reply, Reply::Event { .. })));

    let mut oversized = request.clone();
    if let Request::ExecRaw {
        activation_bytes, ..
    } = &mut oversized
    {
        *activation_bytes = session::MAX_RAW_ACTIVATION_BYTES as u64 + 1;
    }
    let mut wire = raw_exec_wire(&artifact, &oversized, &[]);
    let mut input = std::io::Cursor::new(&mut wire);
    let mut output = Vec::new();
    let error = session::serve(&mut input, &mut output).unwrap_err();
    assert_eq!(error.code, "MESSAGE_TOO_LARGE");
    assert!(
        matches!(replies(&output).last(), Some(Reply::Refused { error }) if error.code == "MESSAGE_TOO_LARGE")
    );

    let mut mismatch = request;
    if let Request::ExecRaw {
        activation_bytes, ..
    } = &mut mismatch
    {
        *activation_bytes += 8;
    }
    let mut wire = raw_exec_wire(&artifact, &mismatch, &[]);
    let mut input = std::io::Cursor::new(&mut wire);
    let mut output = Vec::new();
    let error = session::serve(&mut input, &mut output).unwrap_err();
    assert_eq!(error.code, "INVALID_INPUT");
    assert!(
        matches!(replies(&output).last(), Some(Reply::Refused { error }) if error.code == "INVALID_INPUT")
    );
}

#[test]
fn raw_fp16_exec_worker_refuses_nonfinite_and_truncated_bodies_and_closes() {
    let artifact = artifact(2);
    let problem = routing_problem("a169-raw-fp16-worker-refusal");
    let request = raw_fp16_exec_request(&artifact, &problem);
    let Request::ExecRawFp16 {
        activation_bytes, ..
    } = &request
    else {
        unreachable!("raw_fp16_exec_request returns ExecRawFp16")
    };
    let body_len = usize::try_from(*activation_bytes).unwrap();

    let mut nonfinite_body = vec![0_u8; body_len];
    nonfinite_body[..2].copy_from_slice(&0xfc00_u16.to_le_bytes());
    let mut nonfinite_wire =
        raw_artifact_wire(&artifact, artifact.bytes().len() as u64, artifact.bytes());
    nonfinite_wire.extend_from_slice(&serde_json::to_vec(&request).unwrap());
    nonfinite_wire.push(b'\n');
    nonfinite_wire.extend_from_slice(&nonfinite_body);
    let mut nonfinite_input = std::io::Cursor::new(nonfinite_wire);
    let mut nonfinite_output = Vec::new();
    let error = session::serve(&mut nonfinite_input, &mut nonfinite_output).unwrap_err();
    assert_eq!(error.code, "INVALID_INPUT");
    let nonfinite_replies = replies(&nonfinite_output);
    assert!(matches!(nonfinite_replies[0], Reply::Hello { .. }));
    assert!(matches!(nonfinite_replies[1], Reply::Loaded { .. }));
    assert!(matches!(
        nonfinite_replies.last(),
        Some(Reply::Refused { error }) if error.code == "INVALID_INPUT"
    ));
    assert!(!nonfinite_replies
        .iter()
        .any(|reply| matches!(reply, Reply::Event { .. } | Reply::Bye { .. })));

    let truncated_wire =
        raw_artifact_wire(&artifact, artifact.bytes().len() as u64, artifact.bytes());
    let mut truncated_wire = truncated_wire;
    truncated_wire.extend_from_slice(&serde_json::to_vec(&request).unwrap());
    truncated_wire.push(b'\n');
    truncated_wire.extend(std::iter::repeat_n(0, body_len - 1));
    let mut truncated_input = std::io::Cursor::new(truncated_wire);
    let mut truncated_output = Vec::new();
    let error = session::serve(&mut truncated_input, &mut truncated_output).unwrap_err();
    assert_eq!(error.code, "EXEC_TRUNCATED");
    let truncated_replies = replies(&truncated_output);
    assert!(matches!(
        truncated_replies.last(),
        Some(Reply::Refused { error }) if error.code == "EXEC_TRUNCATED"
    ));
    assert!(!truncated_replies
        .iter()
        .any(|reply| matches!(reply, Reply::Event { .. } | Reply::Bye { .. })));
}

#[test]
fn raw_fp16_exec_process_refuses_nonfinite_before_execution_and_exits() {
    use std::io::Write;

    let artifact = artifact(2);
    let problem = routing_problem("a169-raw-fp16-process-refusal");
    let request = raw_fp16_exec_request(&artifact, &problem);
    let Request::ExecRawFp16 {
        activation_bytes, ..
    } = &request
    else {
        unreachable!("raw_fp16_exec_request returns ExecRawFp16")
    };
    let mut body = vec![0_u8; usize::try_from(*activation_bytes).unwrap()];
    body[..2].copy_from_slice(&0x7c00_u16.to_le_bytes()); // +Inf in IEEE binary16
    let wire = raw_exec_wire(&artifact, &request, &body);

    let mut child = Command::new(env!("CARGO_BIN_EXE_vf-rank-session"))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&wire).unwrap();
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    let replies = replies(&output.stdout);
    assert!(matches!(replies.first(), Some(Reply::Hello { .. })));
    assert!(matches!(replies.get(1), Some(Reply::Loaded { .. })));
    assert!(matches!(
        replies.last(),
        Some(Reply::Refused { error }) if error.code == "INVALID_INPUT"
    ));
    assert!(!replies
        .iter()
        .any(|reply| matches!(reply, Reply::Event { .. } | Reply::Bye { .. })));
    assert!(String::from_utf8_lossy(&output.stderr).contains("INVALID_INPUT"));
}

#[test]
fn raw_exec_nonfinite_values_are_rejected_before_any_completion() {
    let artifact = artifact(2);
    let p = routing_problem("a55-raw-exec-nonfinite");
    let (request, mut body, _) = raw_exec_request(&artifact, &p);
    body[..8].copy_from_slice(&f64::NAN.to_le_bytes());
    let mut wire = raw_exec_wire(&artifact, &request, &body);
    wire.extend_from_slice(&serde_json::to_vec(&Request::Shutdown).unwrap());
    wire.push(b'\n');
    let mut input = std::io::Cursor::new(wire);
    let mut output = Vec::new();
    session::serve(&mut input, &mut output).unwrap();
    let replies = replies(&output);
    assert!(replies.iter().any(|reply| matches!(
        reply,
        Reply::Event { event }
            if matches!(&event.event, process::EventKind::Rejected { error } if error.code == "INVALID_INPUT")
    )));
    assert!(!replies.iter().any(|reply| matches!(
        reply,
        Reply::Event { event }
            if matches!(&event.event, process::EventKind::Completion { .. } | process::EventKind::Finished { .. })
    )));
    assert!(replies
        .iter()
        .any(|reply| matches!(reply, Reply::Bye { execs: 0, .. })));
}

// ------------------------------------------------------------------ artifacts

#[test]
fn artifact_encoding_is_the_exported_file_and_imports_to_the_same_stack() {
    let original = stack(2);
    let bytes = resident_artifact::encode(&original).unwrap();
    let dir = temp_dir();
    let path = dir.join("artifact.safetensors");
    resident_artifact::export(&original, &path).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    let imported = resident_artifact::import(&bytes).unwrap();
    assert_eq!(
        serde_json::to_value(&imported).unwrap(),
        serde_json::to_value(&original).unwrap()
    );
    assert_eq!(resident_artifact::encode(&imported).unwrap(), bytes);
}

#[test]
fn real_checkpoint_expert_round_trips_through_its_artifact() {
    let original = real_stack();
    assert_eq!(original.shape(), (1, 1024, 2048, 2048, 512));
    let bytes = resident_artifact::encode(&original).unwrap();
    let imported = resident_artifact::import(&bytes).unwrap();
    assert_eq!(imported.codes13(), original.codes13());
    assert_eq!(imported.scales13(), original.scales13());
    assert_eq!(imported.codes2(), original.codes2());
    assert_eq!(imported.scales2(), original.scales2());
    assert_eq!(imported.gscales13(), original.gscales13());
    assert_eq!(imported.gscales2(), original.gscales2());
    assert_eq!(imported.payload_bytes(), 1_769_856);
    assert_eq!(resident_artifact::encode(&imported).unwrap(), bytes);
}

type Tensors = BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)>;

/// Re-serialize a valid artifact after `change` edits its tensors or metadata.
fn rebuild(
    bytes: &[u8],
    change: impl FnOnce(&mut Tensors, &mut HashMap<String, String>),
) -> Vec<u8> {
    let file = SafeTensors::deserialize(bytes).unwrap();
    let mut tensors: Tensors = file
        .tensors()
        .into_iter()
        .map(|(name, v)| (name, (v.dtype(), v.shape().to_vec(), v.data().to_vec())))
        .collect();
    let (_, header) = SafeTensors::read_metadata(bytes).unwrap();
    let mut metadata = header.metadata().clone().unwrap();
    change(&mut tensors, &mut metadata);
    let views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, (dtype, shape, data))| {
            (
                name.clone(),
                TensorView::new(*dtype, shape.clone(), data).unwrap(),
            )
        })
        .collect();
    serialize(views, Some(metadata)).unwrap()
}
fn edit_info(meta: &mut HashMap<String, String>, edit: impl FnOnce(&mut serde_json::Value)) {
    let mut info: serde_json::Value = serde_json::from_str(&meta["vf_info"]).unwrap();
    edit(&mut info);
    meta.insert("vf_info".into(), info.to_string());
}
fn set_f32(t: &mut Tensors, name: &str, index: usize, value: f32) {
    t.get_mut(name).unwrap().2[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
}

#[test]
fn import_refuses_foreign_malformed_or_noncanonical_artifacts() {
    let good = resident_artifact::encode(&stack(2)).unwrap();
    let cases: Vec<(&str, Vec<u8>, &str, &str)> = vec![
        (
            "not safetensors",
            b"hello".to_vec(),
            "ARTIFACT_INVALID",
            "not a SafeTensors file",
        ),
        (
            "truncated",
            good[..good.len() - 1].to_vec(),
            "ARTIFACT_INVALID",
            "not a SafeTensors file",
        ),
        (
            "schema",
            rebuild(&good, |_, m| {
                m.insert("vf_schema".into(), "QPN2_RESIDENT_V0".into());
            }),
            "ARTIFACT_INVALID",
            "artifact schema is not QPN2_RESIDENT_V1",
        ),
        (
            "no info",
            rebuild(&good, |_, m| {
                m.remove("vf_info");
            }),
            "ARTIFACT_INVALID",
            "artifact lacks vf_schema or vf_info",
        ),
        (
            "no schema",
            rebuild(&good, |_, m| {
                m.remove("vf_schema");
            }),
            "ARTIFACT_INVALID",
            "artifact lacks vf_schema or vf_info",
        ),
        (
            "missing tensor",
            rebuild(&good, |t, _| {
                t.remove("gscales2");
            }),
            "ARTIFACT_INVALID",
            "a QPN2 resident tensor is missing",
        ),
        (
            "u8 dtype",
            rebuild(&good, |t, _| {
                t.get_mut("codes13").unwrap().0 = Dtype::I8;
            }),
            "ARTIFACT_INVALID",
            "QPN2 code and scale tensors are one-dimensional U8",
        ),
        (
            "u8 rank",
            rebuild(&good, |t, _| {
                let s = t.get_mut("scales2").unwrap();
                s.1 = vec![2, s.2.len() / 2];
            }),
            "ARTIFACT_INVALID",
            "QPN2 code and scale tensors are one-dimensional U8",
        ),
        (
            "f32 shape",
            rebuild(&good, |t, _| {
                let s = t.get_mut("gscales13").unwrap();
                s.1 = vec![s.2.len() / 4];
            }),
            "ARTIFACT_INVALID",
            "tile factor tensors are F32 [experts, tiles]",
        ),
        (
            "f32 dtype",
            rebuild(&good, |t, _| {
                t.get_mut("gscales2").unwrap().0 = Dtype::I32;
            }),
            "ARTIFACT_INVALID",
            "tile factor tensors are F32 [experts, tiles]",
        ),
        (
            "nan scale byte",
            rebuild(&good, |t, _| {
                t.get_mut("scales13").unwrap().2[5] = 0x7f;
            }),
            "ARTIFACT_INVALID",
            "scale byte is not an unsigned finite E4M3 value",
        ),
        (
            "negative scale byte",
            rebuild(&good, |t, _| {
                t.get_mut("scales2").unwrap().2[9] = 0x80;
            }),
            "ARTIFACT_INVALID",
            "scale byte is not an unsigned finite E4M3 value",
        ),
        (
            "mixed factor",
            rebuild(&good, |t, _| set_f32(t, "gscales13", 1, 0.5)),
            "ARTIFACT_INVALID",
            "tile factors must be one positive finite F32 per plane",
        ),
        (
            "zero factor",
            rebuild(&good, |t, _| {
                for i in 0..2 {
                    set_f32(t, "gscales2", 4 + i, 0.0);
                }
            }),
            "ARTIFACT_INVALID",
            "tile factors must be one positive finite F32 per plane",
        ),
        (
            "nan factor",
            rebuild(&good, |t, _| {
                for i in 0..4 {
                    set_f32(t, "gscales2", i, f32::NAN);
                }
            }),
            "ARTIFACT_INVALID",
            "tile factors must be one positive finite F32 per plane",
        ),
        (
            "payload",
            rebuild(&good, |_, m| {
                edit_info(m, |i| {
                    i["payload_bytes"] = (i["payload_bytes"].as_u64().unwrap() + 1).into()
                })
            }),
            "ARTIFACT_INVALID",
            "buffer lengths or payload size differ from the declared shape",
        ),
        (
            "shape",
            rebuild(&good, |_, m| edit_info(m, |i| i["shape"][2] = 96.into())),
            "ARTIFACT_INVALID",
            "shape is not an aligned, bounded, uniform SwiGLU stack",
        ),
        (
            "order",
            rebuild(&good, |_, m| {
                edit_info(m, |i| i["resident_order"] = serde_json::json!([3, 7]))
            }),
            "ARTIFACT_INVALID",
            "duplicate expert or physical order differing from the experts' local IDs",
        ),
        (
            "duplicate expert",
            rebuild(&good, |_, m| {
                edit_info(m, |i| i["physical_experts"][1]["id"] = 10.into())
            }),
            "ARTIFACT_INVALID",
            "duplicate expert or physical order differing from the experts' local IDs",
        ),
        (
            "revision",
            rebuild(&good, |_, m| {
                edit_info(m, |i| i["model_revision"] = " ".into())
            }),
            "ARTIFACT_INVALID",
            "resident identity or metadata exceeds bounded contract",
        ),
        (
            "unknown info field",
            rebuild(&good, |_, m| edit_info(m, |i| i["extra"] = 1.into())),
            "ARTIFACT_INVALID",
            "malformed vf_info",
        ),
        (
            "extra metadata",
            rebuild(&good, |_, m| {
                m.insert("note".into(), "x".into());
            }),
            "ARTIFACT_NOT_CANONICAL",
            "artifact bytes differ from the canonical encoding of their content",
        ),
        (
            "extra tensor",
            rebuild(&good, |t, _| {
                t.insert("padding".into(), (Dtype::U8, vec![1], vec![0]));
            }),
            "ARTIFACT_NOT_CANONICAL",
            "artifact bytes differ from the canonical encoding of their content",
        ),
    ];
    for (name, bytes, code, message) in cases {
        let error = resident_artifact::import(&bytes).unwrap_err();
        assert_eq!(error.code, code, "{name}: {error}");
        assert!(error.message.starts_with(message), "{name}: {error}");
    }
}

#[test]
fn an_empty_rank_artifact_must_carry_the_dimensionless_shape() {
    let mut empty = input(2);
    empty.rank = 11;
    empty.experts.clear();
    empty.resident_order.clear();
    let bytes = resident_artifact::encode(&resident::pack(&empty).unwrap()).unwrap();
    assert!(resident_artifact::import(&bytes).unwrap().is_empty());
    let skewed = rebuild(&bytes, |_, m| edit_info(m, |i| i["shape"][1] = 64.into()));
    assert_reason(
        &resident_artifact::import(&skewed).unwrap_err(),
        "ARTIFACT_INVALID",
        "an empty rank has the shape (0,0,0,0,0)",
    );
    assert_reason(
        &CompressedArtifact::from_bytes(bytes).unwrap_err(),
        "INVALID_INPUT",
        "a session artifact holds at least one expert",
    );
}

// ------------------------------------------------------------------ kernel

#[test]
fn compressed_kernel_equals_the_dense_oracle_bit_for_bit() {
    let source = input(2);
    let stack = stack(2);
    let x = activations(1, HIDDEN as usize, 9).remove(0);
    for (physical, &local) in source.resident_order.iter().enumerate() {
        let e = source.experts.iter().find(|e| e.local == local).unwrap();
        let dense = dense_from_rows(e);
        // The coordinator's oracle path (QPN2 unpack) decodes the same weights.
        let unpacked = stack.dense_expert(physical).unwrap();
        assert_eq!(
            (unpacked.id, &unpacked.gate, &unpacked.up, &unpacked.down),
            (dense.id, &dense.gate, &dense.up, &dense.down)
        );
        let want = moe::expert_value(&dims(HIDDEN, INTER), &dense, &x).unwrap();
        let got = stack.expert_value(physical, &x).unwrap();
        assert_eq!(bits(&got), bits(&want), "physical expert {physical}");
    }
    // The two residents really differ: a wrong physical index cannot pass unseen.
    assert_ne!(
        stack.expert_value(0, &x).unwrap(),
        stack.expert_value(1, &x).unwrap()
    );
}

#[test]
fn real_expert_kernel_equals_the_dense_oracle_bit_for_bit() {
    let stack = real_stack();
    let x = activations(1, 2048, 3).remove(0);
    let dense = stack.dense_expert(0).unwrap();
    let want = moe::expert_value(&dims(2048, 512), &dense, &x).unwrap();
    let got = stack.expert_value(0, &x).unwrap();
    assert_eq!(bits(&got), bits(&want));
    assert!(got.iter().any(|v| *v != 0.0));
}

#[test]
fn kernel_refuses_a_foreign_expert_or_a_malformed_activation() {
    let stack = stack(9);
    let x = activations(1, HIDDEN as usize, 1).remove(0);
    assert_reason(
        &stack.expert_value(2, &x).unwrap_err(),
        "INVALID_INPUT",
        "physical expert outside the resident stack",
    );
    assert_reason(
        &stack.dense_expert(2).unwrap_err(),
        "INVALID_INPUT",
        "physical expert outside the resident stack",
    );
    let message = "activation length differs from the resident hidden size or is nonfinite";
    assert_reason(
        &stack.expert_value(0, &x[1..]).unwrap_err(),
        "INVALID_INPUT",
        message,
    );
    let mut bad = x.clone();
    bad[7] = f64::INFINITY;
    assert_reason(
        &stack.expert_value(1, &bad).unwrap_err(),
        "INVALID_INPUT",
        message,
    );
}

// ------------------------------------------------------------------ sessions

#[test]
fn one_transfer_then_120_exact_batches_with_two_weightless_epoch_renewals() {
    let dir = temp_dir();
    let mut input = series("a53-series", &dir, 120);
    input.record = vec![0, 119];
    let report = session::run_compressed_series(&input).unwrap();
    let s = &report.series;
    assert!(s.ok, "{:?} {:?}", s.error, s.failure);
    assert_eq!(
        (s.batches_ok, s.exact_batches, s.max_abs_diff),
        (120, 120, 0.0)
    );
    assert_eq!(report.artifacts.len(), 2);
    for (summary, traffic) in report.artifacts.iter().zip(&s.traffic) {
        assert_eq!(summary.rank, traffic.rank);
        let t = &traffic.traffic;
        assert_eq!((t.loads_transferred, t.rebinds, t.execs), (1, 2, 120));
        // One raw LOAD body carries the artifact; its bounded header is the only overhead.
        assert!(t.weight_bytes_sent >= summary.file_bytes);
        assert!(t.weight_bytes_sent <= summary.file_bytes + 1024);
    }
    assert!(s
        .close
        .iter()
        .all(|c| c.exit_code == Some(0) && c.bye_execs == Some(120)));
    // Recorded outputs equal the coordinator's dense oracle exactly.
    let artifacts: Vec<_> = [2, 9].map(artifact).into();
    let oracle = session::DenseOracle::new(&artifacts).unwrap();
    assert_eq!(s.recorded.len(), 2);
    for r in &s.recorded {
        let mut p = input.routing.clone();
        p.stamp.batch += r.batch;
        p.stamp.epoch += r.batch / 50;
        p.registry.epoch = p.stamp.epoch;
        let mut x = input.inputs.clone();
        for (t, row) in x.iter_mut().enumerate() {
            for (i, v) in row.iter_mut().enumerate() {
                *v *= 1.0 + ((r.batch as usize * 7 + t * 3 + i) % 11) as f64 / 64.0;
            }
        }
        let want = oracle.output(&p, &x).unwrap();
        assert_eq!(
            bits(&r.output.concat()),
            bits(&want.concat()),
            "batch {}",
            r.batch
        );
    }
}

#[test]
fn compressed_series_cli_accepts_owned_fp16_bits_and_keeps_them_across_batches() {
    let dir = temp_dir();
    let mut base = series("a170-fp16-series-cli", &dir, 3);
    base.epoch_every = 2;
    base.record = vec![0, 1, 2];
    let mut fp16_bits: Vec<Vec<u16>> = base
        .inputs
        .iter()
        .map(|row| {
            row.iter()
                .map(|&value| voltafabric::completion::encode_fp16(value).unwrap())
                .collect()
        })
        .collect();
    // Exercise exact signed-zero and subnormal encodings through JSON -> Rust -> worker.
    fp16_bits[0][0] = 0x8000;
    fp16_bits[0][1] = 0x0001;
    fp16_bits[0][2] = 0x8001;
    let decoded: Vec<Vec<f64>> = fp16_bits
        .iter()
        .map(|row| {
            row.iter()
                .map(|&bits| voltafabric::completion::decode_fp16(bits).unwrap())
                .collect()
        })
        .collect();
    base.inputs = decoded.clone();
    let mut request = serde_json::to_value(&base).unwrap();
    let object = request.as_object_mut().unwrap();
    object.remove("inputs");
    object.insert("inputs_fp16_bits".into(), serde_json::json!(fp16_bits));

    let input: CompressedSeriesInput = serde_json::from_value(request)
        .expect("compressed-series CLI must accept explicit caller-owned FP16 bit rows");
    let request_path = dir.join("typed-input.json");
    std::fs::write(&request_path, serde_json::to_vec(&input).unwrap()).unwrap();
    let cli = Command::new(env!("CARGO_BIN_EXE_vf"))
        .arg("session-run-compressed")
        .arg(&request_path)
        .output()
        .unwrap();
    assert_eq!(
        cli.status.code(),
        Some(0),
        "stderr={}",
        String::from_utf8_lossy(&cli.stderr)
    );
    let envelope: serde_json::Value = serde_json::from_slice(&cli.stdout).unwrap();
    assert_eq!(envelope["ok"], true, "{envelope}");
    let report: CompressedSeriesReport =
        serde_json::from_value(envelope["result"].clone()).unwrap();
    assert!(report.series.ok, "{:?}", report.series.error);
    assert_eq!(
        (report.series.batches_ok, report.series.exact_batches),
        (3, 3)
    );
    assert_eq!(report.series.recorded.len(), 3);
    let artifacts: Vec<_> = [2, 9].map(artifact).into();
    let oracle = session::DenseOracle::new(&artifacts).unwrap();
    for recorded in &report.series.recorded {
        let mut problem = input.routing.clone();
        problem.stamp.batch += recorded.batch;
        problem.stamp.epoch += recorded.batch / input.epoch_every;
        problem.registry.epoch = problem.stamp.epoch;
        let expected = oracle.output(&problem, &decoded).unwrap();
        assert_eq!(output_bits(&recorded.output), output_bits(&expected));
    }
    assert!(report
        .series
        .traffic
        .iter()
        .all(|rank| { rank.traffic.loads_transferred == 1 && rank.traffic.execs == 3 }));
    assert!(report
        .series
        .close
        .iter()
        .all(|close| close.exit_code == Some(0) && close.bye_execs == Some(3)));

    // The legacy f64 mode sees the same exact values, while the worker requests use
    // six-byte smaller activation scalars. Header fields are included in request_bytes.
    let mut f64_baseline = base;
    f64_baseline.routing.stamp.run_id = "a170-fp16-series-cli-f64-baseline".into();
    f64_baseline.inputs_fp16_bits = None;
    let baseline = session::run_compressed_series(&f64_baseline).unwrap();
    assert!(
        baseline.series.ok,
        "error={:?} failure={:?} close={:?}",
        baseline.series.error, baseline.series.failure, baseline.series.close
    );
    let routed_rows: u64 = routing::dispatch(&input.routing)
        .unwrap()
        .ranks
        .iter()
        .map(|rank| {
            rank.slots
                .iter()
                .map(|slot| slot.slot.token)
                .collect::<BTreeSet<_>>()
                .len() as u64
        })
        .sum();
    let expected_body_saving =
        3 * routed_rows * HIDDEN * (std::mem::size_of::<f64>() - std::mem::size_of::<u16>()) as u64;
    let measured_saving = baseline.series.exec_request_bytes - report.series.exec_request_bytes;
    assert!(
        measured_saving > 0 && measured_saving.abs_diff(expected_body_saving) <= 1024,
        "typed mode saved {measured_saving} request bytes; {expected_body_saving} expected from bodies"
    );
}

#[test]
fn compressed_series_cli_accepts_raw_le_fp16_and_reuses_persistent_sessions() {
    let temp = project_temp_dir();
    let dir = temp.0.as_path();
    let mut typed = series("a173-raw-fp16-series-cli", dir, 3);
    typed.epoch_every = 2;
    typed.record = vec![0, 1, 2];
    let mut fp16_bits: Vec<Vec<u16>> = typed
        .inputs
        .iter()
        .map(|row| {
            row.iter()
                .map(|&value| voltafabric::completion::encode_fp16(value).unwrap())
                .collect()
        })
        .collect();
    fp16_bits[0][0] = 0x8000;
    fp16_bits[0][1] = 0x0001;
    fp16_bits[0][2] = 0x8001;
    typed.inputs.clear();
    typed.inputs_fp16_bits = Some(fp16_bits.clone());

    let mut raw_series = typed.clone();
    raw_series.inputs_fp16_bits = None;
    let header = serde_json::json!({
        "series": raw_series,
        "rows": fp16_bits.len(),
        "columns": HIDDEN
    });
    let mut raw_wire = serde_json::to_vec(&header).unwrap();
    raw_wire.push(b'\n');
    for &bits in fp16_bits.iter().flatten() {
        raw_wire.extend_from_slice(&bits.to_le_bytes());
    }
    let typed_json_bytes = serde_json::to_vec(&typed).unwrap().len();
    assert!(raw_wire.len() < typed_json_bytes);

    let mut child = Command::new(env!("CARGO_BIN_EXE_vf"))
        .args(["session-run-compressed", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&raw_wire).unwrap();
    let cli = child.wait_with_output().unwrap();
    assert_eq!(
        cli.status.code(),
        Some(0),
        "stderr={}",
        String::from_utf8_lossy(&cli.stderr)
    );
    let envelope: serde_json::Value = serde_json::from_slice(&cli.stdout).unwrap();
    assert_eq!(envelope["ok"], true, "{envelope}");
    let raw_report: CompressedSeriesReport =
        serde_json::from_value(envelope["result"].clone()).unwrap();
    assert!(raw_report.series.ok, "{:?}", raw_report.series.error);
    assert_eq!(raw_report.series.batches_ok, 3);
    assert_eq!(raw_report.series.exact_batches, 3);
    assert_eq!(raw_report.series.recorded.len(), 3);
    assert!(raw_report
        .series
        .traffic
        .iter()
        .all(|rank| { rank.traffic.loads_transferred == 1 && rank.traffic.execs == 3 }));
    assert!(raw_report
        .series
        .close
        .iter()
        .all(|close| close.exit_code == Some(0) && close.bye_execs == Some(3)));

    // This front-door format changes no request sent over the persistent rank streams.
    let typed_report = session::run_compressed_series(&typed).unwrap();
    assert!(typed_report.series.ok, "{:?}", typed_report.series.error);
    assert_eq!(
        raw_report.series.exec_request_bytes,
        typed_report.series.exec_request_bytes
    );
    for (raw, json) in raw_report
        .series
        .recorded
        .iter()
        .zip(&typed_report.series.recorded)
    {
        assert_eq!(raw.batch, json.batch);
        assert_eq!(output_bits(&raw.output), output_bits(&json.output));
    }

    let artifacts: Vec<_> = [2, 9].map(artifact).into();
    let oracle = session::DenseOracle::new(&artifacts).unwrap();
    for recorded in &raw_report.series.recorded {
        let mut problem = typed.routing.clone();
        problem.stamp.batch += recorded.batch;
        problem.stamp.epoch += recorded.batch / typed.epoch_every;
        problem.registry.epoch = problem.stamp.epoch;
        let decoded: Vec<Vec<f64>> = fp16_bits
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&value| voltafabric::completion::decode_fp16(value).unwrap())
                    .collect()
            })
            .collect();
        assert_eq!(
            output_bits(&recorded.output),
            output_bits(&oracle.output(&problem, &decoded).unwrap())
        );
    }

    let marker = dir.join("worker-started-nonfinite-raw");
    let mut invalid_series = raw_series;
    let shell = format!("touch {}; exit 0", marker.display());
    invalid_series.workers = vec![
        Endpoint {
            rank: 2,
            argv: vec!["/bin/sh".into(), "-c".into(), shell.clone()],
        },
        Endpoint {
            rank: 9,
            argv: vec!["/bin/sh".into(), "-c".into(), shell],
        },
    ];
    let invalid_header = serde_json::json!({
        "series": invalid_series,
        "rows": fp16_bits.len(),
        "columns": HIDDEN
    });
    let mut invalid_wire = serde_json::to_vec(&invalid_header).unwrap();
    invalid_wire.push(b'\n');
    let mut nonfinite_bits = fp16_bits.clone();
    nonfinite_bits[0][0] = 0x7c00;
    for &bits in nonfinite_bits.iter().flatten() {
        invalid_wire.extend_from_slice(&bits.to_le_bytes());
    }
    let mut invalid_child = Command::new(env!("CARGO_BIN_EXE_vf"))
        .args(["session-run-compressed", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    invalid_child
        .stdin
        .take()
        .unwrap()
        .write_all(&invalid_wire)
        .unwrap();
    let invalid_cli = invalid_child.wait_with_output().unwrap();
    assert_eq!(invalid_cli.status.code(), Some(2));
    let invalid_envelope: serde_json::Value = serde_json::from_slice(&invalid_cli.stdout).unwrap();
    assert_eq!(invalid_envelope["error"]["code"], "INVALID_INPUT");
    assert!(!marker.exists(), "nonfinite raw input started a worker");
}

#[test]
fn compressed_series_cli_refuses_malformed_or_oversized_raw_fp16_before_worker_start() {
    let temp = project_temp_dir();
    let dir = temp.0.as_path();
    let mut base = series("a173-raw-fp16-invalid-cli", dir, 1);
    base.inputs.clear();
    let marker = dir.join("worker-started-invalid-raw");
    let shell = format!("touch {}; exit 0", marker.display());
    base.workers = vec![
        Endpoint {
            rank: 2,
            argv: vec!["/bin/sh".into(), "-c".into(), shell.clone()],
        },
        Endpoint {
            rank: 9,
            argv: vec!["/bin/sh".into(), "-c".into(), shell],
        },
    ];
    let rows = base.routing.routes.len();
    let columns = base.routing.hidden;
    let body = vec![0u8; rows * columns * std::mem::size_of::<u16>()];
    let mut ambiguous_f64 = base.clone();
    ambiguous_f64.inputs = vec![vec![0.0; columns]; rows];
    let mut ambiguous_fp16 = base.clone();
    ambiguous_fp16.inputs_fp16_bits = Some(vec![vec![0; columns]; rows]);

    let mut too_many_rows = base.clone();
    too_many_rows.routing.hidden = 65_536;
    let route = too_many_rows.routing.routes[0].clone();
    too_many_rows.routing.routes = vec![route; 257];

    let cases = vec![
        (
            "truncated",
            base.clone(),
            rows,
            columns,
            body[..body.len() - 1].to_vec(),
            "RAW_INPUT_TRUNCATED",
        ),
        (
            "trailing",
            base.clone(),
            rows,
            columns,
            [body.as_slice(), &[0]].concat(),
            "RAW_INPUT_TRAILING_DATA",
        ),
        (
            "shape",
            base.clone(),
            rows + 1,
            columns,
            body.clone(),
            "CAPACITY_EXCEEDED",
        ),
        (
            "ambiguous-f64",
            ambiguous_f64,
            rows,
            columns,
            body.clone(),
            "INVALID_INPUT",
        ),
        (
            "ambiguous-fp16",
            ambiguous_fp16,
            rows,
            columns,
            body.clone(),
            "INVALID_INPUT",
        ),
        (
            "oversized",
            too_many_rows,
            257,
            65_536,
            Vec::new(),
            "MESSAGE_TOO_LARGE",
        ),
    ];

    for (name, series, declared_rows, declared_columns, body, expected_code) in cases {
        let header = serde_json::json!({
            "series": series,
            "rows": declared_rows,
            "columns": declared_columns
        });
        let mut wire = serde_json::to_vec(&header).unwrap();
        wire.push(b'\n');
        wire.extend_from_slice(&body);
        let mut child = Command::new(env!("CARGO_BIN_EXE_vf"))
            .args(["session-run-compressed", "-"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(&wire).unwrap();
        let output = child.wait_with_output().unwrap();
        assert_eq!(output.status.code(), Some(2), "case={name}");
        let envelope: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            envelope["error"]["code"], expected_code,
            "case={name}: {envelope}"
        );
        assert!(!marker.exists(), "case={name} started a worker");
    }
}

#[test]
fn compressed_series_cli_refuses_missing_ambiguous_and_nonfinite_inputs_before_worker_start() {
    let dir = temp_dir();
    let base = series("a170-fp16-series-cli-invalid", &dir, 1);
    let base_json = serde_json::to_value(&base).unwrap();
    let bits: Vec<Vec<u16>> = base
        .inputs
        .iter()
        .map(|row| {
            row.iter()
                .map(|&value| voltafabric::completion::encode_fp16(value).unwrap())
                .collect()
        })
        .collect();
    let cases = ["missing", "both", "nonfinite"];
    for case in cases {
        let marker = dir.join(format!("worker-started-{case}"));
        let shell = format!("touch {}; exit 0", marker.display());
        let mut request = base_json.clone();
        request["workers"] = serde_json::json!([
            {"rank": 2, "argv": ["/bin/sh", "-c", shell.clone()]},
            {"rank": 9, "argv": ["/bin/sh", "-c", shell]}
        ]);
        {
            let fields = request.as_object_mut().unwrap();
            match case {
                "missing" => {
                    fields.remove("inputs");
                    fields.remove("inputs_fp16_bits");
                }
                "both" => {
                    fields.insert("inputs".into(), serde_json::to_value(&base.inputs).unwrap());
                    fields.insert("inputs_fp16_bits".into(), serde_json::json!(bits));
                }
                "nonfinite" => {
                    let mut nonfinite = bits.clone();
                    nonfinite[0][0] = 0x7c00;
                    fields.remove("inputs");
                    fields.insert("inputs_fp16_bits".into(), serde_json::json!(nonfinite));
                }
                _ => unreachable!(),
            }
        }
        let request_path = dir.join(format!("{case}.json"));
        std::fs::write(&request_path, serde_json::to_vec(&request).unwrap()).unwrap();
        let cli = Command::new(env!("CARGO_BIN_EXE_vf"))
            .arg("session-run-compressed")
            .arg(&request_path)
            .output()
            .unwrap();
        assert_eq!(
            cli.status.code(),
            Some(2),
            "case {case}: stdout={} stderr={}",
            String::from_utf8_lossy(&cli.stdout),
            String::from_utf8_lossy(&cli.stderr)
        );
        let response: serde_json::Value = serde_json::from_slice(&cli.stdout).unwrap();
        assert_eq!(response["ok"], false, "case {case}: {response}");
        assert_eq!(
            response["error"]["code"], "INVALID_INPUT",
            "case {case}: {response}"
        );
        assert!(
            !marker.exists(),
            "case {case} started a worker before refusal"
        );
    }
}

#[test]
fn hello_advertises_the_compressed_capability() {
    let s = Session::connect(&endpoint(2), T).unwrap();
    assert!(s
        .hello
        .capabilities
        .contains(&session::CAPABILITY_QPN2.to_string()));
    assert!(s
        .hello
        .capabilities
        .contains(&session::CAPABILITY_QPN2_RAW.to_string()));
    assert!(s
        .hello
        .capabilities
        .contains(&session::CAPABILITY_EXEC_F64_RAW.to_string()));
    assert_eq!(s.close(T).exit_code, Some(0));
}

fn load_request(a: &CompressedArtifact, epoch: u64) -> Request {
    let st = a.stack();
    Request::LoadArtifact {
        artifact_sha256: a.sha256().to_string(),
        model_revision: st.model_revision().into(),
        layer: st.layer(),
        epoch,
        resident_order: st.resident_order().to_vec(),
        artifact: base64::encode(a.bytes()),
    }
}
fn with(request: Request, edit: impl FnOnce(&mut Request)) -> Request {
    let mut request = request;
    edit(&mut request);
    request
}

#[test]
fn load_artifact_refusals_leave_the_session_usable() {
    let a = artifact(2);
    let mut s = Session::connect(&endpoint(2), T).unwrap();
    let other = artifact(9);
    // A different, invalid artifact whose announced SHA-256 is its own: the digest
    // matches, so the refusal comes from the artifact validation itself.
    let corrupted = rebuild(a.bytes(), |t, _| {
        t.get_mut("scales13").unwrap().2[0] = 0x7f;
    });
    let identity = "artifact rank, revision, layer or physical order differs from this LOAD";
    let cases: Vec<(Request, &str, &str)> = vec![
        (
            with(load_request(&a, EPOCH), |r| {
                if let Request::LoadArtifact {
                    artifact_sha256, ..
                } = r
                {
                    *artifact_sha256 = "0".repeat(64);
                }
            }),
            "ARTIFACT_DIGEST_MISMATCH",
            "artifact_sha256 differs from the SHA-256 of the received artifact",
        ),
        (
            with(load_request(&a, EPOCH), |r| {
                if let Request::LoadArtifact { artifact, .. } = r {
                    artifact.push('=');
                }
            }),
            "ARTIFACT_ENCODING",
            "artifact is not canonical padded base64",
        ),
        (
            with(load_request(&a, EPOCH), |r| {
                if let Request::LoadArtifact {
                    artifact,
                    artifact_sha256,
                    ..
                } = r
                {
                    *artifact = base64::encode(&corrupted);
                    *artifact_sha256 = sha256::hex(&corrupted);
                }
            }),
            "ARTIFACT_INVALID",
            "scale byte is not an unsigned finite E4M3 value",
        ),
        (load_request(&other, EPOCH), "INVALID_INPUT", identity),
        (
            with(load_request(&a, EPOCH), |r| {
                if let Request::LoadArtifact { model_revision, .. } = r {
                    *model_revision = "rev-other".into();
                }
            }),
            "INVALID_INPUT",
            identity,
        ),
        (
            with(load_request(&a, EPOCH), |r| {
                if let Request::LoadArtifact { layer, .. } = r {
                    *layer += 1;
                }
            }),
            "INVALID_INPUT",
            identity,
        ),
        (
            with(load_request(&a, EPOCH), |r| {
                if let Request::LoadArtifact { resident_order, .. } = r {
                    resident_order.reverse();
                }
            }),
            "INVALID_INPUT",
            identity,
        ),
        (
            load_request(&a, EPOCH - 1),
            "EPOCH_ROLLBACK",
            "ownership epoch is older than the artifact's",
        ),
    ];
    for (request, code, message) in cases {
        s.send(&request).unwrap();
        let error = refused(s.reply(T).unwrap());
        assert_eq!(error.code, code, "{error}");
        assert!(error.message.starts_with(message), "{error}");
    }
    // Still usable: the real LOAD, its refusal to load twice, weightless rebinds, EVICT.
    s.send(&load_request(&a, EPOCH + 2)).unwrap();
    match s.reply(T).unwrap() {
        Reply::Loaded {
            epoch,
            experts,
            transferred,
            ..
        } => assert_eq!((epoch, experts, transferred), (EPOCH + 2, 2, true)),
        other => panic!("{other:?}"),
    }
    s.send(&load_request(&a, EPOCH + 3)).unwrap();
    assert_reason(
        &refused(s.reply(T).unwrap()),
        "ARTIFACT_LOADED",
        "evict the loaded artifact before loading another",
    );
    let rebind = |epoch| Request::Load {
        artifact_sha256: a.sha256().to_string(),
        model_revision: REVISION.into(),
        layer: LAYER,
        epoch,
        resident_order: order(2),
        model: None,
    };
    s.send(&rebind(EPOCH + 2)).unwrap();
    assert_reason(
        &refused(s.reply(T).unwrap()),
        "EPOCH_NOT_NEWER",
        "rebinding requires a newer ownership epoch",
    );
    s.send(&rebind(EPOCH + 3)).unwrap();
    assert!(matches!(
        s.reply(T).unwrap(),
        Reply::Loaded {
            transferred: false,
            experts: 2,
            ..
        }
    ));
    s.send(&Request::Evict {
        artifact_sha256: a.sha256().to_string(),
    })
    .unwrap();
    assert!(matches!(s.reply(T).unwrap(), Reply::Evicted { .. }));
    assert_eq!(s.close(T).exit_code, Some(0));
}

fn bound(p: &routing::RoutingProblem) -> (Vec<Session>, Vec<CompressedArtifact>) {
    let artifacts: Vec<_> = [2, 9].map(artifact).into();
    let sessions =
        session::prepare_compressed(&[endpoint(2), endpoint(9)], &artifacts, p, T).unwrap();
    (sessions, artifacts)
}
fn rank2(p: &routing::RoutingProblem) -> routing::RankDispatch {
    routing::dispatch(p)
        .unwrap()
        .ranks
        .into_iter()
        .find(|r| r.rank == 2)
        .unwrap()
}
fn token_rows(d: &routing::RankDispatch, x: &[Vec<f64>]) -> Vec<TokenInput> {
    let tokens: std::collections::BTreeSet<_> = d.slots.iter().map(|s| s.slot.token).collect();
    tokens
        .into_iter()
        .map(|token| TokenInput {
            token,
            value: x[token].clone(),
        })
        .collect()
}
static ATTEMPT: AtomicU64 = AtomicU64::new(1000);
fn rejected(
    s: &mut Session,
    d: &routing::RankDispatch,
    inputs: Vec<TokenInput>,
    fault: Fault,
) -> Error {
    let attempt = ATTEMPT.fetch_add(1, Ordering::Relaxed);
    let pending = s.start_exec(d, attempt, inputs, fault).unwrap();
    let receipt = s.finish_exec(pending, None, std::time::Instant::now() + T);
    assert_eq!(receipt.state, ReceiptState::NotStarted, "{receipt:?}");
    assert!(s.is_usable());
    receipt.error.unwrap()
}

#[test]
fn compressed_exec_refusals_are_not_started_and_leave_the_session_usable() {
    let p = routing_problem("a53-exec-refusals");
    let x = activations(6, HIDDEN as usize, 2);
    let (mut sessions, _) = bound(&p);
    let s = &mut sessions[0];
    let d = rank2(&p);
    let foreign = {
        let mut d = d.clone();
        d.slots[0].physical_expert = 1 - d.slots[0].physical_expert;
        d
    };
    assert_reason(
        &rejected(s, &foreign, token_rows(&d, &x), Fault::None),
        "INVALID_INPUT",
        "foreign, duplicate or malformed resident slot",
    );
    let mut missing = token_rows(&d, &x);
    missing.pop();
    assert_reason(
        &rejected(s, &d, missing, Fault::None),
        "INVALID_INPUT",
        "local inputs must cover exactly the dispatched tokens",
    );
    let mut narrow = d.clone();
    narrow.hidden = 64;
    assert_reason(
        &rejected(s, &narrow, token_rows(&d, &x), Fault::None),
        "INVALID_INPUT",
        "dispatch hidden dimension differs from resident weights",
    );
    assert_reason(
        &rejected(
            s,
            &d,
            token_rows(&d, &x),
            Fault::Delay {
                milliseconds: 60_001,
            },
        ),
        "INVALID_INPUT",
        "fault injection outside bounded CPU test contract",
    );
    let mut stale = d.clone();
    stale.stamp.epoch += 1;
    assert_reason(
        &rejected(s, &stale, token_rows(&d, &x), Fault::None),
        "STALE_EXECUTION",
        "dispatch identity differs from the loaded artifact epoch, revision or layer",
    );
    let budget = "compressed EXEC needs 1 slot or more within the rank CPU work budget";
    let mut empty = d.clone();
    empty.slots.clear();
    assert_reason(
        &rejected(s, &empty, vec![], Fault::None),
        "INVALID_INPUT",
        budget,
    );
    // 8192 slots x 3 x 64 x 128 = 201 M multiply-adds: over the 100 M budget.
    let mut huge = d.clone();
    huge.tokens = 4096;
    huge.slots = (0..4096)
        .flat_map(|token| {
            d.slots[..2].iter().enumerate().map(move |(k, slot)| {
                let mut slot = slot.clone();
                slot.slot.token = token;
                slot.slot.slot = k;
                slot.physical_expert = k;
                slot
            })
        })
        .collect();
    for slot in &mut huge.slots {
        let (id, local) = [(10, 7), (99, 3)][slot.physical_expert];
        slot.slot.expert = id;
        slot.slot.local = local;
    }
    assert_reason(
        &rejected(s, &huge, vec![], Fault::None),
        "INVALID_INPUT",
        budget,
    );
    for s in sessions {
        assert_eq!(s.close(T).exit_code, Some(0));
    }
}

#[test]
fn negotiated_completion_batches_reconstruct_one_exact_full_series() {
    let dir = temp_dir();
    let input = series("completion-batch-session-latch", &dir, 2);
    let report = session::run_compressed_series(&input).unwrap();
    assert!(report.series.ok, "{:?}", report.series.failure);
    assert_eq!(report.series.exact_batches, 2);
    assert_eq!(report.series.max_abs_diff, 0.0);
    for rank in &report.series.traffic {
        assert_eq!(rank.traffic.execs, 2);
        assert_eq!(
            rank.traffic.frames_received, 8,
            "HELLO and LOAD plus two STARTED/batch/FINISHED triplets should be eight replies"
        );
    }
}

#[test]
fn opt_in_fp16_completion_frames_match_python_oracle_and_reduce_reply_bytes() {
    let f64_dir = temp_dir();
    let mut f64_input = series("a165-f64", &f64_dir, 9);
    f64_input.warmup = 2;
    f64_input.record = vec![2];
    let f64_report = session::run_compressed_series(&f64_input).unwrap();
    assert!(f64_report.series.ok, "{:?}", f64_report.series.failure);

    let fp16_dir = temp_dir();
    let mut fp16_input = series("a165-f16", &fp16_dir, 9);
    fp16_input.warmup = 2;
    fp16_input.record = vec![2];
    fp16_input.fp16_completions = true;

    let dispatch = routing::dispatch(&fp16_input.routing).unwrap();
    let stacks = BTreeMap::from([(2, stack(2)), (9, stack(9))]);
    let mut oracle_rows = Vec::new();
    let mut slots: Vec<_> = dispatch
        .ranks
        .into_iter()
        .flat_map(|rank| rank.slots)
        .collect();
    slots.sort_by_key(|slot| (slot.slot.token, slot.slot.slot));
    let mut max_rounding_bound = 0.0_f64;
    for batch_index in 0..fp16_input.batches {
        let mut batch_inputs = fp16_input.inputs.clone();
        for (token, row) in batch_inputs.iter_mut().enumerate() {
            for (column, value) in row.iter_mut().enumerate() {
                let step = (batch_index as usize * 7 + token * 3 + column) % 11;
                *value *= 1.0 + step as f64 / 64.0;
            }
        }
        let mut absolute_error_bound = vec![vec![0.0_f64; HIDDEN as usize]; batch_inputs.len()];
        for routed in &slots {
            let weight = routed.slot.weight;
            let values = stacks[&routed.slot.rank]
                .expert_value(routed.physical_expert, &batch_inputs[routed.slot.token])
                .unwrap();
            for (column, value) in values.iter().enumerate() {
                // Binary16 RNE error is <= 2^-11 relative for normals and <= 2^-25
                // absolute for subnormals. The route coefficient is applied afterwards.
                let row_bound = (value.abs() * 2.0_f64.powi(-11)).max(2.0_f64.powi(-25));
                absolute_error_bound[routed.slot.token][column] += weight.abs() * row_bound;
            }
            if batch_index == 2 {
                oracle_rows.push((routed.slot.token, weight, values));
            }
        }
        max_rounding_bound = max_rounding_bound.max(
            absolute_error_bound
                .iter()
                .flatten()
                .copied()
                .fold(0.0_f64, f64::max),
        );
    }
    fp16_input.tolerance_abs = max_rounding_bound + 16.0 * f64::EPSILON;

    // Python's standard-library struct 'e' conversion is independent of the Rust codec
    // and rounds each unweighted completion before the original route weight is applied.
    let oracle_script = "import json,struct,sys\nrows=json.load(sys.stdin)\nout=[[0.0]*128 for _ in range(6)]\nfor token,weight,values in rows:\n  for i,value in enumerate(values):\n    half=struct.unpack('<e',struct.pack('<e',value))[0]\n    out[token][i] += weight*half\njson.dump(out,sys.stdout,separators=(',',':'))\n";
    let mut child = Command::new("python3")
        .args(["-c", oracle_script])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    serde_json::to_writer(child.stdin.as_mut().unwrap(), &oracle_rows).unwrap();
    drop(child.stdin.take());
    let oracle = child.wait_with_output().unwrap();
    assert!(
        oracle.status.success(),
        "{}",
        String::from_utf8_lossy(&oracle.stderr)
    );
    let expected: Vec<Vec<f64>> = serde_json::from_slice(&oracle.stdout).unwrap();

    let fp16_report = session::run_compressed_series(&fp16_input).unwrap();
    assert!(fp16_report.series.ok, "{:?}", fp16_report.series.failure);
    let actual = &fp16_report.series.recorded[0].output;
    assert_eq!(fp16_report.series.recorded[0].batch, 2);
    assert_eq!(actual.len(), expected.len());
    for (actual_row, expected_row) in actual.iter().zip(&expected) {
        assert_eq!(
            actual_row.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected_row.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    let f64_received: u64 = f64_report
        .series
        .traffic
        .iter()
        .map(|rank| rank.traffic.bytes_received)
        .sum();
    let fp16_received: u64 = fp16_report
        .series
        .traffic
        .iter()
        .map(|rank| rank.traffic.bytes_received)
        .sum();
    assert!(
        fp16_received < f64_received,
        "f64={f64_received}, fp16={fp16_received}"
    );
    assert!(fp16_report.series.batch_latency.is_some());
    assert!(fp16_report
        .series
        .close
        .iter()
        .all(|close| close.peak_rss_kib.is_some()));
    eprintln!(
        "A165_METRICS f64_rx_bytes={f64_received} fp16_rx_bytes={fp16_received} f64_p50_us={:?} fp16_p50_us={:?} f64_worker_peak_rss_kib={:?} fp16_worker_peak_rss_kib={:?} rounding_bound={}",
        f64_report.series.batch_latency.as_ref().map(|latency| latency.p50_us),
        fp16_report.series.batch_latency.as_ref().map(|latency| latency.p50_us),
        f64_report.series.close.iter().map(|close| close.peak_rss_kib).collect::<Vec<_>>(),
        fp16_report.series.close.iter().map(|close| close.peak_rss_kib).collect::<Vec<_>>(),
        fp16_input.tolerance_abs,
    );
}

#[test]
fn fp16_session_refuses_a_missing_peer_capability_before_any_exec_is_sent() {
    let artifacts = vec![artifact(2), artifact(9)];
    let problem = routing_problem("a165-peer-capability");
    let inputs = activations(6, HIDDEN as usize, 5);
    let mut sessions =
        session::prepare_compressed(&[endpoint(2), endpoint(9)], &artifacts, &problem, T).unwrap();
    sessions[0]
        .hello
        .capabilities
        .retain(|capability| capability != session::CAPABILITY_COMPLETION_BATCH_FP16_RAW);

    let error = session::run_compressed_batch_fp16(
        &mut sessions,
        &artifacts,
        &problem,
        &inputs,
        &BTreeMap::new(),
        false,
        T,
    )
    .unwrap_err();
    assert_eq!(error.code, "SESSION_CAPABILITY");
    assert!(sessions.iter().all(|worker| worker.traffic.execs == 0));
    for worker in sessions {
        assert_eq!(worker.close(T).exit_code, Some(0));
    }
}

#[test]
fn fp16_overflow_is_reported_and_keeps_the_resident_worker_usable() {
    let observed =
        run_fp16_overflow_transition([endpoint(2), endpoint(9)], "a165-fp16-overflow-recovery", 0);
    assert_eq!(observed.execs, 8);
    assert_eq!(observed.loads, 2);
}

#[derive(Debug)]
struct A168SessionObservation {
    outputs: Vec<Vec<Vec<u64>>>,
    fp16_latency_us: Vec<u128>,
    fp16_overflow_ranks: Vec<u32>,
    worker_arches: BTreeMap<u32, String>,
    rank_token_inputs: BTreeMap<u32, Vec<usize>>,
    exec_bytes_sent: u64,
    exec_bytes_received: u64,
    weight_bytes_sent: u64,
    loads: u64,
    execs: u64,
    worker_peak_rss_kib: Vec<Option<u64>>,
}

fn output_bits(rows: &[Vec<f64>]) -> Vec<Vec<u64>> {
    rows.iter()
        .map(|row| row.iter().map(|value| value.to_bits()).collect())
        .collect()
}

fn run_fp16_overflow_transition(
    endpoints: [Endpoint; 2],
    run_id: &str,
    repeated_valid_fp16_execs: usize,
) -> A168SessionObservation {
    let artifacts = vec![artifact(2), artifact(9)];
    let mut problem = routing_problem(run_id);
    let overflow_inputs = vec![vec![100_000.0; HIDDEN as usize]; 6];

    // Confirm this is a real finite-output overflow for the fixture experts, rather than
    // a mocked writer failure or an activation rejected before the worker starts.
    let dispatch = routing::dispatch(&problem).unwrap();
    let rank_token_inputs = dispatch
        .ranks
        .iter()
        .map(|rank| {
            let tokens: BTreeSet<_> = rank.slots.iter().map(|routed| routed.slot.token).collect();
            (rank.rank, tokens.into_iter().collect())
        })
        .collect();
    let stacks = BTreeMap::from([(2, stack(2)), (9, stack(9))]);
    let mut has_unrepresentable_finite_output = false;
    for rank in &dispatch.ranks {
        for routed in &rank.slots {
            has_unrepresentable_finite_output |= stacks[&rank.rank]
                .expert_value(routed.physical_expert, &overflow_inputs[routed.slot.token])
                .unwrap()
                .iter()
                .any(|value| value.is_finite() && value.abs() >= 65_520.0);
        }
    }
    assert!(
        has_unrepresentable_finite_output,
        "fixture must produce a finite value outside binary16's finite range"
    );

    let mut sessions = session::prepare_compressed(&endpoints, &artifacts, &problem, T).unwrap();
    let worker_arches = sessions
        .iter()
        .map(|worker| (worker.rank, worker.hello.arch.clone()))
        .collect();
    let sent_after_load = sessions
        .iter()
        .map(|worker| worker.traffic.bytes_sent)
        .sum::<u64>();
    let received_after_load = sessions
        .iter()
        .map(|worker| worker.traffic.bytes_received)
        .sum::<u64>();
    let weight_bytes_sent = sessions
        .iter()
        .map(|worker| worker.traffic.weight_bytes_sent)
        .sum::<u64>();
    let loads_after_load = sessions
        .iter()
        .map(|worker| worker.traffic.loads_transferred)
        .sum::<u64>();
    assert_eq!(
        loads_after_load, 2,
        "each rank must receive one resident artifact"
    );
    assert!(weight_bytes_sent > 0);
    let mut outputs = Vec::new();
    let mut fp16_latency_us = Vec::with_capacity(repeated_valid_fp16_execs + 1);

    // Negotiate and execute the session's f64 response path first. Its parser latches remain
    // available across later EXECs, which makes the following FP16 request a real mode switch.
    let f64_before_fp16 = session::run_compressed_batch(
        &mut sessions,
        &artifacts,
        &problem,
        &activations(6, HIDDEN as usize, 5),
        &BTreeMap::new(),
        false,
        T,
    )
    .unwrap();
    assert!(f64_before_fp16.ok, "{:?}", f64_before_fp16.error);
    outputs.push(output_bits(
        f64_before_fp16.output.as_ref().expect("f64 output"),
    ));
    problem.stamp.batch += 1;

    let overflow = session::run_compressed_batch_fp16(
        &mut sessions,
        &artifacts,
        &problem,
        &overflow_inputs,
        &BTreeMap::new(),
        false,
        T,
    )
    .unwrap();
    assert!(!overflow.ok);
    assert!(overflow.output.is_none());
    assert_eq!(overflow.error.as_ref().unwrap().code, "FP16_OVERFLOW");
    let fp16_overflow_ranks: Vec<_> = overflow
        .receipts
        .iter()
        .filter(|receipt| {
            receipt
                .error
                .as_ref()
                .is_some_and(|error| error.code == "FP16_OVERFLOW")
        })
        .map(|receipt| receipt.rank)
        .collect();
    assert!(!fp16_overflow_ranks.is_empty());
    let failed_receipt = overflow
        .receipts
        .iter()
        .find(|receipt| {
            receipt
                .error
                .as_ref()
                .is_some_and(|error| error.code == "FP16_OVERFLOW")
        })
        .expect("overflow must be a known failure after Started, not UNKNOWN");
    assert!(failed_receipt.started);
    assert_eq!(failed_receipt.state, ReceiptState::Started);
    assert_eq!(failed_receipt.received_slots, 0);

    // The same loaded workers must accept the next EXEC without reconnecting or reloading.
    let mut next_problem = problem.clone();
    next_problem.stamp.batch += 1;
    let started = std::time::Instant::now();
    let valid = session::run_compressed_batch_fp16(
        &mut sessions,
        &artifacts,
        &next_problem,
        &activations(6, HIDDEN as usize, 5),
        &BTreeMap::new(),
        false,
        T,
    )
    .unwrap();
    fp16_latency_us.push(started.elapsed().as_micros());
    assert!(valid.ok, "{:?}", valid.error);
    outputs.push(output_bits(
        valid.output.as_ref().expect("valid FP16 output"),
    ));

    for repeat in 0..repeated_valid_fp16_execs {
        let mut repeated_problem = next_problem.clone();
        repeated_problem.stamp.batch += (repeat + 1) as u64;
        let started = std::time::Instant::now();
        let repeated = session::run_compressed_batch_fp16(
            &mut sessions,
            &artifacts,
            &repeated_problem,
            &activations(6, HIDDEN as usize, 5),
            &BTreeMap::new(),
            false,
            T,
        )
        .unwrap();
        fp16_latency_us.push(started.elapsed().as_micros());
        assert!(repeated.ok, "{:?}", repeated.error);
        outputs.push(output_bits(
            repeated.output.as_ref().expect("repeated FP16 output"),
        ));
    }

    // Switching back to the default f64 response mode on the same live session remains
    // compatible after an explicitly negotiated FP16 EXEC.
    let mut f64_problem = next_problem.clone();
    f64_problem.stamp.batch += (repeated_valid_fp16_execs + 1) as u64;
    let f64 = session::run_compressed_batch(
        &mut sessions,
        &artifacts,
        &f64_problem,
        &activations(6, HIDDEN as usize, 5),
        &BTreeMap::new(),
        false,
        T,
    )
    .unwrap();
    assert!(f64.ok, "{:?}", f64.error);
    outputs.push(output_bits(f64.output.as_ref().expect("final f64 output")));

    let expected_execs = 4 + repeated_valid_fp16_execs as u64;
    assert!(sessions
        .iter()
        .all(|worker| worker.traffic.execs == expected_execs));
    let exec_bytes_sent = sessions
        .iter()
        .map(|worker| worker.traffic.bytes_sent)
        .sum::<u64>()
        - sent_after_load;
    let exec_bytes_received = sessions
        .iter()
        .map(|worker| worker.traffic.bytes_received)
        .sum::<u64>()
        - received_after_load;
    let loads = sessions
        .iter()
        .map(|worker| worker.traffic.loads_transferred)
        .sum::<u64>();
    let final_weight_bytes_sent = sessions
        .iter()
        .map(|worker| worker.traffic.weight_bytes_sent)
        .sum::<u64>();
    assert_eq!(
        loads, loads_after_load,
        "EXEC must not reload resident weights"
    );
    assert_eq!(final_weight_bytes_sent, weight_bytes_sent);

    let mut worker_peak_rss_kib = Vec::with_capacity(sessions.len());
    for worker in sessions {
        let close = worker.close(T);
        assert_eq!(close.exit_code, Some(0), "{close:?}");
        assert_eq!(close.bye_execs, Some(expected_execs), "{close:?}");
        worker_peak_rss_kib.push(close.peak_rss_kib);
    }
    A168SessionObservation {
        outputs,
        fp16_latency_us,
        fp16_overflow_ranks,
        worker_arches,
        rank_token_inputs,
        exec_bytes_sent,
        exec_bytes_received,
        weight_bytes_sent,
        loads,
        execs: expected_execs * 2,
        worker_peak_rss_kib,
    }
}

fn upper_median(samples: &[u128]) -> u128 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

#[test]
#[ignore = "requires a current x86-64 worker on PC1 and strict SSH; run only with the recorded VF_PC1_SESSION_WORKER path"]
fn pc6_pc1_fp16_persistent_session_matches_local_baseline() {
    let worker = std::env::var("VF_PC1_SESSION_WORKER")
        .expect("set VF_PC1_SESSION_WORKER to the verified absolute PC1 worker path");
    let remote_first = std::env::var("VF_A168_REMOTE_FIRST").unwrap_or_else(|_| "0".into());
    assert!(
        remote_first == "0" || remote_first == "1",
        "VF_A168_REMOTE_FIRST must be 0 or 1"
    );
    let remote_first = remote_first == "1";
    // Keep identities unique within the coordinator while preserving serialized byte lengths.
    let (local, remote) = if remote_first {
        let remote = run_fp16_overflow_transition(
            [endpoint(2), pc1_endpoint(9, &worker)],
            "a167-pc1-remotely",
            7,
        );
        let local =
            run_fp16_overflow_transition([endpoint(2), endpoint(9)], "a167-pc1-baseline", 7);
        (local, remote)
    } else {
        let local =
            run_fp16_overflow_transition([endpoint(2), endpoint(9)], "a167-pc1-baseline", 7);
        let remote = run_fp16_overflow_transition(
            [endpoint(2), pc1_endpoint(9, &worker)],
            "a167-pc1-remotely",
            7,
        );
        (local, remote)
    };

    let expected_local_arches =
        BTreeMap::from([(2, "aarch64".to_owned()), (9, "aarch64".to_owned())]);
    let expected_remote_arches =
        BTreeMap::from([(2, "aarch64".to_owned()), (9, "x86_64".to_owned())]);
    let expected_token_inputs =
        BTreeMap::from([(2, vec![0, 2, 3, 4, 5]), (9, vec![0, 1, 2, 4, 5])]);
    assert_eq!(local.worker_arches, expected_local_arches);
    assert_eq!(remote.worker_arches, expected_remote_arches);
    assert_eq!(local.rank_token_inputs, expected_token_inputs);
    assert_eq!(remote.rank_token_inputs, expected_token_inputs);
    assert!(
        remote.fp16_overflow_ranks.contains(&9),
        "the remote PC1 rank must own an FP16_OVERFLOW receipt; got {:?}",
        remote.fp16_overflow_ranks
    );

    assert_eq!(
        remote.outputs, local.outputs,
        "PC1/PC6 output bits differ from local baseline"
    );
    assert_eq!(remote.exec_bytes_sent, local.exec_bytes_sent);
    assert_eq!(remote.exec_bytes_received, local.exec_bytes_received);
    assert_eq!(remote.weight_bytes_sent, local.weight_bytes_sent);
    assert_eq!(remote.loads, 2, "two resident artifacts load once each");
    assert_eq!(
        remote.execs, 22,
        "two ranks each serve 11 EXECs on the same sessions"
    );
    assert_eq!(remote.fp16_latency_us.len(), 8);
    assert_eq!(remote.worker_peak_rss_kib.len(), 2);
    assert!(remote.worker_peak_rss_kib.iter().all(Option::is_some));
    eprintln!(
        "A168_INTERPC order={} local_arches={:?} mixed_arches={:?} expected_token_inputs={:?} remote_fp16_overflow_ranks={:?} application_bytes_sent={} application_bytes_received={} weight_bytes_once={} loads={} execs={} local_fp16_samples_us={:?} pc1_fp16_samples_us={:?} local_fp16_median_us={} pc1_fp16_median_us={} local_peak_rss_kib={:?} pc1_worker_peak_rss_kib={:?} output_bits_match=true ssh_framing_counted=false",
        if remote_first { "remote-first" } else { "local-first" },
        local.worker_arches,
        remote.worker_arches,
        remote.rank_token_inputs,
        remote.fp16_overflow_ranks,
        remote.exec_bytes_sent,
        remote.exec_bytes_received,
        remote.weight_bytes_sent,
        remote.loads,
        remote.execs,
        local.fp16_latency_us,
        remote.fp16_latency_us,
        upper_median(&local.fp16_latency_us),
        upper_median(&remote.fp16_latency_us),
        local.worker_peak_rss_kib,
        remote.worker_peak_rss_kib,
    );
}

#[test]
fn fp16_activation_input_wire_matches_f64_and_loads_weights_once() {
    let artifacts = vec![artifact(2), artifact(9)];
    let mut problem = routing_problem("a169-fp16-input-wire");
    let f64_inputs = activations(6, HIDDEN as usize, 5);
    let fp16_inputs: Vec<Vec<u16>> = f64_inputs
        .iter()
        .map(|row| {
            row.iter()
                .map(|&value| voltafabric::completion::encode_fp16(value).unwrap())
                .collect()
        })
        .collect();
    for (f64_row, fp16_row) in f64_inputs.iter().zip(&fp16_inputs) {
        for (&f64_value, &fp16_bits) in f64_row.iter().zip(fp16_row) {
            assert_eq!(
                voltafabric::completion::decode_fp16(fp16_bits)
                    .unwrap()
                    .to_bits(),
                f64_value.to_bits(),
                "fixture inputs must be exactly representable in binary16"
            );
        }
    }

    let mut sessions =
        session::prepare_compressed(&[endpoint(2), endpoint(9)], &artifacts, &problem, T).unwrap();
    let sent_after_load = sessions.iter().map(|s| s.traffic.bytes_sent).sum::<u64>();
    let weights_sent_once = sessions
        .iter()
        .map(|s| s.traffic.weight_bytes_sent)
        .sum::<u64>();
    let baseline = session::run_compressed_batch(
        &mut sessions,
        &artifacts,
        &problem,
        &f64_inputs,
        &BTreeMap::new(),
        false,
        T,
    )
    .unwrap();
    assert!(baseline.ok, "{:?}", baseline.error);
    let baseline_output = output_bits(baseline.output.as_ref().unwrap());
    let baseline_sent =
        sessions.iter().map(|s| s.traffic.bytes_sent).sum::<u64>() - sent_after_load;
    let baseline_request_bytes = baseline
        .receipts
        .iter()
        .map(|receipt| receipt.request_bytes)
        .sum::<u64>();
    assert_eq!(baseline_sent, baseline_request_bytes);

    problem.stamp.batch += 1;
    let compact = session::run_compressed_batch_with_fp16_inputs(
        &mut sessions,
        &artifacts,
        &problem,
        &fp16_inputs,
        &BTreeMap::new(),
        session::CompletionFormat::F64,
        T,
    )
    .unwrap();
    assert!(compact.ok, "{:?}", compact.error);
    assert_eq!(
        output_bits(compact.output.as_ref().unwrap()),
        baseline_output
    );
    let compact_sent = sessions.iter().map(|s| s.traffic.bytes_sent).sum::<u64>()
        - sent_after_load
        - baseline_sent;
    let compact_request_bytes = compact
        .receipts
        .iter()
        .map(|receipt| receipt.request_bytes)
        .sum::<u64>();
    assert_eq!(compact_sent, compact_request_bytes);
    let expected_body_saving =
        10 * 128 * (std::mem::size_of::<f64>() - std::mem::size_of::<u16>()) as u64;
    let measured_saving = baseline_sent - compact_sent;
    assert!(
        measured_saving.abs_diff(expected_body_saving) <= 128,
        "measured wire saving {measured_saving} must be body saving {expected_body_saving} plus only the bounded JSON-header difference"
    );

    problem.stamp.batch += 1;
    let f64_input_fp16_output = session::run_compressed_batch_fp16(
        &mut sessions,
        &artifacts,
        &problem,
        &f64_inputs,
        &BTreeMap::new(),
        false,
        T,
    )
    .unwrap();
    assert!(
        f64_input_fp16_output.ok,
        "{:?}",
        f64_input_fp16_output.error
    );
    problem.stamp.batch += 1;
    let fp16_input_and_output = session::run_compressed_batch_with_fp16_inputs(
        &mut sessions,
        &artifacts,
        &problem,
        &fp16_inputs,
        &BTreeMap::new(),
        session::CompletionFormat::Fp16,
        T,
    )
    .unwrap();
    assert!(
        fp16_input_and_output.ok,
        "{:?}",
        fp16_input_and_output.error
    );
    assert_eq!(
        output_bits(fp16_input_and_output.output.as_ref().unwrap()),
        output_bits(f64_input_fp16_output.output.as_ref().unwrap()),
        "typed FP16 inputs and outputs must match the existing FP16 completion path"
    );
    assert_eq!(weights_sent_once, 57_090);
    assert!(sessions.iter().all(|s| s.traffic.loads_transferred == 1));
    assert!(sessions.iter().all(|s| s.traffic.execs == 4));
    let measured_headers =
        (baseline_request_bytes - 10 * 128 * 8) + (compact_request_bytes - 10 * 128 * 2);
    eprintln!(
        "A169_FP16_INPUT baseline_request_bytes={baseline_request_bytes} compact_request_bytes={compact_request_bytes} measured_saving={measured_saving} expected_body_saving={expected_body_saving} aggregate_header_bytes={measured_headers} output_bits_match=true f64/fp16_input_fp16_output_match=true loads=2"
    );
    for worker in sessions {
        assert_eq!(worker.close(T).exit_code, Some(0));
    }
}

#[test]
fn fp16_activation_input_preflight_rejects_nonfinite_and_missing_capability_atomically() {
    let artifacts = vec![artifact(2), artifact(9)];
    let problem = routing_problem("a169-fp16-input-refusal");
    let valid: Vec<Vec<u16>> = activations(6, HIDDEN as usize, 5)
        .iter()
        .map(|row| {
            row.iter()
                .map(|&value| voltafabric::completion::encode_fp16(value).unwrap())
                .collect()
        })
        .collect();

    let mut nonfinite_sessions =
        session::prepare_compressed(&[endpoint(2), endpoint(9)], &artifacts, &problem, T).unwrap();
    let before = nonfinite_sessions
        .iter()
        .map(|s| (s.traffic.bytes_sent, s.traffic.execs))
        .collect::<Vec<_>>();
    for bits in [0x7c00_u16, 0xfc00, 0x7e00] {
        let mut invalid = valid.clone();
        invalid[0][0] = bits;
        let error = session::run_compressed_batch_with_fp16_inputs(
            &mut nonfinite_sessions,
            &artifacts,
            &problem,
            &invalid,
            &BTreeMap::new(),
            session::CompletionFormat::F64,
            T,
        )
        .unwrap_err();
        assert_eq!(error.code, "INVALID_INPUT");
        assert_eq!(
            nonfinite_sessions
                .iter()
                .map(|s| (s.traffic.bytes_sent, s.traffic.execs))
                .collect::<Vec<_>>(),
            before,
            "NaN and either infinity must fail before either rank receives an EXEC"
        );
    }

    let mut missing_token = valid.clone();
    missing_token.pop();
    let mut wrong_width = valid.clone();
    wrong_width[0].pop();
    for invalid_shape in [&missing_token, &wrong_width] {
        let error = session::run_compressed_batch_with_fp16_inputs(
            &mut nonfinite_sessions,
            &artifacts,
            &problem,
            invalid_shape,
            &BTreeMap::new(),
            session::CompletionFormat::F64,
            T,
        )
        .unwrap_err();
        assert_eq!(error.code, "INVALID_INPUT");
        assert_eq!(
            nonfinite_sessions
                .iter()
                .map(|s| (s.traffic.bytes_sent, s.traffic.execs))
                .collect::<Vec<_>>(),
            before,
            "wrong token count or hidden width must fail before either rank receives an EXEC"
        );
    }
    for worker in nonfinite_sessions {
        assert_eq!(worker.close(T).exit_code, Some(0));
    }

    let mut unsupported_sessions =
        session::prepare_compressed(&[endpoint(2), endpoint(9)], &artifacts, &problem, T).unwrap();
    unsupported_sessions[1]
        .hello
        .capabilities
        .retain(|capability| capability != session::CAPABILITY_EXEC_FP16_RAW);
    let before = unsupported_sessions
        .iter()
        .map(|s| (s.traffic.bytes_sent, s.traffic.execs))
        .collect::<Vec<_>>();
    let error = session::run_compressed_batch_with_fp16_inputs(
        &mut unsupported_sessions,
        &artifacts,
        &problem,
        &valid,
        &BTreeMap::new(),
        session::CompletionFormat::F64,
        T,
    )
    .unwrap_err();
    assert_eq!(error.code, "CAPABILITY_MISSING");
    assert_eq!(
        unsupported_sessions
            .iter()
            .map(|s| (s.traffic.bytes_sent, s.traffic.execs))
            .collect::<Vec<_>>(),
        before,
        "one unsupported rank must prevent the whole batch from being sent"
    );
    for worker in unsupported_sessions {
        assert_eq!(worker.close(T).exit_code, Some(0));
    }
}

#[test]
#[ignore = "requires the verified current x86-64 vf-rank-session worker on PC1 and strict SSH"]
fn a172_pc6_pc1_dynamic_fp16_batches_reuse_one_resident_per_rank() {
    let worker = std::env::var("VF_PC1_SESSION_WORKER")
        .expect("set VF_PC1_SESSION_WORKER to the verified absolute PC1 worker path");
    let artifacts = vec![artifact(2), artifact(9)];
    let endpoints = [endpoint(2), pc1_endpoint(9, &worker)];
    let initial = routing_problem("a172-dynamic-fp16-f64");
    let mut sessions = session::prepare_compressed(&endpoints, &artifacts, &initial, T).unwrap();
    let oracle = session::DenseOracle::new(&artifacts).unwrap();
    let remote_arches: BTreeMap<_, _> = sessions
        .iter()
        .map(|session| (session.rank, session.hello.arch.clone()))
        .collect();
    assert_eq!(
        remote_arches,
        BTreeMap::from([(2, "aarch64".into()), (9, "x86_64".into())])
    );

    let mut f64_request_bytes = BTreeMap::<u32, u64>::new();
    let mut fp16_request_bytes = BTreeMap::<u32, u64>::new();
    let mut expected_saving = BTreeMap::<u32, u64>::new();
    const DYNAMIC_BATCHES: u64 = 6;

    // Start with the f64 baseline, then send distinct caller-owned FP16 bits. The
    // epoch ranges are monotonic on the same worker sessions, so every rebind is
    // weightless and no batch identity is replayed.
    for (typed, run_id, epoch_base) in [
        (false, "a172-dynamic-fp16-f64", EPOCH),
        (true, "a172-dynamic-fp16-raw", EPOCH + 2),
    ] {
        for batch_index in 0..DYNAMIC_BATCHES {
            let mut problem = routing_problem(run_id);
            problem.stamp.batch = batch_index;
            problem.stamp.epoch = epoch_base + batch_index / 3;
            problem.registry.epoch = problem.stamp.epoch;
            session::bind_compressed(&mut sessions, &artifacts, &problem, T).unwrap();

            let values = activations(6, HIDDEN as usize, 103 + batch_index as usize * 19);
            let fp16_bits: Vec<Vec<u16>> = values
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&value| voltafabric::completion::encode_fp16(value).unwrap())
                        .collect()
                })
                .collect();
            let decoded: Vec<Vec<f64>> = fp16_bits
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&value| voltafabric::completion::decode_fp16(value).unwrap())
                        .collect()
                })
                .collect();
            assert_eq!(
                bits(&values.concat()),
                bits(&decoded.concat()),
                "fixture values must be exactly representable before comparing formats"
            );

            let report = if typed {
                session::run_compressed_batch_with_fp16_inputs(
                    &mut sessions,
                    &artifacts,
                    &problem,
                    &fp16_bits,
                    &BTreeMap::new(),
                    session::CompletionFormat::F64,
                    T,
                )
            } else {
                session::run_compressed_batch(
                    &mut sessions,
                    &artifacts,
                    &problem,
                    &values,
                    &BTreeMap::new(),
                    false,
                    T,
                )
            }
            .unwrap();
            assert!(report.ok, "{:?}", report.error);
            let expected = oracle
                .output(&problem, if typed { &decoded } else { &values })
                .unwrap();
            assert_eq!(
                output_bits(report.output.as_ref().unwrap()),
                output_bits(&expected),
                "batch={batch_index} typed={typed}"
            );
            for receipt in &report.receipts {
                if typed {
                    *fp16_request_bytes.entry(receipt.rank).or_default() += receipt.request_bytes;
                } else {
                    *f64_request_bytes.entry(receipt.rank).or_default() += receipt.request_bytes;
                }
            }
            if !typed {
                for rank in routing::dispatch(&problem).unwrap().ranks {
                    let rows = rank
                        .slots
                        .iter()
                        .map(|slot| slot.slot.token)
                        .collect::<BTreeSet<_>>()
                        .len() as u64;
                    *expected_saving.entry(rank.rank).or_default() += rows
                        * HIDDEN
                        * (std::mem::size_of::<f64>() - std::mem::size_of::<u16>()) as u64;
                }
            }
        }
    }

    assert_eq!(f64_request_bytes.len(), 2);
    assert_eq!(fp16_request_bytes.len(), 2);
    for rank in [2, 9] {
        let measured = f64_request_bytes[&rank] - fp16_request_bytes[&rank];
        assert!(
            measured > 0 && measured.abs_diff(expected_saving[&rank]) <= 512,
            "rank {rank}: measured={measured} body_expected={}",
            expected_saving[&rank]
        );
    }
    for session in &sessions {
        assert_eq!(
            session.traffic.loads_transferred, 1,
            "rank {}",
            session.rank
        );
        assert_eq!(session.traffic.rebinds, 3, "rank {}", session.rank);
        assert_eq!(
            session.traffic.execs,
            2 * DYNAMIC_BATCHES,
            "rank {}",
            session.rank
        );
        let artifact = artifacts
            .iter()
            .find(|artifact| artifact.stack().rank() == session.rank)
            .unwrap();
        assert!(session.traffic.weight_bytes_sent >= artifact.bytes().len() as u64);
        assert!(session.traffic.weight_bytes_sent <= artifact.bytes().len() as u64 + 1024);
    }
    eprintln!(
        "A172_DYNAMIC_FP16 batches_per_format={DYNAMIC_BATCHES} f64_request_bytes={f64_request_bytes:?} fp16_request_bytes={fp16_request_bytes:?} expected_body_saving={expected_saving:?} rank_traffic={:?} exact_oracle=true no_replay=true",
        sessions
            .iter()
            .map(|session| (session.rank, session.traffic.clone()))
            .collect::<Vec<_>>()
    );
    for session in sessions {
        let close = session.close(T);
        assert_eq!(close.exit_code, Some(0));
        assert_eq!(close.bye_execs, Some(2 * DYNAMIC_BATCHES));
    }
}

#[test]
#[ignore = "requires the verified current x86-64 vf-rank-session worker on PC1 and strict SSH"]
fn a174_pc6_pc1_raw_fp16_compressed_series_cli_matches_typed_baseline() {
    let worker = std::env::var("VF_PC1_SESSION_WORKER")
        .expect("set VF_PC1_SESSION_WORKER to the verified absolute PC1 worker path");
    let run_prefix = std::env::var("VF_A176_RUN_ID_PREFIX").unwrap_or_else(|_| "a174-r2".into());
    assert!(
        !run_prefix.is_empty()
            && run_prefix.len() <= 48
            && run_prefix
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
        "A176 run ID prefix must be 1-48 ASCII letters, digits, or hyphens"
    );
    let typed_run_id = format!("{run_prefix}-typed-json");
    let raw_run_id = format!("{run_prefix}-raw-binary");
    let temp = project_temp_dir();
    let dir = temp.0.as_path();
    let endpoints = vec![endpoint(2), pc1_endpoint(9, &worker)];

    let mut typed = series(&typed_run_id, dir, 3);
    typed.workers = endpoints.clone();
    typed.epoch_every = 2;
    typed.record = vec![0, 1, 2];
    let mut fp16_bits: Vec<Vec<u16>> = typed
        .inputs
        .iter()
        .map(|row| {
            row.iter()
                .map(|&value| voltafabric::completion::encode_fp16(value).unwrap())
                .collect()
        })
        .collect();
    fp16_bits[0][0] = 0x8000;
    fp16_bits[0][1] = 0x0001;
    fp16_bits[0][2] = 0x8001;
    typed.inputs.clear();
    typed.inputs_fp16_bits = Some(fp16_bits.clone());

    // Preserve a real typed-JSON CLI baseline on the same two workers. The raw
    // invocation below must have a different identity; an uncertain result is never replayed.
    let typed_path = dir.join("typed-series.json");
    std::fs::write(&typed_path, serde_json::to_vec(&typed).unwrap()).unwrap();
    let typed_cli = Command::new(env!("CARGO_BIN_EXE_vf"))
        .arg("session-run-compressed")
        .arg(&typed_path)
        .output()
        .unwrap();
    assert_eq!(
        typed_cli.status.code(),
        Some(0),
        "typed CLI stderr={}",
        String::from_utf8_lossy(&typed_cli.stderr)
    );
    let typed_envelope: serde_json::Value = serde_json::from_slice(&typed_cli.stdout).unwrap();
    assert_eq!(typed_envelope["ok"], true, "{typed_envelope}");
    let typed_report: CompressedSeriesReport =
        serde_json::from_value(typed_envelope["result"].clone()).unwrap();
    assert!(typed_report.series.ok, "{:?}", typed_report.series.error);

    let mut raw_series = typed.clone();
    raw_series.routing.stamp.run_id = raw_run_id.clone();
    raw_series.inputs_fp16_bits = None;
    assert_eq!(
        raw_series.routing.stamp.run_id.len(),
        typed.routing.stamp.run_id.len(),
        "the typed/raw comparison must hold identity-frame size constant"
    );
    let header = serde_json::json!({
        "series": raw_series,
        "rows": fp16_bits.len(),
        "columns": HIDDEN
    });
    let mut raw_wire = serde_json::to_vec(&header).unwrap();
    raw_wire.push(b'\n');
    for &bits in fp16_bits.iter().flatten() {
        raw_wire.extend_from_slice(&bits.to_le_bytes());
    }
    assert!(
        raw_wire.len() < std::fs::metadata(&typed_path).unwrap().len() as usize,
        "raw FP16 CLI input must be smaller than typed-u16 JSON"
    );

    let mut raw_child = Command::new(env!("CARGO_BIN_EXE_vf"))
        .args(["session-run-compressed", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    raw_child
        .stdin
        .take()
        .unwrap()
        .write_all(&raw_wire)
        .unwrap();
    let raw_cli = raw_child.wait_with_output().unwrap();
    assert_eq!(
        raw_cli.status.code(),
        Some(0),
        "raw CLI stderr={}",
        String::from_utf8_lossy(&raw_cli.stderr)
    );
    let raw_envelope: serde_json::Value = serde_json::from_slice(&raw_cli.stdout).unwrap();
    assert_eq!(raw_envelope["ok"], true, "{raw_envelope}");
    let raw_report: CompressedSeriesReport =
        serde_json::from_value(raw_envelope["result"].clone()).unwrap();
    assert!(raw_report.series.ok, "{:?}", raw_report.series.error);

    for report in [&typed_report, &raw_report] {
        assert_eq!(
            (report.series.batches_ok, report.series.exact_batches),
            (3, 3)
        );
        assert_eq!(report.series.recorded.len(), 3);
        assert!(report
            .series
            .traffic
            .iter()
            .all(|rank| { rank.traffic.loads_transferred == 1 && rank.traffic.execs == 3 }));
        assert!(report.series.close.iter().all(|close| {
            close.exit_code == Some(0) && close.bye_execs == Some(3) && !close.killed
        }));
        let arches: BTreeMap<_, _> = report
            .series
            .hellos
            .iter()
            .map(|hello| (hello.rank, hello.arch.as_str()))
            .collect();
        assert_eq!(arches, BTreeMap::from([(2, "aarch64"), (9, "x86_64")]));
    }

    // The caller-side input framing is different, but the persistent worker frames
    // and their application byte totals are unchanged.
    assert_eq!(
        raw_report.series.exec_request_bytes,
        typed_report.series.exec_request_bytes
    );
    assert_eq!(
        raw_report.series.recorded.len(),
        typed_report.series.recorded.len()
    );
    let artifacts: Vec<_> = [2, 9].map(artifact).into();
    let oracle = session::DenseOracle::new(&artifacts).unwrap();
    let decoded: Vec<Vec<f64>> = fp16_bits
        .iter()
        .map(|row| {
            row.iter()
                .map(|&value| voltafabric::completion::decode_fp16(value).unwrap())
                .collect()
        })
        .collect();
    let problem_for = |run_id: &str, batch: u64| {
        let mut problem = routing_problem(run_id);
        problem.stamp.batch += batch;
        problem.stamp.epoch += batch / typed.epoch_every;
        problem.registry.epoch = problem.stamp.epoch;
        problem
    };
    for (raw, typed) in raw_report
        .series
        .recorded
        .iter()
        .zip(&typed_report.series.recorded)
    {
        assert_eq!(raw.batch, typed.batch);
        assert_eq!(
            output_bits(&raw.output),
            output_bits(&typed.output),
            "raw/typed output batch {}",
            raw.batch
        );
        for (run_id, recorded) in [(raw_run_id.as_str(), raw), (typed_run_id.as_str(), typed)] {
            let problem = problem_for(run_id, recorded.batch);
            assert_eq!(
                output_bits(&recorded.output),
                output_bits(&oracle.output(&problem, &decoded).unwrap()),
                "dense oracle run={run_id} batch={}",
                recorded.batch
            );
        }
    }
    eprintln!(
        "A174_PC6_PC1_RAW_FP16 typed_input_bytes={} raw_input_bytes={} worker_exec_request_bytes={} exact_oracle=true architectures=aarch64/x86_64 loads_once=true no_replay=true",
        std::fs::metadata(&typed_path).unwrap().len(),
        raw_wire.len(),
        raw_report.series.exec_request_bytes
    );
}

#[test]
#[ignore = "requires the verified current x86-64 vf-rank-session worker on PC1 and strict SSH"]
fn pc6_pc1_fp16_completion_saves_response_bytes_against_f64() {
    let worker = std::env::var("VF_PC1_SESSION_WORKER")
        .expect("set VF_PC1_SESSION_WORKER to the verified absolute PC1 worker path");
    let run_prefix = std::env::var("VF_A178_RUN_ID_PREFIX")
        .expect("set VF_A178_RUN_ID_PREFIX to a fresh run identifier");
    assert!(
        !run_prefix.is_empty()
            && run_prefix.len() <= 40
            && run_prefix
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
        "A178 run ID prefix must be 1-40 ASCII letters, digits, or hyphens"
    );
    let f64_run_id = format!("{run_prefix}-completion-f64");
    let fp16_run_id = format!("{run_prefix}-completion-f16");
    let local_fp16_problem = routing_problem(&format!("{run_prefix}-local-fp16"));
    assert_eq!(f64_run_id.len(), fp16_run_id.len());

    let artifacts = vec![artifact(2), artifact(9)];
    let f64_problem = routing_problem(&f64_run_id);
    let fp16_problem = routing_problem(&fp16_run_id);
    let inputs = activations(6, HIDDEN as usize, 5);
    let oracle = session::DenseOracle::for_routing(&artifacts, &f64_problem).unwrap();
    let expected_f64 = oracle.output(&f64_problem, &inputs).unwrap();
    let expected_f64_bits = output_bits(&expected_f64);

    let mut local_fp16_sessions = session::prepare_compressed(
        &[endpoint(2), endpoint(9)],
        &artifacts,
        &local_fp16_problem,
        T,
    )
    .unwrap();
    let local_fp16 = session::run_compressed_batch_fp16(
        &mut local_fp16_sessions,
        &artifacts,
        &local_fp16_problem,
        &inputs,
        &BTreeMap::new(),
        false,
        T,
    )
    .unwrap();
    assert!(local_fp16.ok, "{:?}", local_fp16.error);
    let local_fp16_bits = output_bits(local_fp16.output.as_ref().unwrap());
    for session in local_fp16_sessions {
        assert_eq!(session.close(T).exit_code, Some(0));
    }

    let traffic_snapshot = |sessions: &[Session]| {
        sessions
            .iter()
            .map(|session| {
                (
                    session.rank,
                    (session.traffic.bytes_sent, session.traffic.bytes_received),
                )
            })
            .collect::<BTreeMap<_, _>>()
    };

    let mut remote_f64_sessions = session::prepare_compressed(
        &[endpoint(2), pc1_endpoint(9, &worker)],
        &artifacts,
        &f64_problem,
        T,
    )
    .unwrap();
    let remote_f64_after_load = traffic_snapshot(&remote_f64_sessions);
    let remote_f64 = session::run_compressed_batch(
        &mut remote_f64_sessions,
        &artifacts,
        &f64_problem,
        &inputs,
        &BTreeMap::new(),
        false,
        T,
    )
    .unwrap();
    assert!(remote_f64.ok, "{:?}", remote_f64.error);
    let remote_f64_bits = output_bits(remote_f64.output.as_ref().unwrap());
    assert_eq!(
        remote_f64_bits, expected_f64_bits,
        "F64 result differs from independent dense oracle"
    );
    let remote_f64_after_exec = traffic_snapshot(&remote_f64_sessions);
    let mut remote_f64_request_by_rank = BTreeMap::<u32, u64>::new();
    for receipt in &remote_f64.receipts {
        *remote_f64_request_by_rank.entry(receipt.rank).or_default() += receipt.request_bytes;
    }
    assert!(remote_f64_sessions
        .iter()
        .all(|session| session.traffic.execs == 1));
    assert!(remote_f64_sessions
        .iter()
        .all(|session| session.traffic.loads_transferred == 1));
    let remote_f64_arch = remote_f64_sessions
        .iter()
        .find(|session| session.rank == 9)
        .unwrap()
        .hello
        .arch
        .clone();
    for session in remote_f64_sessions {
        assert_eq!(session.close(T).exit_code, Some(0));
    }

    let mut remote_fp16_sessions = session::prepare_compressed(
        &[endpoint(2), pc1_endpoint(9, &worker)],
        &artifacts,
        &fp16_problem,
        T,
    )
    .unwrap();
    let remote_fp16_after_load = traffic_snapshot(&remote_fp16_sessions);
    let remote_fp16 = session::run_compressed_batch_fp16(
        &mut remote_fp16_sessions,
        &artifacts,
        &fp16_problem,
        &inputs,
        &BTreeMap::new(),
        false,
        T,
    )
    .unwrap();
    assert!(remote_fp16.ok, "{:?}", remote_fp16.error);
    let remote_fp16_output = remote_fp16.output.as_ref().unwrap();
    let remote_fp16_bits = output_bits(remote_fp16_output);
    assert_eq!(
        remote_fp16_bits, local_fp16_bits,
        "PC1 x86-64 FP16 completions differ from the PC6 ARM64 CPU control"
    );
    let remote_fp16_after_exec = traffic_snapshot(&remote_fp16_sessions);
    let mut remote_fp16_request_by_rank = BTreeMap::<u32, u64>::new();
    for receipt in &remote_fp16.receipts {
        *remote_fp16_request_by_rank.entry(receipt.rank).or_default() += receipt.request_bytes;
    }
    assert!(remote_fp16_sessions
        .iter()
        .all(|session| session.traffic.execs == 1));
    assert!(remote_fp16_sessions
        .iter()
        .all(|session| session.traffic.loads_transferred == 1));
    let remote_fp16_arch = remote_fp16_sessions
        .iter()
        .find(|session| session.rank == 9)
        .unwrap()
        .hello
        .arch
        .clone();
    for session in remote_fp16_sessions {
        assert_eq!(session.close(T).exit_code, Some(0));
    }
    assert_eq!(remote_f64_arch, "x86_64");
    assert_eq!(remote_fp16_arch, "x86_64");

    let f64_rank9_sent = remote_f64_after_exec[&9].0 - remote_f64_after_load[&9].0;
    let f64_rank9_received = remote_f64_after_exec[&9].1 - remote_f64_after_load[&9].1;
    let fp16_rank9_sent = remote_fp16_after_exec[&9].0 - remote_fp16_after_load[&9].0;
    let fp16_rank9_received = remote_fp16_after_exec[&9].1 - remote_fp16_after_load[&9].1;
    let rank9_slots = routing::dispatch(&fp16_problem)
        .unwrap()
        .ranks
        .into_iter()
        .find(|rank| rank.rank == 9)
        .unwrap()
        .slots
        .len() as u64;
    let expected_response_body_saving =
        rank9_slots * HIDDEN * (std::mem::size_of::<f64>() - std::mem::size_of::<u16>()) as u64;
    let measured_response_saving = f64_rank9_received - fp16_rank9_received;
    assert!(
        measured_response_saving > 0,
        "FP16 must reduce PC1 application response bytes: F64={f64_rank9_received}, FP16={fp16_rank9_received}"
    );

    let mut max_abs_error = 0.0_f64;
    let mut sum_squared_error = 0.0_f64;
    let mut element_count = 0_u64;
    for (reference_row, candidate_row) in expected_f64.iter().zip(remote_fp16_output) {
        assert_eq!(reference_row.len(), candidate_row.len());
        for (&reference, &candidate) in reference_row.iter().zip(candidate_row) {
            assert!(
                candidate.is_finite(),
                "FP16 completion produced a non-finite output"
            );
            let error = (reference - candidate).abs();
            max_abs_error = max_abs_error.max(error);
            sum_squared_error += error * error;
            element_count += 1;
        }
    }
    assert_eq!(
        element_count as usize,
        expected_f64.iter().map(Vec::len).sum::<usize>()
    );
    let rms_abs_error = (sum_squared_error / element_count as f64).sqrt();

    let receipt_path = PathBuf::from(
        std::env::var("VF_A178_RECEIPT_PATH")
            .expect("set VF_A178_RECEIPT_PATH inside the canonical evidence directory"),
    );
    assert!(
        !receipt_path.exists(),
        "refusing to overwrite A178 receipt {receipt_path:?}"
    );
    std::fs::create_dir_all(
        receipt_path
            .parent()
            .expect("receipt path must have a parent"),
    )
    .unwrap();
    let receipt = serde_json::json!({
        "schema_version": 1,
        "phase": "A178",
        "test": "pc6_pc1_fp16_completion_saves_response_bytes_against_f64",
        "run_ids": {"f64": f64_run_id, "fp16": fp16_run_id},
        "remote_rank": 9,
        "remote_arch": remote_fp16_arch,
        "local_fp16_control_arch": "aarch64",
        "same_f64_inputs_and_resident_artifacts": true,
        "weights_loaded_once_per_worker_per_session": true,
        "one_exec_per_worker_per_session": true,
        "application_rank9_exec_bytes": {
            "f64": {"sent": f64_rank9_sent, "received": f64_rank9_received, "request_bytes": remote_f64_request_by_rank[&9]},
            "fp16": {"sent": fp16_rank9_sent, "received": fp16_rank9_received, "request_bytes": remote_fp16_request_by_rank[&9]}
        },
        "rank9_completion_rows": rank9_slots,
        "hidden": HIDDEN,
        "expected_value_body_saving_bytes": expected_response_body_saving,
        "measured_application_response_saving_bytes": measured_response_saving,
        "output_elements": element_count,
        "quality_against_independent_dense_f64_oracle": {"max_abs_error": max_abs_error, "rms_abs_error": rms_abs_error},
        "mixed_fp16_output_bits_match_local_arm64_control": remote_fp16_bits == local_fp16_bits,
        "fp64_remote_output_bits_match_dense_oracle": remote_f64_bits == expected_f64_bits,
        "ssh_totals_are_in_external_A178_SSH_RECEIPT": true,
        "decision_scope": "synthetic CPU fixture only; no model-quality threshold or default FP16 promotion"
    });
    let temp_path = receipt_path.with_extension("json.tmp");
    std::fs::write(&temp_path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
    std::fs::rename(&temp_path, &receipt_path).unwrap();
    eprintln!(
        "A178_FP16_COMPLETION rank9_f64_rx={f64_rank9_received} rank9_fp16_rx={fp16_rank9_received} response_saved={measured_response_saving} expected_body_saved={expected_response_body_saving} quality_max_abs={max_abs_error:.9} quality_rms_abs={rms_abs_error:.9} local_arm64_bits_match=true f64_dense_oracle_bits_match=true"
    );
}

#[test]
#[ignore = "requires the verified current x86-64 vf-rank-session worker on PC1 and strict SSH"]
fn pc6_pc1_combined_fp16_input_completion_transport() {
    let worker = std::env::var("VF_PC1_SESSION_WORKER")
        .expect("set VF_PC1_SESSION_WORKER to the verified absolute PC1 worker path");
    let run_prefix = std::env::var("VF_A179_RUN_ID_PREFIX")
        .expect("set VF_A179_RUN_ID_PREFIX to a fresh run identifier");
    assert!(
        !run_prefix.is_empty()
            && run_prefix.len() <= 32
            && run_prefix
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
        "A179 run ID prefix must be 1-32 ASCII letters, digits, or hyphens"
    );
    let f64_run_id = format!("{run_prefix}-f64-f64");
    let fp16_run_id = format!("{run_prefix}-f16-f16");
    let local_run_id = format!("{run_prefix}-local-f16");
    assert_eq!(f64_run_id.len(), fp16_run_id.len());

    let artifacts = vec![artifact(2), artifact(9)];
    let f64_problem = routing_problem(&f64_run_id);
    let fp16_problem = routing_problem(&fp16_run_id);
    let local_problem = routing_problem(&local_run_id);
    let f64_inputs = activations(6, HIDDEN as usize, 5);
    // These fixture values are multiples of 1/64 and lie in [-0.75, 0.75],
    // so this particular input comparison introduces no FP16 quantization.
    let fp16_inputs: Vec<Vec<u16>> = f64_inputs
        .iter()
        .map(|row| {
            row.iter()
                .map(|&value| {
                    assert!(value.abs() <= 0.75);
                    assert_eq!((value * 64.0).fract(), 0.0);
                    let bits = voltafabric::completion::encode_fp16(value).unwrap();
                    assert_eq!(voltafabric::completion::decode_fp16(bits).unwrap(), value);
                    bits
                })
                .collect()
        })
        .collect();
    let oracle = session::DenseOracle::for_routing(&artifacts, &f64_problem).unwrap();
    let expected_f64 = oracle.output(&f64_problem, &f64_inputs).unwrap();
    let expected_f64_bits = output_bits(&expected_f64);

    // Keep the ARM64 control identity distinct from both later remote sessions:
    // the coordinator intentionally rejects a reused Stamp.
    let mut local_sessions =
        session::prepare_compressed(&[endpoint(2), endpoint(9)], &artifacts, &local_problem, T)
            .unwrap();
    let local_fp16_started = std::time::Instant::now();
    let local_fp16 = session::run_compressed_batch_with_fp16_inputs(
        &mut local_sessions,
        &artifacts,
        &local_problem,
        &fp16_inputs,
        &BTreeMap::new(),
        session::CompletionFormat::Fp16,
        T,
    )
    .unwrap();
    let local_fp16_batch_us = local_fp16_started.elapsed().as_micros() as u64;
    assert!(local_fp16.ok, "{:?}", local_fp16.error);
    let local_fp16_bits = output_bits(local_fp16.output.as_ref().unwrap());
    for session in local_sessions {
        assert_eq!(session.close(T).exit_code, Some(0));
    }

    let traffic_snapshot = |sessions: &[Session]| {
        sessions
            .iter()
            .map(|session| {
                (
                    session.rank,
                    (session.traffic.bytes_sent, session.traffic.bytes_received),
                )
            })
            .collect::<BTreeMap<_, _>>()
    };

    let endpoints = [endpoint(2), pc1_endpoint(9, &worker)];
    let mut f64_sessions =
        session::prepare_compressed(&endpoints, &artifacts, &f64_problem, T).unwrap();
    let f64_after_load = traffic_snapshot(&f64_sessions);
    let f64_started = std::time::Instant::now();
    let f64_report = session::run_compressed_batch(
        &mut f64_sessions,
        &artifacts,
        &f64_problem,
        &f64_inputs,
        &BTreeMap::new(),
        false,
        T,
    )
    .unwrap();
    let f64_batch_us = f64_started.elapsed().as_micros() as u64;
    assert!(f64_report.ok, "{:?}", f64_report.error);
    let f64_output_bits = output_bits(f64_report.output.as_ref().unwrap());
    assert_eq!(f64_output_bits, expected_f64_bits);
    let f64_after_exec = traffic_snapshot(&f64_sessions);
    let mut f64_request_by_rank = BTreeMap::<u32, u64>::new();
    for receipt in &f64_report.receipts {
        *f64_request_by_rank.entry(receipt.rank).or_default() += receipt.request_bytes;
    }
    assert!(f64_sessions
        .iter()
        .all(|session| { session.traffic.execs == 1 && session.traffic.loads_transferred == 1 }));
    let f64_remote_arch = f64_sessions
        .iter()
        .find(|session| session.rank == 9)
        .unwrap()
        .hello
        .arch
        .clone();
    for session in f64_sessions {
        assert_eq!(session.close(T).exit_code, Some(0));
    }

    let mut fp16_sessions =
        session::prepare_compressed(&endpoints, &artifacts, &fp16_problem, T).unwrap();
    let fp16_after_load = traffic_snapshot(&fp16_sessions);
    let fp16_started = std::time::Instant::now();
    let fp16_report = session::run_compressed_batch_with_fp16_inputs(
        &mut fp16_sessions,
        &artifacts,
        &fp16_problem,
        &fp16_inputs,
        &BTreeMap::new(),
        session::CompletionFormat::Fp16,
        T,
    )
    .unwrap();
    let fp16_batch_us = fp16_started.elapsed().as_micros() as u64;
    assert!(fp16_report.ok, "{:?}", fp16_report.error);
    let fp16_output = fp16_report.output.as_ref().unwrap();
    let fp16_output_bits = output_bits(fp16_output);
    assert_eq!(
        fp16_output_bits, local_fp16_bits,
        "PC1 x86-64 FP16 input/completion result must match PC6 ARM64 control"
    );
    let fp16_after_exec = traffic_snapshot(&fp16_sessions);
    let mut fp16_request_by_rank = BTreeMap::<u32, u64>::new();
    for receipt in &fp16_report.receipts {
        *fp16_request_by_rank.entry(receipt.rank).or_default() += receipt.request_bytes;
    }
    assert!(fp16_sessions
        .iter()
        .all(|session| { session.traffic.execs == 1 && session.traffic.loads_transferred == 1 }));
    let fp16_remote_arch = fp16_sessions
        .iter()
        .find(|session| session.rank == 9)
        .unwrap()
        .hello
        .arch
        .clone();
    for session in fp16_sessions {
        assert_eq!(session.close(T).exit_code, Some(0));
    }
    assert_eq!(f64_remote_arch, "x86_64");
    assert_eq!(fp16_remote_arch, "x86_64");

    let f64_rank9_sent = f64_after_exec[&9].0 - f64_after_load[&9].0;
    let f64_rank9_received = f64_after_exec[&9].1 - f64_after_load[&9].1;
    let fp16_rank9_sent = fp16_after_exec[&9].0 - fp16_after_load[&9].0;
    let fp16_rank9_received = fp16_after_exec[&9].1 - fp16_after_load[&9].1;
    assert_eq!(f64_rank9_sent, f64_request_by_rank[&9]);
    assert_eq!(fp16_rank9_sent, fp16_request_by_rank[&9]);
    let rank9 = routing::dispatch(&fp16_problem)
        .unwrap()
        .ranks
        .into_iter()
        .find(|rank| rank.rank == 9)
        .unwrap();
    let rank9_tokens = rank9
        .slots
        .iter()
        .map(|slot| slot.slot.token)
        .collect::<BTreeSet<_>>()
        .len() as u64;
    let rank9_completion_rows = rank9.slots.len() as u64;
    let scalar_bytes_saved = (std::mem::size_of::<f64>() - std::mem::size_of::<u16>()) as u64;
    let expected_request_body_saving = rank9_tokens * HIDDEN * scalar_bytes_saved;
    let expected_response_body_saving = rank9_completion_rows * HIDDEN * scalar_bytes_saved;
    let measured_request_saving = f64_rank9_sent - fp16_rank9_sent;
    let measured_response_saving = f64_rank9_received - fp16_rank9_received;
    assert!(
        measured_request_saving > 0
            && measured_request_saving.abs_diff(expected_request_body_saving) <= 128,
        "rank 9 FP16 input request saved {measured_request_saving}; expected body saving {expected_request_body_saving}"
    );
    assert!(
        measured_response_saving > 0
            && measured_response_saving.abs_diff(expected_response_body_saving) <= 128,
        "rank 9 FP16 completion response saved {measured_response_saving}; expected body saving {expected_response_body_saving}"
    );

    let mut max_abs_error = 0.0_f64;
    let mut sum_squared_error = 0.0_f64;
    let mut element_count = 0_u64;
    for (reference_row, candidate_row) in expected_f64.iter().zip(fp16_output) {
        assert_eq!(reference_row.len(), candidate_row.len());
        for (&reference, &candidate) in reference_row.iter().zip(candidate_row) {
            assert!(
                candidate.is_finite(),
                "FP16 candidate output must be finite"
            );
            let error = (reference - candidate).abs();
            max_abs_error = max_abs_error.max(error);
            sum_squared_error += error * error;
            element_count += 1;
        }
    }
    let rms_abs_error = (sum_squared_error / element_count as f64).sqrt();
    let receipt_path = PathBuf::from(
        std::env::var("VF_A179_RECEIPT_PATH")
            .expect("set VF_A179_RECEIPT_PATH inside the canonical evidence directory"),
    );
    assert!(
        !receipt_path.exists(),
        "refusing to overwrite A179 receipt {receipt_path:?}"
    );
    std::fs::create_dir_all(
        receipt_path
            .parent()
            .expect("receipt path must have a parent"),
    )
    .unwrap();
    let receipt = serde_json::json!({
        "schema_version": 1,
        "phase": "A179",
        "test": "pc6_pc1_combined_fp16_input_completion_transport",
        "run_ids": {"f64_f64": f64_run_id, "fp16_fp16": fp16_run_id, "local_fp16_control": local_run_id},
        "architectures": {"local_control": "aarch64", "remote_rank9_f64_f64": f64_remote_arch, "remote_rank9_fp16_fp16": fp16_remote_arch},
        "same_resident_artifacts_and_exact_representable_inputs": true,
        "weights_loaded_once_and_one_exec_per_rank_per_remote_session": true,
        "rank9": {
            "unique_input_tokens": rank9_tokens,
            "completion_rows": rank9_completion_rows,
            "f64_f64_application_bytes": {"sent": f64_rank9_sent, "received": f64_rank9_received, "request_bytes": f64_request_by_rank[&9]},
            "fp16_fp16_application_bytes": {"sent": fp16_rank9_sent, "received": fp16_rank9_received, "request_bytes": fp16_request_by_rank[&9]},
            "expected_fp16_input_body_saving_bytes": expected_request_body_saving,
            "measured_fp16_input_request_saving_bytes": measured_request_saving,
            "expected_fp16_completion_body_saving_bytes": expected_response_body_saving,
            "measured_fp16_completion_response_saving_bytes": measured_response_saving,
            "measured_combined_application_saving_bytes": measured_request_saving + measured_response_saving
        },
        "quality_against_independent_dense_f64_oracle": {"max_abs_error": max_abs_error, "rms_abs_error": rms_abs_error, "values": element_count},
        "remote_fp16_fp16_matches_local_arm64_control_bitwise": fp16_output_bits == local_fp16_bits,
        "f64_f64_matches_dense_oracle_bitwise": f64_output_bits == expected_f64_bits,
        "latency_diagnostic_microseconds": {"local_arm64_fp16_fp16": local_fp16_batch_us, "remote_x86_f64_f64": f64_batch_us, "remote_x86_fp16_fp16": fp16_batch_us},
        "ssh_totals": "recorded by external OpenSSH per-connection receipt; includes handshake and protocol, excludes TCP/IP/Ethernet headers",
        "scope": "synthetic CPU fixture; no model-level quality threshold, service throughput claim, GPU, CUDA, RDMA, or default format promotion"
    });
    let temp_path = receipt_path.with_extension("json.tmp");
    std::fs::write(&temp_path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
    std::fs::rename(&temp_path, &receipt_path).unwrap();
    eprintln!(
        "A179_COMBINED_FP16 rank9_request_saved={measured_request_saving} expected_input_body={expected_request_body_saving} rank9_response_saved={measured_response_saving} expected_output_body={expected_response_body_saving} quality_max_abs={max_abs_error:.9} quality_rms_abs={rms_abs_error:.9} local_arm64_bits_match=true f64_dense_oracle_bits_match=true latency_us_f64={f64_batch_us} latency_us_fp16={fp16_batch_us}"
    );
}

#[test]
#[ignore = "requires the verified current x86-64 vf-rank-session worker on PC1 and strict SSH"]
fn pc6_pc1_persistent_combined_fp16_interpc_32_batches() {
    let worker = std::env::var("VF_PC1_SESSION_WORKER")
        .expect("set VF_PC1_SESSION_WORKER to the verified absolute PC1 worker path");
    let run_prefix = std::env::var("VF_A180_RUN_ID_PREFIX")
        .expect("set VF_A180_RUN_ID_PREFIX to a fresh run identifier");
    assert!(
        !run_prefix.is_empty()
            && run_prefix.len() <= 28
            && run_prefix
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
        "A180 run ID prefix must be 1-28 ASCII letters, digits, or hyphens"
    );
    const BATCHES: u64 = 32;
    const EPOCH_EVERY: u64 = 8;
    let f64_run_id = format!("{run_prefix}-f64-f64-32b");
    let fp16_run_id = format!("{run_prefix}-f16-f16-32b");
    let local_run_id = format!("{run_prefix}-local-f16-32b");
    assert_eq!(f64_run_id.len(), fp16_run_id.len());

    let artifacts = vec![artifact(2), artifact(9)];
    let inputs_by_batch: Vec<Vec<Vec<f64>>> = (0..BATCHES)
        .map(|batch| activations(6, HIDDEN as usize, 5 + batch as usize * 13))
        .collect();
    let fp16_inputs_by_batch: Vec<Vec<Vec<u16>>> = inputs_by_batch
        .iter()
        .map(|rows| {
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|&value| {
                            assert!(value.abs() <= 0.75);
                            assert_eq!((value * 64.0).fract(), 0.0);
                            let bits = voltafabric::completion::encode_fp16(value).unwrap();
                            assert_eq!(voltafabric::completion::decode_fp16(bits).unwrap(), value);
                            bits
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    let decoded_inputs_by_batch: Vec<Vec<Vec<f64>>> = fp16_inputs_by_batch
        .iter()
        .map(|rows| {
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|&bits| voltafabric::completion::decode_fp16(bits).unwrap())
                        .collect()
                })
                .collect()
        })
        .collect();
    for (source, decoded) in inputs_by_batch.iter().zip(&decoded_inputs_by_batch) {
        assert_eq!(output_bits(source), output_bits(decoded));
    }

    let problem_for = |run_id: &str, batch: u64| {
        let mut problem = routing_problem(run_id);
        problem.stamp.batch = batch;
        problem.stamp.epoch = EPOCH + batch / EPOCH_EVERY;
        problem.registry.epoch = problem.stamp.epoch;
        problem
    };
    let oracle =
        session::DenseOracle::for_routing(&artifacts, &problem_for(&f64_run_id, 0)).unwrap();
    let rank9 = routing::dispatch(&problem_for(&fp16_run_id, 0))
        .unwrap()
        .ranks
        .into_iter()
        .find(|rank| rank.rank == 9)
        .unwrap();
    let rank9_tokens = rank9
        .slots
        .iter()
        .map(|slot| slot.slot.token)
        .collect::<BTreeSet<_>>()
        .len() as u64;
    let rank9_completion_rows = rank9.slots.len() as u64;
    let scalar_bytes_saved = (std::mem::size_of::<f64>() - std::mem::size_of::<u16>()) as u64;
    let expected_request_body_saving_per_batch = rank9_tokens * HIDDEN * scalar_bytes_saved;
    let expected_response_body_saving_per_batch =
        rank9_completion_rows * HIDDEN * scalar_bytes_saved;

    let traffic_snapshot = |sessions: &[Session]| {
        sessions
            .iter()
            .map(|session| {
                (
                    session.rank,
                    (session.traffic.bytes_sent, session.traffic.bytes_received),
                )
            })
            .collect::<BTreeMap<_, _>>()
    };
    let latencies = |samples: &[u64]| {
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        let p95_index = (sorted.len() * 95).div_ceil(100).saturating_sub(1);
        serde_json::json!({
            "samples_microseconds": samples,
            "median_microseconds": sorted[sorted.len() / 2],
            "p95_microseconds_nearest_rank": sorted[p95_index]
        })
    };

    // Local ARM64 FP16/FP16 reference uses its own complete identity and resident
    // sessions. Every epoch transition is bound explicitly; no batch Stamp is replayed.
    let mut local_sessions = session::prepare_compressed(
        &[endpoint(2), endpoint(9)],
        &artifacts,
        &problem_for(&local_run_id, 0),
        T,
    )
    .unwrap();
    let mut local_outputs = Vec::with_capacity(BATCHES as usize);
    let mut local_latencies = Vec::with_capacity(BATCHES as usize);
    for batch in 0..BATCHES {
        let problem = problem_for(&local_run_id, batch);
        session::bind_compressed(&mut local_sessions, &artifacts, &problem, T).unwrap();
        let started = std::time::Instant::now();
        let report = session::run_compressed_batch_with_fp16_inputs(
            &mut local_sessions,
            &artifacts,
            &problem,
            &fp16_inputs_by_batch[batch as usize],
            &BTreeMap::new(),
            session::CompletionFormat::Fp16,
            T,
        )
        .unwrap();
        local_latencies.push(started.elapsed().as_micros() as u64);
        assert!(report.ok, "local batch {batch}: {:?}", report.error);
        local_outputs.push(output_bits(report.output.as_ref().unwrap()));
    }
    assert!(local_sessions.iter().all(|session| {
        session.traffic.loads_transferred == 1
            && session.traffic.execs == BATCHES
            && session.traffic.rebinds == 3
    }));
    for session in local_sessions {
        let close = session.close(T);
        assert_eq!(close.exit_code, Some(0));
        assert_eq!(close.bye_execs, Some(BATCHES));
    }

    // F64/F64 baseline: every returned batch is independently checked against DenseOracle.
    let endpoints = [endpoint(2), pc1_endpoint(9, &worker)];
    let mut f64_sessions =
        session::prepare_compressed(&endpoints, &artifacts, &problem_for(&f64_run_id, 0), T)
            .unwrap();
    let mut f64_rank9_requests = Vec::with_capacity(BATCHES as usize);
    let mut f64_rank9_responses = Vec::with_capacity(BATCHES as usize);
    let mut f64_latencies = Vec::with_capacity(BATCHES as usize);
    for batch in 0..BATCHES {
        let problem = problem_for(&f64_run_id, batch);
        session::bind_compressed(&mut f64_sessions, &artifacts, &problem, T).unwrap();
        let before = traffic_snapshot(&f64_sessions);
        let started = std::time::Instant::now();
        let report = session::run_compressed_batch(
            &mut f64_sessions,
            &artifacts,
            &problem,
            &inputs_by_batch[batch as usize],
            &BTreeMap::new(),
            false,
            T,
        )
        .unwrap();
        f64_latencies.push(started.elapsed().as_micros() as u64);
        assert!(report.ok, "F64 batch {batch}: {:?}", report.error);
        let expected = oracle
            .output(&problem, &inputs_by_batch[batch as usize])
            .unwrap();
        assert_eq!(
            output_bits(report.output.as_ref().unwrap()),
            output_bits(&expected),
            "F64/F64 batch {batch} must equal the independent dense oracle"
        );
        let mut request_by_rank = BTreeMap::<u32, u64>::new();
        for receipt in &report.receipts {
            *request_by_rank.entry(receipt.rank).or_default() += receipt.request_bytes;
        }
        f64_rank9_requests.push(request_by_rank[&9]);
        let after = traffic_snapshot(&f64_sessions);
        f64_rank9_responses.push(after[&9].1 - before[&9].1);
    }
    assert!(f64_sessions.iter().all(|session| {
        session.traffic.loads_transferred == 1
            && session.traffic.execs == BATCHES
            && session.traffic.rebinds == 3
    }));
    let f64_remote_arch = f64_sessions
        .iter()
        .find(|session| session.rank == 9)
        .unwrap()
        .hello
        .arch
        .clone();
    for session in f64_sessions {
        let close = session.close(T);
        assert_eq!(close.exit_code, Some(0));
        assert_eq!(close.bye_execs, Some(BATCHES));
    }

    // FP16/FP16 candidate runs on the same persistent mixed-architecture topology.
    let mut fp16_sessions =
        session::prepare_compressed(&endpoints, &artifacts, &problem_for(&fp16_run_id, 0), T)
            .unwrap();
    let mut fp16_rank9_requests = Vec::with_capacity(BATCHES as usize);
    let mut fp16_rank9_responses = Vec::with_capacity(BATCHES as usize);
    let mut fp16_latencies = Vec::with_capacity(BATCHES as usize);
    let mut max_abs_error = 0.0_f64;
    let mut sum_squared_error = 0.0_f64;
    let mut element_count = 0_u64;
    for batch in 0..BATCHES {
        let problem = problem_for(&fp16_run_id, batch);
        session::bind_compressed(&mut fp16_sessions, &artifacts, &problem, T).unwrap();
        let before = traffic_snapshot(&fp16_sessions);
        let started = std::time::Instant::now();
        let report = session::run_compressed_batch_with_fp16_inputs(
            &mut fp16_sessions,
            &artifacts,
            &problem,
            &fp16_inputs_by_batch[batch as usize],
            &BTreeMap::new(),
            session::CompletionFormat::Fp16,
            T,
        )
        .unwrap();
        fp16_latencies.push(started.elapsed().as_micros() as u64);
        assert!(report.ok, "FP16 batch {batch}: {:?}", report.error);
        let output = report.output.as_ref().unwrap();
        let output_bits = output_bits(output);
        assert_eq!(
            output_bits, local_outputs[batch as usize],
            "FP16/FP16 batch {batch} must match the local ARM64 control"
        );
        let dense = oracle
            .output(&problem, &decoded_inputs_by_batch[batch as usize])
            .unwrap();
        for (reference_row, candidate_row) in dense.iter().zip(output) {
            assert_eq!(reference_row.len(), candidate_row.len());
            for (&reference, &candidate) in reference_row.iter().zip(candidate_row) {
                assert!(
                    candidate.is_finite(),
                    "FP16 batch {batch} output must be finite"
                );
                let error = (reference - candidate).abs();
                max_abs_error = max_abs_error.max(error);
                sum_squared_error += error * error;
                element_count += 1;
            }
        }
        let mut request_by_rank = BTreeMap::<u32, u64>::new();
        for receipt in &report.receipts {
            *request_by_rank.entry(receipt.rank).or_default() += receipt.request_bytes;
        }
        fp16_rank9_requests.push(request_by_rank[&9]);
        let after = traffic_snapshot(&fp16_sessions);
        fp16_rank9_responses.push(after[&9].1 - before[&9].1);
    }
    assert!(fp16_sessions.iter().all(|session| {
        session.traffic.loads_transferred == 1
            && session.traffic.execs == BATCHES
            && session.traffic.rebinds == 3
    }));
    let fp16_remote_arch = fp16_sessions
        .iter()
        .find(|session| session.rank == 9)
        .unwrap()
        .hello
        .arch
        .clone();
    for session in fp16_sessions {
        let close = session.close(T);
        assert_eq!(close.exit_code, Some(0));
        assert_eq!(close.bye_execs, Some(BATCHES));
    }
    assert_eq!(f64_remote_arch, "x86_64");
    assert_eq!(fp16_remote_arch, "x86_64");

    let mut request_savings = Vec::with_capacity(BATCHES as usize);
    let mut response_savings = Vec::with_capacity(BATCHES as usize);
    for batch in 0..BATCHES as usize {
        let request_saved = f64_rank9_requests[batch] - fp16_rank9_requests[batch];
        let response_saved = f64_rank9_responses[batch] - fp16_rank9_responses[batch];
        assert!(
            request_saved > 0
                && request_saved.abs_diff(expected_request_body_saving_per_batch) <= 128,
            "batch {batch}: rank9 request saved {request_saved}, expected {expected_request_body_saving_per_batch} body bytes"
        );
        assert!(
            response_saved > 0
                && response_saved.abs_diff(expected_response_body_saving_per_batch) <= 128,
            "batch {batch}: rank9 response saved {response_saved}, expected {expected_response_body_saving_per_batch} body bytes"
        );
        request_savings.push(request_saved);
        response_savings.push(response_saved);
    }
    assert_eq!(element_count, BATCHES * 6 * HIDDEN);
    let rms_abs_error = (sum_squared_error / element_count as f64).sqrt();
    let total_request_saving: u64 = request_savings.iter().sum();
    let total_response_saving: u64 = response_savings.iter().sum();
    let receipt_path = PathBuf::from(
        std::env::var("VF_A180_RECEIPT_PATH")
            .expect("set VF_A180_RECEIPT_PATH inside the canonical evidence directory"),
    );
    assert!(
        !receipt_path.exists(),
        "refusing to overwrite A180 receipt {receipt_path:?}"
    );
    std::fs::create_dir_all(
        receipt_path
            .parent()
            .expect("receipt path must have a parent"),
    )
    .unwrap();
    let receipt = serde_json::json!({
        "schema_version":1,"phase":"A180",
        "test":"pc6_pc1_persistent_combined_fp16_interpc_32_batches",
        "run_ids":{"f64_f64":f64_run_id,"fp16_fp16":fp16_run_id,"local_fp16_control":local_run_id},
        "architectures":{"local_control":"aarch64","remote_f64_f64":"x86_64","remote_fp16_fp16":"x86_64"},
        "batches":BATCHES,"epoch_every":EPOCH_EVERY,"epoch_renewals":3,
        "all_activations_exactly_representable_in_fp16":true,
        "resident_artifacts_per_worker_per_session":{"loads":1,"execs":BATCHES,"rebinds":3,"reloaded_during_series":false},
        "rank9":{"unique_input_tokens":rank9_tokens,"completion_rows":rank9_completion_rows,
          "expected_request_saving_per_batch":expected_request_body_saving_per_batch,
          "measured_request_saving_by_batch":request_savings,
          "expected_response_saving_per_batch":expected_response_body_saving_per_batch,
          "measured_response_saving_by_batch":response_savings,
          "total_request_saving":total_request_saving,"total_response_saving":total_response_saving,
          "combined_application_saving":total_request_saving+total_response_saving},
        "quality_against_independent_dense_f64_oracle":{"values":element_count,"max_abs_error":max_abs_error,"rms_abs_error":rms_abs_error},
        "all_f64_f64_batches_match_dense_oracle_bitwise":true,
        "all_fp16_fp16_remote_batches_match_local_arm64_control_bitwise":true,
        "latency_diagnostics_microseconds":{"local_arm64_fp16_fp16":latencies(&local_latencies),"remote_x86_f64_f64":latencies(&f64_latencies),"remote_x86_fp16_fp16":latencies(&fp16_latencies)},
        "ssh_totals":"recorded by external OpenSSH per-connection receipt; each remote mode uses one persistent session; includes handshake and protocol",
        "scope":"Synthetic CPU fixture only; no model-level quality threshold, GPU, CUDA, RDMA/RoCE, GPUDirect or default FP16 promotion"
    });
    let temp_path = receipt_path.with_extension("json.tmp");
    std::fs::write(&temp_path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
    std::fs::rename(&temp_path, &receipt_path).unwrap();
    eprintln!(
        "A180_PERSISTENT batches={BATCHES} rank9_request_saved={total_request_saving} rank9_response_saved={total_response_saving} quality_max_abs={max_abs_error:.9} quality_rms_abs={rms_abs_error:.9} local_arm64_bits_match=true f64_dense_oracle_bits_match=true loads_once=true execs={BATCHES} epoch_renewals=3"
    );
}

#[test]
#[ignore = "requires the verified current x86-64 vf-rank-session worker on PC1 and strict SSH"]
fn pc6_pc1_fp16_latency_screen_alternating_order() {
    let worker = std::env::var("VF_PC1_SESSION_WORKER")
        .expect("set VF_PC1_SESSION_WORKER to the verified absolute PC1 worker path");
    let run_prefix = std::env::var("VF_A181_RUN_ID_PREFIX")
        .expect("set VF_A181_RUN_ID_PREFIX to a fresh run identifier");
    assert!(
        !run_prefix.is_empty()
            && run_prefix.len() <= 28
            && run_prefix
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
        "A181 run ID prefix must be 1-28 ASCII letters, digits, or hyphens"
    );
    const PAIRS: usize = 4;
    const WARMUP_BATCHES: u64 = 8;
    const MEASURED_BATCHES: u64 = 24;
    const TOTAL_BATCHES: u64 = WARMUP_BATCHES + MEASURED_BATCHES;
    const EPOCH_EVERY: u64 = 8;
    const MAX_LATENCY_REGRESSION_PERCENT: f64 = 10.0;

    let artifacts = vec![artifact(2), artifact(9)];
    let inputs_by_batch: Vec<Vec<Vec<f64>>> = (0..TOTAL_BATCHES)
        .map(|batch| activations(6, HIDDEN as usize, 11 + batch as usize * 17))
        .collect();
    let fp16_inputs_by_batch: Vec<Vec<Vec<u16>>> = inputs_by_batch
        .iter()
        .map(|rows| {
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|&value| {
                            assert!(value.abs() <= 0.75);
                            assert_eq!((value * 64.0).fract(), 0.0);
                            let bits = voltafabric::completion::encode_fp16(value).unwrap();
                            assert_eq!(voltafabric::completion::decode_fp16(bits).unwrap(), value);
                            bits
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    let decoded_inputs_by_batch: Vec<Vec<Vec<f64>>> = fp16_inputs_by_batch
        .iter()
        .map(|rows| {
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|&bits| voltafabric::completion::decode_fp16(bits).unwrap())
                        .collect()
                })
                .collect()
        })
        .collect();
    for (source, decoded) in inputs_by_batch.iter().zip(&decoded_inputs_by_batch) {
        assert_eq!(output_bits(source), output_bits(decoded));
    }
    let problem_for = |run_id: &str, batch: u64| {
        let mut problem = routing_problem(run_id);
        problem.stamp.batch = batch;
        problem.stamp.epoch = EPOCH + batch / EPOCH_EVERY;
        problem.registry.epoch = problem.stamp.epoch;
        problem
    };
    let oracle = session::DenseOracle::for_routing(
        &artifacts,
        &problem_for(&format!("{run_prefix}-f64-pair0"), 0),
    )
    .unwrap();
    let route = routing::dispatch(&problem_for(&format!("{run_prefix}-f64-pair0"), 0)).unwrap();
    let rank9 = route.ranks.into_iter().find(|rank| rank.rank == 9).unwrap();
    let rank9_input_tokens = rank9
        .slots
        .iter()
        .map(|slot| slot.slot.token)
        .collect::<BTreeSet<_>>()
        .len() as u64;
    let rank9_completion_rows = rank9.slots.len() as u64;
    let traffic_snapshot = |sessions: &[Session]| {
        sessions
            .iter()
            .map(|session| {
                (
                    session.rank,
                    (session.traffic.bytes_sent, session.traffic.bytes_received),
                )
            })
            .collect::<BTreeMap<_, _>>()
    };
    let summarize = |samples: &[u64]| {
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        let p95_index = (sorted.len() * 95).div_ceil(100).saturating_sub(1);
        (sorted[sorted.len() / 2], sorted[p95_index])
    };

    let mut pair_results = Vec::with_capacity(PAIRS);
    let mut f64_latencies_by_order = [Vec::<u64>::new(), Vec::<u64>::new()];
    let mut fp16_latencies_by_order = [Vec::<u64>::new(), Vec::<u64>::new()];
    let mut f64_request_bytes = 0_u64;
    let mut fp16_request_bytes = 0_u64;
    let mut f64_response_bytes = 0_u64;
    let mut fp16_response_bytes = 0_u64;
    let mut candidate_max_abs_error = 0.0_f64;
    let mut candidate_sum_squared_error = 0.0_f64;
    let mut candidate_element_count = 0_u64;
    let mut first_fp16_outputs: Option<Vec<Vec<Vec<u64>>>> = None;

    for pair in 0..PAIRS {
        let f64_first = pair % 2 == 0;
        let order_index = usize::from(!f64_first);
        let modes = if f64_first {
            [false, true]
        } else {
            [true, false]
        };
        let mut pair_f64_latencies = Vec::new();
        let mut pair_fp16_latencies = Vec::new();
        let mut pair_f64_request = 0_u64;
        let mut pair_fp16_request = 0_u64;
        let mut pair_f64_response = 0_u64;
        let mut pair_fp16_response = 0_u64;
        let mut fp16_outputs = Vec::with_capacity(TOTAL_BATCHES as usize);

        for use_fp16 in modes {
            let mode = if use_fp16 { "fp16" } else { "f64" };
            let run_id = format!("{run_prefix}-{mode}-pair{pair}");
            let mut sessions = session::prepare_compressed(
                &[endpoint(2), pc1_endpoint(9, &worker)],
                &artifacts,
                &problem_for(&run_id, 0),
                T,
            )
            .unwrap();
            let mut latency_samples = Vec::with_capacity(MEASURED_BATCHES as usize);
            let mut request_bytes_for_session = 0_u64;
            let mut response_bytes_for_session = 0_u64;

            for batch in 0..TOTAL_BATCHES {
                let problem = problem_for(&run_id, batch);
                session::bind_compressed(&mut sessions, &artifacts, &problem, T).unwrap();
                let before = traffic_snapshot(&sessions);
                let started = std::time::Instant::now();
                let report = if use_fp16 {
                    session::run_compressed_batch_with_fp16_inputs(
                        &mut sessions,
                        &artifacts,
                        &problem,
                        &fp16_inputs_by_batch[batch as usize],
                        &BTreeMap::new(),
                        session::CompletionFormat::Fp16,
                        T,
                    )
                } else {
                    session::run_compressed_batch(
                        &mut sessions,
                        &artifacts,
                        &problem,
                        &inputs_by_batch[batch as usize],
                        &BTreeMap::new(),
                        false,
                        T,
                    )
                }
                .unwrap();
                let elapsed_us = started.elapsed().as_micros() as u64;
                assert!(
                    report.ok,
                    "pair={pair} mode={mode} batch={batch}: {:?}",
                    report.error
                );
                let output = report.output.as_ref().unwrap();
                let expected = oracle
                    .output(&problem, &inputs_by_batch[batch as usize])
                    .unwrap();
                if use_fp16 {
                    assert!(
                        output.iter().flatten().all(|value| value.is_finite()),
                        "pair={pair} batch={batch}: FP16 output must stay finite"
                    );
                    for (reference_row, candidate_row) in expected.iter().zip(output) {
                        for (&reference, &candidate) in reference_row.iter().zip(candidate_row) {
                            let error = (reference - candidate).abs();
                            candidate_max_abs_error = candidate_max_abs_error.max(error);
                            candidate_sum_squared_error += error * error;
                            candidate_element_count += 1;
                        }
                    }
                    fp16_outputs.push(output_bits(output));
                } else {
                    assert_eq!(
                        output_bits(output),
                        output_bits(&expected),
                        "pair={pair} F64 batch={batch} must equal DenseOracle"
                    );
                }
                if batch >= WARMUP_BATCHES {
                    latency_samples.push(elapsed_us);
                    let request_for_rank9 = report
                        .receipts
                        .iter()
                        .filter(|receipt| receipt.rank == 9)
                        .map(|receipt| receipt.request_bytes)
                        .sum::<u64>();
                    assert!(request_for_rank9 > 0);
                    request_bytes_for_session += request_for_rank9;
                    let after = traffic_snapshot(&sessions);
                    response_bytes_for_session += after[&9].1 - before[&9].1;
                }
            }
            assert!(sessions.iter().all(|session| {
                session.traffic.loads_transferred == 1
                    && session.traffic.execs == TOTAL_BATCHES
                    && session.traffic.rebinds == 3
            }));
            let remote_arch = sessions
                .iter()
                .find(|session| session.rank == 9)
                .unwrap()
                .hello
                .arch
                .clone();
            assert_eq!(remote_arch, "x86_64");
            for session in sessions {
                let close = session.close(T);
                assert_eq!(close.exit_code, Some(0));
                assert_eq!(close.bye_execs, Some(TOTAL_BATCHES));
            }

            if use_fp16 {
                pair_fp16_latencies = latency_samples;
                pair_fp16_request = request_bytes_for_session;
                pair_fp16_response = response_bytes_for_session;
            } else {
                pair_f64_latencies = latency_samples;
                pair_f64_request = request_bytes_for_session;
                pair_f64_response = response_bytes_for_session;
            }
        }

        assert_eq!(fp16_outputs.len(), TOTAL_BATCHES as usize);
        if let Some(reference_outputs) = &first_fp16_outputs {
            assert_eq!(
                &fp16_outputs, reference_outputs,
                "fresh FP16 sessions must produce identical outputs across paired trials"
            );
        } else {
            first_fp16_outputs = Some(fp16_outputs);
        }
        let (f64_median, f64_p95) = summarize(&pair_f64_latencies);
        let (fp16_median, fp16_p95) = summarize(&pair_fp16_latencies);
        let median_regression_percent =
            (fp16_median as f64 - f64_median as f64) * 100.0 / f64_median as f64;
        let p95_regression_percent = (fp16_p95 as f64 - f64_p95 as f64) * 100.0 / f64_p95 as f64;
        let pair_pass = median_regression_percent <= MAX_LATENCY_REGRESSION_PERCENT
            && p95_regression_percent <= MAX_LATENCY_REGRESSION_PERCENT;
        pair_results.push(serde_json::json!({
            "pair":pair,"order":if f64_first {"F64/F64 then FP16/FP16"} else {"FP16/FP16 then F64/F64"},
            "f64_median_us":f64_median,"fp16_median_us":fp16_median,
            "f64_p95_us":f64_p95,"fp16_p95_us":fp16_p95,
            "median_regression_percent":median_regression_percent,
            "p95_regression_percent":p95_regression_percent,
            "latency_screen_pass":pair_pass,
            "f64_rank9_request_bytes":pair_f64_request,"fp16_rank9_request_bytes":pair_fp16_request,
            "f64_rank9_response_bytes":pair_f64_response,"fp16_rank9_response_bytes":pair_fp16_response
        }));
        f64_latencies_by_order[order_index].extend(pair_f64_latencies);
        fp16_latencies_by_order[order_index].extend(pair_fp16_latencies);
        f64_request_bytes += pair_f64_request;
        fp16_request_bytes += pair_fp16_request;
        f64_response_bytes += pair_f64_response;
        fp16_response_bytes += pair_fp16_response;
    }

    let order_summary = |index: usize| {
        let (f64_median, f64_p95) = summarize(&f64_latencies_by_order[index]);
        let (fp16_median, fp16_p95) = summarize(&fp16_latencies_by_order[index]);
        serde_json::json!({
            "order":if index==0 {"F64 first"} else {"FP16 first"},
            "f64_samples":f64_latencies_by_order[index].len(),
            "fp16_samples":fp16_latencies_by_order[index].len(),
            "f64_median_us":f64_median,"fp16_median_us":fp16_median,
            "f64_p95_us":f64_p95,"fp16_p95_us":fp16_p95
        })
    };
    let latency_screen_pass = pair_results
        .iter()
        .all(|result| result["latency_screen_pass"] == true);
    let candidate_rms_error = (candidate_sum_squared_error / candidate_element_count as f64).sqrt();
    let measured_batches_total = (PAIRS as u64) * MEASURED_BATCHES;
    let expected_request_body_saving = measured_batches_total
        * rank9_input_tokens
        * HIDDEN
        * (std::mem::size_of::<f64>() - std::mem::size_of::<u16>()) as u64;
    let expected_response_body_saving = measured_batches_total
        * rank9_completion_rows
        * HIDDEN
        * (std::mem::size_of::<f64>() - std::mem::size_of::<u16>()) as u64;
    let request_saving = f64_request_bytes - fp16_request_bytes;
    let response_saving = f64_response_bytes - fp16_response_bytes;
    assert!(request_saving > 0 && response_saving > 0);

    let receipt_path = PathBuf::from(
        std::env::var("VF_A181_RECEIPT_PATH")
            .expect("set VF_A181_RECEIPT_PATH inside the canonical evidence directory"),
    );
    assert!(
        !receipt_path.exists(),
        "refusing to overwrite A181 receipt {receipt_path:?}"
    );
    std::fs::create_dir_all(
        receipt_path
            .parent()
            .expect("receipt path must have a parent"),
    )
    .unwrap();
    let receipt = serde_json::json!({
        "schema_version":1,"phase":"A181","test":"pc6_pc1_fp16_latency_screen_alternating_order",
        "pairs":pair_results,"order_strata":[order_summary(0),order_summary(1)],
        "predeclared_latency_regression_limit_percent":MAX_LATENCY_REGRESSION_PERCENT,
        "latency_screen_pass":latency_screen_pass,
        "workload":{"warmup_batches_per_session":WARMUP_BATCHES,"measured_batches_per_session":MEASURED_BATCHES,"sessions_per_mode":PAIRS,"epochs_per_session":4,"epoch_renewals_per_session":3},
        "quality":{"fp16_values":candidate_element_count,"max_abs_error":candidate_max_abs_error,"rms_abs_error":candidate_rms_error,"f64_batches_exact_against_dense_oracle":true,"fp16_outputs_finite_and_deterministic":true},
        "rank9_bytes":{"unique_input_tokens":rank9_input_tokens,"completion_rows":rank9_completion_rows,
          "f64_requests":f64_request_bytes,"fp16_requests":fp16_request_bytes,"request_saving":request_saving,"expected_request_body_saving":expected_request_body_saving,
          "f64_responses":f64_response_bytes,"fp16_responses":fp16_response_bytes,"response_saving":response_saving,"expected_response_body_saving":expected_response_body_saving},
        "ssh_totals":"recorded externally for eight fresh persistent connections; includes SSH handshake/control bytes",
        "scope":"CPU latency screening only; no model throughput, service-quality, GPU, CUDA or RDMA claim"
    });
    let temp_path = receipt_path.with_extension("json.tmp");
    std::fs::write(&temp_path, serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
    std::fs::rename(&temp_path, &receipt_path).unwrap();
    eprintln!(
        "A181_LATENCY_SCREEN pass={latency_screen_pass} request_saved={request_saving} response_saved={response_saving} quality_max_abs={candidate_max_abs_error:.9} quality_rms_abs={candidate_rms_error:.9} pairs={PAIRS} order_alternated=true"
    );
}

#[test]
#[ignore = "requires the verified current x86-64 vf-rank-session worker on PC1 and strict SSH"]
fn pc6_pc1_fp16_activation_input_matches_f64_and_reduces_request_bytes() {
    let worker = std::env::var("VF_PC1_SESSION_WORKER")
        .expect("set VF_PC1_SESSION_WORKER to the verified absolute PC1 worker path");
    let run_prefix =
        std::env::var("VF_A176_RUN_ID_PREFIX").unwrap_or_else(|_| "a169-interpc".into());
    assert!(
        !run_prefix.is_empty()
            && run_prefix.len() <= 48
            && run_prefix
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
        "A176 run ID prefix must be 1-48 ASCII letters, digits, or hyphens"
    );
    let fp16_first = std::env::var("VF_A169_INTERPC_FP16_FIRST").unwrap_or_else(|_| "0".into());
    assert!(
        fp16_first == "0" || fp16_first == "1",
        "VF_A169_INTERPC_FP16_FIRST must be 0 or 1"
    );
    let fp16_first = fp16_first == "1";
    let artifacts = vec![artifact(2), artifact(9)];
    let f64_inputs = activations(6, HIDDEN as usize, 5);
    let fp16_inputs: Vec<Vec<u16>> = f64_inputs
        .iter()
        .map(|row| {
            row.iter()
                .map(|&value| voltafabric::completion::encode_fp16(value).unwrap())
                .collect()
        })
        .collect();
    let baseline_problem = routing_problem(&format!("{run_prefix}-baseline"));
    let candidate_problem = routing_problem(&format!("{run_prefix}-candidate"));
    let oracle = session::DenseOracle::for_routing(&artifacts, &baseline_problem).unwrap();
    let expected_output = output_bits(&oracle.output(&baseline_problem, &f64_inputs).unwrap());

    let mut local_sessions = session::prepare_compressed(
        &[endpoint(2), endpoint(9)],
        &artifacts,
        &baseline_problem,
        T,
    )
    .unwrap();
    let mut remote_sessions = session::prepare_compressed(
        &[endpoint(2), pc1_endpoint(9, &worker)],
        &artifacts,
        &candidate_problem,
        T,
    )
    .unwrap();
    let local_sent_after_load = local_sessions
        .iter()
        .map(|s| s.traffic.bytes_sent)
        .sum::<u64>();
    let remote_sent_after_load = remote_sessions
        .iter()
        .map(|s| s.traffic.bytes_sent)
        .sum::<u64>();
    let local_weight_bytes = local_sessions
        .iter()
        .map(|s| s.traffic.weight_bytes_sent)
        .sum::<u64>();
    let remote_weight_bytes = remote_sessions
        .iter()
        .map(|s| s.traffic.weight_bytes_sent)
        .sum::<u64>();
    let remote_arches: BTreeMap<_, _> = remote_sessions
        .iter()
        .map(|session| (session.rank, session.hello.arch.clone()))
        .collect();

    let baseline = session::run_compressed_batch(
        &mut local_sessions,
        &artifacts,
        &baseline_problem,
        &f64_inputs,
        &BTreeMap::new(),
        false,
        T,
    )
    .unwrap();
    assert!(baseline.ok, "{:?}", baseline.error);
    let baseline_output = output_bits(baseline.output.as_ref().unwrap());
    let local_request_bytes: u64 = baseline
        .receipts
        .iter()
        .map(|receipt| receipt.request_bytes)
        .sum();

    let mut mixed_f64_output = None;
    let mut mixed_fp16_output = None;
    let mut mixed_f64_request_bytes = 0_u64;
    let mut mixed_fp16_request_bytes = 0_u64;
    let mut mixed_f64_request_bytes_by_rank = BTreeMap::<u32, u64>::new();
    let mut mixed_fp16_request_bytes_by_rank = BTreeMap::<u32, u64>::new();
    let mut mixed_f64_batch_us = 0_u128;
    let mut mixed_fp16_batch_us = 0_u128;
    let mut next_batch = 1;
    for run_fp16 in [fp16_first, !fp16_first] {
        let mut problem = candidate_problem.clone();
        problem.stamp.batch = next_batch;
        next_batch += 1;
        let started = std::time::Instant::now();
        let report = if run_fp16 {
            session::run_compressed_batch_with_fp16_inputs(
                &mut remote_sessions,
                &artifacts,
                &problem,
                &fp16_inputs,
                &BTreeMap::new(),
                session::CompletionFormat::F64,
                T,
            )
        } else {
            session::run_compressed_batch(
                &mut remote_sessions,
                &artifacts,
                &problem,
                &f64_inputs,
                &BTreeMap::new(),
                false,
                T,
            )
        }
        .unwrap();
        let batch_us = started.elapsed().as_micros();
        assert!(report.ok, "{:?}", report.error);
        let output = output_bits(report.output.as_ref().unwrap());
        let request_bytes: u64 = report
            .receipts
            .iter()
            .map(|receipt| receipt.request_bytes)
            .sum();
        let mut request_bytes_by_rank = BTreeMap::new();
        for receipt in &report.receipts {
            *request_bytes_by_rank.entry(receipt.rank).or_insert(0_u64) += receipt.request_bytes;
        }
        if run_fp16 {
            mixed_fp16_output = Some(output);
            mixed_fp16_request_bytes = request_bytes;
            mixed_fp16_request_bytes_by_rank = request_bytes_by_rank;
            mixed_fp16_batch_us = batch_us;
        } else {
            mixed_f64_output = Some(output);
            mixed_f64_request_bytes = request_bytes;
            mixed_f64_request_bytes_by_rank = request_bytes_by_rank;
            mixed_f64_batch_us = batch_us;
        }
    }
    let mixed_f64_output = mixed_f64_output.unwrap();
    let mixed_fp16_output = mixed_fp16_output.unwrap();

    assert_eq!(
        baseline_output, expected_output,
        "f64 baseline differs from independent dense oracle"
    );
    assert_eq!(
        mixed_f64_output, expected_output,
        "mixed-architecture f64 input differs from independent dense oracle"
    );
    assert_eq!(
        mixed_fp16_output, expected_output,
        "mixed-architecture FP16 input differs from independent dense oracle"
    );
    assert_eq!(
        remote_arches,
        BTreeMap::from([(2, "aarch64".into()), (9, "x86_64".into())])
    );
    assert_eq!(local_weight_bytes, 57_090);
    assert_eq!(remote_weight_bytes, 57_090);
    let mut rank_loads = Vec::new();
    for worker in local_sessions.iter().chain(&remote_sessions) {
        let artifact = artifacts
            .iter()
            .find(|artifact| artifact.stack().rank() == worker.rank)
            .unwrap();
        let load_header = serde_json::to_vec(&Request::LoadArtifactRaw {
            artifact_sha256: artifact.sha256().into(),
            epoch: candidate_problem.stamp.epoch,
            artifact_bytes: artifact.bytes().len() as u64,
        })
        .unwrap();
        let expected_load_bytes = load_header.len() as u64 + 1 + artifact.bytes().len() as u64;
        assert_eq!(
            worker.artifact(),
            Some((artifact.sha256(), candidate_problem.stamp.epoch))
        );
        assert_eq!(worker.traffic.loads_transferred, 1);
        assert_eq!(
            worker.traffic.weight_bytes_sent, expected_load_bytes,
            "each rank must receive one complete LOAD_ARTIFACT_RAW frame"
        );
        rank_loads.push((
            worker.rank,
            worker.traffic.loads_transferred,
            worker.traffic.weight_bytes_sent,
        ));
    }
    assert!(local_sessions
        .iter()
        .all(|session| session.traffic.execs == 1));
    assert!(remote_sessions
        .iter()
        .all(|session| session.traffic.execs == 2));
    let local_exec_bytes = local_sessions
        .iter()
        .map(|s| s.traffic.bytes_sent)
        .sum::<u64>()
        - local_sent_after_load;
    let remote_exec_bytes = remote_sessions
        .iter()
        .map(|s| s.traffic.bytes_sent)
        .sum::<u64>()
        - remote_sent_after_load;
    assert_eq!(local_exec_bytes, local_request_bytes);
    assert_eq!(
        remote_exec_bytes,
        mixed_f64_request_bytes + mixed_fp16_request_bytes
    );
    let measured_saving = mixed_f64_request_bytes - mixed_fp16_request_bytes;
    let expected_body_saving =
        10 * HIDDEN * (std::mem::size_of::<f64>() - std::mem::size_of::<u16>()) as u64;
    assert!(measured_saving > 0 && measured_saving.abs_diff(expected_body_saving) <= 128);
    let remote_rank = 9;
    let rank9_f64_bytes = mixed_f64_request_bytes_by_rank[&remote_rank];
    let rank9_fp16_bytes = mixed_fp16_request_bytes_by_rank[&remote_rank];
    let rank9_tokens = routing::dispatch(&candidate_problem)
        .unwrap()
        .ranks
        .into_iter()
        .find(|rank| rank.rank == remote_rank)
        .unwrap()
        .slots
        .into_iter()
        .map(|slot| slot.slot.token)
        .collect::<BTreeSet<_>>()
        .len() as u64;
    let rank9_expected_body_saving =
        rank9_tokens * HIDDEN * (std::mem::size_of::<f64>() - std::mem::size_of::<u16>()) as u64;
    let rank9_measured_saving = rank9_f64_bytes - rank9_fp16_bytes;
    assert!(
        rank9_measured_saving > 0
            && rank9_measured_saving.abs_diff(rank9_expected_body_saving) <= 128,
        "PC1 rank 9 saving {rank9_measured_saving} must match its {rank9_tokens} dispatched token bodies ({rank9_expected_body_saving} bytes)"
    );
    for worker in local_sessions.into_iter().chain(remote_sessions) {
        assert_eq!(worker.close(T).exit_code, Some(0));
    }
    let series_dir = temp_dir();
    let f64_series_run_id = format!("{run_prefix}-f64-series");
    let fp16_series_run_id = format!("{run_prefix}-fp16-series");
    let mut f64_series = series(&f64_series_run_id, &series_dir, 2);
    f64_series.routing = candidate_problem.clone();
    f64_series.routing.stamp.run_id = f64_series_run_id;
    f64_series.inputs = f64_inputs.clone();
    f64_series.epoch_every = 0;
    f64_series.record = vec![0, 1];
    f64_series.workers = vec![endpoint(2), pc1_endpoint(9, &worker)];
    let mut fp16_series = f64_series.clone();
    fp16_series.routing.stamp.run_id = fp16_series_run_id;
    fp16_series.inputs.clear();
    fp16_series.inputs_fp16_bits = Some(fp16_inputs.clone());
    let fp16_series_report = session::run_compressed_series(&fp16_series).unwrap();
    assert!(
        fp16_series_report.series.ok,
        "{:?} {:?}",
        fp16_series_report.series.error, fp16_series_report.series.failure
    );
    let f64_series_report = session::run_compressed_series(&f64_series).unwrap();
    assert!(
        f64_series_report.series.ok,
        "{:?} {:?}",
        f64_series_report.series.error, f64_series_report.series.failure
    );
    assert_eq!(fp16_series_report.series.exact_batches, 2);
    assert_eq!(f64_series_report.series.exact_batches, 2);
    assert!(fp16_series_report
        .series
        .traffic
        .iter()
        .all(|rank| { rank.traffic.loads_transferred == 1 && rank.traffic.execs == 2 }));
    let routed_rows: u64 = routing::dispatch(&fp16_series.routing)
        .unwrap()
        .ranks
        .iter()
        .map(|rank| {
            rank.slots
                .iter()
                .map(|slot| slot.slot.token)
                .collect::<BTreeSet<_>>()
                .len() as u64
        })
        .sum();
    let series_body_saving =
        2 * routed_rows * HIDDEN * (std::mem::size_of::<f64>() - std::mem::size_of::<u16>()) as u64;
    let series_request_saving =
        f64_series_report.series.exec_request_bytes - fp16_series_report.series.exec_request_bytes;
    assert!(
        series_request_saving > 0
            && series_request_saving.abs_diff(series_body_saving) <= 1024,
        "cross-node typed series saved {series_request_saving} request bytes; {series_body_saving} expected from activation bodies"
    );
    eprintln!(
        "A169_INTERPC order={} remote_arches={remote_arches:?} rank_loads={rank_loads:?} local_f64_bytes={local_exec_bytes} mixed_f64_bytes={mixed_f64_request_bytes} mixed_fp16_bytes={mixed_fp16_request_bytes} measured_saving={measured_saving} expected_body_saving={expected_body_saving} pc1_rank9_f64_bytes={rank9_f64_bytes} pc1_rank9_fp16_bytes={rank9_fp16_bytes} pc1_rank9_measured_saving={rank9_measured_saving} pc1_rank9_expected_body_saving={rank9_expected_body_saving} pc1_rank9_tokens={rank9_tokens} mixed_f64_batch_us={mixed_f64_batch_us} mixed_fp16_batch_us={mixed_fp16_batch_us} outputs_match_independent_oracle=true post_load_bytes_equal_exec_requests=true ssh_framing_counted=false latency_scope=whole_batch_diagnostic",
        if fp16_first { "fp16-first" } else { "f64-first" }
    );
    eprintln!(
        "A170_INTERPC_SERIES fp16_exec_request_bytes={} f64_exec_request_bytes={} measured_saving={} expected_body_saving={} rank_traffic={:?} exact_outputs_match=true loaded_once_per_worker=true",
        fp16_series_report.series.exec_request_bytes,
        f64_series_report.series.exec_request_bytes,
        series_request_saving,
        series_body_saving,
        fp16_series_report.series.traffic
    );
}

#[test]
fn worker_latches_completion_batch_after_one_opt_in_until_shutdown() {
    let artifact = artifact(2);
    let p = routing_problem("worker-completion-batch-session-latch");
    let (mut first, body, expected_slots) = raw_exec_request(&artifact, &p);
    if let Request::ExecRaw {
        raw_completions,
        raw_completion_batch,
        ..
    } = &mut first
    {
        *raw_completions = true;
        *raw_completion_batch = true;
    }
    let mut second = first.clone();
    if let Request::ExecRaw {
        attempt,
        raw_completion_batch,
        ..
    } = &mut second
    {
        *attempt = 2;
        *raw_completion_batch = false;
    }

    let mut wire = raw_artifact_wire(&artifact, artifact.bytes().len() as u64, artifact.bytes());
    for request in [&first, &second] {
        wire.extend_from_slice(&serde_json::to_vec(request).unwrap());
        wire.push(b'\n');
        wire.extend_from_slice(&body);
    }
    wire.extend_from_slice(&serde_json::to_vec(&Request::Shutdown).unwrap());
    wire.push(b'\n');
    let mut output = Vec::new();
    session::serve(&mut std::io::Cursor::new(wire), &mut output).unwrap();
    let (replies, values) = raw_aware_replies(&output);
    let batches: Vec<_> = replies
        .iter()
        .filter_map(|reply| match reply {
            Reply::CompletionBatchRaw { attempt, .. } => Some(*attempt),
            _ => None,
        })
        .collect();
    assert_eq!(batches, [1, 2]);
    assert_eq!(values.len(), expected_slots * 2);
    assert_eq!(
        replies
            .iter()
            .filter(|reply| matches!(reply, Reply::CompletionRaw { .. }))
            .count(),
        0,
        "the second raw EXEC must retain batch replies without repeating the opt-in"
    );
}

#[test]
fn faults_over_compressed_sessions_fail_closed_without_partial_output() {
    let cases = [
        (Fault::RejectBeforeCompute, "WORKER_REJECTED", false),
        (Fault::DieAfterStart, "SESSION_CLOSED", true),
        (Fault::Duplicate, "INVALID_COMPLETION", true),
        (Fault::StaleEpoch, "INVALID_COMPLETION", true),
        (Fault::Missing, "BATCH_INCOMPLETE", true),
    ];
    for fp16 in [false, true] {
        for (i, (fault, code, started)) in cases.iter().enumerate() {
            let dir = temp_dir();
            let mode = if fp16 { "f16" } else { "f64" };
            let mut input = series(&format!("a53-fault-{i}-{mode}"), &dir, 5);
            input.fp16_completions = fp16;
            input.faults = vec![SeriesFault {
                batch: 2,
                rank: 9,
                fault: fault.clone(),
            }];
            let report = session::run_compressed_series(&input).unwrap().series;
            assert!(!report.ok);
            assert_eq!(report.batches_ok, 2);
            let failure = report.failure.unwrap();
            assert_eq!(failure.batch, 2);
            assert_eq!(failure.error.unwrap().code, *code);
            let receipt = failure.receipts.iter().find(|r| r.rank == 9).unwrap();
            assert_eq!(receipt.started, *started);
            assert!(report.recorded.is_empty());
        }
    }
}

#[test]
fn coordinator_refusals_precede_any_worker() {
    let dir = temp_dir();
    let ghost = Endpoint {
        rank: 2,
        argv: vec!["/nonexistent/vf-rank-session".into()],
    };
    let run = |edit: &dyn Fn(&mut CompressedSeriesInput)| {
        let mut input = series("a53-coordinator", &dir, 3);
        input.workers = vec![
            ghost.clone(),
            Endpoint {
                rank: 9,
                ..ghost.clone()
            },
        ];
        edit(&mut input);
        session::run_compressed_series(&input).unwrap_err()
    };
    let mismatch =
        "artifact differs from its rank's registry, order, identity, epoch or hidden size";
    assert_reason(
        &run(&|i| {
            i.routing
                .resident_order
                .insert(2, vec![3, 7])
                .map(|_| ())
                .unwrap()
        }),
        "INVALID_INPUT",
        mismatch,
    );
    assert_reason(
        &run(&|i| i.routing.stamp.layer += 1),
        "INVALID_INPUT",
        mismatch,
    );
    assert_reason(
        &run(&|i| {
            i.routing.stamp.epoch = EPOCH - 1;
            i.routing.registry.epoch = EPOCH - 1;
        }),
        "INVALID_INPUT",
        mismatch,
    );
    assert_reason(
        &run(&|i| i.artifacts[1] = i.artifacts[0].clone()),
        "INVALID_INPUT",
        mismatch,
    );
    assert_reason(
        &run(&|i| {
            i.artifacts.pop();
        }),
        "INVALID_INPUT",
        "a rank holding experts has no artifact",
    );
    assert_reason(
        &run(&|i| i.artifacts.clear()),
        "INVALID_INPUT",
        "a compressed series needs 1..64 artifacts",
    );
    assert_reason(
        &run(&|i| {
            i.workers.pop();
        }),
        "INVALID_INPUT",
        "compressed sessions need exactly one endpoint per artifact rank",
    );
    assert_reason(
        &run(&|i| i.record = vec![3]),
        "INVALID_INPUT",
        "a series records at most 16 of its own batches",
    );
    assert_reason(
        &run(&|i| i.record = (0..17).map(|_| 0).collect()),
        "INVALID_INPUT",
        "a series records at most 16 of its own batches",
    );
    let error = run(&|i| i.artifacts[0] = dir.join("absent.safetensors"));
    assert_eq!(error.code, "ARTIFACT_IO");
    // A53-R6: activation rows are the series' own input; they are refused before the
    // oracle decodes, any worker starts or any weight is sent, as the dense series does.
    let rows = "one finite activation row of the hidden size per routed token";
    assert_reason(&run(&|i| i.inputs[1].truncate(5)), "INVALID_INPUT", rows);
    assert_reason(&run(&|i| i.inputs[0].push(0.25)), "INVALID_INPUT", rows);
    assert_reason(
        &run(&|i| {
            i.inputs.pop();
        }),
        "INVALID_INPUT",
        rows,
    );
    assert_reason(
        &run(&|i| i.inputs[2][7] = f64::INFINITY),
        "INVALID_INPUT",
        rows,
    );
    // Each identity the artifact must share with its rank's routing entry.
    assert_reason(
        &run(&|i| i.routing.stamp.model_revision = "rev-other".into()),
        "INVALID_INPUT",
        mismatch,
    );
    assert_reason(&run(&|i| i.routing.hidden = 64), "INVALID_INPUT", mismatch);
    assert_reason(
        &run(&|i| {
            for r in &mut i.routing.registry.copies {
                if r.rank == 2 {
                    r.local = if r.local == 7 { 3 } else { 7 };
                }
            }
        }),
        "INVALID_INPUT",
        mismatch,
    );
    assert_reason(
        &run(&|i| i.artifacts = vec![i.artifacts[0].clone(); 65]),
        "INVALID_INPUT",
        "a compressed series needs 1..64 artifacts",
    );
    assert_reason(
        &run(&|i| i.workers[1].rank = 11),
        "INVALID_INPUT",
        "compressed sessions need exactly one endpoint per artifact rank",
    );
    assert_reason(
        &run(&|i| i.workers.push(i.workers[1].clone())),
        "INVALID_INPUT",
        "compressed sessions need exactly one endpoint per artifact rank",
    );
    let big = dir.join("big.safetensors");
    std::fs::File::create(&big)
        .unwrap()
        .set_len(16 * 1024 * 1024 + 1)
        .unwrap();
    assert_reason(
        &run(&|i| i.artifacts[0] = big.clone()),
        "MESSAGE_TOO_LARGE",
        "artifact exceeds the raw LOAD_ARTIFACT body limit",
    );
}

#[test]
fn an_unbound_epoch_is_refused_before_anything_is_sent() {
    let p = routing_problem("a53-unbound");
    let (mut sessions, artifacts) = bound(&p);
    let mut next = p.clone();
    next.stamp.epoch += 1;
    next.registry.epoch += 1;
    let x = activations(6, HIDDEN as usize, 4);
    let error = session::run_compressed_batch(
        &mut sessions,
        &artifacts,
        &next,
        &x,
        &BTreeMap::new(),
        false,
        T,
    )
    .unwrap_err();
    assert_reason(
        &error,
        "STALE_EXECUTION",
        "session does not hold this batch's resident artifact and epoch; bind first",
    );
    assert_reason(
        &session::run_compressed_batch(
            &mut sessions,
            &artifacts,
            &p,
            &x[1..],
            &BTreeMap::new(),
            false,
            T,
        )
        .unwrap_err(),
        "INVALID_INPUT",
        "one finite activation row of the hidden size per routed token",
    );
    for s in sessions {
        assert_eq!(s.traffic.execs, 0);
        assert_eq!(s.close(T).exit_code, Some(0));
    }
}

#[test]
fn a_real_size_expert_cannot_cross_as_dense_weights_but_does_as_its_artifact() {
    let stack = real_stack();
    let dense = stack.dense_expert(0).unwrap();
    let model = moe::Model {
        hidden: 2048,
        intermediate: 512,
        experts: vec![dense],
    };
    let mut s = Session::connect(&endpoint(0), T).unwrap();
    let error = s
        .load(&model, stack.model_revision(), 0, 1, &[900], T)
        .unwrap_err();
    assert_eq!(error.code, "MESSAGE_TOO_LARGE");
    let a = CompressedArtifact::from_bytes(resident_artifact::encode(&stack).unwrap()).unwrap();
    let receipt = s.load_compressed(&a, 1, T).unwrap();
    assert!(receipt.transferred);
    assert!(
        receipt.request_bytes < 2_500_000,
        "{}",
        receipt.request_bytes
    );
    let close = s.close(T);
    assert_eq!(close.exit_code, Some(0));
}

#[test]
fn cli_runs_a_compressed_series_end_to_end() {
    let dir = temp_dir();
    let input = series("a53-cli", &dir, 4);
    let path = dir.join("input.json");
    std::fs::write(&path, serde_json::to_vec(&input).unwrap()).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_vf"))
        .arg("session-run-compressed")
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["ok"], true);
    assert_eq!(v["result"]["exact_batches"], 4);
    assert_eq!(v["result"]["artifacts"][0]["rank"], 2);
}

#[test]
fn strata_f32_exact_f64_reaches_local_sessions_and_oracle() {
    let p = routing_problem("r368-strata-f64-local");
    let ids: Vec<i32> = p.routes.iter().flatten().map(|c| c.expert as i32).collect();
    let weights: Vec<f32> = p.routes.iter().flatten().map(|c| c.weight as f32).collect();
    let mut values: Vec<f32> = activations(6, HIDDEN as usize, 5)
        .into_iter()
        .flatten()
        .map(|x| x as f32)
        .collect();
    values[0] = 1.0001_f32;
    values[1] = f32::from_bits(1);
    values[2] = -0.0;
    assert!(voltafabric::strata_ingress::admit_pool_batch(
        &p, &p.stamp, LAYER, 6, 2, &values, &ids, &weights
    )
    .is_err());
    let admitted = voltafabric::strata_ingress::admit_pool_batch_f64(
        &p, &p.stamp, LAYER, 6, 2, &values, &ids, &weights,
    )
    .unwrap();
    let (mut sessions, artifacts) = bound(&admitted.routing);
    let report = session::run_compressed_batch(
        &mut sessions,
        &artifacts,
        &admitted.routing,
        &admitted.inputs,
        &BTreeMap::new(),
        false,
        T,
    )
    .unwrap();
    assert!(report.ok, "{:?}", report.error);
    let expected = session::DenseOracle::new(&artifacts)
        .unwrap()
        .output(&admitted.routing, &admitted.inputs)
        .unwrap();
    assert_eq!(
        bits(&report.output.unwrap().concat()),
        bits(&expected.concat())
    );
    assert!(sessions.iter().all(|s| s.traffic.execs == 1));
    for s in sessions {
        assert_eq!(s.close(T).exit_code, Some(0));
    }
}

#[test]
fn admitted_strata_style_batch_reaches_real_local_sessions_and_dense_oracle() {
    let dir = temp_dir();
    let mut input = series("r367-strata-ingress-cpu", &dir, 1);
    let ids: Vec<i32> = input
        .routing
        .routes
        .iter()
        .flatten()
        .map(|c| c.expert as i32)
        .collect();
    let weights: Vec<f32> = input
        .routing
        .routes
        .iter()
        .flatten()
        .map(|c| c.weight as f32)
        .collect();
    let activations: Vec<f32> = input.inputs.iter().flatten().map(|&x| x as f32).collect();
    let original_inputs = input.inputs.clone();
    let admitted = voltafabric::strata_ingress::admit_pool_batch(
        &input.routing,
        &input.routing.stamp,
        LAYER,
        input.inputs.len(),
        2,
        &activations,
        &ids,
        &weights,
    )
    .unwrap();
    input.routing = admitted.routing;
    input.inputs.clear();
    input.inputs_fp16_bits = Some(admitted.inputs_fp16_bits);
    input.record = vec![0];
    let report = session::run_compressed_series(&input).unwrap();
    assert!(report.series.ok, "{:?}", report.series.error);
    assert_eq!(
        (report.series.batches_ok, report.series.exact_batches),
        (1, 1)
    );
    assert_eq!(report.series.traffic.len(), 2);
    assert!(report
        .series
        .traffic
        .iter()
        .all(|rank| rank.traffic.loads_transferred == 1 && rank.traffic.execs == 1));
    let artifacts: Vec<_> = [2, 9].map(artifact).into();
    let oracle = session::DenseOracle::new(&artifacts).unwrap();
    let expected = oracle.output(&input.routing, &original_inputs).unwrap();
    assert_eq!(
        bits(&report.series.recorded[0].output.concat()),
        bits(&expected.concat())
    );
}

#[test]
fn exports_of_one_stack_in_separate_processes_are_one_byte_string() {
    // Until A53 the SafeTensors header listed vf_schema and vf_info in HashMap order,
    // random per process: the same expert exported to two byte strings and SHA-256s.
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut request: serde_json::Value = serde_json::from_slice(
        &std::fs::read(fixture.join("fixtures/resident_import_qwen_expert0.json")).unwrap(),
    )
    .unwrap();
    request["path"] = fixture
        .join(request["path"].as_str().unwrap())
        .to_string_lossy()
        .into_owned()
        .into();
    let dir = temp_dir();
    let mut outputs = std::collections::BTreeSet::new();
    for i in 0..8 {
        let output = dir.join(format!("export-{i}.safetensors"));
        let input = dir.join(format!("input-{i}.json"));
        let body = serde_json::json!({
            "source": {"kind": "checkpoint", "request": request},
            "output": output,
        });
        std::fs::write(&input, serde_json::to_vec(&body).unwrap()).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_vf"))
            .arg("resident-export")
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        outputs.insert(std::fs::read(&output).unwrap());
    }
    assert_eq!(outputs.len(), 1);
    let bytes = outputs.into_iter().next().unwrap();
    assert_eq!(bytes, resident_artifact::encode(&real_stack()).unwrap());
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&bytes[8..8 + n]).unwrap();
    assert!(
        header.starts_with(r#"{"__metadata__":{"vf_info":"#),
        "{header}"
    );
    assert_eq!(n % 8, 0);
}

#[test]
fn artifact_identity_is_the_sha256_of_the_file_bytes() {
    // The SHA-256 a session announces must be the one any tool computes on the file:
    // checked against the system sha256sum, not against this crate's own digest.
    let dir = temp_dir();
    let path = dir.join("rank2.safetensors");
    std::fs::write(&path, resident_artifact::encode(&stack(2)).unwrap()).unwrap();
    let out = Command::new("sha256sum").arg(&path).output().unwrap();
    let system = String::from_utf8(out.stdout).unwrap()[..64].to_string();
    let a = CompressedArtifact::from_bytes(std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(a.sha256(), system);
}

#[test]
fn each_artifact_shape_and_identity_check_refuses_on_its_own() {
    let good = resident_artifact::encode(&stack(2)).unwrap();
    let identity = "resident identity or metadata exceeds bounded contract";
    let order = "duplicate expert or physical order differing from the experts' local IDs";
    let shape = "shape is not an aligned, bounded, uniform SwiGLU stack";
    let shapes: Vec<(serde_json::Value, &str)> = vec![
        (serde_json::json!([3, 128, 128, 128, 64]), "count"),
        (serde_json::json!([2, 0, 128, 128, 0]), "no gate rows"),
        (serde_json::json!([2, 128, 0, 0, 64]), "no hidden"),
        (
            serde_json::json!([2, 64, 128, 128, 32]),
            "gate rows not /64",
        ),
        (serde_json::json!([2, 128, 96, 96, 64]), "hidden not /64"),
        (
            serde_json::json!([2, 192, 128, 128, 64]),
            "gate+up rows not 2x",
        ),
        (
            serde_json::json!([2, 128, 128, 192, 64]),
            "down rows not hidden",
        ),
        (
            // The imported-resident ceiling is derived from the 128 MiB payload cap;
            // this logical matrix exceeds that resident bound without allocating it.
            serde_json::json!([2, 32768, 16384, 16384, 16384]),
            "over the plane bound",
        ),
    ];
    for (value, name) in shapes {
        let bytes = rebuild(&good, |_, m| edit_info(m, |i| i["shape"] = value.clone()));
        assert_reason(
            &resident_artifact::import(&bytes).unwrap_err(),
            "ARTIFACT_INVALID",
            shape,
        );
        let _ = name;
    }
    let cases: Vec<(Vec<u8>, &str)> = vec![
        (
            rebuild(&good, |_, m| {
                edit_info(m, |i| i["model_revision"] = "r".repeat(257).into())
            }),
            identity,
        ),
        (
            // 1025 experts with tile-factor tensors of matching leading dimension.
            rebuild(&good, |t, m| {
                for name in ["gscales13", "gscales2"] {
                    let g = t.get_mut(name).unwrap();
                    g.1 = vec![1025, g.1[1]];
                    g.2 = 0.125_f32.to_le_bytes().repeat(1025 * g.1[1]);
                }
                edit_info(m, |i| {
                    i["physical_experts"] = (0..1025)
                        .map(|k| serde_json::json!({"id": k, "local": 5000 + k}))
                        .collect();
                    i["resident_order"] = (0..1025).map(|k| 5000 + k).collect();
                })
            }),
            identity,
        ),
        (
            rebuild(&good, |_, m| {
                edit_info(m, |i| i["resident_order"] = serde_json::json!([7]))
            }),
            order,
        ),
        (
            rebuild(&good, |t, _| {
                let c = t.get_mut("codes2").unwrap();
                c.2.truncate(c.2.len() - 2);
                c.1 = vec![c.2.len()];
            }),
            "buffer lengths or payload size differ from the declared shape",
        ),
        (
            rebuild(&good, |t, _| {
                t.remove("codes2");
            }),
            "a QPN2 resident tensor is missing",
        ),
        (
            rebuild(&good, |t, _| {
                let g = t.get_mut("gscales13").unwrap();
                g.1 = vec![g.1[0], g.1[1], 1];
            }),
            "tile factor tensors are F32 [experts, tiles]",
        ),
        (
            rebuild(&good, |t, _| {
                let g = t.get_mut("gscales13").unwrap();
                g.1 = vec![1, g.1[0] * g.1[1]];
            }),
            "tile factor tensors are F32 [experts, tiles]",
        ),
    ];
    for (bytes, message) in cases {
        let error = resident_artifact::import(&bytes).unwrap_err();
        assert_reason(&error, "ARTIFACT_INVALID", message);
    }
}

#[test]
fn an_empty_rank_artifact_is_refused_by_a_session_of_that_rank() {
    let mut empty = input(2);
    empty.rank = 11;
    empty.experts.clear();
    empty.resident_order.clear();
    let bytes = resident_artifact::encode(&resident::pack(&empty).unwrap()).unwrap();
    let mut s = Session::connect(&endpoint(11), T).unwrap();
    s.send(&Request::LoadArtifact {
        artifact_sha256: sha256::hex(&bytes),
        model_revision: REVISION.into(),
        layer: LAYER,
        epoch: EPOCH,
        resident_order: vec![],
        artifact: base64::encode(&bytes),
    })
    .unwrap();
    assert_reason(
        &refused(s.reply(T).unwrap()),
        "INVALID_INPUT",
        "artifact rank, revision, layer or physical order differs from this LOAD",
    );
    assert_eq!(s.close(T).exit_code, Some(0));
}

fn wide_artifact() -> CompressedArtifact {
    let mut wide = input(9);
    for e in &mut wide.experts {
        e.gate = plane(128, HIDDEN, 5, 0.125);
        e.up = plane(128, HIDDEN, 6, 0.125);
        e.down = plane(HIDDEN, 128, 7, 0.125);
    }
    CompressedArtifact::from_bytes(
        resident_artifact::encode(&resident::pack(&wide).unwrap()).unwrap(),
    )
    .unwrap()
}

#[test]
fn compressed_batch_refusals_through_the_public_api() {
    let p = routing_problem("a53-batch-refusals");
    let x = activations(6, HIDDEN as usize, 8);
    let (mut sessions, artifacts) = bound(&p);
    let none = BTreeMap::new();
    let mut nan = x.clone();
    nan[2][5] = f64::NAN;
    assert_reason(
        &session::run_compressed_batch(&mut sessions, &artifacts, &p, &nan, &none, false, T)
            .unwrap_err(),
        "INVALID_INPUT",
        "one finite activation row of the hidden size per routed token",
    );
    let stray = BTreeMap::from([(11, Fault::None)]);
    assert_reason(
        &session::run_compressed_batch(&mut sessions, &artifacts, &p, &x, &stray, false, T)
            .unwrap_err(),
        "INVALID_INPUT",
        "fault refers to a nonparticipating rank",
    );
    assert_reason(
        &session::run_compressed_batch(&mut sessions[..1], &artifacts, &p, &x, &none, false, T)
            .unwrap_err(),
        "INVALID_INPUT",
        "no session serves a participating rank",
    );
    let mut extra = vec![Session::connect(&endpoint(11), T).unwrap()];
    assert_reason(
        &session::bind_compressed(&mut extra, &artifacts, &p, T).unwrap_err(),
        "INVALID_INPUT",
        "endpoint rank has no artifact",
    );
    assert_reason(
        &session::DenseOracle::new(&[]).err().unwrap(),
        "INVALID_INPUT",
        "no artifact",
    );
    assert_reason(
        &session::DenseOracle::new(&artifacts[..1])
            .unwrap()
            .output(&p, &x)
            .unwrap_err(),
        "INVALID_INPUT",
        "oracle lacks a routed expert",
    );
    assert_reason(
        &session::DenseOracle::new(&[artifacts[0].clone(), wide_artifact()])
            .err()
            .unwrap(),
        "INVALID_INPUT",
        "artifacts disagree on hidden or intermediate size",
    );
    // A batch that fails after STARTED leaves its session unusable: never reused.
    let crash = BTreeMap::from([(9, Fault::DieAfterStart)]);
    let report =
        session::run_compressed_batch(&mut sessions, &artifacts, &p, &x, &crash, false, T).unwrap();
    assert_eq!(report.error.unwrap().code, "SESSION_CLOSED");
    let mut next = p.clone();
    next.stamp.batch += 1;
    assert_reason(
        &session::run_compressed_batch(&mut sessions, &artifacts, &next, &x, &none, false, T)
            .unwrap_err(),
        "SESSION_UNUSABLE",
        "session must be reopened after a failure",
    );
    for s in extra.into_iter().chain(sessions) {
        s.close(T);
    }
}

// ------------------------------------------------- A53-R2 review corrections

#[test]
fn dense_oracle_refuses_malformed_activation_rows() {
    // Review defect 4: output() indexed rows and passed them to moe::expert_value
    // unchecked (too few rows or a longer row panicked; a shorter row was truncated).
    let p = routing_problem("a53r2-oracle");
    let artifacts: Vec<_> = [2, 9].map(artifact).into();
    let oracle = session::DenseOracle::new(&artifacts).unwrap();
    let x = activations(6, HIDDEN as usize, 3);
    let message = "one finite activation row of the hidden size per routed token";
    let mut long = x.clone();
    long[1].push(0.5);
    let mut short = x.clone();
    short[4].pop();
    let mut nan = x.clone();
    nan[0][7] = f64::NAN;
    for (name, rows) in [
        ("too few rows", x[1..].to_vec()),
        ("too many rows", [x.clone(), vec![x[0].clone()]].concat()),
        ("long row", long),
        ("short row", short),
        ("nan", nan),
    ] {
        let outcome = std::panic::catch_unwind(|| oracle.output(&p, &rows));
        match outcome {
            Ok(Err(error)) => assert_reason(&error, "INVALID_INPUT", message),
            Ok(Ok(_)) => panic!("{name}: malformed rows accepted"),
            Err(_) => panic!("{name}: oracle panicked"),
        }
    }
    assert!(oracle.output(&p, &x).is_ok());
}

/// Bytes of the largest artifact whose canonical base64 leaves 64 KiB of a 16 MiB
/// Legacy base64 LOAD_ARTIFACT frame limit: (16 MiB - 64 KiB) / 4 * 3.
const MAX_ARTIFACT: u64 = 12_533_760;
/// Raw LOAD_ARTIFACT body limit leaves 64 KiB for its separately framed metadata.
const MAX_RAW_ARTIFACT: u64 = 16_711_680;

#[test]
fn artifact_files_are_bounded_by_the_raw_body_before_import() {
    // The raw body has its own declared-length bound; reject before parsing, connecting,
    // or allocating the artifact's resident stack.
    let dir = temp_dir();
    let run = |size: u64| {
        let path = dir.join(format!("sized-{size}.safetensors"));
        std::fs::File::create(&path).unwrap().set_len(size).unwrap();
        let mut input = series("a53r2-size", &dir, 3);
        input.workers = vec![
            Endpoint {
                rank: 2,
                argv: vec!["/nonexistent/vf-rank-session".into()],
            },
            Endpoint {
                rank: 9,
                argv: vec!["/nonexistent/vf-rank-session".into()],
            },
        ];
        input.artifacts[0] = path;
        session::run_compressed_series(&input).unwrap_err()
    };
    assert_reason(
        &run(MAX_RAW_ARTIFACT + 1),
        "MESSAGE_TOO_LARGE",
        "artifact exceeds the raw LOAD_ARTIFACT body limit",
    );
    // Exactly at the raw bound passes size admission and is judged as an artifact.
    let at_bound = run(MAX_RAW_ARTIFACT);
    assert_eq!(at_bound.code, "ARTIFACT_INVALID", "{at_bound}");
    assert!(
        at_bound.message.starts_with("not a SafeTensors file"),
        "{at_bound}"
    );
}

#[test]
fn the_largest_admitted_artifact_fits_one_load_artifact_frame() {
    // Worst case of every other LOAD_ARTIFACT field: 64-hex digest, 256-byte revision,
    // 1024 ten-digit local IDs, u32/u64 maxima. The frame adds one newline.
    let frame = |artifact_bytes: usize| {
        let request = Request::LoadArtifact {
            artifact_sha256: "f".repeat(64),
            model_revision: "r".repeat(256),
            layer: u32::MAX,
            epoch: u64::MAX,
            resident_order: (0..1024).map(|k| u32::MAX - k).collect(),
            artifact: base64::encode(&vec![0xA5; artifact_bytes]),
        };
        serde_json::to_vec(&request).unwrap().len() + 1
    };
    assert_eq!(session::MAX_ARTIFACT_BYTES as u64, MAX_ARTIFACT);
    let cap = voltafabric::process::MAX_MESSAGE_BYTES;
    let largest = frame(MAX_ARTIFACT as usize);
    assert!(largest <= cap, "{largest} bytes");
    // The 64 KiB margin is needed: the raw 16 MiB bound admitted ~4.4 MB more.
    assert!(frame(MAX_ARTIFACT as usize + 64 * 1024) > cap);
}

/// HELLO of an A52 worker: the dense capability only.
const HELLO_A52: &str = r#"{"reply":"hello","hello":{"protocol":1,"rank":2,"arch":"x","os":"y","endian":"little","build":"0","capabilities":["cpu_f64_dense_moe_v1"],"max_message_bytes":16777216,"pid":1}}"#;

#[test]
fn a_worker_without_the_compressed_capability_receives_no_artifact() {
    // Review defect 5: connect() only required the dense capability, so an A52 worker
    // was sent the ~MB LOAD_ARTIFACT frame and failed with a protocol error.
    let script = format!(
        r#"read l; printf '%s\n' '{HELLO_A52}'; read l
case "$l" in
  *'"op":"shutdown"'*) printf '%s\n' '{{"reply":"bye","execs":0}}' ;;
  *) printf '%s\n' '{{"reply":"refused","error":{{"code":"FAKE_UNKNOWN_REQUEST","message":"A52 worker"}}}}' ;;
esac"#
    );
    let mut s = Session::connect(
        &Endpoint {
            rank: 2,
            argv: vec!["sh".into(), "-c".into(), script],
        },
        T,
    )
    .unwrap();
    let sent = s.traffic.bytes_sent;
    let error = s.load_compressed(&artifact(2), EPOCH, T).unwrap_err();
    assert_reason(
        &error,
        "CAPABILITY_MISSING",
        "worker does not advertise cpu_qpn2_resident_v1; nothing was sent",
    );
    assert_eq!(s.traffic.bytes_sent, sent);
    assert!(s.is_usable());
    let close = s.close(T);
    assert_eq!((close.bye_execs, close.exit_code), (Some(0), Some(0)));
}

#[test]
fn legacy_qpn2_capability_uses_base64_without_changing_resident_semantics() {
    // Emulate an A53-v1 worker's HELLO: it supports QPN2 but not the A54 raw-body
    // capability. The existing base64 form must still load the same immutable artifact.
    let mut session = Session::connect(&endpoint(2), T).unwrap();
    session
        .hello
        .capabilities
        .retain(|capability| capability != session::CAPABILITY_QPN2_RAW);
    let artifact = artifact(2);
    let receipt = session.load_compressed(&artifact, EPOCH, T).unwrap();
    assert!(receipt.transferred);
    assert_eq!(receipt.artifact_sha256, artifact.sha256());
    assert_eq!(session.traffic.weight_bytes_sent, receipt.request_bytes);
    assert!(receipt.request_bytes > artifact.bytes().len() as u64 * 4 / 3);
    assert_eq!(session.close(T).exit_code, Some(0));
}

#[test]
fn legacy_exec_capability_uses_json_fallback_and_matches_the_oracle() {
    let artifacts: Vec<_> = [2, 9].map(artifact).into();
    let p = routing_problem("a55-legacy-exec");
    let x = activations(p.routes.len(), HIDDEN as usize, 5);
    let mut sessions: Vec<_> = [2, 9]
        .into_iter()
        .map(|rank| Session::connect(&endpoint(rank), T).unwrap())
        .collect();
    for s in &mut sessions {
        s.hello
            .capabilities
            .retain(|capability| capability != session::CAPABILITY_QPN2_RAW);
        s.hello
            .capabilities
            .retain(|capability| capability != session::CAPABILITY_EXEC_F64_RAW);
    }
    session::bind_compressed(&mut sessions, &artifacts, &p, T).unwrap();
    let report = session::run_compressed_batch(
        &mut sessions,
        &artifacts,
        &p,
        &x,
        &BTreeMap::new(),
        false,
        T,
    )
    .unwrap();
    assert!(report.ok, "{:?}", report.error);
    let expected = session::DenseOracle::new(&artifacts)
        .unwrap()
        .output(&p, &x)
        .unwrap();
    assert_eq!(
        bits(&report.output.unwrap().concat()),
        bits(&expected.concat())
    );
    assert!(sessions.iter().all(|s| s.traffic.execs == 1));
    for s in sessions {
        assert_eq!(s.close(T).exit_code, Some(0));
    }
}

// ------------------------------------------------- A53-R5 oracle memory

#[test]
fn the_series_oracle_decodes_only_the_experts_its_routing_dispatches_to() {
    // DenseOracle::new decodes every held expert to dense f64 (24 MiB per real expert),
    // whatever the routing uses; the series oracle follows the routing instead.
    let artifacts: Vec<_> = [2, 9].map(artifact).into();
    let p = routing_problem("a53r5-oracle");
    let x = activations(p.routes.len(), HIDDEN as usize, 5);
    let all = session::DenseOracle::new(&artifacts).unwrap();
    let held: usize = artifacts
        .iter()
        .map(|a| a.stack().physical_experts().len())
        .sum();
    assert_eq!(all.experts(), held);

    let full = session::DenseOracle::for_routing(&artifacts, &p).unwrap();
    let routed: std::collections::BTreeSet<_> = routing::dispatch(&p)
        .unwrap()
        .slots
        .iter()
        .map(|s| (s.rank, s.expert))
        .collect();
    assert_eq!(full.experts(), routed.len());
    assert_eq!(
        bits(&full.output(&p, &x).unwrap().concat()),
        bits(&all.output(&p, &x).unwrap().concat())
    );

    let mut one = p.clone();
    let first = one.routes[0][0].clone();
    for route in one.routes.iter_mut() {
        *route = vec![first.clone()];
    }
    let narrow = session::DenseOracle::for_routing(&artifacts, &one).unwrap();
    assert_eq!(narrow.experts(), 1);
    assert_eq!(
        bits(&narrow.output(&one, &x).unwrap().concat()),
        bits(&all.output(&one, &x).unwrap().concat())
    );
    // A batch routed elsewhere is refused, never computed without its expert.
    assert_reason(
        &narrow.output(&p, &x).unwrap_err(),
        "INVALID_INPUT",
        "oracle lacks a routed expert",
    );
    // The series builds this oracle: with one routed expert it decodes one, and every
    // batch still matches it exactly.
    let dir = temp_dir();
    let mut input = series("a53r5-series", &dir, 3);
    input.routing = one.clone();
    input.routing.stamp.run_id = "a53r5-series".into();
    let report = session::run_compressed_series(&input).unwrap();
    assert!(report.series.ok, "{:?}", report.series.error);
    assert_eq!((report.series.exact_batches, report.oracle_experts), (3, 1));
    // The routing is validated before anything is decoded.
    let mut bad = one.clone();
    bad.routes[0][0].weight = f64::NAN;
    assert_reason(
        &session::DenseOracle::for_routing(&artifacts, &bad)
            .err()
            .unwrap(),
        "INVALID_INPUT",
        "unknown/duplicate expert or invalid routing weight",
    );
}

// ------------------------------------------------- A53-R8 mixed fleets

#[test]
fn a_mixed_fleet_is_refused_before_any_rank_is_sent_its_artifact() {
    // Each session checked the compressed capability only when its own turn to load
    // came: ranks listed before an A52 worker were sent their whole artifact first.
    let dir = temp_dir();
    let seen = dir.join("rank2.stdin");
    let recorded = Endpoint {
        rank: 2,
        argv: vec![
            "sh".into(),
            "-c".into(),
            r#"exec tee "$0" | "$1""#.into(),
            seen.display().to_string(),
            env!("CARGO_BIN_EXE_vf-rank-session").into(),
        ],
    };
    let hello = HELLO_A52.replace(r#""rank":2"#, r#""rank":9"#);
    let script = format!(
        r#"read l; printf '%s\n' '{hello}'; read l
case "$l" in
  *'"op":"shutdown"'*) printf '%s\n' '{{"reply":"bye","execs":0}}' ;;
  *) printf '%s\n' '{{"reply":"refused","error":{{"code":"FAKE_UNKNOWN_REQUEST","message":"A52 worker"}}}}' ;;
esac"#
    );
    let a52 = Endpoint {
        rank: 9,
        argv: vec!["sh".into(), "-c".into(), script],
    };
    let artifacts: Vec<_> = [2, 9].map(artifact).into();
    let p = routing_problem("a53r8-mixed");
    let missing = "worker does not advertise cpu_qpn2_resident_v1; nothing was sent";
    let ops = || -> Vec<String> {
        std::fs::read(&seen)
            .unwrap()
            .split(|&b| b == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| {
                serde_json::from_slice::<serde_json::Value>(line).unwrap()["op"].to_string()
            })
            .collect()
    };
    let error = session::prepare_compressed(&[recorded.clone(), a52.clone()], &artifacts, &p, T)
        .err()
        .unwrap();
    assert_reason(&error, "CAPABILITY_MISSING", missing);
    // The rank 2 worker answered HELLO, so its request reached the file first.
    assert_eq!(ops(), [r#""hello""#]);
    // The same through the public bind, as a series rebind or a caller would use it.
    std::fs::remove_file(&seen).unwrap();
    let mut sessions = vec![
        Session::connect(&recorded, T).unwrap(),
        Session::connect(&a52, T).unwrap(),
    ];
    let error = session::bind_compressed(&mut sessions, &artifacts, &p, T).unwrap_err();
    assert_reason(&error, "CAPABILITY_MISSING", missing);
    assert!(sessions
        .iter()
        .all(|s| s.is_usable() && s.artifact().is_none()));
    for s in sessions {
        s.close(T);
    }
    assert_eq!(ops(), [r#""hello""#, r#""shutdown""#]);
}

// ------------------------------------------------- A53-R10 rank work budget

#[test]
fn a_rank_over_its_work_budget_is_refused_before_any_rank_is_sent_work() {
    // The worker refuses a compressed EXEC above slots x 3 x hidden x intermediate =
    // 1e8. The coordinator did not check it: other ranks computed the batch, and a series
    // sent every artifact, before that rank refused. One real expert: 3 145 728 per slot,
    // so 31 slots fit and 32 do not.
    let stack = real_stack();
    let a = CompressedArtifact::from_bytes(resident_artifact::encode(&stack).unwrap()).unwrap();
    let problem = |tokens: usize| routing::RoutingProblem {
        hidden: 2048,
        experts: vec![0],
        routes: vec![
            vec![Choice {
                expert: 0,
                weight: 1.0
            }];
            tokens
        ],
        registry: Registry {
            epoch: 1,
            copies: vec![Replica {
                expert: 0,
                rank: 0,
                local: 900,
            }],
        },
        stamp: Stamp {
            run_id: format!("a53r10-{tokens}"),
            model_revision: stack.model_revision().into(),
            plan_id: "plan-a53r10".into(),
            layer: 0,
            batch: 0,
            epoch: 1,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(0, vec![900])]),
    };
    let budget = "compressed EXEC needs 1 slot or more within the rank CPU work budget";
    let none = BTreeMap::new();
    let batch = |tokens: usize| {
        session::run_compressed_batch(
            &mut [],
            std::slice::from_ref(&a),
            &problem(tokens),
            &activations(tokens, 2048, 7),
            &none,
            false,
            T,
        )
        .unwrap_err()
    };
    assert_reason(&batch(32), "INVALID_INPUT", budget);
    assert_reason(
        &batch(31),
        "INVALID_INPUT",
        "no session serves a participating rank",
    );
    // A rank that holds nothing needs no artifact and does no work: never over budget.
    let mut with_empty = problem(31);
    with_empty.resident_order.insert(5, vec![]);
    assert_reason(
        &session::run_compressed_batch(
            &mut [],
            std::slice::from_ref(&a),
            &with_empty,
            &activations(31, 2048, 7),
            &none,
            false,
            T,
        )
        .unwrap_err(),
        "INVALID_INPUT",
        "no session serves a participating rank",
    );
    // The series refuses before reading further, connecting, or sending its artifact.
    let dir = temp_dir();
    let path = dir.join("real.safetensors");
    std::fs::write(&path, a.bytes()).unwrap();
    let series = |tokens: usize| {
        session::run_compressed_series(&CompressedSeriesInput {
            artifacts: vec![path.clone()],
            routing: problem(tokens),
            inputs: activations(tokens, 2048, 7),
            inputs_fp16_bits: None,
            workers: vec![Endpoint {
                rank: 0,
                argv: vec!["/nonexistent/vf-rank-session".into()],
            }],
            batches: 2,
            timeout_ms: 20_000,
            epoch_every: 0,
            warmup: 0,
            tolerance_abs: 0.0,
            fp16_completions: false,
            faults: vec![],
            record: vec![],
        })
        .unwrap_err()
    };
    assert_reason(&series(32), "INVALID_INPUT", budget);
    assert_eq!(series(31).code, "WORKER_IO");
}

// ------------------------------------------------- A53-R11 one artifact per rank

#[test]
fn the_dense_oracle_takes_one_artifact_per_rank() {
    // The oracle keyed its experts by (rank, expert): a second artifact of the same rank
    // silently replaced the first, so its answer depended on the artifacts' order.
    let mut other = input(2);
    for (k, e) in other.experts.iter_mut().enumerate() {
        e.gate = plane(INTER, HIDDEN, 700 + k as u64, 0.125);
    }
    let other = CompressedArtifact::from_bytes(
        resident_artifact::encode(&resident::pack(&other).unwrap()).unwrap(),
    )
    .unwrap();
    let p = routing_problem("a53r11-oracle");
    let one = "an oracle holds one artifact per rank";
    for artifacts in [
        vec![artifact(2), other.clone(), artifact(9)],
        vec![other.clone(), artifact(2), artifact(9)],
        vec![artifact(9), artifact(2), artifact(2)],
    ] {
        assert_reason(
            &session::DenseOracle::new(&artifacts).err().unwrap(),
            "INVALID_INPUT",
            one,
        );
        assert_reason(
            &session::DenseOracle::for_routing(&artifacts, &p)
                .err()
                .unwrap(),
            "INVALID_INPUT",
            one,
        );
    }
    // One artifact per rank, in either order, is one oracle.
    let x = activations(p.routes.len(), HIDDEN as usize, 11);
    let ab = session::DenseOracle::new(&[artifact(2), artifact(9)]).unwrap();
    let ba = session::DenseOracle::new(&[artifact(9), artifact(2)]).unwrap();
    assert_eq!(
        bits(&ab.output(&p, &x).unwrap().concat()),
        bits(&ba.output(&p, &x).unwrap().concat())
    );
}

// ------------------------------------------------- A53-R12 fault targets

#[test]
fn a_compressed_series_fault_on_an_absent_rank_or_out_of_bounds_is_refused_before_any_worker() {
    // Refused only at the faulted batch before R12, after every earlier batch ran.
    let dir = temp_dir();
    let ghost = |rank| Endpoint {
        rank,
        argv: vec!["/nonexistent/vf-rank-session".into()],
    };
    let run = |fault: SeriesFault| {
        let mut input = series("a53r12-faults", &dir, 6);
        input.workers = vec![ghost(2), ghost(9)];
        input.faults = vec![fault];
        session::run_compressed_series(&input).unwrap_err()
    };
    let at = |rank, fault| SeriesFault {
        batch: 5,
        rank,
        fault,
    };
    let bounds = "fault injection outside bounded CPU test contract";
    assert_reason(
        &run(at(7, Fault::Missing)),
        "INVALID_INPUT",
        "fault refers to a nonparticipating rank",
    );
    // Rank 11 is in the routing but holds nothing: it takes no part either.
    let mut input = series("a53r12-faults", &dir, 6);
    input.workers = vec![ghost(2), ghost(9)];
    input.routing.resident_order.insert(11, vec![]);
    input.faults = vec![at(11, Fault::Missing)];
    assert_reason(
        &session::run_compressed_series(&input).unwrap_err(),
        "INVALID_INPUT",
        "fault refers to a nonparticipating rank",
    );
    assert_reason(
        &run(at(
            9,
            Fault::Delay {
                milliseconds: 60_001,
            },
        )),
        "INVALID_INPUT",
        bounds,
    );
    assert_reason(
        &run(at(
            2,
            Fault::StderrFlood {
                bytes: 16 * 1024 * 1024 + 1,
            },
        )),
        "INVALID_INPUT",
        bounds,
    );
    assert_eq!(
        run(at(
            9,
            Fault::Delay {
                milliseconds: 60_000
            }
        ))
        .code,
        "WORKER_IO"
    );
}

// ------------------------------------------------- A53-R13 unusable sessions

#[test]
fn an_unusable_session_sends_nothing_and_takes_no_late_reply() {
    // A missed deadline makes a session unusable: its stream position is no longer
    // known. LOAD, LOAD_ARTIFACT and EVICT still wrote to it and took the next frame as
    // their answer, which can be the late reply to the request that timed out.
    let hello = HELLO_A52.replace(
        r#"["cpu_f64_dense_moe_v1"]"#,
        r#"["cpu_f64_dense_moe_v1","cpu_qpn2_resident_v1"]"#,
    );
    // The first LOAD_ARTIFACT is answered late; any later request is refused.
    let script = format!(
        r#"read l; printf '%s\n' '{hello}'; read l
sha=$(printf '%s' "$l" | sed -n 's/.*"artifact_sha256":"\([0-9a-f]*\)".*/\1/p')
sleep 1
printf '{{"reply":"loaded","artifact_sha256":"%s","epoch":{EPOCH},"experts":2,"transferred":true}}\n' "$sha"
while read l; do
  case "$l" in
    *'"op":"shutdown"'*) printf '%s\n' '{{"reply":"bye","execs":0}}'; exit 0 ;;
    *) printf '%s\n' '{{"reply":"refused","error":{{"code":"FAKE_LATE","message":"second request"}}}}' ;;
  esac
done"#
    );
    let mut s = Session::connect(
        &Endpoint {
            rank: 2,
            argv: vec!["sh".into(), "-c".into(), script],
        },
        T,
    )
    .unwrap();
    let a = artifact(2);
    let error = s
        .load_compressed(&a, EPOCH, Duration::from_millis(100))
        .unwrap_err();
    assert_eq!(error.code, "SESSION_TIMEOUT", "{error}");
    assert!(!s.is_usable());
    // Let the late reply arrive, then retry the same LOAD on the unusable session.
    std::thread::sleep(Duration::from_millis(1500));
    let sent = s.traffic.bytes_sent;
    let unusable = "session must be reopened after a failure";
    assert_reason(
        &s.load_compressed(&a, EPOCH, T).unwrap_err(),
        "SESSION_UNUSABLE",
        unusable,
    );
    let model = moe::Model {
        hidden: HIDDEN as usize,
        intermediate: INTER as usize,
        experts: vec![stack(2).dense_expert(0).unwrap()],
    };
    assert_reason(
        &s.load(&model, REVISION, LAYER, EPOCH, &[7], T).unwrap_err(),
        "SESSION_UNUSABLE",
        unusable,
    );
    assert_eq!(s.traffic.bytes_sent, sent);
    assert!(s.artifact().is_none());
    s.close(T);
}

// ------------------------------------------------- A53-R14 bounded close

#[test]
fn close_returns_by_its_deadline_when_a_leftover_process_holds_the_pipes() {
    // The wrapper runs the worker as a child, then sleeps with the session's stdout and
    // stderr still open. close() killed the wrapper at its deadline but then joined the
    // pipe readers without one, so it waited for the sleep to end.
    let wrapped = Endpoint {
        rank: 2,
        argv: vec![
            "sh".into(),
            "-c".into(),
            r#""$0"; sleep 15"#.into(),
            env!("CARGO_BIN_EXE_vf-rank-session").into(),
        ],
    };
    let s = Session::connect(&wrapped, T).unwrap();
    let started = std::time::Instant::now();
    let receipt = s.close(Duration::from_secs(1));
    let elapsed = started.elapsed();
    assert!(elapsed < Duration::from_secs(5), "close took {elapsed:?}");
    // The worker said BYE; the wrapper did not exit and was killed.
    assert_eq!((receipt.bye_execs, receipt.killed), (Some(0), true));
    assert_reason(
        receipt.error.as_ref().unwrap(),
        "SESSION_PIPES",
        "a process left by the worker kept its output open past the close deadline",
    );
    // A worker that exits on its own still closes cleanly, with nothing to report.
    let receipt = Session::connect(&endpoint(2), T).unwrap().close(T);
    assert_eq!(
        (receipt.exit_code, receipt.killed, receipt.error.is_none()),
        (Some(0), false, true)
    );
}

// ------------------------------------------------- A53-R16 regular artifact files

#[test]
fn an_artifact_path_that_is_not_a_regular_file_is_refused_without_blocking() {
    // read_artifact opened the path with a plain open(): a FIFO without a writer blocked
    // the coordinator forever. The checkpoint importer's policy (A52) applies: a regular
    // file, opened non-blocking and checked again once open.
    let dir = temp_dir();
    let fifo = dir.join("artifact.fifo");
    assert!(Command::new("mkfifo")
        .arg(&fifo)
        .status()
        .unwrap()
        .success());
    let run = |path: PathBuf| {
        let mut input = series("a53r16-fifo", &dir, 2);
        input.artifacts[0] = path;
        let (sender, outcome) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = sender.send(session::run_compressed_series(&input).map(|_| ()));
        });
        outcome.recv_timeout(Duration::from_secs(5))
    };
    let regular = "artifact path must be a regular file";
    match run(fifo.clone()) {
        Ok(Err(error)) => assert_reason(&error, "INVALID_INPUT", regular),
        Ok(Ok(())) => panic!("a FIFO artifact was accepted"),
        Err(_) => {
            // Release the reader blocked in open() before failing.
            let _ = std::fs::OpenOptions::new().write(true).open(&fifo);
            panic!("the coordinator blocked on a FIFO artifact path");
        }
    }
    match run(dir.clone()) {
        Ok(Err(error)) => assert_reason(&error, "INVALID_INPUT", regular),
        other => panic!("a directory artifact path: {other:?}"),
    }
    // The type is checked before any open: a FIFO nobody may read is refused as not
    // regular, not reported as an open failure.
    let locked = dir.join("locked.fifo");
    assert!(Command::new("mkfifo")
        .args(["-m", "000"])
        .arg(&locked)
        .status()
        .unwrap()
        .success());
    match run(locked) {
        Ok(Err(error)) => assert_reason(&error, "INVALID_INPUT", regular),
        other => panic!("an unreadable FIFO artifact path: {other:?}"),
    }
    // A regular file that cannot be opened is an I/O refusal, as before R16.
    let unreadable = dir.join("unreadable.safetensors");
    std::fs::write(&unreadable, b"x").unwrap();
    assert!(Command::new("chmod")
        .args(["000"])
        .arg(&unreadable)
        .status()
        .unwrap()
        .success());
    assert_eq!(run(unreadable).unwrap().unwrap_err().code, "ARTIFACT_IO");
    // A missing file is still an I/O refusal.
    assert_eq!(
        run(dir.join("absent.safetensors"))
            .unwrap()
            .unwrap_err()
            .code,
        "ARTIFACT_IO"
    );
}

// ------------------------------------------------- A53-R17 replicated experts

/// An edit of rank 9's copy of expert 10.
type Change = fn(&mut ResidentExpert);

/// Expert 10 on rank 2 (local 7) and replicated on rank 9 (local 6), where `change` edits
/// rank 9's copy. Returns the artifact files and the routing.
fn replicated(
    dir: &Path,
    tag: &str,
    change: Change,
    owner: u32,
) -> (Vec<PathBuf>, routing::RoutingProblem) {
    let mut reg = registry(EPOCH);
    reg.copies.push(Replica {
        expert: 10,
        rank: 9,
        local: 6,
    });
    let held = |rank: u32| -> Vec<(u32, u32)> {
        if rank == 2 {
            vec![(10, 7), (99, 3)]
        } else {
            vec![(25, 0), (42, 5), (10, 6)]
        }
    };
    let mut paths = Vec::new();
    for rank in [2, 9] {
        let mut experts: Vec<_> = held(rank)
            .into_iter()
            .map(|(id, l)| expert(id, l))
            .collect();
        if rank == 9 {
            change(&mut experts[2]);
        }
        let order: Vec<u32> = if rank == 2 { vec![7, 3] } else { vec![5, 0, 6] };
        let stack = resident::pack(&ResidentInput {
            model_revision: REVISION.into(),
            layer: LAYER,
            rank,
            registry: reg.clone(),
            resident_order: order,
            experts,
        })
        .unwrap();
        let path = dir.join(format!("replica-{tag}-{owner}-rank{rank}.safetensors"));
        std::fs::write(&path, resident_artifact::encode(&stack).unwrap()).unwrap();
        paths.push(path);
    }
    let mut p = routing_problem(&format!("a53r17-{tag}-{owner}"));
    p.registry = reg;
    p.resident_order = BTreeMap::from([(2, vec![7, 3]), (9, vec![5, 0, 6])]);
    p.preferred_owners = BTreeMap::from([(10, owner)]);
    (paths, p)
}

#[test]
fn replicas_of_one_expert_must_hold_the_same_weights() {
    // Each rank's artifact carries its own copy of a replicated expert, and nothing
    // compared them: with diverging copies a series ran exact either way, its output
    // depending on which rank owns the expert.
    let dir = temp_dir();
    let run = |tag: &str, change: Change, owner: u32| {
        let (artifacts, routing) = replicated(&dir, tag, change, owner);
        session::run_compressed_series(&CompressedSeriesInput {
            artifacts,
            routing,
            inputs: activations(6, HIDDEN as usize, 5),
            inputs_fp16_bits: None,
            workers: vec![endpoint(2), endpoint(9)],
            batches: 2,
            timeout_ms: 20_000,
            epoch_every: 0,
            warmup: 0,
            tolerance_abs: 0.0,
            fp16_completions: false,
            faults: vec![],
            record: vec![1],
        })
    };
    let differ = "replicas of one expert differ between artifacts";
    for owner in [2, 9] {
        let gate: Change = |e| e.gate = plane(INTER, HIDDEN, 900, 0.125);
        assert_reason(
            &run("gate", gate, owner).expect_err("diverging replicas must be refused"),
            "INVALID_INPUT",
            differ,
        );
    }
    // Codes, scale bytes and global factors each define the weights.
    let changes: [(&str, Change); 3] = [
        ("code", |e| e.gate.codes[0] ^= 0x11),
        ("scale", |e| {
            e.down.scales[0] = if e.down.scales[0] == 0x30 { 0x31 } else { 0x30 }
        }),
        ("factor", |e| e.up.global_dequant = 0.125),
    ];
    for (tag, change) in changes {
        assert_reason(
            &run(tag, change, 2).expect_err(tag),
            "INVALID_INPUT",
            differ,
        );
    }
    // Identical replicas: one model, whichever rank serves the expert.
    let outputs: Vec<_> = [2, 9]
        .map(|owner| {
            let report = run("same", |_| {}, owner).unwrap();
            assert!(report.series.ok, "{:?}", report.series.error);
            assert_eq!(report.series.exact_batches, 2);
            bits(&report.series.recorded[0].output.concat())
        })
        .into();
    assert_eq!(outputs[0], outputs[1]);
}

// ------------------------------------------------- A53-R21 close receipts in the verdict

#[test]
fn a_close_receipt_error_fails_the_series() {
    // The rank 2 worker answers BYE and exits 0, but leaves a process holding its output
    // pipes: close() records SESSION_PIPES. The series only looked at exit codes and BYE,
    // so it reported ok with an error in its own close receipts.
    let dir = temp_dir();
    let mut input = series("a53r21-close", &dir, 2);
    input.timeout_ms = 2_000;
    input.workers[0] = Endpoint {
        rank: 2,
        argv: vec![
            "sh".into(),
            "-c".into(),
            r#"sleep 15 & exec "$0""#.into(),
            env!("CARGO_BIN_EXE_vf-rank-session").into(),
        ],
    };
    let s = session::run_compressed_series(&input).unwrap().series;
    assert_eq!((s.batches_ok, s.exact_batches), (2, 2));
    let rank2 = s.close.iter().find(|c| c.rank == 2).unwrap();
    assert_eq!((rank2.exit_code, rank2.bye_execs), (Some(0), Some(2)));
    assert_eq!(rank2.error.as_ref().unwrap().code, "SESSION_PIPES");
    assert!(!s.ok);
    assert_reason(
        s.error.as_ref().unwrap(),
        "SESSION_CLOSE",
        "a session worker did not confirm SHUTDOWN and exit cleanly",
    );
}

// ------------------------------------------------- A53-R22 collection bound in the series

/// 64 ranks holding one small expert each (expert e on rank e, local 0), and a routing of
/// 4096 tokens, each to 33 distinct experts: every rank stays within its work budget
/// (2112 slots), but a batch holds 4096 x 33 x 128 = 17 301 504 pending results.
fn wide_fleet(dir: &Path, run_id: &str) -> (Vec<PathBuf>, routing::RoutingProblem, Vec<Vec<f64>>) {
    let reg = Registry {
        epoch: EPOCH,
        copies: (0..64)
            .map(|e| Replica {
                expert: e,
                rank: e,
                local: 0,
            })
            .collect(),
    };
    let paths = (0..64)
        .map(|rank| {
            let stack = resident::pack(&ResidentInput {
                model_revision: REVISION.into(),
                layer: LAYER,
                rank,
                registry: reg.clone(),
                resident_order: vec![0],
                experts: vec![expert(rank, 0)],
            })
            .unwrap();
            let path = dir.join(format!("wide-rank{rank}.safetensors"));
            std::fs::write(&path, resident_artifact::encode(&stack).unwrap()).unwrap();
            path
        })
        .collect();
    let p = routing::RoutingProblem {
        hidden: HIDDEN as usize,
        experts: (0..64).collect(),
        routes: (0..4096u32)
            .map(|t| {
                (0..33u32)
                    .map(|j| Choice {
                        expert: (t + j) % 64,
                        weight: 1.0 / 33.0,
                    })
                    .collect()
            })
            .collect(),
        registry: reg,
        stamp: Stamp {
            run_id: run_id.into(),
            model_revision: REVISION.into(),
            plan_id: "plan-a53r22".into(),
            layer: LAYER,
            batch: 0,
            epoch: EPOCH,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: (0..64).map(|r| (r, vec![0])).collect(),
    };
    let inputs = activations(4096, HIDDEN as usize, 22);
    (paths, p, inputs)
}

#[test]
fn a_series_over_the_collection_bound_is_refused_before_any_worker() {
    // The collector's bound (pending results <= 16 777 216 f64) was checked only per
    // batch, so a series exceeding it was refused at batch 0, after the oracle was built
    // and every artifact sent. Missing workers make any late refusal show as WORKER_IO.
    let dir = temp_dir();
    let (artifacts, routing, inputs) = wide_fleet(&dir, "a53r22-wide");
    let error = session::run_compressed_series(&CompressedSeriesInput {
        artifacts,
        routing,
        inputs,
        inputs_fp16_bits: None,
        workers: (0..64)
            .map(|rank| Endpoint {
                rank,
                argv: vec!["/nonexistent/vf-rank-session".into()],
            })
            .collect(),
        batches: 2,
        timeout_ms: 20_000,
        epoch_every: 0,
        warmup: 0,
        tolerance_abs: 0.0,
        fp16_completions: false,
        faults: vec![],
        record: vec![],
    })
    .unwrap_err();
    assert_reason(
        &error,
        "COLLECTION_TOO_LARGE",
        "pending slot results exceed 16777216 f64 values",
    );
}

#[test]
fn a_series_whose_derived_activations_overflow_is_refused_before_any_worker() {
    // A53-R24: batch b scales each base activation by 1 + ((7b + 3t + i) % 11) / 64. A
    // finite base value the scaling takes past f64::MAX was refused only at the batch
    // deriving it, after every artifact was sent. Missing workers make any late refusal
    // show as WORKER_IO.
    let derived = "series activations must stay finite in every derived batch";
    let dir = temp_dir();
    let run = |batches: u64, column: usize, x: f64| {
        let mut input = series("a53r24-far", &dir, batches);
        input.inputs[0][column] = x;
        input.workers = [2, 9]
            .map(|rank| Endpoint {
                rank,
                argv: vec!["/nonexistent/vf-rank-session".into()],
            })
            .to_vec();
        session::run_compressed_series(&input).unwrap_err()
    };
    // Row 0, column 0: factor 1 at batch 0, 1 + 7/64 at batch 1.
    assert_reason(&run(2, 0, 1.7e308), "INVALID_INPUT", derived);
    assert_eq!(run(1, 0, 1.7e308).code, "WORKER_IO");
    // Row 0, column 6 first meets the largest factor, 1 + 10/64, at batch 10 (factors
    // up to 1 + 9/64 before): the scaling repeats every 11 batches and each is checked.
    let edge = f64::MAX / 1.15;
    assert_eq!(run(10, 6, edge).code, "WORKER_IO");
    assert_reason(&run(11, 6, edge), "INVALID_INPUT", derived);
}

#[test]
fn an_out_of_bounds_fault_is_refused_before_any_compressed_exec() {
    // A53-R27: as for `run_batch`, a fault outside its bounds reached the worker of its
    // rank after the batch identity was claimed and every rank was sent its work.
    let p = routing_problem("a53r27-compressed");
    let (mut sessions, artifacts) = bound(&p);
    let x = activations(6, HIDDEN as usize, 4);
    let over = [
        (
            9,
            Fault::Delay {
                milliseconds: 60_001,
            },
        ),
        (
            2,
            Fault::StderrFlood {
                bytes: 16 * 1024 * 1024 + 1,
            },
        ),
    ];
    for (rank, fault) in over {
        let faults = BTreeMap::from([(rank, fault)]);
        let error =
            session::run_compressed_batch(&mut sessions, &artifacts, &p, &x, &faults, false, T)
                .unwrap_err();
        assert_reason(
            &error,
            "INVALID_INPUT",
            "fault injection outside bounded CPU test contract",
        );
    }
    assert!(sessions.iter().all(|s| s.traffic.execs == 0));
    // Nothing was attempted: the same identity still runs.
    let report = session::run_compressed_batch(
        &mut sessions,
        &artifacts,
        &p,
        &x,
        &BTreeMap::new(),
        false,
        T,
    )
    .unwrap();
    assert!(report.ok, "{:?}", report.error);
    for s in sessions {
        assert_eq!(s.close(T).exit_code, Some(0));
    }
}

#[test]
#[ignore = "requires explicit two-host CPU worker configuration; see cross-node qualification receipt"]
fn a53_crossnode_compressed_session_executes_persistent_150_batch_series() {
    let dir = temp_dir();
    let run_id = std::env::var("VF_A176_RUN_ID_PREFIX")
        .unwrap_or_else(|_| "a53-crossnode-compressed-20261017".into());
    assert!(
        !run_id.is_empty()
            && run_id.len() <= 64
            && run_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
        "cross-node run ID must be 1-64 ASCII letters, digits, or hyphens"
    );
    let mut input = series(&run_id, &dir, 150);
    input.warmup = 10;
    input.record = vec![0, 49, 50, 99, 149];
    let mode = std::env::var("VF_A53_MODE").expect("test mode is required");
    let (rank2_argv, rank2_arch) = if mode == "crossnode" {
        let argv: Vec<String> = serde_json::from_str(
            &std::env::var("VF_A53_RANK2_ARGV_JSON")
                .expect("cross-node rank 2 argv JSON is required"),
        )
        .expect("cross-node rank 2 argv must be a JSON string array");
        assert!(!argv.is_empty(), "cross-node rank 2 argv must not be empty");
        let arch = std::env::var("VF_A53_RANK2_ARCH")
            .expect("expected cross-node rank 2 architecture is required");
        (argv, arch)
    } else {
        assert_eq!(mode, "local");
        (endpoint(2).argv, std::env::consts::ARCH.to_owned())
    };
    input.workers = vec![
        Endpoint {
            rank: 2,
            argv: rank2_argv,
        },
        endpoint(9),
    ];
    let input_path = dir.join("crossnode-input.json");
    std::fs::write(&input_path, serde_json::to_vec(&input).unwrap()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_vf"))
        .arg("session-run-compressed")
        .arg(&input_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let envelope: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(envelope["ok"], true);
    let report: CompressedSeriesReport =
        serde_json::from_value(envelope["result"].clone()).unwrap();
    assert!(report.series.ok, "{:?}", report.series.error);
    assert_eq!(report.series.batches_requested, 150);
    assert_eq!(report.series.batches_ok, 150);
    assert_eq!(report.series.exact_batches, 150);
    assert_eq!(report.series.max_abs_diff, 0.0);
    assert_eq!(report.series.batch_latency.as_ref().unwrap().n, 140);
    assert_eq!(report.series.traffic.len(), 2);
    for worker in &report.series.traffic {
        assert_eq!(worker.traffic.loads_transferred, 1, "rank {}", worker.rank);
        assert_eq!(worker.traffic.execs, 150, "rank {}", worker.rank);
        assert!(worker.traffic.weight_bytes_sent > 0, "rank {}", worker.rank);
    }
    assert_eq!(report.series.close.len(), 2);
    assert!(report
        .series
        .close
        .iter()
        .all(|c| { !c.killed && c.exit_code == Some(0) && c.error.is_none() }));
    let arch = |rank| {
        report
            .series
            .hellos
            .iter()
            .find(|h| h.rank == rank)
            .unwrap()
            .arch
            .as_str()
    };
    assert_eq!(arch(9), std::env::consts::ARCH);
    assert_eq!(arch(2), rank2_arch);
    let receipt = std::env::var("VF_A53_CROSSNODE_RECEIPT").expect("receipt path is required");
    if let Some(parent) = Path::new(&receipt).parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(receipt, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
}

#[test]
#[ignore = "requires explicit two-host CPU worker configuration; see cross-node qualification receipt"]
fn a53_crossnode_compressed_failures_do_not_publish_or_replay() {
    assert_eq!(
        std::env::var("VF_A53_MODE").expect("test mode is required"),
        "crossnode",
        "this test must exercise the configured remote worker"
    );
    let argv: Vec<String> = serde_json::from_str(
        &std::env::var("VF_A53_RANK2_ARGV_JSON").expect("cross-node rank 2 argv JSON is required"),
    )
    .expect("cross-node rank 2 argv must be a JSON string array");
    assert!(!argv.is_empty(), "cross-node rank 2 argv must not be empty");
    let expected_arch = std::env::var("VF_A53_RANK2_ARCH")
        .expect("expected cross-node rank 2 architecture is required");
    let cases = [
        (
            Fault::RejectBeforeCompute,
            "WORKER_REJECTED",
            false,
            ReceiptState::NotStarted,
        ),
        (
            Fault::DieAfterStart,
            "SESSION_CLOSED",
            true,
            ReceiptState::Unknown,
        ),
        (
            Fault::DieAfterCompute,
            "SESSION_CLOSED",
            true,
            ReceiptState::Unknown,
        ),
        (
            Fault::Duplicate,
            "INVALID_COMPLETION",
            true,
            ReceiptState::Unknown,
        ),
        (
            Fault::StaleEpoch,
            "INVALID_COMPLETION",
            true,
            ReceiptState::Unknown,
        ),
        (
            Fault::Missing,
            "BATCH_INCOMPLETE",
            true,
            ReceiptState::Unknown,
        ),
    ];
    let mut receipts = Vec::with_capacity(cases.len());
    let receipt_path = std::env::var("VF_A53_CROSSNODE_FAILURE_RECEIPT")
        .expect("cross-node failure receipt path is required");
    if let Some(parent) = Path::new(&receipt_path).parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    for (index, (fault, expected_error, started, state)) in cases.into_iter().enumerate() {
        // Keep every artifact fixture inside the canonical Naos work tree. This
        // cross-node test is an operator-run qualification, not a reason to leave
        // temporary data under the host's shared /tmp directory.
        let temp = project_temp_dir();
        let dir = temp.0.as_path();
        let run_prefix = std::env::var("VF_A176_RUN_ID_PREFIX")
            .unwrap_or_else(|_| "a53-crossnode-failure".into());
        assert!(
            !run_prefix.is_empty()
                && run_prefix.len() <= 48
                && run_prefix
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'),
            "A176 run ID prefix must be 1-48 ASCII letters, digits, or hyphens"
        );
        let mut input = series(&format!("{run_prefix}-{index}"), dir, 5);
        input.record = (0..5).collect();
        input.workers = vec![
            Endpoint {
                rank: 2,
                argv: argv.clone(),
            },
            endpoint(9),
        ];
        input.faults = vec![SeriesFault {
            batch: 2,
            rank: 2,
            fault,
        }];
        let report = session::run_compressed_series(&input)
            .expect("fault injection is admitted and represented as a series receipt")
            .series;
        receipts.push(report);
        std::fs::write(&receipt_path, serde_json::to_vec_pretty(&receipts).unwrap()).unwrap();
        let report = receipts.last().unwrap();
        assert!(!report.ok, "case {index}: {report:?}");
        assert_eq!(report.batches_ok, 2, "case {index}");
        assert!(
            report.recorded.iter().map(|r| r.batch).collect::<Vec<_>>() == [0, 1],
            "case {index}: only batches before the failure may be published"
        );
        let failure = report.failure.as_ref().expect("case must fail");
        assert_eq!(failure.batch, 2, "case {index}");
        assert_eq!(
            failure.error.as_ref().unwrap().code,
            expected_error,
            "case {index}"
        );
        let receipt = failure
            .receipts
            .iter()
            .find(|r| r.rank == 2)
            .expect("the remote rank must have a terminal receipt");
        assert_eq!(receipt.started, started, "case {index}: {receipt:?}");
        assert_eq!(receipt.state, state, "case {index}: {receipt:?}");
        assert_eq!(report.traffic.len(), 2, "case {index}");
        for rank in [2, 9] {
            let traffic = report.traffic.iter().find(|t| t.rank == rank).unwrap();
            assert_eq!(
                traffic.traffic.loads_transferred, 1,
                "case {index}, rank {rank}"
            );
            assert_eq!(
                traffic.traffic.execs, 3,
                "case {index}, rank {rank}: no retry"
            );
        }
        assert_eq!(
            report.hellos.iter().find(|h| h.rank == 2).unwrap().arch,
            expected_arch,
            "case {index}"
        );
        assert_eq!(
            report.hellos.iter().find(|h| h.rank == 9).unwrap().arch,
            std::env::consts::ARCH,
            "local PC6 rank architecture, case {index}"
        );
    }
}

#[test]
#[ignore = "requires explicit two-host CPU worker configuration and the Qwen3.6 fixture"]
fn a53_crossnode_real_qwen36_checkpoint_artifact_executes_through_cli() {
    let dir = temp_dir();
    let mut input = real_qwen_checkpoint_series(&dir);
    let mode = std::env::var("VF_A53_MODE").expect("test mode is required");
    let (argv, expected_arch) = if mode == "crossnode" {
        let argv: Vec<String> = serde_json::from_str(
            &std::env::var("VF_A53_REAL_WORKER_ARGV_JSON")
                .expect("real checkpoint worker argv JSON is required"),
        )
        .expect("real checkpoint worker argv must be a JSON string array");
        assert!(
            !argv.is_empty(),
            "real checkpoint worker argv must not be empty"
        );
        let arch = std::env::var("VF_A53_REAL_WORKER_ARCH")
            .expect("expected real checkpoint worker architecture is required");
        (argv, arch)
    } else {
        assert_eq!(mode, "local");
        (endpoint(0).argv, std::env::consts::ARCH.to_owned())
    };
    input.workers[0].argv = argv;
    let input_path = dir.join("real-qwen36-input.json");
    std::fs::write(&input_path, serde_json::to_vec(&input).unwrap()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_vf"))
        .arg("session-run-compressed")
        .arg(&input_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let envelope: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(envelope["ok"], true);
    let report: CompressedSeriesReport =
        serde_json::from_value(envelope["result"].clone()).unwrap();
    assert_eq!(report.oracle_experts, 1);
    assert!(report.series.ok, "{:?}", report.series.error);
    assert_eq!(
        (
            report.series.batches_requested,
            report.series.batches_ok,
            report.series.exact_batches
        ),
        (3, 3, 3)
    );
    assert_eq!(report.series.max_abs_diff, 0.0);
    assert_eq!(
        report
            .series
            .recorded
            .iter()
            .map(|r| r.batch)
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
    assert_eq!(report.artifacts.len(), 1);
    assert_eq!(report.artifacts[0].rank, 0);
    assert!(report.artifacts[0].file_bytes > 1_000_000);
    assert_eq!(report.series.traffic.len(), 1);
    let traffic = &report.series.traffic[0];
    assert_eq!(traffic.rank, 0);
    assert_eq!(traffic.traffic.loads_transferred, 1);
    assert_eq!(traffic.traffic.rebinds, 1);
    assert_eq!(traffic.traffic.execs, 3);
    assert!(traffic.traffic.weight_bytes_sent >= report.artifacts[0].file_bytes);
    assert!(
        traffic.traffic.weight_bytes_sent <= report.artifacts[0].file_bytes + 1024,
        "raw resident load should add only bounded metadata: sent {}, artifact {}",
        traffic.traffic.weight_bytes_sent,
        report.artifacts[0].file_bytes
    );
    assert_eq!(report.series.hellos[0].arch, expected_arch);
    assert!(report
        .series
        .close
        .iter()
        .all(|c| !c.killed && c.exit_code == Some(0) && c.error.is_none()));
    let receipt_path =
        std::env::var("VF_A53_REAL_RECEIPT").expect("real checkpoint receipt path is required");
    if let Some(parent) = Path::new(&receipt_path).parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(receipt_path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
}

#[test]
fn a55_real_qwen36_exec_bytes_drop_at_least_25_percent() {
    let dir = temp_dir();
    let mut input = real_qwen_checkpoint_series(&dir);
    input.record.clear();
    let input_path = dir.join("a55-qwen36-input.json");
    std::fs::write(&input_path, serde_json::to_vec(&input).unwrap()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_vf"))
        .arg("session-run-compressed")
        .arg(&input_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let envelope: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let report: CompressedSeriesReport =
        serde_json::from_value(envelope["result"].clone()).unwrap();
    assert!(report.series.ok, "{:?}", report.series.error);
    assert_eq!(report.series.exact_batches, 3);
    assert_eq!(report.series.max_abs_diff, 0.0);
    assert_eq!(report.series.exec_requests, 3);
    // A54's independently executed JSON baseline was 489,019 application bytes for
    // these same three Qwen3.6 fixture batches. A55 must remove at least one quarter.
    let maximum_candidate_bytes = 489_019 * 3 / 4;
    assert!(
        report.series.exec_request_bytes <= maximum_candidate_bytes,
        "A55 target: <= {maximum_candidate_bytes} EXEC bytes (<=75% of A54 baseline 489019), observed {}",
        report.series.exec_request_bytes
    );
}

/// Split a session's output into replies, reading raw completion bodies as well.
fn raw_aware_replies(output: &[u8]) -> (Vec<Reply>, Vec<Vec<f64>>) {
    let mut replies = Vec::new();
    let mut raw_values = Vec::new();
    let mut rest = output;
    while !rest.is_empty() {
        let newline = rest.iter().position(|byte| *byte == b'\n').unwrap();
        let reply: Reply = serde_json::from_slice(&rest[..newline]).unwrap_or_else(|error| {
            panic!(
                "reply parse failed: {error}; header={:?}",
                &rest[..newline.min(80)]
            )
        });
        rest = &rest[newline + 1..];
        if let Reply::CompletionRaw { completion, .. } = &reply {
            let n = completion.values as usize * 8;
            raw_values.push(
                rest[..n]
                    .chunks_exact(8)
                    .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
            );
            rest = &rest[n..];
        } else if let Reply::CompletionBatchRaw { hidden, values, .. } = &reply {
            let n = *values as usize * 8;
            let row_bytes = *hidden as usize * 8;
            assert!(row_bytes > 0);
            assert_eq!(n % row_bytes, 0);
            raw_values.extend(rest[..n].chunks_exact(row_bytes).map(|row| {
                row.chunks_exact(8)
                    .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            }));
            rest = &rest[n..];
        }
        replies.push(reply);
    }
    (replies, raw_values)
}

fn serve_raw_exec(request: &Request, body: &[u8], artifact: &CompressedArtifact) -> Vec<u8> {
    let mut wire = raw_exec_wire(artifact, request, body);
    wire.extend_from_slice(&serde_json::to_vec(&Request::Shutdown).unwrap());
    wire.push(b'\n');
    let mut output = Vec::new();
    session::serve(&mut std::io::Cursor::new(wire), &mut output).unwrap();
    output
}

#[test]
fn a56_raw_completions_are_opt_in_bit_identical_and_smaller() {
    let artifact = artifact(2);
    let p = routing_problem("a56-raw-completions");
    let (legacy, body, expected_slots) = raw_exec_request(&artifact, &p);
    // Legacy: without the flag the header is the A55 header and completions stay JSON.
    let header = serde_json::to_value(&legacy).unwrap();
    assert!(header.get("raw_completions").is_none());
    let json_output = serve_raw_exec(&legacy, &body, &artifact);
    let (json_replies, none) = raw_aware_replies(&json_output);
    assert!(none.is_empty());
    let json_values: Vec<(usize, usize, Vec<f64>)> = json_replies
        .iter()
        .filter_map(|reply| match reply {
            Reply::Event { event } => match &event.event {
                process::EventKind::Completion { completion } => {
                    Some((completion.token, completion.slot, completion.value.clone()))
                }
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(json_values.len(), expected_slots);

    let mut raw = legacy.clone();
    if let Request::ExecRaw {
        raw_completions, ..
    } = &mut raw
    {
        *raw_completions = true;
    }
    let raw_output = serve_raw_exec(&raw, &body, &artifact);
    let (raw_replies, raw_values) = raw_aware_replies(&raw_output);
    assert!(!raw_replies.iter().any(|reply| matches!(
        reply,
        Reply::Event { event } if matches!(event.event, process::EventKind::Completion { .. })
    )));
    let mut raw_slots = Vec::new();
    for (reply, values) in raw_replies
        .iter()
        .filter(|reply| matches!(reply, Reply::CompletionRaw { .. }))
        .zip(&raw_values)
    {
        let Reply::CompletionRaw {
            stamp,
            rank,
            attempt,
            completion,
        } = reply
        else {
            unreachable!()
        };
        assert_eq!((rank, attempt), (&2, &1));
        assert_eq!(stamp, &completion.stamp);
        raw_slots.push((completion.token, completion.slot, values.clone()));
    }
    assert_eq!(raw_slots.len(), json_values.len());
    for ((jt, js, jv), (rt, rs, rv)) in json_values.iter().zip(&raw_slots) {
        assert_eq!((jt, js), (rt, rs));
        let jb: Vec<u64> = jv.iter().map(|x| x.to_bits()).collect();
        let rb: Vec<u64> = rv.iter().map(|x| x.to_bits()).collect();
        assert_eq!(jb, rb);
    }
    assert!(raw_replies.iter().any(|reply| matches!(
        reply,
        Reply::Event { event }
            if matches!(&event.event, process::EventKind::Finished { slots } if *slots == expected_slots)
    )));
    assert!(raw_output.len() < json_output.len());
}

#[test]
fn raw_completion_batch_flag_requires_raw_completion_mode() {
    let artifact = artifact(2);
    let p = routing_problem("batch-completion-requires-raw-mode");
    let (mut request, body, _) = raw_exec_request(&artifact, &p);
    if let Request::ExecRaw {
        raw_completion_batch,
        ..
    } = &mut request
    {
        *raw_completion_batch = true;
    }
    let output = serve_raw_exec(&request, &body, &artifact);
    let replies = replies(&output);
    let reason = replies
        .iter()
        .find_map(|reply| match reply {
            Reply::Refused { error } => Some(error),
            _ => None,
        })
        .unwrap();
    assert_reason(
        reason,
        "SESSION_PROTOCOL",
        "raw completion batching requires raw completion mode",
    );
    assert!(!replies.iter().any(|reply| matches!(
        reply,
        Reply::Event { event } if matches!(event.event, process::EventKind::Started)
    )));
}

#[test]
fn compressed_batch_completion_uses_one_payload_frame() {
    let artifact = artifact(2);
    let p = routing_problem("batch-completion-frames");
    let (request, body, expected_slots) = raw_exec_request(&artifact, &p);
    assert!(expected_slots > 1);
    let json_output = serve_raw_exec(&request, &body, &artifact);
    let (json_replies, _) = raw_aware_replies(&json_output);
    let json_rows: Vec<_> = json_replies
        .iter()
        .filter_map(|reply| match reply {
            Reply::Event { event } => match &event.event {
                process::EventKind::Completion { completion } => Some(completion.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(json_rows.len(), expected_slots);

    let mut per_slot = request.clone();
    if let Request::ExecRaw {
        raw_completions, ..
    } = &mut per_slot
    {
        *raw_completions = true;
    }
    let per_slot_output = serve_raw_exec(&per_slot, &body, &artifact);
    let (per_slot_replies, per_slot_values) = raw_aware_replies(&per_slot_output);
    assert_eq!(
        per_slot_replies
            .iter()
            .filter(|reply| matches!(reply, Reply::CompletionRaw { .. }))
            .count(),
        expected_slots
    );
    assert_eq!(per_slot_values.len(), expected_slots);

    let mut batch = request;
    if let Request::ExecRaw {
        raw_completions,
        raw_completion_batch,
        ..
    } = &mut batch
    {
        *raw_completions = true;
        *raw_completion_batch = true;
    }
    let output = serve_raw_exec(&batch, &body, &artifact);
    let (replies, raw_values) = raw_aware_replies(&output);
    let completion_frames = replies
        .iter()
        .filter(|reply| matches!(reply, Reply::CompletionBatchRaw { .. }))
        .count();
    assert_eq!(
        completion_frames, 1,
        "one rank EXEC should send all validated completion rows in one bounded payload frame"
    );
    assert_eq!(raw_values.len(), expected_slots);
    let (stamp, completion_stamp, rank, attempt, hidden, metadata) = replies
        .iter()
        .find_map(|reply| match reply {
            Reply::CompletionBatchRaw {
                stamp,
                completion_stamp,
                rank,
                attempt,
                hidden,
                completions,
                ..
            } => Some((stamp, completion_stamp, rank, attempt, hidden, completions)),
            _ => None,
        })
        .unwrap();
    assert_eq!((*rank, *attempt), (2, 1));
    assert_eq!(stamp, completion_stamp);
    assert_eq!(*hidden, HIDDEN);
    assert_eq!(metadata.len(), expected_slots);
    for ((json, meta), values) in json_rows.iter().zip(metadata).zip(raw_values) {
        assert_eq!(
            (json.token, json.slot, json.expert, json.rank, json.local),
            (meta.token, meta.slot, meta.expert, 2, meta.local)
        );
        assert_eq!(
            json.value.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            values.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    assert!(output.len() < per_slot_output.len());
}

/// Records the bytes between flushes: one entry per frame `session::serve` emits.
struct Frames(Vec<Vec<u8>>, Vec<u8>);
impl std::io::Write for Frames {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.1.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.push(std::mem::take(&mut self.1));
        Ok(())
    }
}

#[test]
fn the_session_binary_writes_each_binary_frame_whole_in_one_write() {
    use std::io::Write;
    use std::os::unix::net::UnixDatagram;
    let artifact = artifact(2);
    let p = routing_problem("a57-binary-stdout");
    let (mut request, body, _) = raw_exec_request(&artifact, &p);
    if let Request::ExecRaw {
        raw_completions, ..
    } = &mut request
    {
        *raw_completions = true;
    }
    let mut wire = raw_exec_wire(&artifact, &request, &body);
    wire.extend_from_slice(&serde_json::to_vec(&Request::Shutdown).unwrap());
    wire.push(b'\n');

    let mut expected = Frames(Vec::new(), Vec::new());
    session::serve(&mut std::io::Cursor::new(wire.clone()), &mut expected).unwrap();
    assert!(expected.1.is_empty());
    let mut expected = expected.0;
    let raw: Vec<&Vec<u8>> = expected
        .iter()
        .filter(|f| f.starts_with(br#"{"reply":"completion_raw""#))
        .collect();
    assert!(!raw.is_empty(), "the fixture must take the binary path");
    // The bodies carry LF, CR and NUL, and no binary frame ends at a line break.
    let bodies: Vec<u8> = raw
        .iter()
        .flat_map(|f| f[f.iter().position(|b| *b == b'\n').unwrap() + 1..].to_vec())
        .collect();
    for byte in [b'\n', b'\r', 0] {
        assert!(bodies.contains(&byte), "fixture lacks byte {byte:#04x}");
    }
    assert!(raw.iter().all(|f| f.last() != Some(&b'\n')));

    // Each write(2) on a datagram socket arrives as one datagram: the count is exact.
    let (ours, theirs) = UnixDatagram::pair().unwrap();
    ours.set_read_timeout(Some(Duration::from_millis(50)))
        .unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_vf-rank-session"))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::from(std::os::fd::OwnedFd::from(
            theirs,
        )))
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let feeder = std::thread::spawn(move || {
        stdin.write_all(&wire).unwrap();
    });
    let deadline = std::time::Instant::now() + T;
    let mut got = Vec::new();
    let mut buf = vec![0_u8; 1 << 20];
    let mut exited = None;
    loop {
        assert!(
            std::time::Instant::now() < deadline,
            "session did not finish"
        );
        match ours.recv(&mut buf) {
            Ok(n) => got.push(buf[..n].to_vec()),
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                if exited.is_some() {
                    break;
                }
                // Exit is checked before one more drain, so no datagram is left behind.
                exited = child.try_wait().unwrap();
            }
            Err(e) => panic!("{e}"),
        }
    }
    feeder.join().unwrap();
    let status = exited.unwrap();
    let mut stderr = String::new();
    std::io::Read::read_to_string(&mut child.stderr.take().unwrap(), &mut stderr).unwrap();
    assert!(status.success(), "{status} {stderr}");
    // HELLO names the serving process; every other byte is the in-process reply.
    let ours_pid = format!(r#""pid":{}}}}}"#, std::process::id());
    let child_pid = format!(r#""pid":{}}}}}"#, child.id());
    let hello = String::from_utf8(expected[0].clone()).unwrap();
    assert!(hello.ends_with(&format!("{ours_pid}\n")), "{hello}");
    expected[0] = hello.replace(&ours_pid, &child_pid).into_bytes();
    // BYE reports the serving process's peak RSS; the rest of it must agree.
    let bye = |f: &[u8]| {
        let mut v: serde_json::Value = serde_json::from_slice(f).unwrap();
        v.as_object_mut().unwrap().remove("peak_rss_kib").unwrap();
        v
    };
    let (Some(e), Some(g)) = (expected.last_mut(), got.last()) else {
        panic!("no BYE")
    };
    assert_eq!(bye(e), bye(g));
    assert_eq!(g.last(), Some(&b'\n'));
    e.clone_from(g);
    assert_eq!(got.concat(), expected.concat(), "reply bytes differ");
    let sizes = |v: &[Vec<u8>]| v.iter().map(Vec::len).collect::<Vec<_>>();
    assert_eq!(sizes(&got), sizes(&expected), "one write per frame");
    assert_eq!(got, expected);
    assert!(got.last().unwrap().starts_with(br#"{"reply":"bye""#));
}

#[test]
fn engram_rows_use_a_separate_immutable_resident_alongside_qpn2() {
    use voltafabric::engram::{EngramShard, TableIdentity};

    let mut session = Session::connect(&endpoint(2), T).unwrap();
    let weights = artifact(2);
    let weight_receipt = session.load_compressed(&weights, EPOCH, T).unwrap();
    let table = TableIdentity {
        model_revision: REVISION.into(),
        table_sha256: "11".repeat(32),
        tokenizer_map_sha256: "22".repeat(32),
        layer: LAYER,
        epoch: EPOCH,
        num_embeddings: 4,
    };
    let row2: Vec<u8> = (0..264).map(|n| (n ^ 0x5a) as u8).collect();
    let shard = EngramShard::new(table.clone(), 4, 2, row2.clone()).unwrap();
    // Python's independent hashlib/struct oracle computes this over the documented
    // little-endian identity prefix plus the exact 264 row bytes.
    assert_eq!(
        shard.sha256(),
        "875a859a891fe41d1aff9b1426ea37fa7a99f0ad2c61f5d234765761409229c1"
    );
    let first_load = session.load_engram_shard(&shard, T).unwrap();
    let second_load = session.load_engram_shard(&shard, T).unwrap();
    assert!(first_load.transferred);
    assert!(!second_load.transferred);
    assert_eq!(first_load.artifact_sha256, second_load.artifact_sha256);

    let stamp = Stamp {
        run_id: "engram-persistent".into(),
        model_revision: REVISION.into(),
        plan_id: "plan-a64".into(),
        layer: LAYER,
        batch: 0,
        epoch: EPOCH,
    };
    let owner = session
        .fetch_engram_rows(&stamp, &table, 4, &[2, 0, 2], T)
        .unwrap();
    assert_eq!(owner.sender_rank, 2);
    assert_eq!(owner.row_count, 2);
    assert_eq!(owner.payload, [row2.as_slice(), row2.as_slice()].concat());
    assert_eq!(
        session.traffic.loads_transferred, 1,
        "Engram must not evict or reload QPN2"
    );

    let sent_before_local_refusals = session.traffic.bytes_sent;
    let mut stale = stamp.clone();
    stale.epoch += 1;
    assert!(session
        .fetch_engram_rows(&stale, &table, 4, &[2], T)
        .is_err());
    assert!(session
        .fetch_engram_rows(&stamp, &table, 4, &[table.num_embeddings], T)
        .is_err());
    assert_eq!(session.traffic.bytes_sent, sent_before_local_refusals);
    assert!(session.is_usable());

    let empty = session
        .fetch_engram_rows(&stamp, &table, 4, &[], T)
        .unwrap();
    assert_eq!(empty.row_count, 0);
    assert!(empty.payload.is_empty());
    session
        .send(&Request::FetchEngramRows {
            stamp: stamp.clone(),
            table: table.clone(),
            world_size: 4,
            owner_rank: 2,
            row_ids: vec![2],
            attempt: 2,
        })
        .unwrap();
    assert_reason(
        &refused(session.reply(T).unwrap()),
        "STALE_ATTEMPT",
        "Engram row attempt is not newer than the prior lookup",
    );
    assert!(session.is_usable());
    assert_eq!(session.traffic.loads_transferred, 1);

    let p = routing_problem("a64-engram-co-resident");
    let dispatch = rank2(&p);
    let x = token_rows(&dispatch, &activations(p.routes.len(), HIDDEN as usize, 1));
    let pending = session.start_exec(&dispatch, 1, x, Fault::None).unwrap();
    let exec = session.finish_exec(pending, None, std::time::Instant::now() + T);
    assert_eq!(
        exec.state,
        process::ReceiptState::Completed,
        "{:?}",
        exec.error
    );
    assert_eq!(
        session.artifact(),
        Some((weight_receipt.artifact_sha256.as_str(), EPOCH))
    );
    let close = session.close(T);
    assert_eq!(close.exit_code, Some(0));
}

#[test]
fn engram_batch_dispatches_concurrently_and_reconstructs_duplicate_rows_in_order() {
    use voltafabric::engram::{EngramRowCache, EngramShard, TableIdentity};

    let table = TableIdentity {
        model_revision: REVISION.into(),
        table_sha256: "33".repeat(32),
        tokenizer_map_sha256: "44".repeat(32),
        layer: LAYER,
        epoch: EPOCH,
        num_embeddings: 4,
    };
    let rows: Vec<Vec<u8>> = (0..4)
        .map(|row| {
            (0..264)
                .map(|column| (row * 53 + column * 7) as u8)
                .collect()
        })
        .collect();
    // Global worker ranks need not be the dense owner indices in this Engram world.
    let worker_ranks = [2, 9];
    let mut sessions: Vec<_> = worker_ranks
        .into_iter()
        .map(|rank| Session::connect(&endpoint(rank), T).unwrap())
        .collect();
    for rank in 0..2 {
        let local = [rows[rank * 2].as_slice(), rows[rank * 2 + 1].as_slice()].concat();
        let shard = EngramShard::new(table.clone(), 2, rank as u32, local).unwrap();
        assert!(
            sessions[rank]
                .load_engram_shard_for_owner(&shard, rank as u32, T)
                .unwrap()
                .transferred
        );
    }
    let stamp = Stamp {
        run_id: "a64-owner-batch".into(),
        model_revision: REVISION.into(),
        plan_id: "plan-batch".into(),
        layer: LAYER,
        batch: 8,
        epoch: EPOCH,
    };
    let ids = [3, 0, 3, 1, 2];
    let actual =
        Session::fetch_engram_batch_for_owners(&mut sessions, &stamp, &table, &ids, T).unwrap();
    let expected: Vec<u8> = ids
        .iter()
        .flat_map(|&id| rows[id as usize].iter().copied())
        .collect();
    assert_eq!(actual, expected);
    let bytes_before_empty = sessions
        .iter()
        .map(|session| (session.traffic.bytes_sent, session.traffic.bytes_received))
        .collect::<Vec<_>>();
    assert!(
        Session::fetch_engram_batch_for_owners(&mut sessions, &stamp, &table, &[], T)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        sessions
            .iter()
            .map(|session| (session.traffic.bytes_sent, session.traffic.bytes_received))
            .collect::<Vec<_>>(),
        bytes_before_empty,
        "an empty batch should make no worker request"
    );
    assert!(sessions.iter().all(|s| s.is_usable()));
    assert!(sessions.iter().all(|s| s.traffic.execs == 0));
    // A cache smaller than the union of warm hits and new misses must still return the
    // exact request. Inserting the miss may evict the warm row before reconstruction.
    let mut tiny_cache = EngramRowCache::new(264);
    let seeded =
        Session::fetch_engram_batch_cached(&mut sessions, &stamp, &table, &[0], T, &mut tiny_cache)
            .unwrap();
    assert_eq!(seeded.rows, rows[0]);
    let tiny_mixed = Session::fetch_engram_batch_cached(
        &mut sessions,
        &stamp,
        &table,
        &[0, 1],
        T,
        &mut tiny_cache,
    )
    .unwrap();
    assert_eq!(
        tiny_mixed.rows,
        [rows[0].as_slice(), rows[1].as_slice()].concat()
    );
    assert_eq!((tiny_mixed.cache_hits, tiny_mixed.cache_misses), (1, 1));
    assert_eq!(tiny_mixed.owner_payload_bytes, 264);
    let property_ids = [3, 0, 2, 1, 3];
    let property_expected: Vec<u8> = property_ids
        .iter()
        .flat_map(|&id| rows[id as usize].iter().copied())
        .collect();
    // Exhaust all warm-row masks against capacities 1..=4. This exercises FIFO
    // eviction of hits, out-of-order misses, and duplicate caller IDs in the real session.
    for capacity_rows in 1..=4 {
        for warm_mask in 0u8..16 {
            let mut cache = EngramRowCache::new(capacity_rows * 264);
            for (id, row) in rows.iter().enumerate() {
                if warm_mask & (1u8 << id) != 0 {
                    cache.insert(&table, id as u64, row).unwrap();
                }
            }
            let unique_ids = [3u64, 0, 2, 1];
            let expected_hits = unique_ids
                .iter()
                .filter(|&&id| cache.get(&table, id).is_some())
                .count();
            let receipt = Session::fetch_engram_batch_cached(
                &mut sessions,
                &stamp,
                &table,
                &property_ids,
                T,
                &mut cache,
            )
            .unwrap();
            assert_eq!(receipt.rows, property_expected);
            assert_eq!(receipt.cache_hits, expected_hits);
            assert_eq!(receipt.cache_misses, unique_ids.len() - expected_hits);
            assert_eq!(
                receipt.owner_payload_bytes,
                ((unique_ids.len() - expected_hits) * 264) as u64
            );
        }
    }
    for session in sessions {
        assert_eq!(session.close(T).exit_code, Some(0));
    }
}

/// Cross-host CPU qualification entry point. The caller supplies an explicit authenticated
/// transport argv (normally `ssh -T pc1 /isolated/path/vf-rank-session`). This remains ignored
/// in ordinary CI because it requires two live hosts and is not a GPU/RDMA qualification.
#[test]
#[ignore = "requires explicit PC1 x86_64 SSH worker argv and PC6 ARM64 local worker"]
fn a64_crossnode_persistent_engram_rows_over_authenticated_ssh() {
    use voltafabric::engram::{EngramRowCache, EngramShard, TableIdentity};

    let remote_argv: Vec<String> = serde_json::from_str(
        &std::env::var("VF_A64_REMOTE_ARGV_JSON")
            .expect("set VF_A64_REMOTE_ARGV_JSON to the exact SSH worker argv"),
    )
    .expect("remote argv must be a JSON string array");
    assert!(!remote_argv.is_empty());
    let table = TableIdentity {
        model_revision: "a64-crossnode-fixture-v1".into(),
        table_sha256: "55".repeat(32),
        tokenizer_map_sha256: "66".repeat(32),
        layer: 0,
        epoch: 1,
        num_embeddings: 8,
    };
    let rows: Vec<Vec<u8>> = (0..8)
        .map(|row| (0..264).map(|col| (row * 31 + col * 11) as u8).collect())
        .collect();
    // Non-dense global worker ranks intentionally differ from local Engram owners 0 and 1.
    let local = Session::connect(&endpoint(2), T).unwrap();
    let remote = Session::connect(
        &Endpoint {
            rank: 9,
            argv: remote_argv,
        },
        T,
    )
    .unwrap();
    assert_eq!(local.hello.arch, "aarch64");
    assert_eq!(remote.hello.arch, "x86_64");
    let mut sessions = [local, remote];
    let mut initial_load_bytes = Vec::new();
    for owner_rank in 0..2u32 {
        let worker_index = owner_rank as usize;
        let shard_bytes: Vec<u8> = rows[owner_rank as usize * 4..owner_rank as usize * 4 + 4]
            .iter()
            .flatten()
            .copied()
            .collect();
        let shard = EngramShard::new(table.clone(), 2, owner_rank, shard_bytes).unwrap();
        let receipt = sessions[worker_index]
            .load_engram_shard_for_owner(&shard, owner_rank, T)
            .unwrap();
        assert!(receipt.transferred);
        initial_load_bytes.push(receipt.request_bytes);
    }
    let stamp = Stamp {
        run_id: "a64-crossnode-cpu".into(),
        model_revision: table.model_revision.clone(),
        plan_id: "ssh-persistent-engram-v1".into(),
        layer: table.layer,
        batch: 1,
        epoch: table.epoch,
    };
    let ids = [7, 0, 4, 7, 3, 2, 6, 1];
    let received_before_first = sessions
        .iter()
        .map(|session| session.traffic.bytes_received)
        .sum::<u64>();
    let started = std::time::Instant::now();
    let batch_receipt =
        Session::fetch_engram_batch_with_receipt(&mut sessions, &stamp, &table, &ids, T).unwrap();
    let actual = batch_receipt.rows.clone();
    let elapsed_us = started.elapsed().as_micros();
    let first_batch_received_bytes = sessions
        .iter()
        .map(|session| session.traffic.bytes_received)
        .sum::<u64>()
        - received_before_first;
    assert_eq!(batch_receipt.requested_ids, 8);
    assert_eq!(batch_receipt.unique_ids_transferred, 7);
    assert_eq!(batch_receipt.owner_payload_bytes, 7 * 264);
    assert_eq!(batch_receipt.owner_requests, 2);
    assert_eq!(batch_receipt.skipped_empty_owners, 0);
    let expected: Vec<u8> = ids
        .iter()
        .flat_map(|&id| rows[id as usize].iter().copied())
        .collect();
    assert_eq!(
        actual, expected,
        "cross-host row bytes differ from fixture oracle"
    );
    let sent_after_first = sessions.iter().map(|s| s.traffic.bytes_sent).sum::<u64>();
    let second_ids = [1, 7, 1].repeat(40);
    let received_before_second = sessions
        .iter()
        .map(|session| session.traffic.bytes_received)
        .sum::<u64>();
    let second_receipt =
        Session::fetch_engram_batch_with_receipt(&mut sessions, &stamp, &table, &second_ids, T)
            .unwrap();
    let second = second_receipt.rows;
    let second_batch_received_bytes = sessions
        .iter()
        .map(|session| session.traffic.bytes_received)
        .sum::<u64>()
        - received_before_second;
    let expected_second: Vec<u8> = second_ids
        .iter()
        .flat_map(|&id| rows[id as usize].iter().copied())
        .collect();
    assert_eq!(second, expected_second);
    assert_eq!(second_receipt.requested_ids, 120);
    assert_eq!(second_receipt.unique_ids_transferred, 2);
    assert_eq!(second_receipt.owner_payload_bytes, 2 * 264);
    assert_eq!(
        second_receipt.requested_ids * 264 - second_receipt.owner_payload_bytes as usize,
        31_152,
        "duplicate-heavy fixture must remove only repeat row bodies"
    );
    let second_batch_sent_bytes =
        sessions.iter().map(|s| s.traffic.bytes_sent).sum::<u64>() - sent_after_first;
    let repeated_receipts: Vec<_> = sessions
        .iter_mut()
        .enumerate()
        .map(|(owner, session)| {
            let shard_bytes: Vec<u8> = rows[owner * 4..owner * 4 + 4]
                .iter()
                .flatten()
                .copied()
                .collect();
            let shard = EngramShard::new(table.clone(), 2, owner as u32, shard_bytes).unwrap();
            session
                .load_engram_shard_for_owner(&shard, owner as u32, T)
                .unwrap()
        })
        .collect();
    assert!(repeated_receipts
        .iter()
        .all(|r| !r.transferred && r.request_bytes == 0));
    let remote_bytes_before_owner0_only = (
        sessions[1].traffic.bytes_sent,
        sessions[1].traffic.bytes_received,
    );
    let owner0_ids = [0, 1, 1, 3];
    let owner0_only =
        Session::fetch_engram_batch_for_owners(&mut sessions, &stamp, &table, &owner0_ids, T)
            .unwrap();
    let owner0_only_remote_delta = (
        sessions[1].traffic.bytes_sent - remote_bytes_before_owner0_only.0,
        sessions[1].traffic.bytes_received - remote_bytes_before_owner0_only.1,
    );
    let expected_owner0_only: Vec<u8> = owner0_ids
        .iter()
        .flat_map(|&id| rows[id as usize].iter().copied())
        .collect();
    assert_eq!(owner0_only, expected_owner0_only);
    assert_eq!(
        (
            sessions[1].traffic.bytes_sent,
            sessions[1].traffic.bytes_received
        ),
        remote_bytes_before_owner0_only,
        "an owner with no requested rows should receive no request"
    );
    assert_eq!(owner0_only_remote_delta, (0, 0));
    let mut cache = EngramRowCache::new(6 * 264);
    // Missing IDs deliberately arrive out of order and with a duplicate. Reconstruction
    // must retain caller order even when the worker fetch is issued in sorted ID order.
    let mixed_ids = [0, 7, 6, 2, 7, 1, 5];
    let warmup =
        Session::fetch_engram_batch_cached(&mut sessions, &stamp, &table, &[0], T, &mut cache)
            .unwrap();
    assert_eq!(warmup.rows, rows[0]);
    assert_eq!((warmup.cache_hits, warmup.cache_misses), (0, 1));
    let mixed_before = sessions
        .iter()
        .map(|s| s.traffic.bytes_received)
        .sum::<u64>();
    let mixed = Session::fetch_engram_batch_cached(
        &mut sessions,
        &stamp,
        &table,
        &mixed_ids,
        T,
        &mut cache,
    )
    .unwrap();
    let mixed_after = sessions
        .iter()
        .map(|s| s.traffic.bytes_received)
        .sum::<u64>();
    let mixed_expected: Vec<u8> = mixed_ids
        .iter()
        .flat_map(|&id| rows[id as usize].iter().copied())
        .collect();
    assert_eq!(mixed.rows, mixed_expected);
    assert_eq!((mixed.cache_hits, mixed.cache_misses), (1, 5));
    assert_eq!(mixed.owner_payload_bytes, 5 * 264);
    assert!(
        mixed_after - mixed_before >= 5 * 264,
        "stream must carry at least the validated owner row payload"
    );
    let warm_before = sessions
        .iter()
        .map(|s| (s.traffic.bytes_sent, s.traffic.bytes_received))
        .collect::<Vec<_>>();
    let warm = Session::fetch_engram_batch_cached(
        &mut sessions,
        &stamp,
        &table,
        &mixed_ids,
        T,
        &mut cache,
    )
    .unwrap();
    assert_eq!(warm.rows, mixed_expected);
    assert_eq!(
        (
            warm.cache_hits,
            warm.cache_misses,
            warm.owner_payload_bytes,
            warm.owner_requests
        ),
        (6, 0, 0, 0)
    );
    assert_eq!(
        sessions
            .iter()
            .map(|s| (s.traffic.bytes_sent, s.traffic.bytes_received))
            .collect::<Vec<_>>(),
        warm_before,
        "all-hit batch must issue no worker requests"
    );
    assert_eq!(cache.resident_bytes(), 6 * 264);
    let cache_all_hit_worker_bytes_delta = sessions
        .iter()
        .zip(&warm_before)
        .map(|(s, before)| {
            (
                s.traffic.bytes_sent - before.0,
                s.traffic.bytes_received - before.1,
            )
        })
        .collect::<Vec<_>>();
    let sent: Vec<u64> = sessions.iter().map(|s| s.traffic.bytes_sent).collect();
    let received: Vec<u64> = sessions.iter().map(|s| s.traffic.bytes_received).collect();
    let closes: Vec<_> = sessions.into_iter().map(|s| s.close(T)).collect();
    assert!(closes.iter().all(|receipt| receipt.exit_code == Some(0)));
    let receipt = serde_json::json!({
        "architectures": ["aarch64", "x86_64"],
        "workers": [2, 9],
        "owners": [0, 1],
        "ids": ids,
        "payload_bytes": actual.len(),
        "requested_row_ids": ids.len(),
        "unique_row_ids": batch_receipt.unique_ids_transferred,
        "unique_response_payload_bytes": batch_receipt.owner_payload_bytes,
        "owner_requests": batch_receipt.owner_requests,
        "skipped_empty_owners": batch_receipt.skipped_empty_owners,
        "first_batch_response_stream_bytes": first_batch_received_bytes,
        "initial_load_bytes": initial_load_bytes,
        "second_batch_sent_bytes": second_batch_sent_bytes,
        "second_batch_response_stream_bytes": second_batch_received_bytes,
        "second_batch_requested_ids": second_receipt.requested_ids,
        "second_batch_unique_ids_transferred": second_receipt.unique_ids_transferred,
        "second_batch_owner_payload_bytes": second_receipt.owner_payload_bytes,
        "second_batch_duplicate_payload_bytes_avoided": second_receipt.requested_ids * 264 - second_receipt.owner_payload_bytes as usize,
        "owner0_only_request_remote_delta": owner0_only_remote_delta,
        "cache_mixed_hits": mixed.cache_hits,
        "cache_mixed_misses": mixed.cache_misses,
        "cache_mixed_owner_payload_bytes": mixed.owner_payload_bytes,
        "cache_mixed_response_stream_bytes": mixed_after - mixed_before,
        "cache_all_hit_worker_bytes_delta": cache_all_hit_worker_bytes_delta,
        "session_bytes_sent": sent,
        "session_bytes_received": received,
        "first_batch_elapsed_us": elapsed_us,
        "worker_peak_rss_kib": [
            closes[0].peak_rss_kib.unwrap_or(0),
            closes[1].peak_rss_kib.unwrap_or(0)
        ],
        "result": "resident_load_owner_filter_dedup_and_bounded_cache_verified",
        "gpu": false,
        "rdma": false
    });
    println!(
        "A67_CROSSNODE_RECEIPT {}",
        serde_json::to_string(&receipt).unwrap()
    );
}

#[test]
fn engram_cache_does_not_commit_rows_when_worker_executes_but_reply_is_lost() {
    use voltafabric::engram::{EngramRowCache, EngramShard, TableIdentity};

    let dir = temp_dir();
    let receipt_path = dir.join("dropped-reply.json");
    let proxy =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/support/drop_engram_reply_proxy.py");
    let worker = env!("CARGO_BIN_EXE_vf-rank-session");
    let table = TableIdentity {
        model_revision: "a68-drop-reply-v1".into(),
        table_sha256: "71".repeat(32),
        tokenizer_map_sha256: "82".repeat(32),
        layer: 0,
        epoch: 9,
        num_embeddings: 256,
    };
    let rows: Vec<Vec<u8>> = (0..256)
        .map(|id| (0..264).map(|byte| (id * 37 + byte * 11) as u8).collect())
        .collect();
    let healthy = Session::connect(&endpoint(2), T).unwrap();
    let failed_endpoint = Endpoint {
        rank: 9,
        argv: vec![
            "python3".into(),
            proxy.to_string_lossy().into_owned(),
            worker.into(),
            receipt_path.to_string_lossy().into_owned(),
        ],
    };
    let lost = Session::connect(&failed_endpoint, T).unwrap();
    let mut sessions = [healthy, lost];
    for (owner, session) in sessions.iter_mut().enumerate() {
        let shard = EngramShard::new(
            table.clone(),
            2,
            owner as u32,
            rows[owner * 128..owner * 128 + 128]
                .iter()
                .flatten()
                .copied()
                .collect(),
        )
        .unwrap();
        assert!(
            session
                .load_engram_shard_for_owner(&shard, owner as u32, T)
                .unwrap()
                .transferred
        );
    }
    let stamp = Stamp {
        run_id: "a68-drop-reply".into(),
        model_revision: table.model_revision.clone(),
        plan_id: "atomic-cache-insert".into(),
        layer: table.layer,
        batch: 1,
        epoch: table.epoch,
    };
    let mut cache = EngramRowCache::new(4 * 264);
    let row_ids: Vec<u64> = (0..256).collect();
    let result =
        Session::fetch_engram_batch_cached(&mut sessions, &stamp, &table, &row_ids, T, &mut cache);
    assert!(
        result.is_err(),
        "lost reply must not become a successful batch"
    );
    assert!(
        cache.is_empty(),
        "healthy owner's partial success must not enter cache"
    );
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&receipt_path).unwrap()).unwrap();
    assert_eq!(receipt["request"], "fetch_engram_rows");
    assert_eq!(receipt["wire_request"], "fetch_engram_rows_varint");
    assert_eq!(receipt["attempt"], 1);
    assert_eq!(receipt["worker_reply_consumed"], true);
    assert!(receipt["reply_bytes_discarded"].as_u64().unwrap() >= 264);
    println!("A68_UNKNOWN_AFTER_EXECUTION {}", receipt);
    let mut sessions = sessions.into_iter();
    let healthy_close = sessions.next().unwrap().close(T);
    assert_eq!(healthy_close.exit_code, Some(0));
    let lost_close = sessions.next().unwrap().close_input(T);
    assert!(lost_close.exit_code.is_some());
    std::fs::remove_dir_all(dir).unwrap();
}

/// Large PC6→PC1 owner batches exercise both peers on the real SSH stream. The A78 worker
/// must select the smaller varint form; an older worker must retain JSON. The byte counter
/// must match an independently assembled request for the negotiated form. CPU-only evidence.
#[test]
#[ignore = "requires explicit PC1 x86_64 SSH worker argv and PC6 ARM64 local worker"]
fn a78_crossnode_large_engram_request_uses_peer_compatible_wire_format() {
    use voltafabric::engram::{EngramShard, TableIdentity};

    let remote_argv: Vec<String> = serde_json::from_str(
        &std::env::var("VF_A64_REMOTE_ARGV_JSON")
            .expect("set VF_A64_REMOTE_ARGV_JSON to the exact SSH worker argv"),
    )
    .expect("remote argv must be a JSON string array");
    assert!(!remote_argv.is_empty());
    let table = TableIdentity {
        model_revision: "a78-crossnode-large-fixture-v1".into(),
        table_sha256: "a7".repeat(32),
        tokenizer_map_sha256: "b8".repeat(32),
        layer: 0,
        epoch: 78,
        num_embeddings: 256,
    };
    let rows: Vec<Vec<u8>> = (0..256)
        .map(|row| (0..264).map(|byte| (row * 37 + byte * 11) as u8).collect())
        .collect();
    let local = Session::connect(&endpoint(2), T).unwrap();
    let remote = Session::connect(
        &Endpoint {
            rank: 9,
            argv: remote_argv,
        },
        T,
    )
    .unwrap();
    assert_eq!(local.hello.arch, "aarch64");
    assert_eq!(remote.hello.arch, "x86_64");
    let remote_supports_varint = remote
        .hello
        .capabilities
        .iter()
        .any(|capability| capability == session::CAPABILITY_ENGRAM_ROW_IDS_VARINT);
    let mut sessions = [local, remote];
    for owner in 0..2usize {
        let shard = EngramShard::new(
            table.clone(),
            2,
            owner as u32,
            rows[owner * 128..(owner + 1) * 128]
                .iter()
                .flatten()
                .copied()
                .collect(),
        )
        .unwrap();
        assert!(
            sessions[owner]
                .load_engram_shard_for_owner(&shard, owner as u32, T)
                .unwrap()
                .transferred
        );
    }
    let stamp = Stamp {
        run_id: "a78-crossnode-large".into(),
        model_revision: table.model_revision.clone(),
        plan_id: "adaptive-row-id-wire".into(),
        layer: table.layer,
        batch: 1,
        epoch: table.epoch,
    };
    let ids: Vec<u64> = (0..256).collect();
    let owner_ids: Vec<u64> = (128..256).collect();
    let request_attempt = 1;
    let mut varint_body_bytes = 0usize;
    for &row_id in &owner_ids {
        let mut row_id = row_id;
        varint_body_bytes += 1;
        while row_id >= 0x80 {
            row_id >>= 7;
            varint_body_bytes += 1;
        }
    }
    let varint_header = serde_json::to_vec(&Request::FetchEngramRowsVarint {
        stamp: stamp.clone(),
        table: table.clone(),
        world_size: 2,
        owner_rank: 1,
        row_id_count: owner_ids.len() as u64,
        row_id_bytes: varint_body_bytes as u64,
        attempt: request_attempt,
    })
    .unwrap();
    let json_request = serde_json::to_vec(&Request::FetchEngramRows {
        stamp: stamp.clone(),
        table: table.clone(),
        world_size: 2,
        owner_rank: 1,
        row_ids: owner_ids.clone(),
        attempt: request_attempt,
    })
    .unwrap();

    let remote_sent_before = sessions[1].traffic.bytes_sent;
    let started = std::time::Instant::now();
    let receipt =
        Session::fetch_engram_batch_with_receipt(&mut sessions, &stamp, &table, &ids, T).unwrap();
    let elapsed_us = started.elapsed().as_micros();
    let remote_request_bytes = sessions[1].traffic.bytes_sent - remote_sent_before;
    // Session framing appends one newline to either independently serialized JSON header.
    let expected_varint_request_bytes = (varint_header.len() + varint_body_bytes + 1) as u64;
    let expected_json_request_bytes = (json_request.len() + 1) as u64;
    let expected_varint =
        remote_supports_varint && expected_varint_request_bytes < expected_json_request_bytes;
    let expected_request_bytes = if expected_varint {
        expected_varint_request_bytes
    } else {
        expected_json_request_bytes
    };
    let expected_rows: Vec<u8> = ids
        .iter()
        .flat_map(|&id| rows[id as usize].iter().copied())
        .collect();
    assert_eq!(receipt.rows, expected_rows);
    assert_eq!(receipt.requested_ids, 256);
    assert_eq!(receipt.unique_ids_transferred, 256);
    assert_eq!(receipt.owner_payload_bytes, 256 * 264);
    assert_eq!(receipt.owner_requests, 2);
    assert_eq!(
        remote_request_bytes, expected_request_bytes,
        "PC1 byte counter must equal the exact negotiated request frame"
    );
    assert_eq!(expected_varint, remote_supports_varint);
    let closes: Vec<_> = sessions
        .into_iter()
        .map(|session| session.close(T))
        .collect();
    assert!(closes.iter().all(|close| close.exit_code == Some(0)));
    println!(
        "A78_CROSSNODE_VARINT {}",
        serde_json::json!({
            "architectures": ["aarch64", "x86_64"],
            "transport": "authenticated SSH",
            "remote_varint_capability": remote_supports_varint,
            "selected_format": if expected_varint { "varint" } else { "json" },
            "requested_ids": receipt.requested_ids,
            "unique_ids": receipt.unique_ids_transferred,
            "owner_payload_bytes": receipt.owner_payload_bytes,
            "pc6_to_pc1_varint_request_bytes": remote_request_bytes,
            "pc6_to_pc1_json_request_bytes": expected_json_request_bytes,
            "pc6_to_pc1_varint_candidate_bytes": expected_varint_request_bytes,
            "request_bytes_saved": expected_json_request_bytes - remote_request_bytes,
            "varint_body_bytes": varint_body_bytes,
            "elapsed_us": elapsed_us,
            "worker_peak_rss_kib": [
                closes[0].peak_rss_kib.unwrap_or(0),
                closes[1].peak_rss_kib.unwrap_or(0)
            ],
            "exact_rows_match": true,
            "gpu": false,
            "rdma": false
        })
    );
}

#[test]
fn released_transfer_bytes_preserve_bound_batch_and_refuse_fresh_load() {
    let p = routing_problem("r293-release-transfer");
    let (mut sessions, mut artifacts) = bound(&p);
    let x = activations(6, HIDDEN as usize, 4);
    let before = session::run_compressed_batch(
        &mut sessions,
        &artifacts,
        &p,
        &x,
        &BTreeMap::new(),
        false,
        T,
    )
    .unwrap();
    assert!(before.ok);
    let hashes: Vec<_> = artifacts.iter().map(|a| a.sha256().to_owned()).collect();
    for a in &mut artifacts {
        assert!(!a.bytes().is_empty());
        a.discard_transfer_bytes();
        assert!(a.bytes().is_empty());
    }
    assert_eq!(
        hashes,
        artifacts
            .iter()
            .map(|a| a.sha256().to_owned())
            .collect::<Vec<_>>()
    );
    let mut next = p.clone();
    next.stamp.batch += 1;
    let after = session::run_compressed_batch(
        &mut sessions,
        &artifacts,
        &next,
        &x,
        &BTreeMap::new(),
        false,
        T,
    )
    .unwrap();
    assert!(after.ok);
    assert_eq!(before.output, after.output);
    let mut fresh = Session::connect(&endpoint(artifacts[0].stack().rank()), T).unwrap();
    let sent = fresh.traffic.bytes_sent;
    let error = fresh.load_compressed(&artifacts[0], EPOCH, T).unwrap_err();
    assert_eq!(error.code, "ARTIFACT_BYTES_RELEASED");
    assert_eq!(fresh.traffic.bytes_sent, sent);
    assert_eq!(fresh.close(T).exit_code, Some(0));
    for s in sessions {
        assert_eq!(s.close(T).exit_code, Some(0));
    }
}

#[test]
fn released_bytes_refuse_mixed_fleet_before_any_rebind() {
    let p = routing_problem("r293-mixed-preflight");
    let (mut sessions, mut artifacts) = bound(&p);
    let old = sessions.pop().unwrap();
    assert_eq!(old.close(T).exit_code, Some(0));
    sessions.push(Session::connect(&endpoint(9), T).unwrap());
    for a in &mut artifacts {
        a.discard_transfer_bytes();
    }
    let mut next = p.clone();
    next.stamp.epoch += 1;
    next.registry.epoch += 1;
    let sent: Vec<_> = sessions.iter().map(|s| s.traffic.bytes_sent).collect();
    let error = session::bind_compressed(&mut sessions, &artifacts, &next, T).unwrap_err();
    assert_eq!(error.code, "ARTIFACT_BYTES_RELEASED");
    assert_eq!(
        sent,
        sessions
            .iter()
            .map(|s| s.traffic.bytes_sent)
            .collect::<Vec<_>>()
    );
    let error =
        session::prepare_compressed_multipart(&[endpoint(2), endpoint(9)], &artifacts, &p, 1, T)
            .err()
            .unwrap();
    assert_eq!(error.code, "INVALID_INPUT");
    for s in sessions {
        assert_eq!(s.close(T).exit_code, Some(0));
    }
}

#[test]
fn compact_fleet_executes_after_drop_and_rejects_changed_identity_without_sending() {
    let p = routing_problem("r296-compact");
    let (mut sessions, artifacts) = bound(&p);
    let inputs = vec![vec![0x3c00_u16; HIDDEN as usize]; p.routes.len()];
    let mut routed = p.clone();
    routed.routes[0][0].weight = 0.125;
    let reference = session::run_compressed_batch_with_fp16_inputs(
        &mut sessions,
        &artifacts,
        &routed,
        &inputs,
        &BTreeMap::new(),
        session::CompletionFormat::Fp16,
        T,
    )
    .unwrap();
    assert!(reference.ok);
    let compact = session::CompactCompressedFleet::from_loaded(&sessions, &artifacts, &p).unwrap();
    drop(artifacts);
    let mut next = routed.clone();
    next.stamp.batch += 1;
    let result = compact.run_fp16(&mut sessions, &next, &inputs, T).unwrap();
    assert!(result.ok);
    assert_eq!(result.output, reference.output);
    let before: Vec<_> = sessions.iter().map(|s| s.traffic.bytes_sent).collect();
    let mut bad = next.clone();
    bad.stamp.batch += 1;
    bad.stamp.plan_id.push('x');
    assert!(compact.run_fp16(&mut sessions, &bad, &inputs, T).is_err());
    let mut bad = next.clone();
    bad.stamp.batch += 1;
    bad.registry.copies.swap(0, 1);
    assert!(compact.run_fp16(&mut sessions, &bad, &inputs, T).is_err());
    let mut bad = next.clone();
    bad.stamp.batch += 1;
    bad.stamp.epoch += 1;
    bad.registry.epoch += 1;
    assert!(compact.run_fp16(&mut sessions, &bad, &inputs, T).is_err());
    assert_eq!(
        before,
        sessions
            .iter()
            .map(|s| s.traffic.bytes_sent)
            .collect::<Vec<_>>()
    );
    for s in sessions {
        assert_eq!(s.close(T).exit_code, Some(0));
    }
}
