//! A49 return-Err sweep: each refusal below returns a reason no test checked. A40
//! removed the guards in front of most of them, which proves a test reaches the
//! refusal, not that any test reads what it says. Every test pins code and message.
#![cfg(target_os = "linux")]
mod common;

use serde_json::{json, Value};
use std::{
    fs,
    io::{Cursor, Write},
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::atomic::{AtomicU64, Ordering},
};
use voltafabric::{
    checkpoint, completion, launch, memory, moe, pipeline, process::*, qpn2, quant, resident,
    resident_artifact, routing, Error,
};

static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
fn temp_dir() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "vf-a49-{}-{}",
        std::process::id(),
        NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&path).unwrap();
    path
}
fn assert_reason(error: &Error, code: &str, message: &str) {
    assert_eq!(error.code, code, "{error}");
    assert_eq!(error.message, message, "{error}");
}
/// The refusal of a call whose success value need not be printable.
fn refused<T>(result: Result<T, Error>) -> Error {
    match result {
        Ok(_) => panic!("accepted"),
        Err(error) => error,
    }
}
fn invalid(error: &Error, message: &str) {
    assert_reason(error, "INVALID_INPUT", message);
}

fn checkpoint_request() -> checkpoint::Request {
    let bytes = fs::read(manifest().join("fixtures/resident_import_qwen_expert0.json")).unwrap();
    let mut request: checkpoint::Request = serde_json::from_slice(&bytes).unwrap();
    request.path = manifest().join(&request.path);
    request
}

#[test]
fn checkpoint_offsets_that_miss_the_file_length_are_refused() {
    // One U8 tensor of two bytes, then three data bytes: the header parses, the
    // offsets are valid, and one byte is left over.
    let mut header =
        serde_json::to_vec(&json!({"t": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}}))
            .unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend([0; 3]);
    let error = refused(checkpoint::read(
        &mut Cursor::new(bytes),
        &checkpoint_request().spec,
    ));
    invalid(&error, "safetensors offsets do not cover exact file length");
}

#[test]
fn checkpoint_label_mismatch_names_the_label() {
    let mut request = checkpoint_request();
    request
        .spec
        .expected_metadata
        .insert("license".into(), "MIT".into());
    let error = refused(checkpoint::load(&request));
    invalid(&error, "checkpoint label differs: license");
}

fn plane(rows: u64, columns: u64) -> quant::PackedRows {
    quant::PackedRows {
        rows,
        columns,
        codes: vec![0x22; (rows * columns / 2) as usize],
        scales: vec![56; (rows * columns / 16) as usize],
        global_dequant: 1.0,
    }
}
fn expert(id: u32, local: u32) -> resident::ResidentExpert {
    resident::ResidentExpert {
        id,
        local,
        gate: plane(128, 64),
        up: plane(128, 64),
        down: plane(64, 128),
    }
}
fn replica(expert: u32, rank: u32, local: u32) -> moe::Replica {
    moe::Replica {
        expert,
        rank,
        local,
    }
}
/// Rank 17 holds experts 23 (local 3) and 99 (local 900), in physical order 900, 3.
fn weights() -> resident::ResidentInput {
    resident::ResidentInput {
        model_revision: "m".into(),
        layer: 3,
        rank: 17,
        registry: moe::Registry {
            epoch: 12,
            copies: vec![replica(23, 17, 3), replica(99, 17, 900)],
        },
        resident_order: vec![900, 3],
        experts: vec![expert(23, 3), expert(99, 900)],
    }
}
fn dispatch() -> routing::RankDispatch {
    let stamp = moe::Stamp {
        run_id: "r".into(),
        model_revision: "m".into(),
        plan_id: "p".into(),
        layer: 3,
        batch: 4,
        epoch: 12,
    };
    let slots = [(0, 0, 23, 3, 1), (0, 1, 99, 900, 0), (1, 0, 99, 900, 0)]
        .into_iter()
        .map(
            |(token, slot, expert, local, physical_expert)| routing::RankSlot {
                slot: moe::Slot {
                    token,
                    slot,
                    expert,
                    rank: 17,
                    local,
                    weight: 0.5,
                },
                physical_expert,
            },
        )
        .collect();
    routing::RankDispatch {
        stamp,
        rank: 17,
        hidden: 64,
        tokens: 2,
        resident_order: vec![900, 3],
        slots,
    }
}
fn options() -> launch::Options {
    launch::Options {
        splitk13: 4,
        splitk2: 8,
        max_workspace_bytes: 128 * 1024 * 1024,
    }
}

#[test]
fn compact_output_with_missing_values_is_refused() {
    let w = resident::pack(&weights()).unwrap();
    let d = dispatch();
    let p = launch::prepare(&w, &d, &d.stamp, &options()).unwrap();
    let mut raw = completion::RawOutput {
        stamp: d.stamp.clone(),
        rank: d.rank,
        rows: d.slots.len(),
        columns: 64,
        fp16_bits: vec![0x3c00; d.slots.len() * 64],
    };
    assert!(completion::from_output(&p, &raw).is_ok());
    raw.fp16_bits.pop();
    let error = refused(completion::from_output(&p, &raw));
    invalid(
        &error,
        "compact FP16 output length differs; no padding rows allowed",
    );
}

#[test]
fn splitk_that_does_not_divide_the_projection_is_refused() {
    // Gate/up K is 64 columns: 4 blocks of 16, which split-K 8 cannot divide.
    let w = resident::pack(&weights()).unwrap();
    let d = dispatch();
    let o = launch::Options {
        splitk13: 8,
        ..options()
    };
    let error = refused(launch::prepare(&w, &d, &d.stamp, &o));
    invalid(
        &error,
        "projection dimensions or splitk outside compiled FFI contract",
    );
}

#[test]
fn rank_dispatch_bound_to_other_weights_is_refused() {
    let w = resident::pack(&weights()).unwrap();
    w.bind_dispatch(&dispatch()).unwrap();

    let mut d = dispatch();
    d.stamp.epoch += 1;
    invalid(
        &w.bind_dispatch(&d).unwrap_err(),
        "dispatch does not match static resident identity or physical order",
    );

    let mut d = dispatch();
    d.hidden = 128;
    invalid(
        &w.bind_dispatch(&d).unwrap_err(),
        "dispatch hidden dimension differs from resident weights",
    );

    let mut d = dispatch();
    d.slots[1].slot.weight = -1.0;
    invalid(
        &w.bind_dispatch(&d).unwrap_err(),
        "foreign, duplicate or malformed resident slot",
    );

    // A rank without resident experts binds only an empty dispatch.
    let empty = resident::pack(&resident::ResidentInput {
        registry: moe::Registry {
            epoch: 12,
            copies: vec![replica(23, 18, 3)],
        },
        resident_order: vec![],
        experts: vec![],
        ..weights()
    })
    .unwrap();
    let mut d = dispatch();
    d.resident_order.clear();
    invalid(
        &empty.bind_dispatch(&d).unwrap_err(),
        "empty rank cannot bind compute slots",
    );
    d.slots.clear();
    empty.bind_dispatch(&d).unwrap();
}

#[test]
fn resident_registry_and_scale_refusals_name_their_reason() {
    let mut input = weights();
    input.registry.copies[1].rank = 18;
    invalid(
        &refused(resident::pack(&input)),
        "registry rank does not exactly match local resident experts",
    );

    let mut input = weights();
    input.experts[1].down.global_dequant = 0.1;
    invalid(
        &refused(resident::pack(&input)),
        "global dequant factor must be positive, finite and exactly representable as F32",
    );
}

#[test]
fn artifact_output_must_name_a_new_file() {
    let stack = resident::pack(&weights()).unwrap();
    let dir = temp_dir();
    let error = refused(resident_artifact::export(&stack, &dir.join("..")));
    assert_reason(&error, "INVALID_INPUT", "output path must name a file");

    let output = dir.join("taken.safetensors");
    fs::write(&output, b"keep").unwrap();
    let error = refused(resident_artifact::export(&stack, &output));
    let left: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    let kept = fs::read(&output).unwrap();
    fs::remove_file(&output).unwrap();
    fs::remove_dir(&dir).unwrap();
    assert_reason(
        &error,
        "OUTPUT_EXISTS",
        "cannot atomically publish artifact: File exists (os error 17)",
    );
    assert_eq!(kept, b"keep");
    assert_eq!(left, vec![std::ffi::OsString::from("taken.safetensors")]);
}

#[test]
fn qpn2_plane_refusals_name_their_reason() {
    let mut p = plane(32, 64);
    p.global_dequant = 0.0;
    invalid(
        &refused(qpn2::pack(&p)),
        "QPN2 global dequant factor must be finite and positive",
    );
    let mut p = plane(32, 64);
    p.codes.pop();
    invalid(
        &refused(qpn2::pack(&p)),
        "QPN2 packed plane length mismatch",
    );
    invalid(
        &quant::decode_fp4(16).unwrap_err(),
        "FP4 nibble outside 0..15",
    );
}

#[test]
fn layout_refusals_name_their_reason() {
    for (rows, columns) in [(0, 64), (64, 0)] {
        invalid(
            &refused(memory::nvfp4_layout(rows, columns, 256, 0)),
            "matrix dimensions must be positive",
        );
    }
    for alignment in [0, 3, 96] {
        invalid(
            &refused(memory::nvfp4_layout(64, 64, alignment, 0)),
            "alignment must be a nonzero power of two",
        );
    }
}

#[test]
fn moe_problem_refusals_name_their_reason() {
    let mut p = common::problem();
    p.model.experts[1].down[2] = f64::NAN;
    invalid(&moe::validate(&p).unwrap_err(), "nonfinite model weights");

    // Expert 10 lives on rank 2 only.
    let mut p = common::problem();
    p.preferred_owners.insert(10, 9);
    invalid(
        &moe::validate(&p).unwrap_err(),
        "preferred owner has no resident replica",
    );
}

#[test]
fn pipeline_whose_last_stage_has_a_boundary_is_refused() {
    let stage = |rank, boundary_after_us| pipeline::Stage {
        rank,
        available_bytes: 10,
        boundary_after_us,
    };
    let layer = || pipeline::Layer {
        bytes: 1,
        compute_us_by_stage: vec![1.0, 1.0],
    };
    let p = pipeline::PipelineProblem {
        layers: vec![layer(), layer()],
        stages: vec![stage(0, 0.5), stage(1, 0.5)],
    };
    invalid(
        &refused(pipeline::partition(&p)),
        "last stage cannot have a following boundary",
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
fn resident_export_of_an_unknown_source_kind_is_refused() {
    let dir = temp_dir();
    let input = dir.join("input.json");
    let output = dir.join("out.safetensors");
    let request = json!({
        "source": {"kind": "safetensors", "request": {}},
        "output": output,
    });
    fs::write(&input, serde_json::to_vec(&request).unwrap()).unwrap();
    let result = vf(&["resident-export".as_ref(), input.as_ref()]);
    fs::remove_file(&input).unwrap();
    fs::remove_dir(&dir).unwrap();
    invalid(
        &cli_error(&result),
        "source.kind must be checkpoint, resident or deepseek-mxfp4-expert",
    );
}

fn problem(run_id: &str, faults: &[(u32, Fault)]) -> ProcessProblem {
    let mut problem = common::problem();
    problem.stamp.run_id = run_id.into();
    ProcessProblem {
        problem,
        faults: faults.iter().cloned().collect(),
        timeout_ms: 5000,
        recovery: false,
    }
}
fn worker() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_vf-rank-worker"))
}

#[test]
fn fault_on_a_rank_without_work_is_refused() {
    let error = refused(run(
        &problem("a49-idle-fault", &[(5, Fault::None)]),
        worker(),
    ));
    invalid(&error, "fault refers to a nonparticipating rank");
}

#[test]
fn worker_rejection_reaches_the_report_unchanged() {
    // Rank 2 holds experts 10 and 99; the worker refuses before computing.
    let p = problem("a49-reject", &[(2, Fault::RejectBeforeCompute)]);
    let report = run(&p, worker()).unwrap();
    assert!(!report.ok, "{report:?}");
    assert_reason(
        report.error.as_ref().unwrap(),
        "WORKER_REJECTED",
        "injected rejection before computation",
    );
}

#[test]
fn rank_that_finishes_without_all_results_is_refused() {
    let p = problem("a49-missing", &[(2, Fault::Missing)]);
    let report = run(&p, worker()).unwrap();
    assert!(!report.ok, "{report:?}");
    assert_reason(
        report.error.as_ref().unwrap(),
        "BATCH_INCOMPLETE",
        "rank finished with missing results",
    );
}

#[test]
fn worker_request_without_its_newline_is_an_unterminated_frame() {
    let p = common::problem();
    let dispatch = routing::dispatch(&routing::from_dense(&p)).unwrap();
    let request = rank_request(&p, &dispatch.ranks[0], 1, Fault::None).unwrap();
    let line = serde_json::to_vec(&request).unwrap();
    let mut child = Command::new(worker())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(&line).unwrap();
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert!(output.stdout.is_empty(), "{output:?}");
    assert_eq!(
        String::from_utf8_lossy(&output.stderr),
        "WORKER_PROTOCOL: unterminated JSON frame\n"
    );
}

#[test]
fn staged_artifact_that_cannot_be_reopened_is_never_published() {
    // umask 0477 creates the staging file write-only (0200): SafeTensors writes it,
    // then the reopen for sync and size fails. A subprocess, since umask is per process.
    let dir = temp_dir();
    let input = dir.join("input.json");
    let output = dir.join("out.safetensors");
    let request = fs::read(manifest().join("fixtures/resident_import_qwen_expert0.json")).unwrap();
    let mut request: Value = serde_json::from_slice(&request).unwrap();
    request["path"] = json!(manifest().join(request["path"].as_str().unwrap()));
    let body = json!({"source": {"kind": "checkpoint", "request": request}, "output": output});
    fs::write(&input, serde_json::to_vec(&body).unwrap()).unwrap();
    let result = Command::new("sh")
        .args(["-c", "umask 0477 && exec \"$@\"", "sh"])
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
    assert_reason(
        &cli_error(&result),
        "ARTIFACT_IO",
        "cannot reopen staged artifact: Permission denied (os error 13)",
    );
    assert_eq!(left, vec![std::ffi::OsString::from("input.json")]);
}

#[test]
fn artifact_linked_into_an_unreadable_directory_is_reported_uncertain() {
    // Write and search permission let the stage be created, linked and removed; only
    // opening the directory to sync it fails. Needs a user without CAP_DAC_OVERRIDE.
    use std::os::unix::fs::PermissionsExt;
    let stack = resident::pack(&weights()).unwrap();
    let dir = temp_dir();
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o300)).unwrap();
    assert!(fs::read_dir(&dir).is_err(), "test needs a non-root user");
    let output = dir.join("out.safetensors");
    let error = refused(resident_artifact::export(&stack, &output));
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).unwrap();
    let left: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    let published = fs::metadata(&output).unwrap().len();
    fs::remove_file(&output).unwrap();
    fs::remove_dir(&dir).unwrap();
    assert_reason(
        &error,
        "PUBLICATION_UNCERTAIN",
        "artifact is linked but directory sync failed: Permission denied (os error 13)",
    );
    assert_eq!(left, vec![std::ffi::OsString::from("out.safetensors")]);
    // The artifact is published even though the call fails: header plus payload.
    assert!(published > stack.payload_bytes(), "{published}");
}
