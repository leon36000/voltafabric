use safetensors::SafeTensors;
use serde_json::{json, Value};
use std::{
    fs,
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};
use voltafabric::{checkpoint, moe, quant, resident, resident_artifact};

static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    temp_dir_with_nonce(nonce)
}

fn temp_dir_with_nonce(nonce: u128) -> std::path::PathBuf {
    let serial = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("vf-a19-{}-{nonce}-{serial}", std::process::id()));
    fs::create_dir(&path).unwrap();
    path
}

fn checkpoint_request() -> (Value, checkpoint::Request) {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/resident_import_qwen_expert0.json");
    let bytes = fs::read(fixture).unwrap();
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    let request: checkpoint::Request = serde_json::from_value(value.clone()).unwrap();
    (value, request)
}

fn cli_input(source: Value, output: &std::path::Path) -> Value {
    json!({"source": source, "output": output})
}

fn run_export(input: &Value, dir: &std::path::Path) -> std::process::Output {
    let path = dir.join("input.json");
    fs::write(&path, serde_json::to_vec(input).unwrap()).unwrap();
    Command::new(env!("CARGO_BIN_EXE_vf"))
        .arg("resident-export")
        .arg(path)
        .output()
        .unwrap()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn checkpoint_export_has_exact_safetensors_payload_and_metadata() {
    let (request_value, request) = checkpoint_request();
    let stack = checkpoint::load(&request).unwrap().stack;
    let dir = temp_dir();
    let output = dir.join("resident.safetensors");
    let result = run_export(
        &cli_input(
            json!({"kind":"checkpoint","request":request_value}),
            &output,
        ),
        &dir,
    );
    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let receipt: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(receipt["result"]["scope"], "CPU_COMPRESSED_EXPORT_ONLY");
    assert_eq!(receipt["result"]["cuda_executed"], false);
    assert_eq!(
        receipt["result"]["meta"]["payload_bytes"],
        stack.payload_bytes()
    );
    assert_eq!(
        receipt["result"]["meta"]["shape"],
        json!([
            stack.shape().0,
            stack.shape().1,
            stack.shape().2,
            stack.shape().3,
            stack.shape().4
        ])
    );

    let bytes = fs::read(output).unwrap();
    assert_eq!(receipt["result"]["file_bytes"], bytes.len());
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    assert_eq!(tensors.tensor("codes13").unwrap().data(), stack.codes13());
    assert_eq!(tensors.tensor("scales13").unwrap().data(), stack.scales13());
    assert_eq!(tensors.tensor("codes2").unwrap().data(), stack.codes2());
    assert_eq!(tensors.tensor("scales2").unwrap().data(), stack.scales2());
    assert_eq!(
        tensors.tensor("gscales13").unwrap().data(),
        f32_bytes(stack.gscales13())
    );
    assert_eq!(
        tensors.tensor("gscales2").unwrap().data(),
        f32_bytes(stack.gscales2())
    );

    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    let metadata = header["__metadata__"].as_object().unwrap();
    assert_eq!(metadata.len(), 2);
    assert_eq!(metadata["vf_schema"], "QPN2_RESIDENT_V1");
    let info: Value = serde_json::from_str(metadata["vf_info"].as_str().unwrap()).unwrap();
    assert_eq!(info["model_revision"], stack.model_revision());
    assert_eq!(info["layer"], stack.layer());
    assert_eq!(info["rank"], stack.rank());
    assert_eq!(info["epoch"], stack.epoch());
    assert_eq!(info["resident_order"], json!(stack.resident_order()));
    assert_eq!(
        info["physical_experts"][0]["id"],
        stack.physical_experts()[0].id()
    );
    assert_eq!(
        info["physical_experts"][0]["local"],
        stack.physical_experts()[0].local()
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn malformed_input_existing_target_and_missing_parent_do_not_publish() {
    let (request_value, _) = checkpoint_request();
    let dir = temp_dir();
    let existing = dir.join("already.safetensors");
    fs::write(&existing, b"keep me").unwrap();
    let result = run_export(
        &cli_input(
            json!({"kind":"checkpoint","request":request_value.clone()}),
            &existing,
        ),
        &dir,
    );
    assert!(!result.status.success());
    assert_eq!(fs::read(&existing).unwrap(), b"keep me");

    let missing = dir.join("missing").join("artifact.safetensors");
    let result = run_export(
        &cli_input(
            json!({"kind":"checkpoint","request":request_value}),
            &missing,
        ),
        &dir,
    );
    assert!(!result.status.success());
    assert!(!missing.exists());

    let bad_output = dir.join("malformed.safetensors");
    let result = run_export(
        &json!({"source":{"kind":"bogus","request":{}},"output":bad_output}),
        &dir,
    );
    assert!(!result.status.success());
    assert!(!bad_output.exists());
    assert_eq!(
        fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().contains("vf-stage"))
            .count(),
        0
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn concurrent_processes_publish_exactly_one_artifact() {
    let (request_value, _) = checkpoint_request();
    let dir = temp_dir();
    let output = dir.join("race.safetensors");
    let input = dir.join("race.json");
    fs::write(
        &input,
        serde_json::to_vec(&cli_input(
            json!({"kind":"checkpoint","request":request_value}),
            &output,
        ))
        .unwrap(),
    )
    .unwrap();
    let spawn = || {
        Command::new(env!("CARGO_BIN_EXE_vf"))
            .arg("resident-export")
            .arg(&input)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap()
    };
    let mut first = spawn();
    let mut second = spawn();
    let statuses = [first.wait().unwrap(), second.wait().unwrap()];
    assert_eq!(statuses.iter().filter(|s| s.success()).count(), 1);
    assert!(output.is_file());
    let bytes = fs::read(&output).unwrap();
    SafeTensors::deserialize(&bytes).unwrap();
    assert_eq!(
        fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().contains("vf-stage"))
            .count(),
        0
    );
    fs::remove_dir_all(dir).unwrap();
}

fn matrix(rows: u64, columns: u64, seed: u8, factor: f64) -> quant::PackedRows {
    quant::PackedRows {
        rows,
        columns,
        codes: (0..rows * columns / 2)
            .map(|i| (i as u8).wrapping_mul(37).wrapping_add(seed))
            .collect(),
        scales: (0..rows * columns / 16)
            .map(|i| ((i + u64::from(seed)) % 127) as u8)
            .collect(),
        global_dequant: factor,
    }
}

fn resident_expert(id: u32, local: u32, seed: u8) -> resident::ResidentExpert {
    resident::ResidentExpert {
        id,
        local,
        gate: matrix(128, 64, seed, 0.5),
        up: matrix(128, 64, seed + 1, 2.0),
        down: matrix(64, 128, seed + 2, 4.0),
    }
}

#[test]
fn reversed_sparse_order_is_preserved_in_metadata_and_planes() {
    let input = resident::ResidentInput {
        model_revision: "reverse-model".into(),
        layer: 6,
        rank: 3,
        registry: moe::Registry {
            epoch: 17,
            copies: vec![
                moe::Replica {
                    expert: 41,
                    rank: 3,
                    local: 2,
                },
                moe::Replica {
                    expert: 77,
                    rank: 3,
                    local: 99,
                },
                moe::Replica {
                    expert: 88,
                    rank: 0,
                    local: 0,
                },
            ],
        },
        resident_order: vec![99, 2],
        experts: vec![resident_expert(41, 2, 5), resident_expert(77, 99, 9)],
    };
    let stack = resident::pack(&input).unwrap();
    let dir = temp_dir();
    let output = dir.join("reverse.safetensors");
    let receipt = resident_artifact::export(&stack, &output).unwrap();
    assert_eq!(receipt.meta.resident_order, vec![99, 2]);
    assert_eq!(
        receipt
            .meta
            .physical_experts
            .iter()
            .map(|e| (e.id, e.local))
            .collect::<Vec<_>>(),
        vec![(77, 99), (41, 2)]
    );
    let bytes = fs::read(&output).unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    assert_eq!(tensors.tensor("codes13").unwrap().data(), stack.codes13());
    assert_eq!(tensors.tensor("scales13").unwrap().data(), stack.scales13());
    assert_eq!(tensors.tensor("codes2").unwrap().data(), stack.codes2());
    assert_eq!(tensors.tensor("scales2").unwrap().data(), stack.scales2());
    assert_eq!(
        tensors.tensor("gscales13").unwrap().data(),
        f32_bytes(stack.gscales13())
    );
    assert_eq!(
        tensors.tensor("gscales2").unwrap().data(),
        f32_bytes(stack.gscales2())
    );
    fs::remove_dir_all(dir).unwrap();
}

fn empty_resident() -> resident::ResidentInput {
    resident::ResidentInput {
        model_revision: "empty-model".into(),
        layer: 0,
        rank: 4,
        registry: moe::Registry {
            epoch: 9,
            copies: vec![],
        },
        resident_order: vec![],
        experts: vec![],
    }
}

#[test]
fn empty_resident_uses_sentinel_and_zero_length_tensors() {
    let stack = resident::pack(&empty_resident()).unwrap();
    assert_eq!(stack.shape(), (0, 0, 0, 0, 0));
    let dir = temp_dir();
    let output = dir.join("empty.safetensors");
    let receipt = resident_artifact::export(&stack, &output).unwrap();
    assert_eq!(receipt.meta.shape, [0, 0, 0, 0, 0]);
    let bytes = fs::read(&output).unwrap();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    for name in [
        "codes13",
        "scales13",
        "codes2",
        "scales2",
        "gscales13",
        "gscales2",
    ] {
        assert!(tensors.tensor(name).unwrap().data().is_empty());
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn independent_test_directories_do_not_depend_on_clock_resolution() {
    let threads: Vec<_> = (0..8)
        .map(|_| std::thread::spawn(|| temp_dir_with_nonce(0)))
        .collect();
    let paths: Vec<_> = threads.into_iter().map(|h| h.join().unwrap()).collect();
    let unique: std::collections::BTreeSet<_> = paths.iter().collect();
    assert_eq!(unique.len(), paths.len());
    for path in paths {
        fs::remove_dir(path).unwrap();
    }
}

#[test]
fn export_into_a_missing_directory_reports_the_staging_failure() {
    // A41 reason sweep: the CLI test only checked that this fails; a panic would
    // have failed too. The library must name the staging step.
    let (_, mut request) = checkpoint_request();
    request.path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(&request.path);
    let stack = checkpoint::load(&request).unwrap().stack;
    let dir = temp_dir();
    let error = resident_artifact::export(&stack, &dir.join("missing").join("out.safetensors"))
        .unwrap_err();
    fs::remove_dir(&dir).unwrap();
    assert_eq!(error.code, "ARTIFACT_IO");
    assert!(
        error.message.starts_with("cannot create staging file"),
        "{error}"
    );
}
