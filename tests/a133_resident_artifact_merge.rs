//! A133: compose validated compressed resident artifacts without decoding weights.
#![cfg(target_os = "linux")]

use serde_json::{json, Value};
use std::{path::PathBuf, process::Command};
use voltafabric::{
    moe::{Registry, Replica},
    quant,
    resident::{self, ResidentExpert, ResidentInput},
    resident_artifact, sha256,
};

struct TempDir(PathBuf);
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn temp_dir() -> TempDir {
    let path = std::env::temp_dir().join(format!(
        "vf-a133-merge-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&path).unwrap();
    TempDir(path)
}

fn plane(seed: u8) -> quant::PackedRows {
    quant::PackedRows {
        rows: 64,
        columns: 64,
        codes: vec![seed; 64 * 64 / 2],
        scales: vec![0x38; 64 * 64 / 16],
        global_dequant: 1.0,
    }
}

fn input(experts: &[(u32, u32, u8)], epoch: u64) -> ResidentInput {
    ResidentInput {
        model_revision: "a133-merge-test".into(),
        layer: 0,
        rank: 2,
        registry: Registry {
            epoch,
            copies: experts
                .iter()
                .map(|&(expert, local, _)| Replica {
                    expert,
                    rank: 2,
                    local,
                })
                .collect(),
        },
        resident_order: experts.iter().map(|&(_, local, _)| local).collect(),
        experts: experts
            .iter()
            .map(|&(id, local, seed)| ResidentExpert {
                id,
                local,
                gate: plane(seed),
                up: plane(seed.wrapping_add(1)),
                down: plane(seed.wrapping_add(2)),
            })
            .collect(),
    }
}

fn write_source(path: &std::path::Path, expert: (u32, u32, u8), epoch: u64) -> String {
    let stack = resident::pack(&input(&[expert], epoch)).unwrap();
    let bytes = resident_artifact::encode(&stack).unwrap();
    std::fs::write(path, &bytes).unwrap();
    sha256::hex(&bytes)
}

fn invoke(request_path: &std::path::Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_vf"))
        .arg("resident-artifact-merge")
        .arg(request_path)
        .output()
        .unwrap()
}

#[test]
fn cli_merges_compressed_planes_preserves_input_order_and_refuses_bad_inputs() {
    let temp = temp_dir();
    let left = temp.0.join("expert11.safetensors");
    let right = temp.0.join("expert29.safetensors");
    let left_sha = write_source(&left, (11, 4, 0x12), 7);
    let right_sha = write_source(&right, (29, 9, 0x42), 7);
    let output_path = temp.0.join("merged.safetensors");
    let request_path = temp.0.join("merge.json");
    let request = json!({
        "artifacts": [
            {"path": right, "expected_sha256": right_sha},
            {"path": left, "expected_sha256": left_sha}
        ],
        "output": output_path
    });
    std::fs::write(&request_path, serde_json::to_vec(&request).unwrap()).unwrap();
    let result = invoke(&request_path);
    assert_eq!(
        result.status.code(),
        Some(0),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let receipt: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(receipt["ok"], true);
    assert_eq!(receipt["result"]["cuda_executed"], false);
    assert_eq!(receipt["result"]["scope"], "CPU_COMPRESSED_EXPORT_ONLY");
    assert_eq!(
        receipt["result"]["meta"]["physical_experts"],
        json!([{"id":29,"local":9},{"id":11,"local":4}])
    );
    assert_eq!(receipt["result"]["meta"]["resident_order"], json!([9, 4]));
    let merged_bytes = std::fs::read(&output_path).unwrap();
    let merged = resident_artifact::import(&merged_bytes).unwrap();
    let expected = resident::pack(&input(&[(29, 9, 0x42), (11, 4, 0x12)], 7)).unwrap();
    assert_eq!(merged.codes13(), expected.codes13());
    assert_eq!(merged.scales13(), expected.scales13());
    assert_eq!(merged.gscales13(), expected.gscales13());
    assert_eq!(merged.codes2(), expected.codes2());
    assert_eq!(merged.scales2(), expected.scales2());
    assert_eq!(merged.gscales2(), expected.gscales2());
    assert_eq!(merged.payload_bytes(), expected.payload_bytes());
    assert_eq!(merged_bytes, resident_artifact::encode(&expected).unwrap());

    let duplicate_output = temp.0.join("duplicate.safetensors");
    let duplicate_request_path = temp.0.join("duplicate.json");
    let duplicate = json!({
        "artifacts": [
            {"path": left, "expected_sha256": left_sha},
            {"path": left, "expected_sha256": left_sha}
        ],
        "output": duplicate_output
    });
    std::fs::write(
        &duplicate_request_path,
        serde_json::to_vec(&duplicate).unwrap(),
    )
    .unwrap();
    let rejected = invoke(&duplicate_request_path);
    assert_eq!(rejected.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&rejected.stdout).contains("repeat"),
        "stdout={}",
        String::from_utf8_lossy(&rejected.stdout)
    );
    assert!(!temp.0.join("duplicate.safetensors").exists());

    let digest_output = temp.0.join("bad-digest.safetensors");
    let digest_request_path = temp.0.join("bad-digest.json");
    let bad_digest = json!({
        "artifacts": [{"path": left, "expected_sha256": "0".repeat(64)}],
        "output": digest_output
    });
    std::fs::write(
        &digest_request_path,
        serde_json::to_vec(&bad_digest).unwrap(),
    )
    .unwrap();
    let rejected = invoke(&digest_request_path);
    assert_eq!(rejected.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&rejected.stdout).contains("ARTIFACT_DIGEST_MISMATCH"),
        "stdout={}",
        String::from_utf8_lossy(&rejected.stdout)
    );
    assert!(!temp.0.join("bad-digest.safetensors").exists());

    let stale_output = temp.0.join("stale.safetensors");
    let stale_request_path = temp.0.join("stale.json");
    let stale = json!({
        "artifacts": [
            {"path": left, "expected_sha256": left_sha},
            {"path": right, "expected_sha256": right_sha}
        ],
        "output": stale_output
    });
    // Same content hashes, but an artifact with a stale epoch is substituted.
    let stale_source = temp.0.join("stale-epoch.safetensors");
    let stale_sha = write_source(&stale_source, (29, 9, 0x42), 8);
    let mut stale: Value = stale;
    stale["artifacts"][1]["path"] = json!(stale_source);
    stale["artifacts"][1]["expected_sha256"] = json!(stale_sha);
    std::fs::write(&stale_request_path, serde_json::to_vec(&stale).unwrap()).unwrap();
    let rejected = invoke(&stale_request_path);
    assert_eq!(rejected.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&rejected.stdout).contains("identity or expert geometry"),
        "stdout={}",
        String::from_utf8_lossy(&rejected.stdout)
    );
    assert!(!temp.0.join("stale.safetensors").exists());
}
