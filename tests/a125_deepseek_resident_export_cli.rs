//! Exercise the real bounded resident-export CLI from a single-expert MXFP4 artifact.

use safetensors::tensor::{serialize, SafeTensors, TensorView};
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};
use voltafabric::{deepseek_v41::Mxfp4ExpertShard, resident_artifact};

const FIXTURE_SHA256: &str = "e6d39f1821c0e3f7f1cc4827d206391aa428a6f3944731069d45dee4135f8c00";

struct OwnedTempDir(PathBuf);

impl Drop for OwnedTempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn one_expert_artifact(source: &[u8], expert: u32) -> Vec<u8> {
    let safetensors = SafeTensors::deserialize(source).unwrap();
    let prefix = format!("layers.0.ffn.experts.{expert}");
    let names = [
        format!("{prefix}.w1.weight"),
        format!("{prefix}.w1.scale"),
        format!("{prefix}.w2.weight"),
        format!("{prefix}.w2.scale"),
        format!("{prefix}.w3.weight"),
        format!("{prefix}.w3.scale"),
    ];
    let views: Vec<(String, TensorView<'_>)> = names
        .into_iter()
        .map(|name| {
            let source = safetensors.tensor(&name).unwrap();
            (
                name,
                TensorView::new(source.dtype(), source.shape().to_vec(), source.data()).unwrap(),
            )
        })
        .collect();
    serialize(views, None).unwrap()
}

fn workspace(root: &Path) -> OwnedTempDir {
    std::fs::create_dir_all(root.join(".work")).unwrap();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = root
        .join(".work")
        .join(format!("a125-cli-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&path).unwrap();
    OwnedTempDir(path)
}

#[test]
fn workspace_creates_its_parent_in_a_fresh_source_tree() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("a150-fresh-source-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let _owned_root = OwnedTempDir(root.clone());
    assert!(!root.join(".work").exists());
    let work = workspace(&root);
    assert!(work.0.is_dir());
    assert_eq!(work.0.parent(), Some(root.join(".work").as_path()));
    drop(work);
    assert!(root.join(".work").is_dir());
}

#[test]
fn resident_export_cli_accepts_a_bounded_deepseek_expert_artifact() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture =
        std::fs::read(root.join("fixtures/deepseek_v41_layer0_experts96-192_314113f8.safetensors"))
            .unwrap();
    let shard = Mxfp4ExpertShard::from_safetensors_shard_artifact(
        fixture.clone(),
        FIXTURE_SHA256,
        64 * 1024 * 1024,
    )
    .unwrap();
    let single = one_expert_artifact(&fixture, 96);
    drop(fixture);
    let source_sha256 = voltafabric::sha256::hex(&single);
    let work = workspace(root);
    let source_path = work.0.join("expert96.safetensors");
    let output_path = work.0.join("expert96.qpn2.safetensors");
    let request_path = work.0.join("request.json");
    std::fs::write(&source_path, &single).unwrap();
    drop(single);

    let request = json!({
        "source": {
            "kind": "deepseek-mxfp4-expert",
            "request": {
                "artifact_path": source_path,
                "expected_sha256": source_sha256,
                "layer": 0,
                "expert": 96,
                "rank": 0,
                "local": 0,
                "epoch": 41
            }
        },
        "output": output_path
    });
    std::fs::write(&request_path, serde_json::to_vec(&request).unwrap()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_vf"))
        .arg("resident-export")
        .arg(&request_path)
        .current_dir(root)
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(0),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let receipt: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(receipt["ok"], true);
    assert_eq!(receipt["result"]["scope"], "CPU_COMPRESSED_EXPORT_ONLY");
    assert_eq!(receipt["result"]["cuda_executed"], false);

    let resident_bytes = std::fs::read(&output_path).unwrap();
    let stack = resident_artifact::import(&resident_bytes).unwrap();
    let expert = shard.expert(0, 96).unwrap();
    let expected = expert.to_qpn2_resident_stack(0, 0, 41).unwrap();
    assert_eq!(stack.epoch(), 41);
    assert_eq!(stack.payload_bytes(), 19_907_776);
    assert_eq!(stack.physical_experts()[0].id(), 96);
    assert_eq!(stack.codes13(), expected.codes13());
    assert_eq!(stack.scales13(), expected.scales13());
    assert_eq!(stack.gscales13(), expected.gscales13());
    assert_eq!(stack.codes2(), expected.codes2());
    assert_eq!(stack.scales2(), expected.scales2());
    assert_eq!(stack.gscales2(), expected.gscales2());

    let mut bad_request = request;
    bad_request["source"]["request"]["expected_sha256"] = json!("0".repeat(64));
    bad_request["output"] = json!(work.0.join("bad.qpn2.safetensors"));
    let bad_request_path = work.0.join("bad-request.json");
    std::fs::write(&bad_request_path, serde_json::to_vec(&bad_request).unwrap()).unwrap();
    let rejected = Command::new(env!("CARGO_BIN_EXE_vf"))
        .arg("resident-export")
        .arg(&bad_request_path)
        .current_dir(root)
        .output()
        .unwrap();
    assert_eq!(rejected.status.code(), Some(2));
    assert!(!work.0.join("bad.qpn2.safetensors").exists());
}
