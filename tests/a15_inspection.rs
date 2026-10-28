use serde_json::{json, Value};
use std::{
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};
fn invoke_command(command: &str, mut request: Value) -> (bool, Value, usize) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    request["path"] = json!(root.join(request["path"].as_str().unwrap()));
    let mut child = Command::new(env!("CARGO_BIN_EXE_vf"))
        .args([command, "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(serde_json::to_string(&request).unwrap().as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    (
        out.status.success(),
        serde_json::from_slice(&out.stdout).unwrap(),
        out.stdout.len(),
    )
}
fn invoke(request: Value) -> (bool, Value, usize) {
    invoke_command("resident-inspect", request)
}
#[test]
fn real_checkpoint_inspection_is_small_and_preserves_exact_identity_and_byte_counts() {
    let request: Value = serde_json::from_str(include_str!(
        "../fixtures/resident_import_qwen_expert0.json"
    ))
    .unwrap();
    let (ok, out, bytes) = invoke(request);
    assert!(ok, "{out}");
    let r = &out["result"];
    assert_eq!(r["scope"], "CPU_COMPRESSED_INSPECTION_ONLY");
    assert_eq!(r["cuda_executed"], false);
    assert_eq!(
        r["model_revision"],
        "1355db6a052410cfd62085d94b58866fd0f2c3c5"
    );
    assert_eq!(r["resident_order"], json!([900]));
    assert_eq!(r["physical_experts"], json!([{"id":0,"local":900}]));
    assert_eq!(r["shape"], json!([1, 1024, 2048, 2048, 512]));
    assert_eq!(r["payload_bytes"], 1_769_856);
    assert_eq!(
        r["plane_elements"],
        json!({"codes13":1048576,"codes2":524288,
        "scales13":131072,"scales2":65536,"gscales13":32,"gscales2":64})
    );
    for field in [
        "codes13",
        "codes2",
        "scales13",
        "scales2",
        "gscales13",
        "gscales2",
    ] {
        assert!(r.get(field).is_none());
    }
    assert!(bytes < 4096, "{bytes}");
}
#[test]
fn inspector_preserves_import_rejections_and_empty_rank_semantics() {
    let mut request: Value = serde_json::from_str(include_str!(
        "../fixtures/resident_import_qwen_expert0.json"
    ))
    .unwrap();
    request["spec"]["expected_metadata"]["license"] = json!("wrong");
    let (ok, out, _) = invoke(request.clone());
    assert!(!ok);
    // Compare the established import contract instead of inventing a new error code.
    let (old_ok, old_error, _) = invoke_command("resident-import", request.clone());
    assert!(!old_ok);
    assert_eq!(out, old_error);
    request["spec"]["expected_metadata"]["license"] = json!("Apache-2.0");
    request["spec"]["registry"]["copies"] = json!([]);
    request["spec"]["resident_order"] = json!([]);
    request["spec"]["experts"] = json!([]);
    let (ok, out, bytes) = invoke(request);
    assert!(ok, "{out}");
    assert_eq!(out["result"]["shape"], json!([0, 0, 0, 0, 0]));
    assert_eq!(out["result"]["payload_bytes"], 0);
    assert!(bytes < 4096);
}

#[cfg(unix)]
#[test]
fn closed_output_pipe_never_reports_success_for_partial_snapshot() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut child = Command::new(env!("CARGO_BIN_EXE_vf"))
        .current_dir(root)
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
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("JSON output failed"));
}
