mod common;
use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};
use voltafabric::{
    moe,
    process::{Fault, ProcessProblem},
};

fn invoke(p: &ProcessProblem) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_vf"))
        .args(["rank-run", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(p).unwrap())
        .unwrap();
    child.wait_with_output().unwrap()
}
#[test]
fn cli_reports_receipts_and_exact_cpu_output() {
    let p = ProcessProblem {
        problem: common::problem(),
        faults: BTreeMap::new(),
        timeout_ms: 5000,
        recovery: false,
    };
    let output = invoke(&p);
    assert!(output.status.success());
    let data: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(data["ok"], true);
    let report: voltafabric::process::ProcessReport =
        serde_json::from_value(data["result"].clone()).unwrap();
    assert_eq!(report.output.unwrap(), moe::reference(&p.problem).unwrap());
    assert!(report
        .receipts
        .iter()
        .all(|r| r.reaped && r.children_reaped == r.children_spawned));
}
#[test]
fn cli_failed_wave_is_nonzero_and_retains_uncertain_receipts_without_output() {
    let mut p = ProcessProblem {
        problem: common::problem(),
        faults: BTreeMap::new(),
        timeout_ms: 5000,
        recovery: false,
    };
    p.faults.insert(2, Fault::DieAfterCompute);
    let output = invoke(&p);
    assert_eq!(output.status.code(), Some(2));
    let data: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(data["ok"], false);
    assert_eq!(data["result"]["output"], serde_json::Value::Null);
    assert_eq!(data["result"]["ok"], false);
    let report: voltafabric::process::ProcessReport =
        serde_json::from_value(data["result"].clone()).unwrap();
    assert!(report.receipts.iter().all(|r| r.reaped));
    assert!(report
        .receipts
        .iter()
        .any(|r| r.state == voltafabric::process::ReceiptState::Unknown));
}
#[test]
fn json_global_factor_preserves_all_bits() {
    let p: serde_json::Value = serde_json::from_str(r#"{"factor":0.0037869367688414092}"#).unwrap();
    assert_eq!(
        p["factor"].as_f64().unwrap().to_bits(),
        0.0037869367688414092_f64.to_bits()
    );
}

// A40 guard sweep: the CLI argument and input-size contract. With each clause
// disabled, vf either panicked on args[0], ignored an extra argument, accepted
// input for `capabilities`, or reported a JSON error instead of the size bound.
fn vf(args: &[&str], stdin: &[u8]) -> (i32, serde_json::Value) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_vf"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let _ = input.write_all(stdin);
    drop(input);
    let output = child.wait_with_output().unwrap();
    let json = serde_json::from_slice(&output.stdout).unwrap_or(serde_json::Value::Null);
    (output.status.code().unwrap_or(-1), json)
}
fn usage_error(args: &[&str], stdin: &[u8], message: &str) {
    let (code, json) = vf(args, stdin);
    assert_eq!(code, 2, "{args:?}: {json}");
    assert_eq!(json["ok"], false, "{args:?}");
    assert_eq!(json["error"]["code"], "USAGE", "{args:?}: {json}");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .starts_with(message),
        "{args:?}: {json}"
    );
}

fn raw_fp16_completion_header() -> serde_json::Value {
    let stamp = serde_json::json!({
        "run_id": "raw-cli-run",
        "model_revision": "raw-cli-model",
        "plan_id": "raw-cli-plan",
        "layer": 3,
        "batch": 4,
        "epoch": 12
    });
    let plane = |rows: u64, columns: u64| {
        serde_json::json!({
            "rows": rows,
            "columns": columns,
            "codes": vec![0x22_u8; (rows * columns / 2) as usize],
            "scales": vec![0x38_u8; (rows * columns / 16) as usize],
            "global_dequant": 1.0
        })
    };
    serde_json::json!({
        "launch": {
            "weights": {
                "model_revision": "raw-cli-model",
                "layer": 3,
                "rank": 17,
                "registry": {
                    "epoch": 12,
                    "copies": [{"expert": 23, "rank": 17, "local": 3}]
                },
                "resident_order": [3],
                "experts": [{
                    "id": 23,
                    "local": 3,
                    "gate": plane(128, 64),
                    "up": plane(128, 64),
                    "down": plane(64, 128)
                }]
            },
            "dispatch": {
                "stamp": stamp,
                "rank": 17,
                "hidden": 64,
                "tokens": 1,
                "resident_order": [3],
                "slots": [{
                    "slot": {
                        "token": 0,
                        "slot": 0,
                        "expert": 23,
                        "rank": 17,
                        "local": 3,
                        "weight": 1.0
                    },
                    "physical_expert": 0
                }]
            },
            "expected_stamp": stamp,
            "options": {
                "splitk13": 4,
                "splitk2": 4,
                "max_workspace_bytes": 134217728
            }
        },
        "output": {
            "stamp": stamp,
            "rank": 17,
            "rows": 1,
            "columns": 64
        }
    })
}

fn raw_fp16_wire(header: &serde_json::Value, body: &[u8]) -> Vec<u8> {
    let mut wire = serde_json::to_vec(header).unwrap();
    wire.push(b'\n');
    wire.extend_from_slice(body);
    wire
}

#[test]
fn raw_fp16_completion_cli_matches_json_and_refuses_bad_binary_bodies() {
    let header = raw_fp16_completion_header();
    let body: Vec<u8> = (0..64).flat_map(|_| 0x3c00_u16.to_le_bytes()).collect();
    let (legacy_code, legacy) = vf(
        &["resident-completions", "-"],
        &serde_json::to_vec(&serde_json::json!({
            "launch": header["launch"],
            "output": {
                "stamp": header["output"]["stamp"],
                "rank": 17,
                "rows": 1,
                "columns": 64,
                "fp16_bits": vec![0x3c00_u16; 64]
            }
        }))
        .unwrap(),
    );
    assert_eq!(legacy_code, 0, "{legacy}");

    let wire = raw_fp16_wire(&header, &body);
    let (code, raw) = vf(&["resident-completions", "-"], &wire);
    assert_eq!(code, 0, "{raw}");
    assert_eq!(
        raw["result"]["completions"],
        legacy["result"]["completions"]
    );
    assert_eq!(raw["result"]["completions"][0]["value"][0], 1.0);

    let (code, truncated) = vf(
        &["resident-completions", "-"],
        &raw_fp16_wire(&header, &body[..body.len() - 1]),
    );
    assert_eq!(code, 2, "{truncated}");
    assert_eq!(truncated["error"]["code"], "RAW_OUTPUT_TRUNCATED");

    let mut trailing_body = body.clone();
    trailing_body.push(0);
    let (code, trailing) = vf(
        &["resident-completions", "-"],
        &raw_fp16_wire(&header, &trailing_body),
    );
    assert_eq!(code, 2, "{trailing}");
    assert_eq!(trailing["error"]["code"], "RAW_OUTPUT_TRAILING_DATA");

    let mut nonfinite_body = body;
    nonfinite_body[..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
    let (code, nonfinite) = vf(
        &["resident-completions", "-"],
        &raw_fp16_wire(&header, &nonfinite_body),
    );
    assert_eq!(code, 2, "{nonfinite}");
    assert_eq!(nonfinite["error"]["code"], "NON_FINITE_RESULT");
}

#[test]
fn cli_argument_contract_is_enforced_before_any_input_is_read() {
    usage_error(&[], b"", "vf <");
    usage_error(&["layout", "-", "extra"], b"{}", "vf <");
    usage_error(&["capabilities", "-"], b"", "capabilities takes no input");
}
#[test]
fn cli_input_over_16_mib_is_refused_by_its_size_bound() {
    let (code, json) = vf(&["layout", "-"], &vec![b' '; 16 * 1024 * 1024 + 1]);
    assert_eq!(code, 2, "{json}");
    assert_eq!(json["error"]["code"], "INPUT_TOO_LARGE", "{json}");
}

#[cfg(unix)]
#[test]
fn cli_names_a_non_utf8_or_unknown_command() {
    // A41 reason sweep: neither refusal was reached by any test before.
    use std::os::unix::ffi::OsStrExt;
    let output = Command::new(env!("CARGO_BIN_EXE_vf"))
        .arg(std::ffi::OsStr::from_bytes(b"\xff"))
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["error"]["message"], "non-UTF8 command", "{json}");
    let (code, json) = vf(&["no-such-command", "-"], b"{}");
    assert_eq!(code, 2);
    assert_eq!(json["error"]["code"], "USAGE");
    assert_eq!(
        json["error"]["message"],
        "unsupported command: no-such-command"
    );
}

#[test]
fn cli_usage_names_every_command_and_only_commands_it_accepts() {
    // A53-R31: A53 added `session-run-compressed` without naming it in the usage message,
    // the CLI's only help. Every top-level command arm of src/main.rs must be named there,
    // and every command named there must be accepted (refused on its input, not as USAGE).
    let (_, json) = vf(&[], b"");
    let message = json["error"]["message"].as_str().unwrap().to_string();
    let named: Vec<&str> = message
        .strip_prefix("vf <")
        .and_then(|rest| rest.split_once('>'))
        .map(|(list, _)| list.split('|').collect())
        .unwrap();
    let mut dispatched: Vec<&str> = include_str!("../src/main.rs")
        .lines()
        .filter_map(|line| line.strip_prefix("        \""))
        .filter_map(|rest| rest.split_once("\" =>").map(|(name, _)| name))
        .collect();
    dispatched.push("capabilities");
    for command in &dispatched {
        assert!(
            named.contains(command),
            "`{command}` missing from: {message}"
        );
    }
    for command in named.iter().filter(|c| **c != "capabilities") {
        let (code, json) = vf(&[command, "-"], b"{}");
        assert_eq!(code, 2, "{command}: {json}");
        assert_ne!(json["error"]["code"], "USAGE", "{command}: {json}");
    }
    assert!(dispatched.contains(&"session-run-compressed"));
}
