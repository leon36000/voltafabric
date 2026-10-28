//! One real native-worker transport truncation, isolated behind a lab-only proxy.
#![cfg(target_os = "linux")]

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use voltafabric::moe::{Choice, Registry, Replica, Stamp};
use voltafabric::preweighted::Collector;
use voltafabric::process::ReceiptState;
use voltafabric::routing::RoutingProblem;
use voltafabric::session::{mxfp4, Endpoint, Session};
use voltafabric::sha256;

const RANK: u32 = 7;
const HIDDEN: usize = 32;
const EXPERTS: [u32; 3] = [17, 93, 241];
const LOCALS: [u32; 3] = [4, 1, 9];
const TIMEOUT: Duration = Duration::from_secs(30);

fn require_reserved_gpu() {
    assert_eq!(
        std::env::var("VF_RUN_GPU_TESTS").as_deref(),
        Ok("1"),
        "set VF_RUN_GPU_TESTS=1 only on the reserved native GPU runner"
    );
    let identity = env!("VF_MXFP4_LIBRARY_SHA256");
    assert_eq!(identity.len(), 64);
    assert!(identity
        .bytes()
        .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
}

fn fixture_dir() -> PathBuf {
    std::env::var_os("VF_FIXTURE_DIR").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures"),
        PathBuf::from,
    )
}

fn endpoint() -> Endpoint {
    let worker = std::env::var("VF_GPU_WORKER")
        .unwrap_or_else(|_| env!("CARGO_BIN_EXE_vf-mxfp4-gpu-worker").to_owned());
    let proxy = std::env::var("VF_TRUNCATION_PROXY").unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../.runs/persistent-gpu-worker-20261024-r14/truncation_proxy.py")
            .display()
            .to_string()
    });
    Endpoint {
        rank: RANK,
        argv: vec!["python3".into(), "-B".into(), proxy, worker],
    }
}

fn producer() -> mxfp4::Producer {
    mxfp4::Producer::GpuR13ResidentFp32V1 {
        library_sha256: env!("VF_MXFP4_LIBRARY_SHA256").into(),
    }
}

fn artifact(epoch: u64) -> mxfp4::Artifact {
    let bytes = fs::read(fixture_dir().join("preweighted_mxfp4_tiny.safetensors"))
        .expect("canonical tiny artifact must be present");
    let binding = mxfp4::Binding {
        artifact_sha256: sha256::hex(&bytes),
        model_revision: "r13-tiny-independent-golden".into(),
        rank: RANK,
        layer: 0,
        epoch,
        experts: EXPERTS
            .iter()
            .zip(LOCALS)
            .map(|(&expert, local)| mxfp4::Expert { expert, local })
            .collect(),
    };
    mxfp4::Artifact::new(bytes, binding).expect("canonical tiny shard must validate")
}

fn f32_bits(values: &[serde_json::Value]) -> Vec<u32> {
    values
        .iter()
        .map(|value| {
            u32::from_str_radix(value.as_str().expect("golden FP32 bit string"), 16)
                .expect("eight-digit FP32 bit string")
        })
        .collect()
}

fn audit_records(stderr: &str, kind: &str) -> Vec<serde_json::Value> {
    stderr
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|value| value.get("gpu_audit").and_then(|v| v.as_str()) == Some(kind))
        .collect()
}

#[test]
#[ignore = "requires reserved native GPU and VF_RUN_GPU_TESTS=1"]
fn native_worker_partial_completion_body_is_unknown_and_never_published() {
    require_reserved_gpu();
    let artifact = artifact(70);
    let golden: serde_json::Value = serde_json::from_slice(
        &fs::read(fixture_dir().join("preweighted_mxfp4_tiny_golden.json"))
            .expect("canonical independent golden must be present"),
    )
    .unwrap();
    let input = f32_bits(
        golden["inputs"]["pattern"]["bf16_as_f32_bits"]
            .as_array()
            .unwrap(),
    );
    assert_eq!(input.len(), HIDDEN);
    let problem = RoutingProblem {
        hidden: HIDDEN,
        experts: EXPERTS.to_vec(),
        routes: vec![vec![Choice {
            expert: EXPERTS[0],
            weight: 0.375,
        }]],
        registry: Registry {
            epoch: 70,
            copies: EXPERTS
                .iter()
                .zip(LOCALS)
                .map(|(&expert, local)| Replica {
                    expert,
                    rank: RANK,
                    local,
                })
                .collect(),
        },
        stamp: Stamp {
            run_id: "r14-native-gpu-truncation".into(),
            model_revision: "r13-tiny-independent-golden".into(),
            plan_id: "one-slot-partial-body".into(),
            layer: 0,
            batch: 0,
            epoch: 70,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(RANK, LOCALS.to_vec())]),
    };
    let mut worker = Session::connect_mxfp4(&endpoint(), TIMEOUT, &producer()).unwrap();
    assert!(
        worker
            .load_mxfp4_as(&artifact, 70, TIMEOUT, &producer())
            .unwrap()
            .transferred
    );
    let mut collector = Collector::new(&problem).unwrap();
    let pending = worker
        .start_preweighted_fp32(
            &problem,
            &collector,
            1,
            vec![mxfp4::TokenInput {
                token: 0,
                value_bits: input,
            }],
            voltafabric::process::Fault::None,
        )
        .unwrap();
    let receipt = worker.finish_preweighted_fp32(pending, &mut collector, Instant::now() + TIMEOUT);
    assert!(receipt.started, "native worker must pass the Started gate");
    assert_eq!(receipt.state, ReceiptState::Unknown, "{receipt:?}");
    assert!(receipt.error.is_some(), "{receipt:?}");
    assert_eq!(receipt.expected_slots, 1);
    assert_eq!(receipt.received_slots, 0);
    assert!(
        collector.finish().is_err(),
        "partial body must expire the collector"
    );
    assert_eq!(
        worker.traffic.execs, 1,
        "the coordinator must not retry or fall back"
    );

    // The proxy privately drains the real body, consumes Finished, sends native
    // SHUTDOWN and exits; the coordinator itself never receives BYE.
    let close = worker.close(TIMEOUT);
    assert_eq!(close.exit_code, Some(0), "{close:?}");
    assert_eq!(close.bye_execs, None);
    let truncations = close
        .stderr
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|value| value.get("truncation_proxy").is_some())
        .collect::<Vec<_>>();
    assert_eq!(
        truncations.len(),
        1,
        "proxy must report its bounded cut and child exit"
    );
    let marker = &truncations[0]["truncation_proxy"];
    assert_eq!(marker["forwarded_bytes"], 17);
    assert!(marker["declared_bytes"].as_u64().unwrap() > 17);
    assert_eq!(marker["child_exit"], 0);
    let evicted = audit_records(&close.stderr, "evicted");
    assert_eq!(evicted.len(), 1);
    assert_eq!(evicted[0]["lifecycle"], serde_json::json!([0, 0, 0]));
    let owners = evicted[0]["experts"].as_array().unwrap();
    assert_eq!(owners.len(), 3);
    let calls = owners
        .iter()
        .map(|owner| owner["exec_calls"].as_u64().unwrap())
        .sum::<u64>();
    let kernels = owners
        .iter()
        .map(|owner| owner["kernel_launches"].as_u64().unwrap())
        .sum::<u64>();
    assert_eq!(calls, 1);
    assert!(kernels > 0);
    assert!(owners.iter().all(|owner| owner["poisoned"] == 0));
    println!(
        "{{\"test\":\"partial_completion_body\",\"receipt\":\"unknown\",\"received_slots\":0,\"execs\":{},\"gpu_kernel_launches\":{},\"cut_bytes\":17,\"child_exit\":0,\"lifecycle\":[0,0,0]}}",
        calls, kernels
    );
}
