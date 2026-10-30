//! Native GPU worker acceptance tests.
//!
//! These tests require the reserved hardware runner and the exact shared-library
//! identity supplied at compile time. They are intentionally never part of the
//! ordinary CPU test suite.
#![cfg(target_os = "linux")]

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use voltafabric::moe::{Choice, Registry, Replica, Stamp};
use voltafabric::preweighted::Collector;
use voltafabric::process::{Fault, ReceiptState};
use voltafabric::routing::RoutingProblem;
use voltafabric::session::{self, mxfp4, Endpoint, Session};
use voltafabric::sha256;

const RANK: u32 = 7;
const LAYER: u32 = 0;
const HIDDEN: usize = 32;
const TIMEOUT: Duration = Duration::from_secs(30);
const EXPERTS: [u32; 3] = [17, 93, 241];
// These logical locals intentionally differ from the experts' order.
const LOCALS: [u32; 3] = [4, 1, 9];

fn test_run_id() -> String {
    let value = std::env::var("VF_GPU_TEST_RUN_ID")
        .expect("VF_GPU_TEST_RUN_ID must supply a fresh qualification namespace");
    assert!(
        value.starts_with("r53-")
            && (12..=64).contains(&value.len())
            && value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
        "VF_GPU_TEST_RUN_ID is outside the declared fresh-namespace syntax"
    );
    value
}

fn require_reserved_gpu() {
    let _fresh_id = test_run_id();
    assert_eq!(
        std::env::var("VF_RUN_GPU_TESTS").as_deref(),
        Ok("1"),
        "set VF_RUN_GPU_TESTS=1 only on the reserved native GPU runner"
    );
    let identity = env!("VF_MXFP4_LIBRARY_SHA256");
    assert_eq!(
        identity.len(),
        64,
        "compiled GPU library identity is malformed"
    );
    assert!(
        identity
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "compiled GPU library identity must be lowercase hexadecimal"
    );
}

fn endpoint() -> Endpoint {
    let worker = std::env::var("VF_GPU_WORKER")
        .unwrap_or_else(|_| env!("CARGO_BIN_EXE_vf-mxfp4-gpu-worker").to_owned());
    Endpoint {
        rank: RANK,
        argv: vec![worker, "--audit-counters".into()],
    }
}

fn producer() -> mxfp4::Producer {
    mxfp4::Producer::GpuR13ResidentFp32V1 {
        library_sha256: env!("VF_MXFP4_LIBRARY_SHA256").into(),
    }
}

fn fixture_dir() -> PathBuf {
    std::env::var_os("VF_FIXTURE_DIR").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures"),
        PathBuf::from,
    )
}

fn artifact(epoch: u64) -> mxfp4::Artifact {
    let bytes = fs::read(fixture_dir().join("preweighted_mxfp4_tiny.safetensors"))
        .expect("canonical tiny artifact must be present");
    let binding = mxfp4::Binding {
        artifact_sha256: sha256::hex(&bytes),
        model_revision: "r13-tiny-independent-golden".into(),
        rank: RANK,
        layer: LAYER,
        epoch,
        experts: EXPERTS
            .iter()
            .zip(LOCALS)
            .map(|(&expert, local)| mxfp4::Expert { expert, local })
            .collect(),
    };
    mxfp4::Artifact::new(bytes, binding).expect("canonical tiny shard must validate")
}

fn golden() -> serde_json::Value {
    let bytes = fs::read(fixture_dir().join("preweighted_mxfp4_tiny_golden.json"))
        .expect("canonical independent golden must be present");
    let value: serde_json::Value = serde_json::from_slice(&bytes).expect("golden JSON must parse");
    let artifact = fs::read(fixture_dir().join("preweighted_mxfp4_tiny.safetensors"))
        .expect("canonical tiny artifact must be present");
    assert_eq!(value["case_count"], 45);
    assert_eq!(value["cases"].as_array().unwrap().len(), 45);
    assert_eq!(value["weights_sha256"], sha256::hex(&artifact));
    value
}

fn bits(values: &[serde_json::Value]) -> Vec<u32> {
    values
        .iter()
        .map(|s| {
            u32::from_str_radix(s.as_str().expect("hex string"), 16)
                .expect("golden uses 8-digit FP32 hex")
        })
        .collect()
}

fn problem(batch: u64, epoch: u64, routes: Vec<Vec<Choice>>) -> RoutingProblem {
    let mut copies: Vec<_> = EXPERTS
        .iter()
        .zip(LOCALS)
        .map(|(&expert, local)| Replica {
            expert,
            rank: RANK,
            local,
        })
        .collect();
    // A second placement proves owner selection is explicit and still resolves to
    // the loaded local on rank 7; this test does not silently require another worker.
    copies.push(Replica {
        expert: EXPERTS[0],
        rank: 8,
        local: 44,
    });
    RoutingProblem {
        hidden: HIDDEN,
        experts: EXPERTS.to_vec(),
        routes,
        registry: Registry { epoch, copies },
        stamp: Stamp {
            run_id: test_run_id(),
            model_revision: "r13-tiny-independent-golden".into(),
            plan_id: "tiny-45-independent-cases".into(),
            layer: LAYER,
            batch,
            epoch,
        },
        preferred_owners: BTreeMap::from([(EXPERTS[0], RANK)]),
        resident_order: BTreeMap::from([(RANK, LOCALS.to_vec()), (8, vec![44])]),
    }
}

fn golden_batch(
    g: &serde_json::Value,
    batch: u64,
    epoch: u64,
) -> (RoutingProblem, Vec<mxfp4::TokenInput>, Vec<Vec<f32>>) {
    let mut routes = Vec::new();
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for case in g["cases"].as_array().unwrap() {
        let expert = case["expert"].as_u64().unwrap() as u32;
        let coefficient = case["coefficient"].as_f64().unwrap();
        let pattern = case["input"].as_str().unwrap();
        let input = bits(g["inputs"][pattern]["bf16_as_f32_bits"].as_array().unwrap());
        assert_eq!(input.len(), HIDDEN);
        routes.push(
            EXPERTS
                .iter()
                .map(|&routed_expert| Choice {
                    expert: routed_expert,
                    weight: if routed_expert == expert {
                        coefficient
                    } else {
                        0.0
                    },
                })
                .collect(),
        );
        inputs.push(mxfp4::TokenInput {
            token: routes.len() - 1,
            value_bits: input,
        });
        expected.push(
            bits(case["output_bits"].as_array().unwrap())
                .into_iter()
                .map(f32::from_bits)
                .collect(),
        );
    }

    // One separate multi-route token checks the collector's ordered FP32 sum.
    let pattern = "pattern";
    let coefficient = 0.375_f64;
    let token = routes.len();
    routes.push(
        EXPERTS
            .iter()
            .map(|&expert| Choice {
                expert,
                weight: coefficient,
            })
            .collect(),
    );
    inputs.push(mxfp4::TokenInput {
        token,
        value_bits: bits(g["inputs"][pattern]["bf16_as_f32_bits"].as_array().unwrap()),
    });
    let mut sum = vec![0.0_f32; HIDDEN];
    for &expert in &EXPERTS {
        let case = g["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| {
                case["expert"].as_u64() == Some(u64::from(expert))
                    && case["coefficient"].as_f64() == Some(coefficient)
                    && case["input"].as_str() == Some(pattern)
            })
            .unwrap();
        for (acc, value) in sum
            .iter_mut()
            .zip(bits(case["output_bits"].as_array().unwrap()))
        {
            *acc += f32::from_bits(value);
        }
    }
    expected.push(sum);
    (problem(batch, epoch, routes), inputs, expected)
}

fn assert_unpublished(collector: Collector) {
    assert!(
        collector.finish().is_err(),
        "failed or uncertain attempt must not expose staged output"
    );
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
fn native_worker_matches_all_45_independent_golden_slots_twice_and_sums_routes() {
    require_reserved_gpu();
    let fixture = golden();
    let artifact = artifact(20);
    let mut worker = Session::connect_mxfp4(&endpoint(), TIMEOUT, &producer()).unwrap();
    r70_observe(&worker);
    assert!(
        worker
            .load_mxfp4_as(&artifact, 20, TIMEOUT, &producer())
            .unwrap()
            .transferred
    );
    let weight_bytes = worker.traffic.weight_bytes_sent;

    for batch in 0..2 {
        let (p, inputs, expected) = golden_batch(&fixture, batch, 20);
        let mut collector = Collector::new(&p).unwrap();
        let pending = worker
            .start_preweighted_fp32(&p, &collector, batch + 1, inputs, Fault::None)
            .unwrap();
        let receipt =
            worker.finish_preweighted_fp32(pending, &mut collector, Instant::now() + TIMEOUT);
        assert_eq!(receipt.state, ReceiptState::Completed, "{receipt:?}");
        assert!(receipt.error.is_none(), "{receipt:?}");
        assert_eq!(receipt.expected_slots, 138);
        assert_eq!(receipt.received_slots, 138);
        let actual = collector.finish().unwrap();
        assert_eq!(
            actual
                .iter()
                .map(|row| row.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|row| row.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            "collector must sum the independent per-slot oracle values in route order"
        );
        assert_eq!(worker.traffic.weight_bytes_sent, weight_bytes);
    }

    assert_eq!(worker.traffic.loads_transferred, 1);
    assert_eq!(worker.traffic.execs, 2);
    // SHUTDOWN itself must evict the resident native owners before BYE.
    let close = worker.close(TIMEOUT);
    assert_eq!(close.exit_code, Some(0), "{close:?}");
    assert_eq!(close.bye_execs, Some(2));
    let startup = audit_records(&close.stderr, "startup");
    assert_eq!(startup.len(), 1);
    assert_eq!(
        startup[0]["library_sha256"],
        env!("VF_MXFP4_LIBRARY_SHA256")
    );
    let evicted = audit_records(&close.stderr, "evicted");
    assert_eq!(
        evicted.len(),
        1,
        "stderr must retain the EVICT audit record"
    );
    let record = &evicted[0];
    assert_eq!(
        record["producer"],
        serde_json::to_value(producer()).unwrap()
    );
    assert_eq!(record["lifecycle"], serde_json::json!([0, 0, 0]));
    assert!(record["cuda_api_calls"].as_u64().unwrap() > 0);
    let owners = record["experts"].as_array().expect("audit owners array");
    assert_eq!(owners.len(), 3);
    for expert in EXPERTS {
        let owner = owners
            .iter()
            .find(|row| row["expert"] == expert)
            .expect("owner counter");
        assert_eq!(
            owner["weight_upload_copies"], 6,
            "one upload for each of six planes"
        );
        assert!(owner["weight_upload_bytes"].as_u64().unwrap() > 0);
        assert_eq!(owner["exec_calls"], 92);
        assert!(owner["kernel_launches"].as_u64().unwrap() > 0);
        assert_eq!(owner["poisoned"], 0);
    }
    println!("{{\"test\":\"tiny45_twice\",\"execs\":2,\"slots_per_exec\":138,\"upload_plane_copies_per_expert\":6,\"kernel_launches\":[{},{},{}],\"cuda_api_calls\":{}}}",
        owners[0]["kernel_launches"], owners[1]["kernel_launches"], owners[2]["kernel_launches"], record["cuda_api_calls"]);
}

#[test]
#[ignore = "requires reserved native GPU and VF_RUN_GPU_TESTS=1"]
fn native_worker_rebind_preserves_loaded_owner_and_never_resends_weights() {
    require_reserved_gpu();
    let fixture = golden();
    let artifact = artifact(30);
    let mut worker = Session::connect_mxfp4(&endpoint(), TIMEOUT, &producer()).unwrap();
    r70_observe(&worker);
    let first = worker
        .load_mxfp4_as(&artifact, 30, TIMEOUT, &producer())
        .unwrap();
    assert!(first.transferred);
    let bytes_after_load = worker.traffic.weight_bytes_sent;
    let rebound = worker
        .load_mxfp4_as(&artifact, 31, TIMEOUT, &producer())
        .unwrap();
    assert!(!rebound.transferred);
    assert_eq!(rebound.artifact_sha256, first.artifact_sha256);
    assert_eq!(worker.traffic.weight_bytes_sent, bytes_after_load);
    assert_eq!(worker.traffic.loads_transferred, 1);
    assert_eq!(worker.traffic.rebinds, 1);

    let (p, inputs, expected) = golden_batch(&fixture, 1, 31);
    let mut collector = Collector::new(&p).unwrap();
    let pending = worker
        .start_preweighted_fp32(&p, &collector, 1, inputs, Fault::None)
        .unwrap();
    let receipt = worker.finish_preweighted_fp32(pending, &mut collector, Instant::now() + TIMEOUT);
    assert_eq!(receipt.state, ReceiptState::Completed, "{receipt:?}");
    let actual = collector.finish().unwrap();
    assert_eq!(
        actual
            .iter()
            .map(|row| row.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|row| row.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    );
    assert_eq!(worker.traffic.weight_bytes_sent, bytes_after_load);
    worker.evict(TIMEOUT).unwrap();
    let close = worker.close(TIMEOUT);
    assert_eq!(close.exit_code, Some(0), "{close:?}");
    let evicted = audit_records(&close.stderr, "evicted");
    assert_eq!(evicted.len(), 1);
    assert_eq!(evicted[0]["lifecycle"], serde_json::json!([0, 0, 0]));
    for owner in evicted[0]["experts"].as_array().unwrap() {
        assert_eq!(
            owner["weight_upload_copies"], 6,
            "one upload for each of six planes"
        );
        assert_eq!(owner["exec_calls"], 46);
    }
    println!("{{\"test\":\"weightless_rebind\",\"loads_transferred\":1,\"rebinds\":1,\"weight_bytes\":{}}}", bytes_after_load);
}

#[test]
#[ignore = "requires reserved native GPU and VF_RUN_GPU_TESTS=1"]
fn native_worker_rejects_stale_duplicate_and_missing_rows_atomically() {
    require_reserved_gpu();
    let fixture = golden();
    for (attempt, fault) in [Fault::StaleEpoch, Fault::Duplicate, Fault::Missing]
        .into_iter()
        .enumerate()
    {
        let artifact = artifact(40 + attempt as u64);
        let mut worker = Session::connect_mxfp4(&endpoint(), TIMEOUT, &producer()).unwrap();
    r70_observe(&worker);
        worker
            .load_mxfp4_as(&artifact, 40 + attempt as u64, TIMEOUT, &producer())
            .unwrap();
        let (p, inputs, _) = golden_batch(&fixture, attempt as u64, 40 + attempt as u64);
        let mut collector = Collector::new(&p).unwrap();
        let pending = worker
            .start_preweighted_fp32(&p, &collector, attempt as u64 + 1, inputs, fault)
            .unwrap();
        let receipt =
            worker.finish_preweighted_fp32(pending, &mut collector, Instant::now() + TIMEOUT);
        assert!(
            receipt.started,
            "fault {attempt} must pass the native start gate"
        );
        assert_ne!(
            receipt.state,
            ReceiptState::Completed,
            "fault {attempt}: {receipt:?}"
        );
        assert!(receipt.error.is_some(), "fault {attempt}: {receipt:?}");
        assert_unpublished(collector);
        // Invalid completion identity/cardinality poisons the typed stream; close
        // must reap the native process even when graceful EVICT is no longer safe.
        let close = worker.close(Duration::from_secs(2));
        assert!(close.exit_code.is_some(), "fault {attempt}: {close:?}");
        println!("{{\"test\":\"identity_or_cardinality_refusal\",\"case\":{},\"state\":\"{:?}\",\"received\":{}}}", attempt, receipt.state, receipt.received_slots);
    }
}

#[test]
#[ignore = "requires reserved native GPU and VF_RUN_GPU_TESTS=1"]
fn native_worker_crash_receipts_never_publish_partial_or_unknown_outputs() {
    require_reserved_gpu();
    let fixture = golden();
    for (attempt, fault) in [Fault::DieAfterStart, Fault::DieAfterCompute]
        .into_iter()
        .enumerate()
    {
        let artifact = artifact(50 + attempt as u64);
        let mut worker = Session::connect_mxfp4(&endpoint(), TIMEOUT, &producer()).unwrap();
    r70_observe(&worker);
        worker
            .load_mxfp4_as(&artifact, 50 + attempt as u64, TIMEOUT, &producer())
            .unwrap();
        let (p, inputs, _) = golden_batch(&fixture, attempt as u64, 50 + attempt as u64);
        let mut collector = Collector::new(&p).unwrap();
        let pending = worker
            .start_preweighted_fp32(&p, &collector, attempt as u64 + 1, inputs, fault)
            .unwrap();
        let receipt =
            worker.finish_preweighted_fp32(pending, &mut collector, Instant::now() + TIMEOUT);
        assert!(
            receipt.started,
            "crash case {attempt} must have a Started receipt"
        );
        assert_ne!(
            receipt.state,
            ReceiptState::Completed,
            "crash case {attempt}: {receipt:?}"
        );
        assert!(receipt.error.is_some(), "crash case {attempt}: {receipt:?}");
        assert_unpublished(collector);
        // The worker exits from the injected crash; close must reap it. GPU audit
        // counters are intentionally not asserted here because no EVICT ran.
        let close = worker.close(TIMEOUT);
        assert!(close.exit_code.is_some(), "crash case {attempt}: {close:?}");
        assert!(
            close.exit_code != Some(0),
            "faulted worker unexpectedly survived"
        );
        println!(
            "{{\"test\":\"crash_atomicity\",\"case\":{},\"state\":\"{:?}\",\"started\":{}}}",
            attempt, receipt.state, receipt.started
        );
    }
}

#[test]
#[ignore = "requires reserved native GPU and VF_RUN_GPU_TESTS=1"]
fn native_load_numeric_refusal_rolls_back_each_partial_owner_prefix() {
    require_reserved_gpu();
    let declared: Vec<String> = serde_json::from_str(
        &std::env::var("VF_R72_ROLLBACK_SHA256").expect("predeclared numeric LOAD hashes required"),
    ).expect("numeric LOAD hashes must be JSON strings");
    assert_eq!(declared.len(), 3, "exactly three numeric LOAD hashes required");
    assert!(declared.iter().all(|s| s.len()==64 && s.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())), "malformed numeric LOAD hash");

    for (position, &bad_expert) in EXPERTS.iter().enumerate() {
        let epoch = 60 + position as u64;
        let mut bytes = fs::read(fixture_dir().join("preweighted_mxfp4_tiny.safetensors"))
            .expect("canonical tiny artifact must be present");
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let data_start = 8_usize.checked_add(header_len).expect("header bound");
        assert!(
            data_start <= bytes.len(),
            "SafeTensors header must fit the file"
        );
        let header: serde_json::Value =
            serde_json::from_slice(&bytes[8..data_start]).expect("SafeTensors header JSON");
        let weight_name = format!("layers.{LAYER}.ffn.experts.{bad_expert}.w2.weight");
        let scale_name = format!("layers.{LAYER}.ffn.experts.{bad_expert}.w2.scale");
        let (weight_index, scale_index) = {
            let data_offset = |name: &str| -> usize {
                let start = header[name]["data_offsets"][0]
                    .as_u64()
                    .expect("tensor has a numeric start offset")
                    as usize;
                let index = data_start.checked_add(start).expect("tensor offset bound");
                assert!(
                    index < bytes.len(),
                    "tensor must lie in the SafeTensors data section"
                );
                index
            };
            (data_offset(&weight_name), data_offset(&scale_name))
        };
        // E2M1 code 4 has magnitude 2; E8M0 254 is finite in the CPU parser's
        // f64 domain, but its exact product 2 * 2^127 overflows an RN32 decode.
        bytes[weight_index] = (bytes[weight_index] & 0xf0) | 0x04;
        bytes[scale_index] = 254;
        let bad_artifact = mxfp4::Artifact::new(
            bytes.clone(),
            mxfp4::Binding {
                artifact_sha256: sha256::hex(&bytes),
                model_revision: "r13-tiny-independent-golden".into(),
                rank: RANK,
                layer: LAYER,
                epoch,
                experts: EXPERTS
                    .iter()
                    .zip(LOCALS)
                    .map(|(&expert, local)| mxfp4::Expert { expert, local })
                    .collect(),
            },
        )
        .expect("CPU FP64 artifact admission accepts this RN32-only overflow case");
        let valid = artifact(epoch);
        assert_eq!(bad_artifact.binding().artifact_sha256, declared[position], "predeclared bad artifact SHA mismatch");
        println!("{}", serde_json::json!({"r72_bad_load_hash":true,"position":position,"epoch":epoch,"expert":bad_expert,"bad_sha256":bad_artifact.binding().artifact_sha256,"good_sha256":valid.binding().artifact_sha256}));


        let worker_path = endpoint().argv.remove(0);
        let mut child = Command::new(worker_path)
            .arg("--audit-counters")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("start the actual GPU worker executable");
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        let send =
            |stdin: &mut std::process::ChildStdin, request: &session::Request, body: &[u8]| {
                let mut header = serde_json::to_vec(request).unwrap();
                header.push(b'\n');
                stdin.write_all(&header).unwrap();
                stdin.write_all(body).unwrap();
                stdin.flush().unwrap();
            };
        let mut read_reply = || {
            let mut line = Vec::new();
            assert!(
                stdout.read_until(b'\n', &mut line).unwrap() > 0,
                "worker closed stdout"
            );
            serde_json::from_slice::<session::Reply>(&line).expect("valid session reply frame")
        };

        send(
            &mut stdin,
            &session::Request::Hello {
                protocol: session::PROTOCOL,
                rank: RANK,
            },
            &[],
        );
        match read_reply() {
            session::Reply::Hello { hello } => {
                r72_observe_hello(&hello);
                assert!(hello.capabilities.contains(&producer().capability()));
            }
            reply => panic!("expected native HELLO, received {reply:?}"),
        }

        send(
            &mut stdin,
            &session::Request::LoadMxfp4Raw {
                binding: bad_artifact.binding().clone(),
                artifact_bytes: bad_artifact.bytes().len() as u64,
            },
            bad_artifact.bytes(),
        );
        match read_reply() {
            session::Reply::Refused { error } => {
                assert_eq!(error.code, "GPU_ADMISSION", "expert {bad_expert}: {error}");
            }
            reply => panic!("failed numeric LOAD must be refused before Loaded: {reply:?}"),
        }

        // Keep the raw owner protocol open after the pre-CUDA refusal, then send
        // a different valid LOAD. This exercises backend rollback without asking
        // the strict typed Session client to recover from its quarantined stream.
        send(
            &mut stdin,
            &session::Request::LoadMxfp4Raw {
                binding: valid.binding().clone(),
                artifact_bytes: valid.bytes().len() as u64,
            },
            valid.bytes(),
        );
        match read_reply() {
            session::Reply::Loaded {
                artifact_sha256,
                epoch: loaded_epoch,
                experts,
                transferred,
            } => {
                assert_eq!(artifact_sha256, valid.binding().artifact_sha256);
                assert_eq!(loaded_epoch, epoch);
                assert_eq!(experts, 3);
                assert!(transferred);
            }
            reply => panic!("valid LOAD after rollback must succeed: {reply:?}"),
        }

        send(&mut stdin, &session::Request::Shutdown, &[]);
        match read_reply() {
            session::Reply::Bye { execs, .. } => assert_eq!(execs, 0),
            reply => panic!("expected clean SHUTDOWN: {reply:?}"),
        }
        drop(stdin);
        let status = child.wait().unwrap();
        assert!(status.success(), "expert {bad_expert}: {status}");
        let mut stderr = String::new();
        child
            .stderr
            .take()
            .unwrap()
            .read_to_string(&mut stderr)
            .unwrap();
        let evicted = audit_records(&stderr, "evicted");
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0]["lifecycle"], serde_json::json!([0, 0, 0]));
        let owners = evicted[0]["experts"].as_array().unwrap();
        assert_eq!(
            owners.len(),
            3,
            "rollback must leave exactly the good LOAD owners"
        );
        for owner in owners {
            assert_eq!(owner["weight_upload_copies"], 6);
            assert_eq!(owner["poisoned"], 0);
        }
        println!("{{\"test\":\"numeric_load_rollback\",\"failed_expert\":{},\"prefix_owners\":{},\"good_owners\":3,\"execs\":0,\"lifecycle\":[0,0,0]}}", bad_expert, position);
    }
}

#[test]
#[ignore = "requires reserved native GPU and VF_RUN_GPU_TESTS=1"]
fn native_raw_exec_rejects_duplicate_token_expert_slots_before_any_gpu_call() {
    require_reserved_gpu();
    let epoch = 80;
    let artifact = artifact(epoch);
    let stamp = Stamp {
        run_id: format!("{}-duplicate-slot", test_run_id()),
        model_revision: "r13-tiny-independent-golden".into(),
        plan_id: "duplicate-token-expert".into(),
        layer: LAYER,
        batch: 0,
        epoch,
    };
    let make_slot = |slot| voltafabric::moe::Slot {
        token: 0,
        slot,
        expert: EXPERTS[0],
        rank: RANK,
        local: LOCALS[0],
        weight: 0.375,
    };
    let slot0 = make_slot(0);
    let slot1 = make_slot(1);
    let dispatch = voltafabric::routing::RankDispatch {
        stamp: stamp.clone(),
        rank: RANK,
        hidden: HIDDEN,
        tokens: 1,
        resident_order: LOCALS.to_vec(),
        slots: vec![
            voltafabric::routing::RankSlot {
                slot: slot0.clone(),
                physical_expert: 0,
            },
            voltafabric::routing::RankSlot {
                slot: slot1.clone(),
                physical_expert: 0,
            },
        ],
    };
    let canonical_dispatch = voltafabric::routing::Dispatch {
        stamp: stamp.clone(),
        hidden: HIDDEN,
        tokens: 1,
        slots: vec![slot0, slot1],
        ranks: vec![dispatch.clone()],
    };
    let format = voltafabric::preweighted::Format::R9Mxfp4PreweightedFp32V1;
    let contract = voltafabric::preweighted::Contract {
        format,
        sha256: sha256::hex(
            &serde_json::to_vec(&("VoltaFabric/preweighted/1", format, &canonical_dispatch))
                .unwrap(),
        ),
    };

    let worker_path = endpoint().argv.remove(0);
    let mut child = Command::new(worker_path)
        .arg("--audit-counters")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start the actual GPU worker executable");
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let send = |stdin: &mut std::process::ChildStdin, request: &session::Request, body: &[u8]| {
        let mut header = serde_json::to_vec(request).unwrap();
        header.push(b'\n');
        stdin.write_all(&header).unwrap();
        stdin.write_all(body).unwrap();
        stdin.flush().unwrap();
    };
    let mut read_reply = || {
        let mut line = Vec::new();
        assert!(
            stdout.read_until(b'\n', &mut line).unwrap() > 0,
            "worker closed stdout"
        );
        serde_json::from_slice::<session::Reply>(&line).expect("valid session reply frame")
    };

    send(
        &mut stdin,
        &session::Request::Hello {
            protocol: session::PROTOCOL,
            rank: RANK,
        },
        &[],
    );
    match read_reply() {
        session::Reply::Hello { hello } => {
                r72_observe_hello(&hello);
            assert!(hello.capabilities.contains(&producer().capability()));
        }
        reply => panic!("expected native HELLO, received {reply:?}"),
    }
    send(
        &mut stdin,
        &session::Request::LoadMxfp4Raw {
            binding: artifact.binding().clone(),
            artifact_bytes: artifact.bytes().len() as u64,
        },
        artifact.bytes(),
    );
    match read_reply() {
        session::Reply::Loaded {
            artifact_sha256,
            epoch: loaded_epoch,
            experts,
            transferred,
        } => {
            assert_eq!(artifact_sha256, artifact.binding().artifact_sha256);
            assert_eq!(loaded_epoch, epoch);
            assert_eq!(experts, 3);
            assert!(transferred);
        }
        reply => panic!("expected valid tiny artifact LOAD, received {reply:?}"),
    }

    let fixture = golden();
    let input = bits(
        fixture["inputs"]["pattern"]["bf16_as_f32_bits"]
            .as_array()
            .unwrap(),
    );
    assert_eq!(input.len(), HIDDEN);
    let mut activation = Vec::with_capacity(HIDDEN * 4);
    for value in input {
        activation.extend_from_slice(&value.to_le_bytes());
    }
    send(
        &mut stdin,
        &session::Request::ExecPreweightedFp32Raw {
            artifact_sha256: artifact.binding().artifact_sha256.clone(),
            dispatch,
            contract,
            attempt: 1,
            inputs: vec![session::RawTokenInputMeta {
                token: 0,
                values: HIDDEN as u64,
            }],
            activation_bytes: activation.len() as u64,
            fault: Fault::None,
        },
        &activation,
    );
    match read_reply() {
        session::Reply::Event { event } => match event.event {
            voltafabric::process::EventKind::Rejected { error } => {
                assert!(!error.code.is_empty());
            }
            voltafabric::process::EventKind::Started => {
                panic!("duplicate (token, expert) reached the Started gate")
            }
            other => panic!("expected pre-Started Rejected, received {other:?}"),
        },
        reply => panic!("duplicate raw route must be refused before any body: {reply:?}"),
    }

    send(&mut stdin, &session::Request::Shutdown, &[]);
    match read_reply() {
        session::Reply::Bye { execs, .. } => assert_eq!(execs, 0),
        reply => panic!("expected clean SHUTDOWN, received {reply:?}"),
    }
    drop(stdin);
    let status = child.wait().unwrap();
    assert!(status.success(), "{status}");
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    let evicted = audit_records(&stderr, "evicted");
    assert_eq!(evicted.len(), 1);
    assert_eq!(evicted[0]["lifecycle"], serde_json::json!([0, 0, 0]));
    let owners = evicted[0]["experts"]
        .as_array()
        .expect("audit owners array");
    assert_eq!(owners.len(), 3);
    for owner in owners {
        // R13 exposes native forward invocations as exec_calls in its audit ABI.
        assert_eq!(owner["exec_calls"], 0);
        assert_eq!(owner["published"], 0);
        assert_eq!(owner["kernel_launches"], 0);
    }
    println!("{{\"test\":\"duplicate_raw_token_expert\",\"event\":\"rejected_before_started\",\"native_forward_calls\":0,\"kernel_launches\":0,\"lifecycle\":[0,0,0]}}");
}

fn r70_observe(worker: &Session) {
    r72_observe_hello(&worker.hello);
}

// R72 passive HELLO provenance for typed and raw clients; no numeric oracle.
fn r72_observe_hello(hello: &session::ServerHello) {
    let pid = hello.pid;
    let path = PathBuf::from(format!("/proc/{pid}"));
    let stat = fs::read_to_string(path.join("stat")).expect("live native worker stat");
    let exe = fs::read_link(path.join("exe")).expect("live native worker exe");
    let bytes = fs::read(path.join("exe")).expect("live native executable bytes");
    let again = fs::read_to_string(path.join("stat")).expect("native stat recheck");
    let start = |s: &str| s.rsplit_once(") ").unwrap().1.split_whitespace().nth(19).unwrap().to_owned();
    assert_eq!(start(&stat), start(&again), "native lifetime changed");
    println!("{}", serde_json::json!({"r70_worker_hello":true,"pid":pid,"rank":hello.rank,
        "runner_pid":std::process::id(),"stat_before":stat,"stat_after":again,
        "exe":exe,"exe_sha256":sha256::hex(&bytes),"hello":hello}));
}
