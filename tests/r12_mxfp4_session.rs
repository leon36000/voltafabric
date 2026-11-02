//! Actual persistent subprocess tests for the explicit CPU R9 producer.
#![cfg(target_os = "linux")]

use safetensors::tensor::{serialize, Dtype, TensorView};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};
use voltafabric::moe::{Choice, Registry, Replica, Stamp};
use voltafabric::preweighted;
use voltafabric::process::{Fault, ReceiptState};
use voltafabric::routing::RoutingProblem;
use voltafabric::session::{self, mxfp4, Endpoint, Session};
use voltafabric::sha256;

const RANK: u32 = 7;
const LAYER: u32 = 0;
const HIDDEN: usize = 32;
const TIMEOUT: Duration = Duration::from_secs(10);
const EXPERTS: [u32; 3] = [17, 93, 241];
const LOCALS: [u32; 3] = [4, 1, 9]; // deliberately non-sorted physical resident order

fn endpoint() -> Endpoint {
    Endpoint {
        rank: RANK,
        argv: vec![env!("CARGO_BIN_EXE_vf-rank-session").into()],
    }
}

/// Produce three experts with 32x32 identity MXFP4 projections in one SafeTensors shard.
/// E2M1 code 2 is +1; E8M0 byte 127 is scale 1. For each weight row, the diagonal
/// element gets code 2 and every other nibble is zero.
fn identity32_artifact(gate_scale: u8) -> (Vec<u8>, String) {
    let mut names = Vec::with_capacity(EXPERTS.len() * 6);
    let mut buffers = Vec::with_capacity(EXPERTS.len() * 6);
    let mut dtypes = Vec::with_capacity(EXPERTS.len() * 6);
    let mut shapes = Vec::with_capacity(EXPERTS.len() * 6);

    for &expert in &EXPERTS {
        for projection in ["w1", "w2", "w3"] {
            let mut packed = vec![0_u8; HIDDEN * (HIDDEN / 2)];
            for row in 0..HIDDEN {
                let col = row;
                let byte = row * (HIDDEN / 2) + col / 2;
                let code = 2_u8 << ((col % 2) * 4);
                packed[byte] |= code;
            }
            names.push(format!(
                "layers.{LAYER}.ffn.experts.{expert}.{projection}.weight"
            ));
            buffers.push(packed);
            dtypes.push(Dtype::I8);
            shapes.push(vec![HIDDEN, HIDDEN / 2]);

            names.push(format!(
                "layers.{LAYER}.ffn.experts.{expert}.{projection}.scale"
            ));
            buffers.push(vec![
                if projection == "w1" { gate_scale } else { 127 };
                HIDDEN
            ]);
            dtypes.push(Dtype::F8_E8M0);
            shapes.push(vec![HIDDEN, 1]);
        }
    }

    // All buffers are fully allocated before TensorViews borrow them.
    let entries: Vec<_> = names
        .into_iter()
        .zip(buffers.iter())
        .zip(dtypes)
        .zip(shapes)
        .map(|(((name, bytes), dtype), shape)| {
            (name, TensorView::new(dtype, shape, bytes).unwrap())
        })
        .collect();
    let bytes = serialize(entries, None).unwrap();
    let digest = sha256::hex(&bytes);
    (bytes, digest)
}

fn artifact(epoch: u64) -> mxfp4::Artifact {
    artifact_with_gate_scale(epoch, 127)
}

fn artifact_with_gate_scale(epoch: u64, gate_scale: u8) -> mxfp4::Artifact {
    let (bytes, digest) = identity32_artifact(gate_scale);
    mxfp4::Artifact::new(
        bytes,
        mxfp4::Binding {
            artifact_sha256: digest,
            model_revision: "r12-identity32".into(),
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
    .unwrap()
}

fn problem(batch: u64, epoch: u64, plan_id: &str) -> RoutingProblem {
    let choices = EXPERTS
        .iter()
        .zip([0.25_f64, 0.5, 1.0])
        .map(|(&expert, weight)| Choice { expert, weight })
        .collect::<Vec<_>>();
    RoutingProblem {
        hidden: HIDDEN,
        experts: EXPERTS.to_vec(),
        routes: vec![choices.clone(), choices],
        registry: Registry {
            epoch,
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
            run_id: "r12-session-smoke".into(),
            model_revision: "r12-identity32".into(),
            plan_id: plan_id.into(),
            layer: LAYER,
            batch,
            epoch,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(RANK, LOCALS.to_vec())]),
    }
}

fn zero_inputs(p: &RoutingProblem) -> Vec<mxfp4::TokenInput> {
    // +0 is finite and exactly representable in BF16, FP32, and the test oracle.
    (0..p.routes.len())
        .map(|token| mxfp4::TokenInput {
            token,
            value_bits: vec![0_u32; HIDDEN],
        })
        .collect()
}

fn assert_error_code<T>(result: voltafabric::Result<T>, expected: &str) {
    match result {
        Err(error) => assert_eq!(error.code, expected, "{error}"),
        Ok(_) => panic!("expected {expected} refusal"),
    }
}

fn run_zero_batch(
    worker: &mut Session,
    p: &RoutingProblem,
    attempt: u64,
    fault: Fault,
) -> session::ExecReceipt {
    let mut collector = preweighted::Collector::new(p).unwrap();
    let pending = worker
        .start_preweighted_fp32(p, &collector, attempt, zero_inputs(p), fault)
        .unwrap();
    let receipt = worker.finish_preweighted_fp32(pending, &mut collector, Instant::now() + TIMEOUT);
    if receipt.state == ReceiptState::Completed && receipt.error.is_none() {
        // Golden for identity projections and zero input: every preweighted row/output is +0.
        assert_eq!(collector.finish().unwrap(), vec![vec![0.0_f32; HIDDEN]; 2]);
    } else {
        assert!(
            collector.finish().is_err(),
            "failed/unknown attempt exposed output"
        );
    }
    receipt
}

#[test]
fn bf16_wire_negotiates_with_real_worker_and_preserves_legacy_raw32() {
    let legacy = Endpoint {
        rank: RANK,
        argv: vec![
            "python3".into(),
            format!(
                "{}/tests/reference/bf16_legacy_hello_proxy.py",
                env!("CARGO_MANIFEST_DIR")
            ),
            env!("CARGO_BIN_EXE_vf-rank-session").into(),
        ],
    };
    let artifact = artifact(4);
    let p = problem(0, 4, "bf16-wire-negotiation");
    let inputs: Vec<_> = (0..2)
        .map(|token| mxfp4::TokenInput {
            token,
            value_bits: (0..HIDDEN)
                .map(|j| {
                    [
                        0_u32,
                        0x8000_0000,
                        0x0001_0000,
                        0x8001_0000,
                        0x3f80_0000,
                        0xbf00_0000,
                        0x4000_0000,
                        0xc000_0000,
                    ][j % 8]
                })
                .collect(),
        })
        .collect();
    let mut results = Vec::new();
    for (endpoint, supports_bf16) in [(endpoint(), true), (legacy, false)] {
        let mut worker = Session::connect(&endpoint, TIMEOUT).unwrap();
        worker.prefer_mxfp4_bf16_activations();
        assert_eq!(
            worker
                .hello
                .capabilities
                .iter()
                .any(|c| c == "mxfp4_exec_bf16_exact_raw_v1"),
            supports_bf16
        );
        worker.load_mxfp4(&artifact, 4, TIMEOUT).unwrap();
        let mut collector = preweighted::Collector::new(&p).unwrap();
        let pending = worker
            .start_preweighted_fp32(&p, &collector, 1, inputs.clone(), Fault::None)
            .unwrap();
        let receipt =
            worker.finish_preweighted_fp32(pending, &mut collector, Instant::now() + TIMEOUT);
        assert_eq!(receipt.state, ReceiptState::Completed, "{receipt:?}");
        assert!(receipt.error.is_none(), "{receipt:?}");
        let bits: Vec<Vec<_>> = collector
            .finish()
            .unwrap()
            .iter()
            .map(|r| r.iter().map(|v| v.to_bits()).collect())
            .collect();
        results.push((bits, receipt.request_bytes));
        assert_eq!(worker.traffic.loads_transferred, 1);
        assert_eq!(worker.close(TIMEOUT).bye_execs, Some(1));
    }
    assert_eq!(
        results[0].0, results[1].0,
        "same FP32 output bits and slot arithmetic"
    );
    // Headers differ only by operation name and body-byte decimal length. The
    // two-byte encoding must save close to 2*values, without compressing outputs.
    let saved = results[1].1 - results[0].1;
    assert!(
        (120..=140).contains(&saved),
        "saved {saved}, expected ~128 body bytes"
    );
}

#[test]
fn bf16_stream_loss_during_input_is_unknown_without_fallback_or_replay() {
    let artifact = artifact(4);
    let p = problem(0, 4, "bf16-input-cut");
    let mut worker = Session::connect(&fault_endpoint("bf16_cut", None), TIMEOUT).unwrap();
    worker.prefer_mxfp4_bf16_activations();
    worker.load_mxfp4(&artifact, 4, TIMEOUT).unwrap();
    let mut collector = preweighted::Collector::new(&p).unwrap();
    let pending = worker
        .start_preweighted_fp32(&p, &collector, 1, zero_inputs(&p), Fault::None)
        .unwrap();
    let receipt = worker.finish_preweighted_fp32(pending, &mut collector, Instant::now() + TIMEOUT);
    assert_eq!(receipt.state, ReceiptState::Unknown, "{receipt:?}");
    assert!(!receipt.started);
    assert!(receipt.error.is_some());
    assert!(collector.finish().is_err());
    assert!(!worker.is_usable());
    let before = worker.traffic.bytes_sent;
    assert_error_code(
        worker.start_preweighted_fp32(
            &p,
            &preweighted::Collector::new(&p).unwrap(),
            2,
            zero_inputs(&p),
            Fault::None,
        ),
        "SESSION_UNUSABLE",
    );
    assert_eq!(worker.traffic.bytes_sent, before);
    assert_eq!(worker.traffic.execs, 1);
    worker.close(TIMEOUT);
}

#[test]
fn bf16_capability_does_not_change_default_fp32_wire() {
    let artifact = artifact(4);
    let p = problem(0, 4, "bf16-default-wire");
    let mut bytes = Vec::new();
    for endpoint in [
        endpoint(),
        Endpoint {
            rank: RANK,
            argv: vec![
                "python3".into(),
                format!(
                    "{}/tests/reference/bf16_legacy_hello_proxy.py",
                    env!("CARGO_MANIFEST_DIR")
                ),
                env!("CARGO_BIN_EXE_vf-rank-session").into(),
            ],
        },
    ] {
        let mut worker = Session::connect(&endpoint, TIMEOUT).unwrap();
        worker.load_mxfp4(&artifact, 4, TIMEOUT).unwrap();
        bytes.push(run_zero_batch(&mut worker, &p, 1, Fault::None).request_bytes);
        assert_eq!(worker.close(TIMEOUT).bye_execs, Some(1));
    }
    assert_eq!(
        bytes[0], bytes[1],
        "BF16 requires explicit selection, not just peer capability"
    );
}

#[test]
fn persistent_worker_runs_three_batches_with_one_artifact_transfer_and_weightless_rebind() {
    let artifact = artifact(4);
    let mut worker = Session::connect(&endpoint(), TIMEOUT).unwrap();
    let first = worker.load_mxfp4(&artifact, 4, TIMEOUT).unwrap();
    assert!(first.transferred);
    assert_eq!(first.artifact_sha256, artifact.binding().artifact_sha256);
    let bytes_after_load = worker.traffic.weight_bytes_sent;
    assert!(bytes_after_load >= artifact.bytes().len() as u64);

    for batch in 0..3 {
        let epoch = if batch < 2 { 4 } else { 5 };
        let p = problem(batch, epoch, "r12-plan");
        if batch == 2 {
            let rebound = worker.load_mxfp4(&artifact, epoch, TIMEOUT).unwrap();
            assert!(!rebound.transferred);
            assert_eq!(rebound.artifact_sha256, first.artifact_sha256);
        }
        let receipt = run_zero_batch(&mut worker, &p, batch + 1, Fault::None);
        assert_eq!(receipt.state, ReceiptState::Completed, "{receipt:?}");
        assert!(receipt.error.is_none(), "{receipt:?}");
        assert_eq!(receipt.expected_slots, 6);
        assert_eq!(receipt.received_slots, 6);
        assert_eq!(worker.traffic.weight_bytes_sent, bytes_after_load);
    }

    assert_eq!(worker.traffic.loads_transferred, 1);
    assert_eq!(worker.traffic.rebinds, 1);
    assert_eq!(worker.traffic.execs, 3);
    let close = worker.close(TIMEOUT);
    assert_eq!(close.exit_code, Some(0), "{close:?}");
    assert_eq!(close.bye_execs, Some(3));
}

#[test]
fn artifact_epoch_rollback_plan_and_coefficient_mutations_send_no_exec_bytes() {
    let artifact = artifact(8);
    let mut worker = Session::connect(&endpoint(), TIMEOUT).unwrap();
    worker.load_mxfp4(&artifact, 8, TIMEOUT).unwrap();
    let loaded_weight_bytes = worker.traffic.weight_bytes_sent;

    // A rollback attempt is refused before writing any rebind request.
    let before_rollback = worker.traffic.bytes_sent;
    assert_eq!(
        worker.load_mxfp4(&artifact, 7, TIMEOUT).unwrap_err().code,
        "EPOCH_ROLLBACK"
    );
    assert_eq!(worker.traffic.bytes_sent, before_rollback);
    assert_eq!(worker.traffic.weight_bytes_sent, loaded_weight_bytes);

    let p = problem(0, 8, "plan-original");
    let collector = preweighted::Collector::new(&p).unwrap();
    let mut changed_plan = problem(0, 8, "plan-mutated");
    let mut stale_collector = preweighted::Collector::new(&p).unwrap();
    let before = worker.traffic.bytes_sent;
    assert_error_code(
        worker.start_preweighted_fp32(
            &changed_plan,
            &stale_collector,
            1,
            zero_inputs(&changed_plan),
            Fault::None,
        ),
        "INVALID_INPUT",
    );
    assert_eq!(worker.traffic.bytes_sent, before);

    // Route coefficient bits are part of the arithmetic contract even if the full Stamp
    // remains identical. Old collector contract + changed route must be refused pre-send.
    changed_plan = p.clone();
    changed_plan.routes[0][0].weight = 0.375;
    let before = worker.traffic.bytes_sent;
    assert_error_code(
        worker.start_preweighted_fp32(
            &changed_plan,
            &collector,
            1,
            zero_inputs(&changed_plan),
            Fault::None,
        ),
        "INVALID_INPUT",
    );
    assert_eq!(worker.traffic.bytes_sent, before);

    // The unused mutable binding keeps the draft explicit that collectors are per batch.
    stale_collector.expire();
    assert!(stale_collector.finish().is_err());
    assert_eq!(worker.traffic.bytes_sent, before);
    assert_eq!(worker.close(TIMEOUT).exit_code, Some(0));
}

#[test]
fn missing_duplicate_stale_attempt_and_worker_death_expire_the_whole_collector() {
    let artifact = artifact(4);
    for (attempt, fault, started) in [
        (1, Fault::Missing, true),
        (1, Fault::Duplicate, true),
        (1, Fault::StaleEpoch, true),
        (1, Fault::OldAttempt { attempt: 0 }, true),
        (1, Fault::DieBeforeStart, false),
        (1, Fault::DieAfterStart, true),
        (1, Fault::DieAfterCompute, true),
    ] {
        // Each fault gets a fresh process because protocol corruption/death poisons its stream.
        let mut worker = Session::connect(&endpoint(), TIMEOUT).unwrap();
        worker.load_mxfp4(&artifact, 4, TIMEOUT).unwrap();
        let p = problem(0, 4, "r12-plan");
        let mut collector = preweighted::Collector::new(&p).unwrap();
        let pending = worker
            .start_preweighted_fp32(&p, &collector, attempt, zero_inputs(&p), fault)
            .unwrap();
        let receipt =
            worker.finish_preweighted_fp32(pending, &mut collector, Instant::now() + TIMEOUT);
        assert_ne!(receipt.state, ReceiptState::Completed, "{receipt:?}");
        assert_eq!(receipt.started, started, "{receipt:?}");
        assert!(receipt.error.is_some(), "{receipt:?}");
        assert!(
            collector.finish().is_err(),
            "failed attempt leaked partial rows"
        );
        assert_eq!(worker.traffic.execs, 1, "no replay is allowed");
        let close = worker.close(TIMEOUT);
        assert_ne!(
            close.bye_execs,
            Some(2),
            "worker must not have replayed the EXEC"
        );
    }
}

#[test]
fn old_peer_without_r9_mxfp4_capability_is_refused_before_load_bytes() {
    // HELLO advertises the ordinary session protocol but omits the R12 producer capability.
    // This mutates the negotiated capability list before load to isolate the local gate;
    // a dedicated frozen old-worker executable can replace this smoke harness later.
    let artifact = artifact(4);
    let mut worker = Session::connect(&endpoint(), TIMEOUT).unwrap();
    worker
        .hello
        .capabilities
        .retain(|capability| capability != mxfp4::CAPABILITY);
    let before = worker.traffic.bytes_sent;
    let error = worker.load_mxfp4(&artifact, 4, TIMEOUT).unwrap_err();
    assert_eq!(error.code, "CAPABILITY_MISSING");
    assert_eq!(worker.traffic.bytes_sent, before);
    assert_eq!(worker.traffic.loads_transferred, 0);
    assert_eq!(worker.close(TIMEOUT).exit_code, Some(0));
}

fn fault_endpoint(mode: &str, socket: Option<&std::path::Path>) -> Endpoint {
    let mut argv = vec![
        "python3".into(),
        "-B".into(),
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/reference/preweighted_fault_peer.py"
        )
        .into(),
        mode.into(),
    ];
    if let Some(socket) = socket {
        argv.push(socket.to_str().unwrap().into());
    }
    Endpoint { rank: RANK, argv }
}

fn assert_expired(collector: &mut preweighted::Collector, p: &RoutingProblem) {
    let slot = voltafabric::routing::dispatch(p).unwrap().slots.remove(0);
    let row = preweighted::Completion {
        contract: collector.contract().clone(),
        attempt: None,
        stamp: p.stamp.clone(),
        token: slot.token,
        slot: slot.slot,
        expert: slot.expert,
        rank: slot.rank,
        local: slot.local,
        weight_bits: (slot.weight as f32).to_bits(),
        value_bits: vec![0; HIDDEN],
    };
    assert_eq!(collector.accept(row).unwrap_err().code, "BATCH_ABORTED");
}

#[test]
fn real_worker_preserves_all_45_nonzero_and_zero_golden_cases_without_reload() {
    let golden: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../fixtures/preweighted_mxfp4_tiny_golden.json"
    ))
    .unwrap();
    let binding = mxfp4::Binding {
        artifact_sha256: golden["weights_sha256"].as_str().unwrap().into(),
        model_revision: "r12-identity32".into(),
        rank: RANK,
        layer: 0,
        epoch: 4,
        experts: EXPERTS
            .iter()
            .zip(LOCALS)
            .map(|(&expert, local)| mxfp4::Expert { expert, local })
            .collect(),
    };
    let artifact = mxfp4::Artifact::new(
        include_bytes!("../fixtures/preweighted_mxfp4_tiny.safetensors").to_vec(),
        binding,
    )
    .unwrap();
    let mut worker = Session::connect(&endpoint(), TIMEOUT).unwrap();
    worker.load_mxfp4(&artifact, 4, TIMEOUT).unwrap();
    let weight_bytes = worker.traffic.weight_bytes_sent;
    for (index, case) in golden["cases"].as_array().unwrap().iter().enumerate() {
        let mut p = problem(index as u64, 4, "nonzero-golden");
        p.routes = vec![vec![Choice {
            expert: case["expert"].as_u64().unwrap() as u32,
            weight: case["coefficient"].as_f64().unwrap(),
        }]];
        let input = golden["inputs"][case["input"].as_str().unwrap()]["bf16_as_f32_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| u32::from_str_radix(v.as_str().unwrap(), 16).unwrap())
            .collect();
        let mut collector = preweighted::Collector::new(&p).unwrap();
        let pending = worker
            .start_preweighted_fp32(
                &p,
                &collector,
                index as u64 + 1,
                vec![mxfp4::TokenInput {
                    token: 0,
                    value_bits: input,
                }],
                Fault::None,
            )
            .unwrap();
        let receipt =
            worker.finish_preweighted_fp32(pending, &mut collector, Instant::now() + TIMEOUT);
        assert_eq!(receipt.state, ReceiptState::Completed, "{receipt:?}");
        let output: Vec<_> = collector.finish().unwrap()[0]
            .iter()
            .map(|x| x.to_bits())
            .collect();
        let expected: Vec<_> = case["output_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                let bits = u32::from_str_radix(v.as_str().unwrap(), 16).unwrap();
                if bits & 0x7fff_ffff == 0 {
                    0
                } else {
                    bits
                }
            })
            .collect();
        assert_eq!(output, expected, "case {index}");
        assert_eq!(worker.traffic.weight_bytes_sent, weight_bytes);
    }
    let close = worker.close(TIMEOUT);
    assert_eq!(close.bye_execs, Some(45));
    assert_eq!(close.exit_code, Some(0));
}

#[test]
fn incomplete_terminal_and_malformed_frames_never_publish_a_batch() {
    let artifact = artifact(4);
    for mode in [
        "valid",
        "missing_finished",
        "truncate_body",
        "truncated_json",
        "wrong_artifact",
        "wrong_contract",
        "old_epoch",
        "old_attempt",
        "foreign_rank",
        "wrong_weight",
        "nan",
        "oversized_values",
        "padded_header",
    ] {
        let mut worker = Session::connect(&fault_endpoint(mode, None), TIMEOUT).unwrap();
        worker.load_mxfp4(&artifact, 4, TIMEOUT).unwrap();
        let p = problem(0, 4, mode);
        let mut collector = preweighted::Collector::new(&p).unwrap();
        let pending = worker
            .start_preweighted_fp32(&p, &collector, 1, zero_inputs(&p), Fault::None)
            .unwrap();
        let receipt =
            worker.finish_preweighted_fp32(pending, &mut collector, Instant::now() + TIMEOUT);
        if mode == "valid" {
            assert_eq!(receipt.state, ReceiptState::Completed, "{receipt:?}");
            assert_eq!(collector.finish().unwrap(), vec![vec![0.0; 32]; 2]);
        } else {
            assert_eq!(
                receipt.state,
                ReceiptState::Unknown,
                "mode={mode} {receipt:?}"
            );
            assert!(receipt.error.is_some());
            assert_expired(&mut collector, &p);
            if mode == "missing_finished" {
                assert_eq!(receipt.received_slots, 6);
            }
            if mode == "padded_header" {
                assert_eq!(receipt.error.as_ref().unwrap().code, "MESSAGE_TOO_LARGE");
            }
            assert!(collector.finish().is_err(), "mode={mode}");
            assert!(!worker.is_usable());
        }
        worker.close(TIMEOUT);
    }
}

#[test]
fn late_response_after_timeout_is_not_replayed_or_reused() {
    use std::io::{Read, Write};
    use std::os::linux::net::SocketAddrExt;
    use std::os::unix::net::{SocketAddr, UnixListener};
    let name = format!("vf-r12-late-{}", std::process::id());
    let address = SocketAddr::from_abstract_name(name.as_bytes()).unwrap();
    let listener = UnixListener::bind_addr(&address).unwrap();
    let socket = std::path::PathBuf::from(format!("@{name}"));
    let mut worker = Session::connect(&fault_endpoint("late", Some(&socket)), TIMEOUT).unwrap();
    let (mut control, _) = listener.accept().unwrap();
    control.set_read_timeout(Some(TIMEOUT)).unwrap();
    let artifact = artifact(4);
    worker.load_mxfp4(&artifact, 4, TIMEOUT).unwrap();
    let p = problem(0, 4, "late");
    let mut collector = preweighted::Collector::new(&p).unwrap();
    let pending = worker
        .start_preweighted_fp32(&p, &collector, 1, zero_inputs(&p), Fault::None)
        .unwrap();
    let mut signal = [0];
    control.read_exact(&mut signal).unwrap();
    assert_eq!(signal, [b'S']);
    // Peer has actually Started and cannot produce rows until explicitly released.
    // Receipt.started is only what reached our reader, never proof of non-execution.
    let receipt = worker.finish_preweighted_fp32(pending, &mut collector, Instant::now());
    assert_eq!(receipt.state, ReceiptState::Unknown);
    assert_expired(&mut collector, &p);
    assert!(!worker.is_usable());
    let sent = worker.traffic.bytes_sent;
    control.write_all(b"R").unwrap();
    control.read_exact(&mut signal).unwrap();
    assert_eq!(signal, [b'D']);
    let new = preweighted::Collector::new(&p).unwrap();
    assert_error_code(
        worker.start_preweighted_fp32(&p, &new, 2, zero_inputs(&p), Fault::None),
        "SESSION_UNUSABLE",
    );
    assert_eq!(worker.traffic.bytes_sent, sent);
    assert_eq!(worker.traffic.execs, 1);
    worker.close(TIMEOUT);
}

#[test]
fn unsolicited_body_after_finished_poisoned_without_a_new_exec() {
    use std::io::{Read, Write};
    use std::os::linux::net::SocketAddrExt;
    use std::os::unix::net::{SocketAddr, UnixListener};
    let name = format!("vf-r12-extra-{}", std::process::id());
    let address = SocketAddr::from_abstract_name(name.as_bytes()).unwrap();
    let listener = UnixListener::bind_addr(&address).unwrap();
    let socket = std::path::PathBuf::from(format!("@{name}"));
    let mut worker =
        Session::connect(&fault_endpoint("after_finished", Some(&socket)), TIMEOUT).unwrap();
    let (mut control, _) = listener.accept().unwrap();
    control.set_read_timeout(Some(TIMEOUT)).unwrap();
    let artifact = artifact(4);
    worker.load_mxfp4(&artifact, 4, TIMEOUT).unwrap();
    let p = problem(0, 4, "extra");
    let mut collector = preweighted::Collector::new(&p).unwrap();
    let pending = worker
        .start_preweighted_fp32(&p, &collector, 1, zero_inputs(&p), Fault::None)
        .unwrap();
    let receipt = worker.finish_preweighted_fp32(pending, &mut collector, Instant::now() + TIMEOUT);
    assert_eq!(receipt.state, ReceiptState::Completed);
    assert!(receipt.error.is_none());
    let mut signal = [0];
    control.read_exact(&mut signal).unwrap();
    assert_eq!(signal, [b'F']);
    control.write_all(b"R").unwrap();
    control.read_exact(&mut signal).unwrap();
    assert_eq!(signal, [b'D']);
    assert!(worker.reply(TIMEOUT).is_err());
    assert!(!worker.is_usable());
    assert_eq!(worker.traffic.execs, 1);
    worker.close(TIMEOUT);
}

#[test]
fn rebind_compares_the_current_session_epoch_and_old_hello_is_refused() {
    let artifact = artifact(1);
    let mut worker = Session::connect(&endpoint(), TIMEOUT).unwrap();
    worker.load_mxfp4(&artifact, 1, TIMEOUT).unwrap();
    worker.load_mxfp4(&artifact, 5, TIMEOUT).unwrap();
    let bytes = worker.traffic.bytes_sent;
    assert_error_code(worker.load_mxfp4(&artifact, 4, TIMEOUT), "EPOCH_NOT_NEWER");
    assert_error_code(worker.load_mxfp4(&artifact, 5, TIMEOUT), "EPOCH_NOT_NEWER");
    assert_eq!(worker.traffic.bytes_sent, bytes);
    assert_eq!(worker.artifact().unwrap().1, 5);
    worker.close(TIMEOUT);
    let mut old = Session::connect(&fault_endpoint("old_peer", None), TIMEOUT).unwrap();
    let bytes = old.traffic.bytes_sent;
    assert_error_code(old.load_mxfp4(&artifact, 1, TIMEOUT), "CAPABILITY_MISSING");
    assert_eq!(old.traffic.bytes_sent, bytes);
    old.close(TIMEOUT);
}

#[test]
fn draining_a_completed_rank_after_collector_abort_preserves_execution_certainty() {
    let artifact = artifact(4);
    let mut worker = Session::connect(&endpoint(), TIMEOUT).unwrap();
    worker.load_mxfp4(&artifact, 4, TIMEOUT).unwrap();
    let p = problem(0, 4, "drain");
    let mut collector = preweighted::Collector::new(&p).unwrap();
    let pending = worker
        .start_preweighted_fp32(&p, &collector, 1, zero_inputs(&p), Fault::None)
        .unwrap();
    collector.expire(); // another rank failed while this rank was running
    let receipt = worker.finish_preweighted_fp32(pending, &mut collector, Instant::now() + TIMEOUT);
    assert_eq!(receipt.state, ReceiptState::Completed, "{receipt:?}");
    assert_eq!(receipt.error.unwrap().code, "BATCH_ABORTED");
    assert!(worker.is_usable());
    assert!(collector.finish().is_err());
    let mut next = p;
    next.stamp.batch += 1;
    assert_eq!(
        run_zero_batch(&mut worker, &next, 2, Fault::None).state,
        ReceiptState::Completed
    );
    assert_eq!(worker.close(TIMEOUT).bye_execs, Some(2));
}

#[test]
fn malformed_typed_inputs_refuse_before_any_provider_or_request() {
    let artifact = artifact(4);
    let mut worker = Session::connect(&endpoint(), TIMEOUT).unwrap();
    worker.load_mxfp4(&artifact, 4, TIMEOUT).unwrap();
    let p = problem(0, 4, "input-admission");
    let collector = preweighted::Collector::new(&p).unwrap();
    let sent = worker.traffic.bytes_sent;
    for bits in [0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0x3f80_0001] {
        let mut inputs = zero_inputs(&p);
        inputs[0].value_bits[0] = bits;
        assert!(worker
            .start_preweighted_fp32(&p, &collector, 1, inputs, Fault::None)
            .is_err());
        assert_eq!(worker.traffic.bytes_sent, sent);
        assert_eq!(worker.traffic.execs, 0);
    }
    assert_error_code(
        worker.start_preweighted_fp32_until(
            &p,
            &collector,
            1,
            zero_inputs(&p),
            Fault::None,
            Instant::now(),
        ),
        "SESSION_DEADLINE",
    );
    assert_eq!(worker.traffic.bytes_sent, sent);
    assert!(worker.is_usable());
    assert_eq!(
        run_zero_batch(&mut worker, &p, 1, Fault::Reverse).state,
        ReceiptState::Completed
    );
    assert_eq!(worker.close(TIMEOUT).bye_execs, Some(1));
}

#[test]
fn a_dead_worker_is_replaced_for_a_distinct_batch_without_replaying_the_unknown_batch() {
    let artifact = artifact(4);
    let p = problem(0, 4, "explicit-new-batch");
    let mut first = Session::connect(&endpoint(), TIMEOUT).unwrap();
    first.load_mxfp4(&artifact, 4, TIMEOUT).unwrap();
    let receipt = run_zero_batch(&mut first, &p, 1, Fault::DieAfterStart);
    assert_eq!(receipt.state, ReceiptState::Unknown);
    assert!(receipt.started);
    assert_eq!(first.traffic.execs, 1);
    assert_eq!(first.close(TIMEOUT).exit_code, Some(72));
    let mut next = p;
    next.stamp.batch = 1;
    let mut replacement = Session::connect(&endpoint(), TIMEOUT).unwrap();
    replacement.load_mxfp4(&artifact, 4, TIMEOUT).unwrap();
    assert_eq!(
        run_zero_batch(&mut replacement, &next, 2, Fault::None).state,
        ReceiptState::Completed
    );
    assert_eq!(replacement.traffic.loads_transferred, 1);
    assert_eq!(replacement.close(TIMEOUT).bye_execs, Some(1));
}

#[test]
fn a_known_compute_failure_is_terminal_and_next_distinct_batch_can_use_the_resident() {
    let artifact = artifact_with_gate_scale(4, 254);
    let mut worker = Session::connect(&endpoint(), TIMEOUT).unwrap();
    worker.load_mxfp4(&artifact, 4, TIMEOUT).unwrap();
    let p = problem(0, 4, "known-compute-failure");
    let mut collector = preweighted::Collector::new(&p).unwrap();
    let mut inputs = zero_inputs(&p);
    for row in &mut inputs {
        row.value_bits.fill(2.0_f32.to_bits());
    }
    let pending = worker
        .start_preweighted_fp32(&p, &collector, 1, inputs, Fault::None)
        .unwrap();
    let receipt = worker.finish_preweighted_fp32(pending, &mut collector, Instant::now() + TIMEOUT);
    assert_eq!(receipt.state, ReceiptState::Started);
    assert!(receipt.started);
    assert_eq!(receipt.received_slots, 0);
    assert_eq!(receipt.error.unwrap().code, "NUMERICAL_OVERFLOW");
    assert_expired(&mut collector, &p);
    assert!(collector.finish().is_err());
    assert!(worker.is_usable());
    let mut next = p;
    next.stamp.batch = 1;
    assert_eq!(
        run_zero_batch(&mut worker, &next, 2, Fault::None).state,
        ReceiptState::Completed
    );
    assert_eq!(worker.traffic.loads_transferred, 1);
    assert_eq!(worker.close(TIMEOUT).bye_execs, Some(2));
}

#[test]
fn a_peer_that_stops_reading_a_large_load_cannot_block_the_coordinator() {
    use safetensors::tensor::SafeTensors;
    use std::os::linux::net::SocketAddrExt;
    use std::os::unix::net::{SocketAddr, UnixListener};
    let (original, _) = identity32_artifact(127);
    let tensors = SafeTensors::deserialize(&original).unwrap();
    let pad = vec![0; 1024 * 1024];
    let mut entries: Vec<_> = tensors
        .names()
        .into_iter()
        .map(|name| (name.to_owned(), tensors.tensor(name).unwrap()))
        .collect();
    entries.push((
        "r12_non_expert_padding".into(),
        TensorView::new(Dtype::U8, vec![pad.len()], &pad).unwrap(),
    ));
    let bytes = serialize(entries, None).unwrap();
    let digest = sha256::hex(&bytes);
    let mut binding = artifact(4).binding().clone();
    binding.artifact_sha256 = digest;
    let artifact = mxfp4::Artifact::new(bytes, binding).unwrap();
    let name = format!("vf-r12-load-{}", std::process::id());
    let listener =
        UnixListener::bind_addr(&SocketAddr::from_abstract_name(name.as_bytes()).unwrap()).unwrap();
    let socket = std::path::PathBuf::from(format!("@{name}"));
    let mut worker =
        Session::connect(&fault_endpoint("no_load_read", Some(&socket)), TIMEOUT).unwrap();
    let (_control, _) = listener.accept().unwrap();
    let pid = worker.pid();
    let (send, done) = std::sync::mpsc::channel();
    let join = std::thread::spawn(move || {
        let result = worker.load_mxfp4(&artifact, 4, Duration::from_millis(300));
        send.send((worker, result)).unwrap();
    });
    let (worker, result) = match done.recv_timeout(Duration::from_secs(5)) {
        Ok(value) => value,
        Err(error) => {
            // Bounded cleanup also makes removing the write guard a failing test,
            // instead of leaving a blocked test thread/owned peer behind.
            let _ = std::process::Command::new("kill")
                .args(["-KILL", &pid.to_string()])
                .status();
            if let Ok((worker, _)) = done.recv_timeout(TIMEOUT) {
                worker.close(TIMEOUT);
            }
            let _ = join.join();
            panic!("large LOAD write guard did not terminate: {error}");
        }
    };
    assert_eq!(result.unwrap_err().code, "SESSION_WRITE_TIMEOUT");
    assert!(!worker.is_usable());
    let close = worker.close(TIMEOUT);
    assert!(close.exit_code.is_none() || close.exit_code == Some(137));
    join.join().unwrap();
}
