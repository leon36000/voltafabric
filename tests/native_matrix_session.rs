#![cfg(all(feature = "native-ggml-cpu", target_os = "linux"))]

use std::{collections::BTreeMap, path::PathBuf, time::Duration};
use voltafabric::{
    moe::{Choice, Registry, Replica, Stamp},
    native_ggml_cpu::GroupedMatrixBinding,
    native_ledger::NativeBatchLedger,
    routing::RoutingProblem,
    session::{Endpoint, Session},
    sha256,
};

const H: usize = 32;
const Q8_0_ROW_BYTES: usize = 34;
const TIMEOUT: Duration = Duration::from_secs(10);

fn endpoint(rank: u32) -> Endpoint {
    Endpoint {
        rank,
        argv: vec![env!("CARGO_BIN_EXE_vf-rank-session").into()],
    }
}

fn stack(rows: usize, scale: [u8; 2], code: u8) -> Vec<u8> {
    let mut bytes = vec![0_u8; rows * Q8_0_ROW_BYTES];
    for row in 0..rows {
        let offset = row * Q8_0_ROW_BYTES;
        bytes[offset..offset + 2].copy_from_slice(&scale);
        bytes[offset + 2] = code;
    }
    bytes
}

fn binding(
    rank: u32,
    experts: &[(u32, u32)],
    bytes: &[u8],
    model: &str,
    layer: u32,
) -> GroupedMatrixBinding {
    GroupedMatrixBinding {
        model_revision: model.into(),
        layer,
        tensor: "ffn.experts.gate_proj.weight".into(),
        ggml_type: 8,
        rank,
        epoch: 9,
        rows: H,
        columns: H,
        residents: experts.to_vec(),
        sha256: sha256::hex(bytes),
    }
}

fn ledger(label: &str) -> NativeBatchLedger {
    let runs = std::env::current_dir().unwrap().join(".runs");
    std::fs::create_dir_all(&runs).unwrap();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory: PathBuf = runs.join(format!("r657-{label}-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    NativeBatchLedger::open(&directory).unwrap()
}

fn routed_problem(batch: u64) -> RoutingProblem {
    RoutingProblem {
        hidden: H,
        experts: vec![3, 18, 40],
        routes: vec![
            vec![
                Choice {
                    expert: 18,
                    weight: 0.25,
                },
                Choice {
                    expert: 40,
                    weight: 0.75,
                },
            ],
            vec![
                Choice {
                    expert: 3,
                    weight: 0.5,
                },
                Choice {
                    expert: 18,
                    weight: 0.25,
                },
            ],
        ],
        registry: Registry {
            epoch: 9,
            copies: vec![
                Replica {
                    expert: 40,
                    rank: 7,
                    local: 0,
                },
                Replica {
                    expert: 3,
                    rank: 7,
                    local: 1,
                },
                Replica {
                    expert: 18,
                    rank: 7,
                    local: 2,
                },
            ],
        },
        stamp: Stamp {
            run_id: "r657-session-run".into(),
            model_revision: "r657-session-model".into(),
            plan_id: "r657-session-plan".into(),
            layer: 5,
            batch,
            epoch: 9,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(7, vec![1, 2, 0])]),
    }
}

#[test]
fn grouped_ggml_matrix_stays_resident_across_identified_session_batches() {
    let mut weights = Vec::with_capacity(3 * H * Q8_0_ROW_BYTES);
    for code in [1, 2, 4] {
        weights.extend(stack(H, [0x00, 0x3c], code));
    }
    let binding = binding(
        7,
        &[(3, 1), (18, 2), (40, 0)],
        &weights,
        "r657-session-model",
        5,
    );
    let mut session = Session::connect(&endpoint(7), TIMEOUT).unwrap();
    assert!(session
        .hello
        .capabilities
        .iter()
        .any(|c| c == voltafabric::session::CAPABILITY_NATIVE_GROUPED_MATRIX_CPU));
    session
        .load_grouped_matrix_raw(&binding, &weights, TIMEOUT)
        .unwrap();
    let mut sessions = vec![session];
    let ledger = ledger("success");

    let first_problem = routed_problem(41);
    let mut first_inputs = vec![vec![0.0_f32; H]; 2];
    first_inputs[0][0] = 1.0;
    first_inputs[1][0] = 2.0;
    let first = Session::exec_grouped_matrix_batch(
        &mut sessions,
        &first_problem,
        &first_inputs,
        &ledger,
        TIMEOUT,
    )
    .unwrap();
    assert_eq!(first[0][0], 3.5);
    assert_eq!(first[1][0], 2.0);

    let second_problem = routed_problem(42);
    let mut second_inputs = vec![vec![0.0_f32; H]; 2];
    second_inputs[0][0] = 2.0;
    second_inputs[1][0] = 1.0;
    let second = Session::exec_grouped_matrix_batch(
        &mut sessions,
        &second_problem,
        &second_inputs,
        &ledger,
        TIMEOUT,
    )
    .unwrap();
    assert_eq!(second[0][0], 7.0);
    assert_eq!(second[1][0], 1.0);

    let replay = Session::exec_grouped_matrix_batch(
        &mut sessions,
        &first_problem,
        &first_inputs,
        &ledger,
        TIMEOUT,
    )
    .unwrap_err();
    assert_eq!(replay.code, "INVALID_INPUT");
    assert!(replay.message.contains("identity already attempted"));
    assert!(sessions[0].is_usable());
    let _ = sessions.pop().unwrap().close(TIMEOUT);
}

#[test]
fn grouped_ggml_session_failure_discards_all_rank_outputs_and_poisons_workers() {
    let ordinary = stack(H, [0x00, 0x3c], 1);
    let maximum = stack(H, [0xff, 0x7b], 1);
    let b0 = binding(7, &[(3, 0)], &ordinary, "r657-failure-model", 0);
    let b1 = binding(8, &[(18, 0)], &maximum, "r657-failure-model", 0);
    let mut sessions = vec![
        Session::connect(&endpoint(7), TIMEOUT).unwrap(),
        Session::connect(&endpoint(8), TIMEOUT).unwrap(),
    ];
    sessions[0]
        .load_grouped_matrix_raw(&b0, &ordinary, TIMEOUT)
        .unwrap();
    sessions[1]
        .load_grouped_matrix_raw(&b1, &maximum, TIMEOUT)
        .unwrap();
    let problem = RoutingProblem {
        hidden: H,
        experts: vec![3, 18],
        routes: vec![vec![
            Choice {
                expert: 3,
                weight: 0.5,
            },
            Choice {
                expert: 18,
                weight: 0.5,
            },
        ]],
        registry: Registry {
            epoch: 9,
            copies: vec![
                Replica {
                    expert: 3,
                    rank: 7,
                    local: 0,
                },
                Replica {
                    expert: 18,
                    rank: 8,
                    local: 0,
                },
            ],
        },
        stamp: Stamp {
            run_id: "r657-failure-run".into(),
            model_revision: "r657-failure-model".into(),
            plan_id: "r657-failure-plan".into(),
            layer: 0,
            batch: 1,
            epoch: 9,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(7, vec![0]), (8, vec![0])]),
    };
    let mut input = vec![vec![0.0_f32; H]];
    input[0][0] = 2.0;
    let ledger = ledger("failure");
    let error =
        Session::exec_grouped_matrix_batch(&mut sessions, &problem, &input, &ledger, TIMEOUT)
            .unwrap_err();
    assert_eq!(error.code, "FP16_OVERFLOW");
    assert!(sessions.iter().all(|session| !session.is_usable()));
    for session in sessions {
        let _ = session.close_input(TIMEOUT);
    }
}
