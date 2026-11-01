#![cfg(all(feature = "native-ggml-cpu", target_os = "linux"))]

use std::{
    collections::BTreeMap,
    io::{self, Read, Write},
    path::PathBuf,
    time::Duration,
};
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
    binding_shape(rank, experts, bytes, model, layer, H, H)
}

fn binding_shape(
    rank: u32,
    experts: &[(u32, u32)],
    bytes: &[u8],
    model: &str,
    layer: u32,
    rows: usize,
    columns: usize,
) -> GroupedMatrixBinding {
    GroupedMatrixBinding {
        model_revision: model.into(),
        layer,
        tensor: "ffn.experts.gate_proj.weight".into(),
        ggml_type: 8,
        rank,
        epoch: 9,
        rows,
        columns,
        residents: experts.to_vec(),
        sha256: sha256::hex(bytes),
    }
}

fn gguf_fixture(label: &str, tensor_name: &str, dimensions: [u64; 3], payload: &[u8]) -> PathBuf {
    let runs = std::env::current_dir().unwrap().join(".runs");
    std::fs::create_dir_all(&runs).unwrap();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = runs.join(format!("r658-gguf-{label}-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let mut header = Vec::new();
    header.extend_from_slice(b"GGUF");
    header.extend_from_slice(&3_u32.to_le_bytes());
    header.extend_from_slice(&1_u64.to_le_bytes()); // tensors
    header.extend_from_slice(&0_u64.to_le_bytes()); // key/value entries
    header.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
    header.extend_from_slice(tensor_name.as_bytes());
    header.extend_from_slice(&3_u32.to_le_bytes());
    for dimension in dimensions {
        header.extend_from_slice(&dimension.to_le_bytes());
    }
    header.extend_from_slice(&8_i32.to_le_bytes()); // GGML_TYPE_Q8_0
    header.extend_from_slice(&0_u64.to_le_bytes()); // tensor-relative offset
    while header.len() % 32 != 0 {
        header.push(0);
    }
    let path = directory.join("tiny.gguf");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(&header).unwrap();
    file.write_all(payload).unwrap();
    file.sync_all().unwrap();
    path
}

fn q8_0_one_hot_oracle(
    problem: &RoutingProblem,
    binding: &GroupedMatrixBinding,
    weights: &[u8],
    inputs: &[Vec<f32>],
) -> Vec<Vec<f64>> {
    let row_bytes = (binding.columns / 32) * Q8_0_ROW_BYTES;
    let expert_bytes = binding.rows * row_bytes;
    problem
        .routes
        .iter()
        .zip(inputs)
        .map(|(routes, input)| {
            assert!(input.iter().skip(1).all(|x| *x == 0.0));
            let mut row = vec![0.0; binding.rows];
            for route in routes {
                let physical = binding
                    .residents
                    .iter()
                    .position(|(expert, _)| *expert == route.expert)
                    .unwrap();
                for (output, value) in row.iter_mut().enumerate() {
                    let offset = physical * expert_bytes + output * row_bytes;
                    let scale = u16::from_le_bytes([weights[offset], weights[offset + 1]]);
                    let q8_value = weights[offset + 2] as i8;
                    *value += f64::from(input[0])
                        * voltafabric::completion::decode_fp16(scale).unwrap()
                        * f64::from(q8_value)
                        * route.weight;
                }
            }
            row
        })
        .collect()
}

struct ShortReader {
    remaining: usize,
}
impl Read for ShortReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Ok(0);
        }
        let count = self.remaining.min(out.len());
        out[..count].fill(0);
        self.remaining -= count;
        Ok(count)
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
    let checkpoint = gguf_fixture("small", &binding.tensor, [H as u64, H as u64, 3], &weights);
    let mut session = Session::connect(&endpoint(7), TIMEOUT).unwrap();
    assert!(session
        .hello
        .capabilities
        .iter()
        .any(|c| c == voltafabric::session::CAPABILITY_NATIVE_GROUPED_MATRIX_CPU));
    session
        .load_grouped_matrix_gguf_tensor(&binding, &checkpoint, &binding.tensor, TIMEOUT)
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
    assert_eq!(
        first,
        q8_0_one_hot_oracle(&first_problem, &binding, &weights, &first_inputs)
    );
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
fn gguf_tensor_span_streams_more_than_one_session_frame_into_resident_memory() {
    let rows = 644;
    let columns = 6144;
    let residents = [(3, 1), (18, 2), (40, 0), (55, 3)];
    let row_bytes = (columns / 32) * Q8_0_ROW_BYTES;
    let expert_bytes = rows * row_bytes;
    let mut weights = vec![0_u8; residents.len() * expert_bytes];
    for (expert_index, code) in [1_u8, 2, 3, 4].into_iter().enumerate() {
        for row in 0..rows {
            let offset = expert_index * expert_bytes + row * row_bytes;
            weights[offset..offset + 2].copy_from_slice(&[0x00, 0x3c]);
            weights[offset + 2] = code;
        }
    }
    assert!(weights.len() > voltafabric::session::MAX_RAW_ARTIFACT_BYTES);
    let binding = binding_shape(
        7,
        &residents,
        &weights,
        "r658-large-gguf-model",
        5,
        rows,
        columns,
    );
    let checkpoint = gguf_fixture(
        "large",
        &binding.tensor,
        [columns as u64, rows as u64, residents.len() as u64],
        &weights,
    );
    let mut session = Session::connect(&endpoint(7), Duration::from_secs(60)).unwrap();
    session
        .load_grouped_matrix_gguf_tensor(
            &binding,
            &checkpoint,
            &binding.tensor,
            Duration::from_secs(60),
        )
        .unwrap();
    assert_eq!(session.traffic.weight_bytes_sent, weights.len() as u64);
    let receipt = session.close(Duration::from_secs(5));
    assert_eq!(receipt.bye_execs, Some(0));
    assert_eq!(receipt.exit_code, Some(0));
}

#[test]
fn short_checkpoint_span_poison_closes_session_without_loading_a_resident() {
    let full = stack(H, [0x00, 0x3c], 1);
    let binding = binding(7, &[(3, 0)], &full, "r658-short-span-model", 0);
    let mut session = Session::connect(&endpoint(7), TIMEOUT).unwrap();
    let error = session
        .load_grouped_matrix_reader(
            &binding,
            &mut ShortReader { remaining: 4096 },
            (2 * 1024 * 1024) as u64,
            TIMEOUT,
        )
        .unwrap_err();
    assert_eq!(error.code, "CHECKPOINT_IO");
    assert!(!session.is_usable());
    let receipt = session.close_input(Duration::from_secs(5));
    assert!(receipt.exit_code.is_some());
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
