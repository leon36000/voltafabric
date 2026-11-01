#![cfg(all(feature = "native-ggml-cpu", target_os = "linux"))]

use std::{
    collections::BTreeMap,
    io::{self, Read, Write},
    path::PathBuf,
    time::Duration,
};
use voltafabric::{
    gguf_expert_slices::ExpertCheckpointSlice,
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

// Independent scalar rounding for the normal positive values used by this Q8_0 fixture.
// It models one Q8_0 activation block's stored FP16 scale and the Collector's FP16 result
// format without calling VoltaFabric's production conversion helpers.
fn oracle_fp16_round_trip(value: f32) -> f32 {
    let bits = value.to_bits();
    let sign = bits & 0x8000_0000;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    assert_eq!(sign, 0, "fixture oracle expects positive values");
    assert!((-14..=15).contains(&exponent), "fixture value must be normal FP16");

    let significand = bits & 0x007f_ffff;
    let mut mantissa = significand >> 13;
    let remainder = significand & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && mantissa & 1 != 0) {
        mantissa += 1;
    }
    let mut half_exponent = exponent + 15;
    if mantissa == 0x400 {
        mantissa = 0;
        half_exponent += 1;
    }
    assert!(half_exponent <= 30, "fixture output must remain finite FP16");
    let f32_exponent = (half_exponent - 15 + 127) as u32;
    let decoded = (f32_exponent << 23) | (mantissa << 13);
    f32::from_bits(decoded)
}

fn oracle_q8_0_positive_block_value(value: f32) -> f32 {
    assert!(value >= 0.0 && value.is_finite());
    oracle_fp16_round_trip(value / 127.0) * 127.0
}

fn oracle_qwen2moe_down(value: f32, gate_code: u8, up_code: u8, down_code: u8) -> f32 {
    // The input is one-hot, so GGML Q8_0 reconstructs its maximum from the stored
    // half-precision scale. Gate and up each have one nonzero Q8_0 tap.
    let input = oracle_q8_0_positive_block_value(value);
    let gate = input * f32::from(gate_code);
    let up = input * f32::from(up_code);
    let swiglu = gate / (1.0 + (-gate).exp()) * up;

    // The 64-wide down activation consists of two identical 32-value Q8_0 blocks.
    // Each down weight block has one nonzero tap, so the two dot contributions add.
    let down_input = oracle_q8_0_positive_block_value(swiglu);
    let partial = down_input * f32::from(down_code);
    partial + partial
}

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
fn gguf_rank_load_selects_only_explicit_checkpoint_experts_in_resident_order() {
    let mut checkpoint_weights = Vec::new();
    for code in [1, 2, 4] {
        checkpoint_weights.extend(stack(H, [0x00, 0x3c], code));
    }
    let expert_bytes = H * Q8_0_ROW_BYTES;
    let mut resident_weights = checkpoint_weights[2 * expert_bytes..].to_vec();
    resident_weights.extend_from_slice(&checkpoint_weights[..expert_bytes]);
    let mut binding = binding_shape(
        7,
        &[(40, 0), (3, 1)],
        &resident_weights,
        "r659-gguf-slices-model",
        5,
        H,
        H,
    );
    binding.tensor = "blk.5.ffn_gate_exps.weight".into();
    let checkpoint = gguf_fixture(
        "rank-local",
        &binding.tensor,
        [H as u64, H as u64, 3],
        &checkpoint_weights,
    );
    let slices = [
        ExpertCheckpointSlice {
            resident_expert: 40,
            resident_local: 0,
            checkpoint_expert: 2,
        },
        ExpertCheckpointSlice {
            resident_expert: 3,
            resident_local: 1,
            checkpoint_expert: 0,
        },
    ];

    let mut session = Session::connect(&endpoint(7), TIMEOUT).unwrap();
    session
        .load_grouped_matrix_gguf_expert_slices(
            &binding,
            &checkpoint,
            &binding.tensor,
            &slices,
            TIMEOUT,
        )
        .unwrap();
    assert_eq!(
        session.traffic.weight_bytes_sent,
        resident_weights.len() as u64
    );

    let problem = RoutingProblem {
        hidden: H,
        experts: vec![3, 40],
        routes: vec![
            vec![Choice {
                expert: 40,
                weight: 0.25,
            }],
            vec![Choice {
                expert: 3,
                weight: 0.75,
            }],
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
            ],
        },
        stamp: Stamp {
            run_id: "r659-gguf-slices-run".into(),
            model_revision: "r659-gguf-slices-model".into(),
            plan_id: "r659-gguf-slices-plan".into(),
            layer: 5,
            batch: 69,
            epoch: 9,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(7, vec![0, 1])]),
    };
    let mut inputs = vec![vec![0.0_f32; H]; 2];
    inputs[0][0] = 2.0;
    inputs[1][0] = 1.0;
    let mut sessions = vec![session];
    let output = Session::exec_grouped_matrix_batch(
        &mut sessions,
        &problem,
        &inputs,
        &ledger("rank-local-gguf"),
        TIMEOUT,
    )
    .unwrap();
    assert_eq!(
        output,
        q8_0_one_hot_oracle(&problem, &binding, &resident_weights, &inputs)
    );
    assert_eq!(output[0][0], 2.0);
    assert_eq!(output[1][0], 0.75);
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


#[test]
fn grouped_ggml_per_slot_projection_keeps_rectangular_shape_and_unweighted_outputs() {
    use voltafabric::native_ggml_cpu::GroupedProjectionInput;

    let rows = 64;
    let columns = H;
    let mut weights = Vec::new();
    for code in [1_u8, 2, 4] {
        weights.extend(stack(rows, [0x00, 0x3c], code));
    }
    let binding = binding_shape(
        7,
        &[(3, 1), (18, 2), (40, 0)],
        &weights,
        "r659-session-model",
        5,
        rows,
        columns,
    );
    let checkpoint = gguf_fixture(
        "rectangular-projection",
        &binding.tensor,
        [columns as u64, rows as u64, 3],
        &weights,
    );
    let mut session = Session::connect(&endpoint(7), TIMEOUT).unwrap();
    session
        .load_grouped_matrix_gguf_tensor(&binding, &checkpoint, &binding.tensor, TIMEOUT)
        .unwrap();
    let mut sessions = vec![session];
    let mut problem = routed_problem(91);
    problem.stamp.model_revision = "r659-session-model".into();
    let mut inputs = Vec::new();
    for (token, routes) in problem.routes.iter().enumerate() {
        for (slot, route) in routes.iter().enumerate() {
            let local = binding
                .residents
                .iter()
                .find(|(expert, _)| *expert == route.expert)
                .map(|(_, local)| *local)
                .unwrap();
            let mut values = vec![0.0_f32; columns];
            values[0] = (token * 3 + slot + 2) as f32;
            inputs.push(GroupedProjectionInput {
                token,
                slot,
                expert: route.expert,
                local,
                values,
            });
        }
    }


    // Preflight rejects incomplete slot coverage before the worker starts or identity is consumed.
    let ledger = ledger("rectangular-projection");
    let mut incomplete = inputs.clone();
    incomplete.pop();
    assert!(Session::exec_grouped_projection_batch(
        &mut sessions,
        &problem,
        &incomplete,
        &ledger,
        TIMEOUT,
    )
    .is_err());
    assert!(sessions[0].is_usable());

    let output =
        Session::exec_grouped_projection_batch(&mut sessions, &problem, &inputs, &ledger, TIMEOUT)
            .unwrap();
    assert_eq!(output.dispatch_hidden, H);
    assert_eq!(output.input_columns, columns);
    assert_eq!(output.output_columns, rows);
    assert_eq!(output.tensor, binding.tensor);
    assert_eq!(output.slots.len(), inputs.len());

    // Independent scalar Q8_0 oracle; route coefficients are intentionally absent.
    for (actual, input) in output.slots.iter().zip(&inputs) {
        assert_eq!(
            (actual.token, actual.slot, actual.expert, actual.local),
            (input.token, input.slot, input.expert, input.local)
        );
        let physical = binding
            .residents
            .iter()
            .position(|(expert, _)| *expert == input.expert)
            .unwrap();
        for row in 0..rows {
            let offset = physical * rows * Q8_0_ROW_BYTES + row * Q8_0_ROW_BYTES;
            let scale = u16::from_le_bytes([weights[offset], weights[offset + 1]]);
            let quant = weights[offset + 2] as i8;
            let expected = f64::from(oracle_q8_0_positive_block_value(input.values[0]))
                * voltafabric::completion::decode_fp16(scale).unwrap()
                * f64::from(quant);
            assert_eq!(actual.values[row], expected as f32);
        }
    }
    // The first route weight is 0.25; this intermediate result stays unweighted.
    assert_eq!(
        output.slots[0].values[0],
        oracle_q8_0_positive_block_value(2.0) * 2.0
    );
    let _ = sessions.pop().unwrap().close(TIMEOUT);
}



#[cfg(feature = "native-ggml-cpu")]
#[test]
fn qwen2moe_gate_up_preserves_slot_identity_and_builds_unweighted_down_inputs() {
    use voltafabric::{
        moe::Stamp,
        session::{
            qwen2moe_gate_up_to_down_inputs, GroupedProjectionBatchOutput,
            GroupedProjectionSlotOutput,
        },
    };

    let stamp = Stamp {
        run_id: "qwen2moe-run".into(),
        model_revision: "qwen2moe-model".into(),
        plan_id: "qwen2moe-plan".into(),
        layer: 5,
        batch: 17,
        epoch: 9,
    };
    let gate = GroupedProjectionBatchOutput {
        stamp: stamp.clone(),
        tensor: "blk.5.ffn_gate_exps.weight".into(),
        attempt: 101,
        dispatch_hidden: H,
        input_columns: H,
        output_columns: 3,
        slots: vec![
            GroupedProjectionSlotOutput {
                token: 0,
                slot: 0,
                expert: 3,
                rank: 7,
                local: 1,
                values: vec![-2.0, 0.0, 2.0],
            },
            GroupedProjectionSlotOutput {
                token: 0,
                slot: 1,
                expert: 18,
                rank: 8,
                local: 2,
                values: vec![0.5, -1.0, 1.0],
            },
        ],
    };
    let up = GroupedProjectionBatchOutput {
        stamp: stamp.clone(),
        tensor: "blk.5.ffn_up_exps.weight".into(),
        attempt: 101,
        dispatch_hidden: H,
        input_columns: H,
        output_columns: 3,
        slots: vec![
            GroupedProjectionSlotOutput {
                token: 0,
                slot: 0,
                expert: 3,
                rank: 7,
                local: 1,
                values: vec![2.0, 3.0, 4.0],
            },
            GroupedProjectionSlotOutput {
                token: 0,
                slot: 1,
                expert: 18,
                rank: 8,
                local: 2,
                values: vec![5.0, 6.0, 7.0],
            },
        ],
    };

    let actual = qwen2moe_gate_up_to_down_inputs(&gate, &up, None).unwrap();
    assert_eq!(actual.len(), gate.slots.len());
    for ((actual, gate_slot), up_slot) in actual.iter().zip(&gate.slots).zip(&up.slots) {
        assert_eq!(
            (actual.token, actual.slot, actual.expert, actual.local),
            (
                gate_slot.token,
                gate_slot.slot,
                gate_slot.expert,
                gate_slot.local
            )
        );
        for ((&gate_value, &up_value), &value) in gate_slot
            .values
            .iter()
            .zip(&up_slot.values)
            .zip(&actual.values)
        {
            let oracle = f64::from(gate_value) / (1.0 + (-f64::from(gate_value)).exp())
                * f64::from(up_value);
            assert!((f64::from(value) - oracle).abs() <= 2e-6);
        }
    }
    // Zero gate gives zero output; no route coefficient is present in this helper.
    assert_eq!(actual[0].values[1], 0.0);

    let clamped = qwen2moe_gate_up_to_down_inputs(&gate, &up, Some(1.0)).unwrap();
    assert_eq!(clamped[0].values[2], 1.0);
}

#[cfg(feature = "native-ggml-cpu")]
#[test]
fn qwen2moe_gate_up_rejects_wrong_tensor_role_and_misaligned_batch() {
    use voltafabric::{
        moe::Stamp,
        session::{
            qwen2moe_gate_up_to_down_inputs, GroupedProjectionBatchOutput,
            GroupedProjectionSlotOutput,
        },
    };
    let make = |tensor: &str, batch: u64| GroupedProjectionBatchOutput {
        stamp: Stamp {
            run_id: "qwen2moe-run".into(),
            model_revision: "qwen2moe-model".into(),
            plan_id: "qwen2moe-plan".into(),
            layer: 5,
            batch,
            epoch: 9,
        },
        tensor: tensor.into(),
        attempt: 103,
        dispatch_hidden: H,
        input_columns: H,
        output_columns: 1,
        slots: vec![GroupedProjectionSlotOutput {
            token: 0,
            slot: 0,
            expert: 3,
            rank: 7,
            local: 1,
            values: vec![1.0],
        }],
    };
    let gate = make("blk.5.ffn_gate_exps.weight", 18);
    let up = make("blk.5.ffn_up_exps.weight", 18);
    let wrong_role = make("blk.5.ffn_gate_up_exps.weight", 18);
    let wrong_batch = make("blk.5.ffn_up_exps.weight", 19);
    assert!(qwen2moe_gate_up_to_down_inputs(&gate, &wrong_role, None).is_err());
    assert!(qwen2moe_gate_up_to_down_inputs(&gate, &wrong_batch, None).is_err());
    assert_eq!(
        qwen2moe_gate_up_to_down_inputs(&gate, &up, None)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn pinned_ggml_qwen2moe_route_ops_feed_the_voltafabric_adapter_in_tensor_slot_order() {
    use voltafabric::routing::{qwen2moe_routes_from_donor, Qwen2MoEDonorSelection};

    // GGML [experts, tokens] layout; each token has a distinct, untied top two.
    let logits = [4.0_f32, 1.0, 3.0, 0.0, 5.0, 2.0];
    let fixture = vf_ggml_native_ffi::qwen2moe_router_fixture(&logits, 3, 2, 2, 0.5).unwrap();
    assert_eq!(fixture.selected_experts, vec![0, 2, 1, 2]);

    let expected = [
        0.5 * (4.0_f64.exp() / (4.0_f64.exp() + 1.0_f64.exp() + 3.0_f64.exp())),
        0.5 * (3.0_f64.exp() / (4.0_f64.exp() + 1.0_f64.exp() + 3.0_f64.exp())),
        0.5 * (5.0_f64.exp() / (1.0 + 5.0_f64.exp() + 2.0_f64.exp())),
        0.5 * (2.0_f64.exp() / (1.0 + 5.0_f64.exp() + 2.0_f64.exp())),
    ];
    for (&actual, &expected) in fixture.final_weights.iter().zip(&expected) {
        assert!((f64::from(actual) - expected).abs() < 1e-6);
    }

    let base = routed_problem(667);
    let mapped_experts: Vec<i32> = fixture
        .selected_experts
        .iter()
        .map(|&expert| i32::try_from(base.experts[expert as usize]).unwrap())
        .collect();
    let imported = qwen2moe_routes_from_donor(
        Qwen2MoEDonorSelection {
            stamp: base.stamp.clone(),
            attempt: 667_003,
            tokens: 2,
            top_k: 2,
            selected_experts: mapped_experts,
            final_weights: fixture.final_weights.clone(),
        },
        base.hidden,
        base.experts.clone(),
        base.registry.clone(),
        base.preferred_owners.clone(),
        base.resident_order.clone(),
    )
    .unwrap();
    assert_eq!(
        imported
            .problem()
            .routes
            .iter()
            .map(|route| route.iter().map(|choice| choice.expert).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        vec![
            vec![base.experts[0], base.experts[2]],
            vec![base.experts[1], base.experts[2]]
        ]
    );
    assert_eq!(
        imported
            .problem()
            .routes
            .iter()
            .flatten()
            .map(|choice| choice.weight)
            .collect::<Vec<_>>(),
        fixture
            .final_weights
            .iter()
            .copied()
            .map(f64::from)
            .collect::<Vec<_>>()
    );
    assert!(imported
        .problem()
        .routes
        .iter()
        .all(|route| { route.iter().map(|choice| choice.weight).sum::<f64>() < 1.0 }));
}

#[test]
fn qwen2moe_donor_route_import_preserves_slot_order_scaled_weights_and_attempt() {
    use voltafabric::routing::{qwen2moe_routes_from_donor, Qwen2MoEDonorSelection};

    let mut base = routed_problem(661);
    base.stamp.run_id = "r666-qwen2moe-route-run".into();
    base.stamp.model_revision = "r666-qwen2moe-route-model".into();
    base.stamp.plan_id = "r666-qwen2moe-route-plan".into();
    let imported = qwen2moe_routes_from_donor(
        Qwen2MoEDonorSelection {
            stamp: base.stamp.clone(),
            attempt: 661_004,
            tokens: 2,
            top_k: 2,
            // GGML [top_k, tokens], with slot fastest: do not re-sort by expert ID.
            selected_experts: vec![18, 40, 3, 18],
            // These are final donor weights after expert_weights_scale. Their sums are
            // deliberately below one; the adapter must not normalize or scale again.
            final_weights: vec![0.125, 0.375, 0.25, 0.125],
        },
        base.hidden,
        base.experts.clone(),
        base.registry.clone(),
        base.preferred_owners.clone(),
        base.resident_order.clone(),
    )
    .unwrap();

    assert_eq!(imported.problem().stamp, base.stamp);
    assert_eq!(imported.attempt(), 661_004);
    assert_eq!(
        imported
            .problem()
            .routes
            .iter()
            .map(|route| route.iter().map(|choice| choice.expert).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        vec![vec![18, 40], vec![3, 18]]
    );
    assert_eq!(
        imported.problem().routes[0]
            .iter()
            .map(|choice| choice.weight)
            .collect::<Vec<_>>(),
        vec![f64::from(0.125_f32), f64::from(0.375_f32)]
    );
    assert_eq!(
        imported.problem().routes[0]
            .iter()
            .map(|choice| choice.weight)
            .sum::<f64>(),
        0.5
    );
    let dispatched = voltafabric::routing::dispatch(imported.problem()).unwrap();
    assert_eq!(
        dispatched
            .slots
            .iter()
            .map(|slot| (slot.token, slot.slot, slot.expert, slot.weight))
            .collect::<Vec<_>>(),
        vec![
            (0, 0, 18, f64::from(0.125_f32)),
            (0, 1, 40, f64::from(0.375_f32)),
            (1, 0, 3, f64::from(0.25_f32)),
            (1, 1, 18, f64::from(0.125_f32)),
        ]
    );
}

#[test]
fn qwen2moe_donor_route_import_rejects_incomplete_invalid_and_duplicate_selection() {
    use voltafabric::routing::{qwen2moe_routes_from_donor, Qwen2MoEDonorSelection};

    let base = routed_problem(662);
    let make = |attempt, selected_experts: Vec<i32>, final_weights: Vec<f32>| {
        qwen2moe_routes_from_donor(
            Qwen2MoEDonorSelection {
                stamp: base.stamp.clone(),
                attempt,
                tokens: 2,
                top_k: 2,
                selected_experts,
                final_weights,
            },
            base.hidden,
            base.experts.clone(),
            base.registry.clone(),
            base.preferred_owners.clone(),
            base.resident_order.clone(),
        )
    };

    assert!(make(662_001, vec![18, 40, 3], vec![0.25, 0.75, 0.5, 0.25]).is_err());
    assert!(make(0, vec![18, 40, 3, 18], vec![0.25, 0.75, 0.5, 0.25]).is_err());
    assert!(make(662_002, vec![-1, 40, 3, 18], vec![0.25, 0.75, 0.5, 0.25]).is_err());
    assert!(make(662_003, vec![18, 40, 3, 18], vec![0.25, f32::NAN, 0.5, 0.25]).is_err());
    assert!(make(662_004, vec![18, 18, 3, 18], vec![0.25, 0.75, 0.5, 0.25]).is_err());
    assert!(make(662_005, vec![18, 99, 3, 18], vec![0.25, 0.75, 0.5, 0.25]).is_err());
}


#[test]
fn qwen2moe_compound_ffn_uses_one_identity_and_returns_unweighted_down_rows_and_preflights_all_stages() {
    let residents = [(3, 1), (18, 2), (40, 0)];
    let make_weights = |rows: usize, columns: usize, codes: &[u8]| {
        let row_bytes = (columns / 32) * Q8_0_ROW_BYTES;
        let mut weights = vec![0_u8; codes.len() * rows * row_bytes];
        for (physical, &code) in codes.iter().enumerate() {
            for row in 0..rows {
                for block in 0..columns / 32 {
                    let offset = physical * rows * row_bytes + row * row_bytes
                        + block * Q8_0_ROW_BYTES;
                    weights[offset..offset + 2].copy_from_slice(&[0x00, 0x3c]);
                    weights[offset + 2] = code;
                }
            }
        }
        weights
    };
    let load = |label: &str, tensor: &str, rows: usize, columns: usize, codes: &[u8]| {
        let weights = make_weights(rows, columns, codes);
        let mut binding = binding_shape(7, &residents, &weights,
            "r662-qwen2moe-synthetic-model", 5, rows, columns);
        binding.tensor = tensor.into();
        let checkpoint = gguf_fixture(label, &binding.tensor,
            [columns as u64, rows as u64, residents.len() as u64], &weights);
        let mut session = Session::connect(&endpoint(7), TIMEOUT).unwrap();
        session.load_grouped_matrix_gguf_tensor(
            &binding, &checkpoint, &binding.tensor, TIMEOUT,
        ).unwrap();
        vec![session]
    };
    let mut problem = routed_problem(662);
    problem.stamp.run_id = "r662-qwen2moe-run".into();
    problem.stamp.model_revision = "r662-qwen2moe-synthetic-model".into();
    problem.stamp.plan_id = "r662-qwen2moe-plan".into();
    let route_batch = voltafabric::routing::qwen2moe_routes_from_donor(
        voltafabric::routing::Qwen2MoEDonorSelection {
            stamp: problem.stamp.clone(),
            attempt: 662_006,
            tokens: problem.routes.len(),
            top_k: problem.routes.first().unwrap().len(),
            selected_experts: problem
                .routes
                .iter()
                .flatten()
                .map(|choice| i32::try_from(choice.expert).unwrap())
                .collect(),
            final_weights: problem
                .routes
                .iter()
                .flatten()
                .map(|choice| choice.weight as f32)
                .collect(),
        },
        problem.hidden,
        problem.experts.clone(),
        problem.registry.clone(),
        problem.preferred_owners.clone(),
        problem.resident_order.clone(),
    )
    .unwrap();
    problem = route_batch.problem().clone();
    let mut hidden = vec![vec![0.0_f32; H]; 2];
    hidden[0][0] = 2.0;
    hidden[1][0] = 1.0;
    let ledger = ledger("qwen2moe-compound-ffn");
    // Wrong role fails before claiming the Stamp or sending an execution request.
    let mut bad_gate = load("bad-gate", "blk.5.ffn_gate_up_exps.weight", 64, H, &[1, 2, 4]);
    let mut preflight_up = load("preflight-up", "blk.5.ffn_up_exps.weight", 64, H, &[2, 3, 5]);
    let mut preflight_down = load("preflight-down", "blk.5.ffn_down_exps.weight", H, 64, &[3, 1, 2]);
    let error = Session::exec_qwen2moe_ffn_batch(
        &mut bad_gate, &mut preflight_up, &mut preflight_down,
        &problem, &hidden, &ledger, TIMEOUT, None,
    ).unwrap_err();
    assert_eq!(error.code, "INVALID_INPUT");
    assert!(bad_gate.iter().all(Session::is_usable));
    assert!(preflight_up.iter().all(Session::is_usable));
    assert!(preflight_down.iter().all(Session::is_usable));
    let _ = bad_gate.pop().unwrap().close(TIMEOUT);
    let _ = preflight_up.pop().unwrap().close(TIMEOUT);
    let _ = preflight_down.pop().unwrap().close(TIMEOUT);
    let mut gate = load("gate", "blk.5.ffn_gate_exps.weight", 64, H, &[1, 2, 4]);
    let mut up = load("up", "blk.5.ffn_up_exps.weight", 64, H, &[2, 3, 5]);
    let mut down = load("down", "blk.5.ffn_down_exps.weight", H, 64, &[3, 1, 2]);
    let output = Session::exec_qwen2moe_ffn_from_donor_routes(
        &mut gate, &mut up, &mut down, &route_batch, &hidden, &ledger, TIMEOUT, None,
    ).unwrap();
    assert_eq!(output.stamp, problem.stamp);
    assert_eq!(output.attempt, route_batch.attempt());
    assert_eq!(output.tensor, "blk.5.ffn_down_exps.weight");
    assert_eq!(output.dispatch_hidden, H);
    assert_eq!(output.input_columns, 64);
    assert_eq!(output.output_columns, H);
    assert_eq!(output.slots.len(), problem.routes.iter().map(Vec::len).sum::<usize>());
    let gate_codes = [1_u8, 2, 4];
    let up_codes = [2_u8, 3, 5];
    let down_codes = [3_u8, 1, 2];
    let mut weighted_oracle = vec![vec![0.0_f64; H]; problem.routes.len()];
    for actual in &output.slots {
        let route = &problem.routes[actual.token][actual.slot];
        assert_eq!(actual.expert, route.expert);
        let physical = residents
            .iter()
            .position(|(expert, _)| *expert == actual.expert)
            .unwrap();
        let expected_unweighted = oracle_qwen2moe_down(
            hidden[actual.token][0],
            gate_codes[physical],
            up_codes[physical],
            down_codes[physical],
        );
        for &value in &actual.values {
            let expected = f64::from(expected_unweighted);
            assert!((f64::from(value) - expected).abs() <= 2e-5 * expected.abs().max(1.0));
            assert!((f64::from(value) - expected * route.weight).abs() > 1e-3);
        }
        let fp16 = f64::from(oracle_fp16_round_trip(expected_unweighted));
        for oracle in &mut weighted_oracle[actual.token] {
            *oracle += route.weight * fp16;
        }
    }
    let mut stale_problem = problem.clone();
    stale_problem.stamp.batch = stale_problem.stamp.batch.checked_add(1).unwrap();
    assert!(voltafabric::session::collect_qwen2moe_ffn_output(&stale_problem, &output).is_err());
    let mut fp16_overflow = output.clone();
    fp16_overflow.slots[0].values[0] = f32::MAX;
    assert!(voltafabric::session::collect_qwen2moe_ffn_output(&problem, &fp16_overflow).is_err());
    let collected = voltafabric::session::collect_qwen2moe_ffn_output(&problem, &output).unwrap();
    assert_eq!(collected.len(), weighted_oracle.len());
    for (actual, expected) in collected.iter().zip(&weighted_oracle) {
        assert_eq!(actual.len(), H);
        for (&value, &oracle) in actual.iter().zip(expected) {
            assert_eq!(value, oracle);
        }
    }
    let replay = Session::exec_qwen2moe_ffn_from_donor_routes(
        &mut gate, &mut up, &mut down, &route_batch, &hidden, &ledger, TIMEOUT, None,
    ).unwrap_err();
    assert!(replay.message.contains("identity already attempted"));
    assert!(gate.iter().all(Session::is_usable));
    assert!(up.iter().all(Session::is_usable));
    assert!(down.iter().all(Session::is_usable));
    let _ = gate.pop().unwrap().close(TIMEOUT);
    let _ = up.pop().unwrap().close(TIMEOUT);
    let _ = down.pop().unwrap().close(TIMEOUT);
}
