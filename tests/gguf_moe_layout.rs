#![cfg(all(feature = "native-ggml-cpu", target_os = "linux"))]

use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::PathBuf,
    time::Duration,
};
use vf_ggml_native_ffi::GgufIndex;
use voltafabric::{
    completion::{decode_fp16, encode_fp16},
    gguf_expert_slices::ExpertCheckpointSlice,
    gguf_moe_layout::{layer_spans, resolve_expert_ffn_layout},
    moe::{Choice, Registry, Replica, Stamp},
    native_ggml_cpu::GroupedMatrixBinding,
    native_ledger::NativeBatchLedger,
    routing::RoutingProblem,
    session::{Endpoint, Session},
};

const TIMEOUT: Duration = Duration::from_secs(10);
const HIDDEN: usize = 32;

enum Value<'a> {
    String(&'a str),
    U32(u32),
}

struct Tensor<'a> {
    name: &'a str,
    dimensions: [u64; 3],
    payload: Vec<u8>,
}

fn q8_payload(rows: usize, columns: usize, experts: usize, first_code: i8) -> Vec<u8> {
    let row_bytes = columns / 32 * 34;
    let mut payload = vec![0_u8; rows * experts * row_bytes];
    for expert in 0..experts {
        for row in 0..rows {
            let row_offset = (expert * rows + row) * row_bytes;
            for block in 0..columns / 32 {
                let block_offset = row_offset + block * 34;
                payload[block_offset..block_offset + 2].copy_from_slice(&[0x00, 0x3c]);
                if row < columns && row / 32 == block {
                    payload[block_offset + 2 + row % 32] = (first_code + expert as i8) as u8;
                }
            }
        }
    }
    payload
}

fn put_string(output: &mut Vec<u8>, value: &str) {
    output.extend_from_slice(&(value.len() as u64).to_le_bytes());
    output.extend_from_slice(value.as_bytes());
}

fn fixture(fused: bool, ffn_size: usize) -> PathBuf {
    let tensors = if fused {
        vec![
            Tensor {
                name: "blk.0.ffn_gate_up_exps.weight",
                dimensions: [32, (2 * ffn_size) as u64, 2],
                payload: q8_payload(2 * ffn_size, 32, 2, 1),
            },
            Tensor {
                name: "blk.0.ffn_down_exps.weight",
                dimensions: [ffn_size as u64, 32, 2],
                payload: q8_payload(32, ffn_size, 2, 1),
            },
        ]
    } else {
        vec![
            Tensor {
                name: "blk.0.ffn_gate_exps.weight",
                dimensions: [32, ffn_size as u64, 2],
                payload: q8_payload(ffn_size, 32, 2, 1),
            },
            Tensor {
                name: "blk.0.ffn_up_exps.weight",
                dimensions: [32, ffn_size as u64, 2],
                payload: q8_payload(ffn_size, 32, 2, 2),
            },
            Tensor {
                name: "blk.0.ffn_down_exps.weight",
                dimensions: [ffn_size as u64, 32, 2],
                payload: q8_payload(32, ffn_size, 2, 1),
            },
        ]
    };
    let metadata = [
        ("general.architecture", Value::String("qwen2moe")),
        ("qwen2moe.block_count", Value::U32(1)),
        ("qwen2moe.embedding_length", Value::U32(32)),
        ("qwen2moe.expert_count", Value::U32(2)),
        (
            "qwen2moe.expert_feed_forward_length",
            Value::U32(ffn_size as u32),
        ),
    ];

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        put_string(&mut bytes, key);
        match value {
            Value::String(value) => {
                bytes.extend_from_slice(&8_u32.to_le_bytes());
                put_string(&mut bytes, value);
            }
            Value::U32(value) => {
                bytes.extend_from_slice(&4_u32.to_le_bytes());
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    let mut relative = 0_u64;
    for tensor in &tensors {
        put_string(&mut bytes, tensor.name);
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        for dimension in tensor.dimensions {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        bytes.extend_from_slice(&8_i32.to_le_bytes()); // GGML_TYPE_Q8_0
        bytes.extend_from_slice(&relative.to_le_bytes());
        relative += tensor.payload.len() as u64;
    }
    while bytes.len() % 32 != 0 {
        bytes.push(0);
    }
    for tensor in tensors {
        bytes.extend_from_slice(&tensor.payload);
    }

    let root = std::env::current_dir().unwrap().join(".runs");
    std::fs::create_dir_all(&root).unwrap();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = root.join(format!(
        "r659-moe-layout-{}-{nonce}.gguf",
        std::process::id()
    ));
    let mut file = File::create(&path).unwrap();
    file.write_all(&bytes).unwrap();
    file.sync_all().unwrap();
    path
}

fn endpoint(rank: u32) -> Endpoint {
    Endpoint {
        rank,
        argv: vec![env!("CARGO_BIN_EXE_vf-rank-session").into()],
    }
}

fn ledger() -> NativeBatchLedger {
    let root = std::env::current_dir().unwrap().join(".runs");
    std::fs::create_dir_all(&root).unwrap();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = root.join(format!(
        "r659-moe-ffn-ledger-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir(&path).unwrap();
    NativeBatchLedger::open(&path).unwrap()
}

fn binding(tensor: &str, residents: &[(u32, u32)], bytes: &[u8]) -> GroupedMatrixBinding {
    GroupedMatrixBinding {
        model_revision: "r659-qwen2moe-cpu-fixture".into(),
        layer: 0,
        tensor: tensor.into(),
        ggml_type: 8,
        rank: 7,
        epoch: 9,
        rows: HIDDEN,
        columns: HIDDEN,
        residents: residents.to_vec(),
        sha256: voltafabric::sha256::hex(bytes),
    }
}

fn routed_problem(expert: u32, weight: f64, batch: u64) -> RoutingProblem {
    RoutingProblem {
        hidden: HIDDEN,
        experts: vec![3, 40],
        routes: vec![vec![Choice { expert, weight }]],
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
            run_id: "r659-qwen2moe-run".into(),
            model_revision: "r659-qwen2moe-cpu-fixture".into(),
            plan_id: "r659-qwen2moe-plan".into(),
            layer: 0,
            batch,
            epoch: 9,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(7, vec![0, 1])]),
    }
}

fn run_projection(
    session: Session,
    input: &[f32],
    expert: u32,
    route_weight: f64,
    batch: u64,
    ledger: &NativeBatchLedger,
) -> (Session, Vec<f64>) {
    let problem = routed_problem(expert, route_weight, batch);
    let mut sessions = vec![session];
    let output = Session::exec_grouped_matrix_batch(
        &mut sessions,
        &problem,
        &[input.to_vec()],
        ledger,
        TIMEOUT,
    )
    .unwrap()
    .remove(0);
    (sessions.pop().unwrap(), output)
}

/// Independent scalar Q8_0 decoder/matvec: it reads the checkpoint's original
/// physical tensor bytes and does not call the GGML provider or Session collector.
fn scalar_q8_expert_matvec(
    file: &mut File,
    span: vf_ggml_native_ffi::GgufTensorSpan,
    expert_count: usize,
    checkpoint_expert: usize,
    rows: usize,
    columns: usize,
    input: &[f32],
) -> Vec<f64> {
    assert_eq!(input.len(), columns);
    let expert_bytes = span.bytes as usize / expert_count;
    assert_eq!(expert_bytes, rows * (columns / 32) * 34);
    let offset = span.offset + (checkpoint_expert * expert_bytes) as u64;
    file.seek(SeekFrom::Start(offset)).unwrap();
    let mut bytes = vec![0_u8; expert_bytes];
    file.read_exact(&mut bytes).unwrap();
    let row_bytes = columns / 32 * 34;
    let mut output = vec![0.0_f64; rows];
    for row in 0..rows {
        let row_start = row * row_bytes;
        for block in 0..columns / 32 {
            let block_start = row_start + block * 34;
            let scale = u16::from_le_bytes([bytes[block_start], bytes[block_start + 1]]);
            let scale = decode_fp16(scale).unwrap();
            for lane in 0..32 {
                let quant = bytes[block_start + 2 + lane] as i8;
                output[row] += f64::from(input[block * 32 + lane]) * scale * f64::from(quant);
            }
        }
    }
    output
}

fn selected_payload(
    file: &mut File,
    span: vf_ggml_native_ffi::GgufTensorSpan,
    expert_count: usize,
    indices: &[usize],
) -> Vec<u8> {
    let expert_bytes = span.bytes as usize / expert_count;
    let mut output = Vec::with_capacity(expert_bytes * indices.len());
    for &index in indices {
        file.seek(SeekFrom::Start(span.offset + (index * expert_bytes) as u64))
            .unwrap();
        let start = output.len();
        output.resize(start + expert_bytes, 0);
        file.read_exact(&mut output[start..]).unwrap();
    }
    output
}

fn collector_fp16(value: f64) -> f64 {
    decode_fp16(encode_fp16(value).unwrap()).unwrap()
}

#[test]
fn pinned_gguf_reader_resolves_separate_expert_ffn_metadata_and_spans() {
    let path = fixture(false, 32);
    let file = File::open(&path).unwrap();
    let index = GgufIndex::open(&file).unwrap();
    let layout = resolve_expert_ffn_layout(&index, file.metadata().unwrap().len()).unwrap();
    assert_eq!(layout.architecture, "qwen2moe");
    assert_eq!(
        (
            layout.hidden_size,
            layout.expert_count,
            layout.expert_ffn_size
        ),
        (32, 2, 32)
    );
    assert_eq!(layout.layers.len(), 1);
    let layer = &layout.layers[0];
    assert!(!layer.fused_gate_up);
    assert_eq!(layer.gate.dimensions, [32, 32, 2, 1]);
    assert_eq!(layer.up.dimensions, [32, 32, 2, 1]);
    assert_eq!(layer.down.dimensions, [32, 32, 2, 1]);
    assert_eq!(layer_spans(layer).len(), 3);
}

#[test]
fn pinned_gguf_reader_recognizes_the_fused_gate_up_layout_without_guessing_halves() {
    let path = fixture(true, 32);
    let file = File::open(&path).unwrap();
    let index = GgufIndex::open(&file).unwrap();
    let layout = resolve_expert_ffn_layout(&index, file.metadata().unwrap().len()).unwrap();
    let layer = &layout.layers[0];
    assert!(layer.fused_gate_up);
    assert_eq!(layer.gate_name, "blk.0.ffn_gate_up_exps.weight");
    assert_eq!(layer.up_name, "blk.0.ffn_gate_up_exps.weight");
    assert_eq!(layer.gate.dimensions, [32, 64, 2, 1]);
    assert_eq!(layer_spans(layer).len(), 2);
}

#[test]
fn pinned_gguf_reader_resolves_rectangular_ffn_shapes_without_claiming_execution() {
    let path = fixture(false, 64);
    let file = File::open(&path).unwrap();
    let index = GgufIndex::open(&file).unwrap();
    let layout = resolve_expert_ffn_layout(&index, file.metadata().unwrap().len()).unwrap();
    let layer = &layout.layers[0];
    assert_eq!((layout.hidden_size, layout.expert_ffn_size), (32, 64));
    assert_eq!(layer.gate.dimensions, [32, 64, 2, 1]);
    assert_eq!(layer.up.dimensions, [32, 64, 2, 1]);
    assert_eq!(layer.down.dimensions, [64, 32, 2, 1]);
}

#[test]
fn separate_qwen2moe_gate_up_down_rank_sessions_match_scalar_swiglu_oracle() {
    let path = fixture(false, 32);
    let mut checkpoint = File::open(&path).unwrap();
    let index = GgufIndex::open(&checkpoint).unwrap();
    let layout = resolve_expert_ffn_layout(&index, checkpoint.metadata().unwrap().len()).unwrap();
    let layer = &layout.layers[0];
    assert!(!layer.fused_gate_up);

    // The checkpoint has two experts. The VoltaFabric rank's explicit local order
    // [40, 3] maps to checkpoint order [1, 0]; expert IDs are never inferred as file indices.
    let residents = [(40, 0), (3, 1)];
    let slices = [
        ExpertCheckpointSlice {
            resident_expert: 40,
            resident_local: 0,
            checkpoint_expert: 1,
        },
        ExpertCheckpointSlice {
            resident_expert: 3,
            resident_local: 1,
            checkpoint_expert: 0,
        },
    ];
    let checkpoint_indices = [1_usize, 0];
    let mut load_matrix = |tensor_name: &str, span: vf_ggml_native_ffi::GgufTensorSpan| {
        let payload = selected_payload(
            &mut checkpoint,
            span,
            layout.expert_count as usize,
            &checkpoint_indices,
        );
        let binding = binding(tensor_name, &residents, &payload);
        let mut session = Session::connect(&endpoint(7), TIMEOUT).unwrap();
        session
            .load_grouped_matrix_gguf_expert_slices(&binding, &path, tensor_name, &slices, TIMEOUT)
            .unwrap();
        assert_eq!(session.traffic.weight_bytes_sent, payload.len() as u64);
        session
    };
    let mut gate = load_matrix(&layer.gate_name, layer.gate);
    let mut up = load_matrix(&layer.up_name, layer.up);
    let mut down = load_matrix(&layer.down_name, layer.down);
    drop(load_matrix);
    let batch_ledger = ledger();
    let mut batch = 1_u64;
    let mut actual = vec![0.0_f64; HIDDEN];
    let mut expected = vec![0.0_f64; HIDDEN];

    for (expert, checkpoint_expert, route_weight) in [(40_u32, 1_usize, 0.25_f64), (3, 0, 0.75)] {
        let mut input = vec![0.0_f32; HIDDEN];
        input[0] = 1.0;
        let (next_gate, gate_output) =
            run_projection(gate, &input, expert, 1.0, batch, &batch_ledger);
        gate = next_gate;
        batch += 1;
        let (next_up, up_output) = run_projection(up, &input, expert, 1.0, batch, &batch_ledger);
        up = next_up;
        batch += 1;

        let scalar_gate = scalar_q8_expert_matvec(
            &mut checkpoint,
            layer.gate,
            layout.expert_count as usize,
            checkpoint_expert,
            HIDDEN,
            HIDDEN,
            &input,
        );
        let scalar_up = scalar_q8_expert_matvec(
            &mut checkpoint,
            layer.up,
            layout.expert_count as usize,
            checkpoint_expert,
            HIDDEN,
            HIDDEN,
            &input,
        );
        assert_eq!(
            gate_output,
            scalar_gate
                .iter()
                .map(|&v| collector_fp16(v))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            up_output,
            scalar_up
                .iter()
                .map(|&v| collector_fp16(v))
                .collect::<Vec<_>>()
        );

        // Qwen2Moe's synthetic fixture uses SiLU(gate) * up. The VoltaFabric
        // grouped projection returns decoded FP16 values; the route coefficient
        // is applied exactly once, by the down projection's ordinary Collector.
        let activation: Vec<f32> = gate_output
            .iter()
            .zip(&up_output)
            .map(|(&g, &u)| {
                let g = g as f32;
                (g / (1.0 + (-g).exp())) * (u as f32)
            })
            .collect();
        let oracle_activation: Vec<f32> = scalar_gate
            .iter()
            .zip(&scalar_up)
            .map(|(&g, &u)| {
                let g = collector_fp16(g) as f32;
                (g / (1.0 + (-g).exp())) * (collector_fp16(u) as f32)
            })
            .collect();
        assert_eq!(activation, oracle_activation);

        let (next_down, down_output) = run_projection(
            down,
            &activation,
            expert,
            route_weight,
            batch,
            &batch_ledger,
        );
        down = next_down;
        batch += 1;
        let scalar_down = scalar_q8_expert_matvec(
            &mut checkpoint,
            layer.down,
            layout.expert_count as usize,
            checkpoint_expert,
            HIDDEN,
            HIDDEN,
            &oracle_activation,
        );
        let expected_slot: Vec<f64> = scalar_down
            .iter()
            .map(|&value| collector_fp16(value) * route_weight)
            .collect();
        assert_eq!(down_output, expected_slot);
        for column in 0..HIDDEN {
            actual[column] += down_output[column];
            expected[column] += expected_slot[column];
        }
    }
    assert_eq!(actual, expected);

    for session in [gate, up, down] {
        let receipt = session.close(TIMEOUT);
        assert_eq!(receipt.bye_execs, Some(2));
        assert_eq!(receipt.exit_code, Some(0));
    }
}
