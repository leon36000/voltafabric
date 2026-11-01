#![cfg(feature = "native-ggml-cpu")]
use std::collections::BTreeMap;
use vf_ggml_native_ffi::NativeRoutedMatrix;
use voltafabric::{
    completion::{self, OutputSlot, RoutedRawOutput},
    moe::{self, Choice, Registry, Replica, Stamp},
    native_ggml_cpu::{MatrixBinding, QuantizedMatrix},
    routing::{self, RoutingProblem},
    sha256,
};
#[test]
fn bound_matrices_are_resident_and_verify_identity() {
    for (kind, row_bytes) in [(8, 272), (12, 144), (13, 176), (14, 210), (23, 136)] {
        let bytes = vec![0_u8; row_bytes * 3];
        let binding = MatrixBinding {
            model_revision: "model-observed".into(),
            layer: 0,
            tensor: "blk.0.attn_qkv.weight".into(),
            ggml_type: kind,
            rows: 3,
            columns: 256,
            sha256: sha256::hex(&bytes),
        };
        let mut resident = QuantizedMatrix::load(binding.clone(), &bytes).unwrap();
        let first = resident.forward(&vec![0.125; 256]).unwrap();
        assert_eq!(first, vec![0.0; 3]);
        assert_eq!(resident.forward(&vec![0.125; 256]).unwrap(), first);
        assert_eq!(resident.binding(), &binding);
        assert!(resident.forward(&[f32::NAN; 256]).is_err());
        assert!(resident.forward(&[0.0; 255]).is_err());
        let mut altered = bytes.clone();
        altered[0] = 1;
        assert!(QuantizedMatrix::load(binding.clone(), &altered).is_err());
        let mut wrong = binding.clone();
        wrong.columns = 255;
        assert!(QuantizedMatrix::load(wrong, &bytes).is_err());
        let mut wrong = binding;
        wrong.ggml_type = 39;
        assert!(QuantizedMatrix::load(wrong, &bytes).is_err());
    }
}

#[test]
fn grouped_ggml_rows_enter_the_collector_unweighted_and_atomically() {
    const H: usize = 32;
    const ROW_BYTES: usize = 34; // Q8_0: FP16 scale followed by 32 signed bytes.
    let physical_values = [1_u8, 2, 4];
    let mut weights = vec![0_u8; 3 * H * ROW_BYTES];
    for (expert, &value) in physical_values.iter().enumerate() {
        for row in 0..H {
            let start = (expert * H + row) * ROW_BYTES;
            weights[start] = 0x00;
            weights[start + 1] = 0x3c; // scale = 1.0
            weights[start + 2] = value;
        }
    }

    let problem = RoutingProblem {
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
            run_id: "r655-run".into(),
            model_revision: "r655-model".into(),
            plan_id: "r655-plan".into(),
            layer: 5,
            batch: 41,
            epoch: 9,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(7, vec![1, 2, 0])]),
    };
    let dispatch = routing::dispatch(&problem).unwrap();
    let rank = &dispatch.ranks[0];
    assert_eq!(rank.rank, 7);
    assert_eq!(rank.slots.len(), dispatch.slots.len());

    let mut inputs = vec![vec![0.0_f32; H]; 2];
    inputs[0][0] = 1.0;
    inputs[1][0] = 2.0;
    let slot_inputs = dispatch
        .slots
        .iter()
        .flat_map(|slot| inputs[slot.token].iter().copied())
        .collect::<Vec<_>>();
    let ids = rank
        .slots
        .iter()
        .map(|slot| slot.physical_expert as u32)
        .collect::<Vec<_>>();
    let mut matrix = NativeRoutedMatrix::load(8, 3, H, H, &weights).unwrap();
    let raw_rows = matrix.forward(&slot_inputs, &ids).unwrap();
    assert_eq!(raw_rows.len(), dispatch.slots.len() * H);
    for (slot, expected) in [2.0_f32, 4.0, 2.0, 4.0].iter().enumerate() {
        assert!(
            (raw_rows[slot * H] - expected).abs() < 1e-3,
            "slot {slot} got {}, expected {expected}",
            raw_rows[slot * H]
        );
    }

    let fp16_bits = raw_rows
        .iter()
        .map(|&value| completion::encode_fp16(f64::from(value)).unwrap())
        .collect::<Vec<_>>();
    let raw = RoutedRawOutput {
        stamp: problem.stamp.clone(),
        rank: 7,
        attempt: 13,
        rows: rank.slots.len(),
        columns: H,
        slots: rank
            .slots
            .iter()
            .map(|slot| OutputSlot {
                token: slot.slot.token,
                slot: slot.slot.slot,
                expert: slot.slot.expert,
                local: slot.slot.local,
            })
            .collect(),
        fp16_bits,
    };
    let mut collector = moe::Collector::from_routing(&problem).unwrap();
    assert_eq!(
        completion::accept_routed_output(12, raw.clone(), &mut collector)
            .unwrap_err()
            .code,
        "STALE_EXECUTION"
    );
    assert_eq!(
        completion::accept_routed_output(13, raw.clone(), &mut collector).unwrap(),
        4
    );

    let mut expected = vec![vec![0.0_f64; H]; 2];
    for (index, slot) in dispatch.slots.iter().enumerate() {
        for column in 0..H {
            let decoded = completion::decode_fp16(raw.fp16_bits[index * H + column]).unwrap();
            expected[slot.token][column] += slot.weight * decoded;
        }
    }
    let collected = collector.finish().unwrap();
    assert_eq!(collected, expected);
    assert!((collected[0][0] - 3.5).abs() < 1e-6);
    assert!((collected[1][0] - 2.0).abs() < 1e-6);

    let mut partial = moe::Collector::from_routing(&problem).unwrap();
    let mut incomplete = raw;
    incomplete.rows = 1;
    incomplete.slots.truncate(1);
    incomplete.fp16_bits.truncate(H);
    assert_eq!(
        completion::accept_routed_output(13, incomplete, &mut partial).unwrap(),
        1
    );
    assert_eq!(partial.finish().unwrap_err().code, "BATCH_INCOMPLETE");
}
