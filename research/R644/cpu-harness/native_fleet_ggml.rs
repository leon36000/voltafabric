use std::collections::BTreeMap;

use voltafabric::{
    completion,
    moe::{Choice, Registry, Replica, Stamp},
    native_ggml_cpu::{Binding, Fleet, Resident},
    routing::RoutingProblem,
    sha256,
    strata_native_layout::{NativeLayout, VerifiedNativeExpert},
};

const PACK_BYTES: u64 = 50_292_326_400;
const LAYER: u32 = 0;
const EXPERT: u32 = 8;
const EPOCH: u64 = 77;

fn synthetic_resident() -> Resident {
    let layout = NativeLayout::parse_sc117_v3(
        include_str!("../../../fixtures/strata_sc117_native_experts_v3.txt"),
        PACK_BYTES,
    )
    .unwrap();
    let span = layout.expert(LAYER as usize, u64::from(EXPERT)).unwrap();
    assert_eq!((span.gate_up_type, span.down_type), (18, 20));

    // Match the bounded synthetic block initialization used by the native
    // provider test; this is format data, not a model/checkpoint payload.
    let gate_up_bytes = (2560 / 256 * 98) * 640 * 2;
    let mut bytes = vec![0_u8; span.bytes as usize];
    for offset in (0..gate_up_bytes).step_by(98) {
        bytes[offset + 1] = 0x28;
    }
    for offset in (gate_up_bytes..bytes.len()).step_by(18) {
        bytes[offset + 1] = 0x28;
    }
    let binding = Binding {
        model_revision: "r644-synthetic-sc117".into(),
        rank: 0,
        layer: LAYER,
        expert: EXPERT,
        local: 0,
        epoch: EPOCH,
        index_sha256: layout.index_sha256().into(),
        blob_sha256: sha256::hex(&bytes),
    };
    Resident::load(
        &layout,
        VerifiedNativeExpert {
            layer: LAYER as usize,
            expert: u64::from(EXPERT),
            span,
            bytes,
        },
        binding,
    )
    .unwrap()
}

fn problem(batch: u64, weight: f64) -> RoutingProblem {
    RoutingProblem {
        hidden: 2560,
        experts: vec![EXPERT],
        routes: vec![vec![Choice {
            expert: EXPERT,
            weight,
        }]],
        registry: Registry {
            epoch: EPOCH,
            copies: vec![Replica {
                expert: EXPERT,
                rank: 0,
                local: 0,
            }],
        },
        stamp: Stamp {
            run_id: "r644-run".into(),
            model_revision: "r644-synthetic-sc117".into(),
            plan_id: "r644-plan".into(),
            layer: LAYER,
            batch,
            epoch: EPOCH,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: [(0, vec![0])].into_iter().collect(),
    }
}

#[test]
fn native_ggml_resident_flows_through_rust_owner_and_collector_once() {
    let mut resident = synthetic_resident();
    let input: Vec<f32> = (0..2560)
        .map(|i| ((i as f32) * 0.17).sin() / 16.0)
        .collect();
    let unweighted = resident.forward(&input).unwrap();
    assert!(unweighted.iter().all(|x| x.is_finite()));
    assert!(unweighted.iter().any(|x| *x != 0.0));

    let mut fleet = Fleet::new(vec![resident]).unwrap();
    let routed = problem(9001, 0.25);

    let mut wrong_epoch = routed.clone();
    wrong_epoch.stamp.epoch += 1;
    assert!(fleet
        .run_fp16_collected(&wrong_epoch, std::slice::from_ref(&input), 1)
        .is_err());

    let receipt = fleet
        .run_fp16_collected(&routed, std::slice::from_ref(&input), 1)
        .unwrap();
    let expected: Vec<f64> = unweighted
        .into_iter()
        .map(|value| {
            completion::decode_fp16(completion::encode_fp16(f64::from(value)).unwrap()).unwrap()
                * 0.25
        })
        .collect();
    assert_eq!(receipt.stamp, routed.stamp);
    assert_eq!(receipt.attempt, 1);
    assert_eq!(receipt.completed_slots, 1);
    assert_eq!(receipt.output, vec![expected]);

    let mut wrong_plan = problem(9002, 0.25);
    wrong_plan.stamp.plan_id = "other-plan".into();
    assert!(fleet
        .run_fp16_collected(&wrong_plan, std::slice::from_ref(&input), 2)
        .is_err());
    assert!(fleet
        .run_fp16_collected(&routed, std::slice::from_ref(&input), 2)
        .is_err());
}
