use std::collections::BTreeMap;
use voltafabric::{
    completion::{OutputSlot, RoutedRawOutput},
    moe::{Choice, Registry, Replica, Stamp},
    routing::RoutingProblem,
    sc117_layer_completion::{finish, LayerInputs, StampedValues},
};
fn fixture() -> LayerInputs {
    let stamp = Stamp {
        run_id: "r".into(),
        model_revision: "m".into(),
        plan_id: "p".into(),
        layer: 2,
        batch: 7,
        epoch: 1,
    };
    let p = RoutingProblem {
        hidden: 2560,
        experts: vec![10, 20],
        routes: vec![vec![
            Choice {
                expert: 10,
                weight: 0.25,
            },
            Choice {
                expert: 20,
                weight: 0.75,
            },
        ]],
        registry: Registry {
            epoch: 1,
            copies: vec![
                Replica {
                    expert: 10,
                    rank: 2,
                    local: 0,
                },
                Replica {
                    expert: 20,
                    rank: 3,
                    local: 0,
                },
            ],
        },
        stamp: stamp.clone(),
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(2, vec![0]), (3, vec![0])]),
    };
    let raw = |rank, slot, expert, bits| RoutedRawOutput {
        stamp: stamp.clone(),
        rank,
        attempt: 5,
        rows: 1,
        columns: 2560,
        slots: vec![OutputSlot {
            token: 0,
            slot,
            expert,
            local: 0,
        }],
        fp16_bits: vec![bits; 2560],
    };
    let values = |v| StampedValues {
        stamp: stamp.clone(),
        attempt: 5,
        values: v,
    };
    LayerInputs {
        problem: p,
        attempt: 5,
        outputs: vec![raw(3, 1, 20, 0x4000), raw(2, 0, 10, 0x3c00)],
        shared: values(vec![0.5; 2560]),
        attention_residual: values(vec![3.0; 10240]),
        injection: values(vec![0.0; 4]),
    }
}
#[test]
fn completes_only_once_weighted_rows_and_plain_shared() {
    let out = finish(fixture()).unwrap();
    assert_eq!(out, vec![5.25; 10240]);
}
#[test]
fn rejects_missing_duplicate_stale_and_bad_shared() {
    let mut x = fixture();
    x.outputs.pop();
    assert!(finish(x).is_err());
    let mut x = fixture();
    x.outputs.push(x.outputs[0].clone());
    assert!(finish(x).is_err());
    let mut x = fixture();
    x.outputs[0].attempt = 4;
    assert!(finish(x).is_err());
    let mut x = fixture();
    x.shared.stamp.batch += 1;
    assert!(finish(x).is_err());
    let mut x = fixture();
    x.injection.values.pop();
    assert!(finish(x).is_err());
    let mut x = fixture();
    x.shared.values[2559] = f32::NAN;
    assert!(finish(x).is_err());
    let mut x = fixture();
    x.outputs[1].fp16_bits[2559] = 0x7c00;
    assert!(finish(x).is_err());
}
