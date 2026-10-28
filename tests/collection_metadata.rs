mod common;
use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};
use voltafabric::{
    moe::{self, Choice, Collector, Completion, Replica},
    routing::{self, RoutingProblem},
};

fn complete(stamp: &moe::Stamp, slot: &moe::Slot, value: Vec<f64>) -> Completion {
    Completion {
        stamp: stamp.clone(),
        token: slot.token,
        slot: slot.slot,
        expert: slot.expert,
        rank: slot.rank,
        local: slot.local,
        value,
    }
}
#[test]
fn metadata_collector_needs_neither_dense_weights_nor_activations() {
    let mut dense = common::problem();
    for expert in &mut dense.model.experts {
        expert.gate.clear();
        expert.up.clear();
        expert.down.clear();
    }
    dense.batch.inputs[0] = vec![f64::NAN];
    assert!(Collector::new(&dense).is_err());
    let p = routing::from_dense(&dense);
    let dispatched = routing::dispatch(&p).unwrap();
    let mut collector = Collector::from_routing(&p).unwrap();
    for s in dispatched.slots.iter().rev() {
        collector
            .accept(complete(&p.stamp, s, vec![s.expert as f64, 2.]))
            .unwrap();
    }
    assert_eq!(
        collector.finish().unwrap(),
        vec![vec![81.2, 2.], vec![3., 0.6]]
    );
}
#[test]
fn metadata_and_dense_collectors_match_existing_numerical_oracle_exactly() {
    for seed in [0, 1, 8] {
        let p: moe::Problem = serde_json::from_str(
            &std::fs::read_to_string(format!("fixtures/moe_seed_{seed}.json")).unwrap(),
        )
        .unwrap();
        let mut a = Collector::new(&p).unwrap();
        let mut b = Collector::from_routing(&routing::from_dense(&p)).unwrap();
        for slot in moe::dispatch(&p).unwrap().iter().rev() {
            let c = moe::compute_slot(&p, slot).unwrap();
            a.accept(c.clone()).unwrap();
            b.accept(c).unwrap();
        }
        assert_eq!(a.finish().unwrap(), b.finish().unwrap());
    }
}
#[test]
fn metadata_collector_replicas_and_physical_order_are_explicit() {
    let mut p = routing::from_dense(&common::problem());
    p.registry.copies.push(Replica {
        expert: 10,
        rank: 9,
        local: 900,
    });
    p.resident_order.get_mut(&9).unwrap().insert(0, 900);
    p.preferred_owners.insert(10, 9);
    p.resident_order.get_mut(&2).unwrap().reverse();
    let d = routing::dispatch(&p).unwrap();
    let s = d.slots.iter().find(|s| s.expert == 10).unwrap();
    assert_eq!((s.rank, s.local), (9, 900));
    let mut c = Collector::from_routing(&p).unwrap();
    let mut foreign = complete(&p.stamp, s, vec![1., 2.]);
    foreign.rank = 2;
    foreign.local = 0;
    assert!(c.accept(foreign).is_err());
    assert!(c.accept(complete(&p.stamp, s, vec![1., 2.])).is_err());
    assert!(c.finish().is_err());
}

#[test]
fn routed_fp16_collection_accepts_out_of_order_ranks_and_selected_replica_once() {
    let mut p = routing::from_dense(&common::problem());
    p.registry.copies.push(Replica {
        expert: 10,
        rank: 9,
        local: 900,
    });
    p.resident_order.get_mut(&9).unwrap().push(900);
    p.preferred_owners.insert(10, 9);
    p.resident_order.get_mut(&2).unwrap().reverse();
    let dispatch = routing::dispatch(&p).unwrap();
    let owner = dispatch.ranks.iter().find(|rank| rank.rank == 9).unwrap();
    assert_eq!(
        owner
            .slots
            .iter()
            .filter(|route| route.slot.expert == 10)
            .map(|route| (route.slot.token, route.slot.rank, route.slot.local))
            .collect::<Vec<_>>(),
        vec![(0, 9, 900), (1, 9, 900)]
    );

    let mut collector = Collector::from_routing(&p).unwrap();
    for rank in dispatch.ranks.iter().rev() {
        let output = voltafabric::completion::RoutedRawOutput {
            stamp: rank.stamp.clone(),
            rank: rank.rank,
            attempt: 17,
            rows: rank.slots.len(),
            columns: p.hidden,
            slots: rank
                .slots
                .iter()
                .map(|route| voltafabric::completion::OutputSlot {
                    token: route.slot.token,
                    slot: route.slot.slot,
                    expert: route.slot.expert,
                    local: route.slot.local,
                })
                .collect(),
            fp16_bits: vec![0x3c00; rank.slots.len() * p.hidden],
        };
        assert_eq!(
            voltafabric::completion::accept_routed_output(17, output, &mut collector).unwrap(),
            rank.slots.len()
        );
    }
    assert_eq!(
        collector.finish().unwrap(),
        vec![vec![1.0; 2], vec![0.3; 2]]
    );
}

#[test]
fn metadata_collector_rejects_every_stale_stamp_field() {
    let p = routing::from_dense(&common::problem());
    let s = routing::dispatch(&p).unwrap().slots.remove(0);
    for field in 0..6 {
        let mut c = Collector::from_routing(&p).unwrap();
        let mut r = complete(&p.stamp, &s, vec![1., 2.]);
        match field {
            0 => r.stamp.run_id.push('x'),
            1 => r.stamp.model_revision.push('x'),
            2 => r.stamp.plan_id.push('x'),
            3 => r.stamp.layer += 1,
            4 => r.stamp.batch += 1,
            _ => r.stamp.epoch += 1,
        }
        assert!(c.accept(r).is_err());
        assert!(c.accept(complete(&p.stamp, &s, vec![1., 2.])).is_err());
        assert!(c.finish().is_err());
    }
}
#[test]
fn metadata_collector_rejects_slot_and_value_corruption_terminally() {
    let p = routing::from_dense(&common::problem());
    let s = routing::dispatch(&p).unwrap().slots.remove(0);
    for kind in 0..8 {
        let mut c = Collector::from_routing(&p).unwrap();
        let mut r = complete(&p.stamp, &s, vec![1., 2.]);
        match kind {
            0 => r.token += 20,
            1 => r.slot += 1,
            2 => r.expert = 25,
            3 => r.rank += 1,
            4 => r.local += 2,
            5 => r.value.pop().map(|_| ()).unwrap(),
            6 => r.value[0] = f64::NAN,
            _ => r.value[0] = f64::INFINITY,
        }
        assert!(c.accept(r).is_err());
        assert!(c.finish().is_err());
    }
}
#[test]
fn metadata_collector_empty_missing_duplicate_and_expiry_have_no_partial_result() {
    let p = routing::from_dense(&common::problem());
    let s = routing::dispatch(&p).unwrap().slots.remove(0);
    assert!(Collector::from_routing(&p).unwrap().finish().is_err());
    let mut c = Collector::from_routing(&p).unwrap();
    let r = complete(&p.stamp, &s, vec![1., 2.]);
    c.accept(r.clone()).unwrap();
    assert!(c.accept(r.clone()).is_err());
    assert!(c.finish().is_err());
    let mut c = Collector::from_routing(&p).unwrap();
    c.expire();
    assert!(c.accept(r).is_err());
    assert!(c.finish().is_err());
    let mut empty = p;
    empty.routes.clear();
    assert_eq!(
        Collector::from_routing(&empty).unwrap().finish().unwrap(),
        Vec::<Vec<f64>>::new()
    );
}
#[test]
fn metadata_collector_preserves_original_slot_order_for_roundoff_sensitive_values() {
    let mut p = routing::from_dense(&common::problem());
    p.routes = vec![vec![
        Choice {
            expert: 10,
            weight: 1.,
        },
        Choice {
            expert: 25,
            weight: 1.,
        },
        Choice {
            expert: 99,
            weight: 1.,
        },
    ]];
    let d = routing::dispatch(&p).unwrap();
    let values = [1e16, -1e16, 1.];
    let mut c = Collector::from_routing(&p).unwrap();
    for s in d.slots.iter().rev() {
        c.accept(complete(&p.stamp, s, vec![values[s.slot], 0.]))
            .unwrap();
    }
    assert_eq!(c.finish().unwrap(), vec![vec![1., 0.]]);
}
#[test]
fn metadata_collector_zero_coefficients_are_not_silently_dropped_and_overflow_rejects() {
    let mut p = routing::from_dense(&common::problem());
    for choices in &mut p.routes {
        for choice in choices {
            choice.weight = 0.;
        }
    }
    assert!(Collector::from_routing(&p).unwrap().finish().is_err());
    let mut c = Collector::from_routing(&p).unwrap();
    for s in routing::dispatch(&p).unwrap().slots {
        c.accept(complete(&p.stamp, &s, vec![f64::MAX, f64::MAX]))
            .unwrap();
    }
    assert_eq!(c.finish().unwrap(), vec![vec![0., 0.], vec![0., 0.]]);
    p.routes[0][0].weight = f64::MAX;
    let mut c = Collector::from_routing(&p).unwrap();
    for s in routing::dispatch(&p).unwrap().slots {
        c.accept(complete(&p.stamp, &s, vec![2., 2.])).unwrap();
    }
    assert!(c.finish().is_err());
}
fn wide(tokens: usize) -> RoutingProblem {
    RoutingProblem {
        hidden: 1024,
        experts: (0..64).collect(),
        routes: vec![
            (0..64)
                .map(|expert| Choice { expert, weight: 1. })
                .collect();
            tokens
        ],
        registry: moe::Registry {
            epoch: 7,
            copies: (0..64)
                .map(|expert| Replica {
                    expert,
                    rank: 2,
                    local: expert,
                })
                .collect(),
        },
        stamp: common::problem().stamp,
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(2, (0..64).collect())]),
    }
}
#[test]
fn metadata_collection_bounds_pending_slot_values_before_dispatch_and_buffers() {
    let too_large = wide(257);
    routing::validate(&too_large).unwrap();
    assert_eq!(
        Collector::from_routing(&too_large).err().unwrap().code,
        "COLLECTION_TOO_LARGE"
    );
    assert!(Collector::from_routing(&wide(256)).is_ok());
}
#[test]
fn metadata_collector_rejects_invalid_registry_before_creating_collector() {
    let mut p = routing::from_dense(&common::problem());
    p.registry.epoch += 1;
    assert!(Collector::from_routing(&p).is_err());
}
#[test]
fn metadata_only_collect_cli_supports_numeric_results_without_model_weights() {
    let p = routing::from_dense(&common::problem());
    let completions: Vec<_> = routing::dispatch(&p)
        .unwrap()
        .slots
        .into_iter()
        .map(|s| complete(&p.stamp, &s, vec![1., 2.]))
        .collect();
    let payload = serde_json::json!({"problem":p,"completions":completions});
    let mut child = Command::new(env!("CARGO_BIN_EXE_vf"))
        .args(["collect-routing", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(&payload).unwrap())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{:?}",
        String::from_utf8_lossy(&out.stdout)
    );
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(json["result"], serde_json::json!([[1., 2.], [0.3, 0.6]]));
}
