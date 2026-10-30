use std::collections::BTreeMap;
use voltafabric::{moe, routing};

fn problem() -> routing::RoutingProblem {
    routing::RoutingProblem {
        hidden: 8,
        experts: vec![0, 1, 2, 3],
        routes: vec![vec![
            moe::Choice {
                expert: 0,
                weight: 0.25,
            },
            moe::Choice {
                expert: 1,
                weight: 0.75,
            },
        ]],
        registry: moe::Registry {
            epoch: 7,
            copies: vec![
                moe::Replica {
                    expert: 0,
                    rank: 0,
                    local: 0,
                },
                moe::Replica {
                    expert: 1,
                    rank: 1,
                    local: 0,
                },
                moe::Replica {
                    expert: 2,
                    rank: 1,
                    local: 1,
                },
                moe::Replica {
                    expert: 3,
                    rank: 0,
                    local: 1,
                },
            ],
        },
        stamp: moe::Stamp {
            run_id: "run".into(),
            model_revision: "model".into(),
            plan_id: "plan".into(),
            layer: 0,
            batch: 0,
            epoch: 7,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(0, vec![0, 1]), (1, vec![0, 1])]),
    }
}

#[test]
fn resident_first_then_heat_and_prior_without_changing_dispatch() {
    let p = problem();
    let before = routing::dispatch(&p).unwrap();
    let heat = BTreeMap::from([(0, 3), (1, 200), (2, 200), (3, 1)]);
    assert_eq!(
        routing::rank_resident_candidates(&p, 0, &heat, &[2, 1]).unwrap(),
        vec![0, 3, 2, 1]
    );
    assert_eq!(routing::dispatch(&p).unwrap(), before);
    assert_eq!(
        before.slots.iter().map(|s| s.weight).collect::<Vec<_>>(),
        vec![0.25, 0.75]
    );
}

#[test]
fn invalid_metadata_never_yields_a_plan() {
    let mut p = problem();
    assert!(routing::rank_resident_candidates(&p, 2, &BTreeMap::new(), &[]).is_err());
    assert!(routing::rank_resident_candidates(&p, 0, &BTreeMap::from([(9, 1)]), &[]).is_err());
    assert!(routing::rank_resident_candidates(&p, 0, &BTreeMap::new(), &[1, 1]).is_err());
    assert!(routing::rank_resident_candidates(&p, 0, &BTreeMap::new(), &[9]).is_err());
    p.registry.epoch = 8;
    assert!(routing::rank_resident_candidates(&p, 0, &BTreeMap::new(), &[]).is_err());
}

#[test]
fn ties_are_stable_and_duplicate_replica_is_rejected() {
    let mut p = problem();
    p.experts.reverse();
    assert_eq!(
        routing::rank_resident_candidates(&p, 0, &BTreeMap::new(), &[]).unwrap(),
        vec![0, 3, 1, 2]
    );
    p.registry.copies.push(moe::Replica {
        expert: 0,
        rank: 0,
        local: 2,
    });
    p.resident_order.get_mut(&0).unwrap().push(2);
    assert!(routing::rank_resident_candidates(&p, 0, &BTreeMap::new(), &[]).is_err());
}
