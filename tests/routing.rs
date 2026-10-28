use std::collections::BTreeMap;
use voltafabric::{moe, routing::*};

fn problem() -> RoutingProblem {
    RoutingProblem {
        hidden: 8,
        experts: vec![10, 25, 99],
        routes: vec![
            vec![
                moe::Choice {
                    expert: 10,
                    weight: 2.5,
                },
                moe::Choice {
                    expert: 99,
                    weight: 0.0,
                },
            ],
            vec![
                moe::Choice {
                    expert: 25,
                    weight: 4.0,
                },
                moe::Choice {
                    expert: 10,
                    weight: 0.25,
                },
            ],
        ],
        registry: moe::Registry {
            epoch: 7,
            copies: vec![
                moe::Replica {
                    expert: 10,
                    rank: 2,
                    local: 3,
                },
                moe::Replica {
                    expert: 25,
                    rank: 2,
                    local: 900,
                },
                moe::Replica {
                    expert: 99,
                    rank: 4,
                    local: 3,
                },
                moe::Replica {
                    expert: 10,
                    rank: 4,
                    local: 900,
                },
            ],
        },
        stamp: moe::Stamp {
            run_id: "run".into(),
            model_revision: "model".into(),
            plan_id: "plan".into(),
            layer: 2,
            batch: 8,
            epoch: 7,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(2, vec![900, 3]), (4, vec![3, 900]), (17, vec![])]),
    }
}

#[test]
fn metadata_only_sparse_ids_and_physical_order() {
    let p = problem();
    let d = dispatch(&p).unwrap();
    assert_eq!((d.hidden, d.tokens), (8, 2));
    assert_eq!(d.stamp, p.stamp);
    assert_eq!(
        d.ranks.iter().map(|r| r.rank).collect::<Vec<_>>(),
        vec![2, 4, 17]
    );
    assert_eq!(d.ranks[0].resident_order, vec![900, 3]);
    assert_eq!(
        d.ranks[0]
            .slots
            .iter()
            .map(|s| s.physical_expert)
            .collect::<Vec<_>>(),
        vec![1, 0, 1]
    );
    assert_eq!(
        d.slots
            .iter()
            .map(|s| (s.token, s.slot, s.expert, s.rank, s.local))
            .collect::<Vec<_>>(),
        vec![
            (0, 0, 10, 2, 3),
            (0, 1, 99, 4, 3),
            (1, 0, 25, 2, 900),
            (1, 1, 10, 2, 3)
        ]
    );
    assert_eq!(
        d.slots.iter().map(|s| s.weight).collect::<Vec<_>>(),
        vec![2.5, 0.0, 4.0, 0.25]
    );
    assert!(d.ranks[2].slots.is_empty());
    for rank in d.ranks {
        assert_eq!((rank.hidden, rank.tokens), (8, 2));
        assert_eq!(rank.stamp, p.stamp);
        for slot in rank.slots {
            assert_eq!(rank.resident_order[slot.physical_expert], slot.slot.local);
            assert_eq!(slot.slot.rank, rank.rank);
        }
    }
}

#[test]
fn replicas_selected_once_and_unused_experts_keep_residency() {
    let mut p = problem();
    p.preferred_owners.insert(10, 4);
    p.routes = vec![vec![moe::Choice {
        expert: 10,
        weight: 3.0,
    }]];
    let d = dispatch(&p).unwrap();
    assert_eq!(d.slots.len(), 1);
    assert_eq!(
        (d.slots[0].rank, d.ranks[1].slots[0].physical_expert),
        (4, 1)
    );
    assert!(d.ranks[0].slots.is_empty());
    assert_eq!(d.ranks[0].resident_order, vec![900, 3]);
    p.routes.clear();
    let empty = dispatch(&p).unwrap();
    assert!(empty.slots.is_empty());
    assert_eq!(empty.ranks.len(), 3);
}

#[test]
fn routing_is_independent_of_metadata_permutations() {
    let p = problem();
    let d = dispatch(&p).unwrap();
    let mut permuted = p.clone();
    permuted.experts.reverse();
    permuted.registry.copies.reverse();
    assert_eq!(dispatch(&permuted).unwrap(), d);
}

fn rejected(change: impl FnOnce(&mut RoutingProblem)) {
    let mut p = problem();
    change(&mut p);
    assert_eq!(dispatch(&p).unwrap_err().code, "INVALID_INPUT");
}

#[test]
fn malformed_residency_and_aliases_rejected() {
    rejected(|p| {
        p.resident_order.remove(&2);
    });
    rejected(|p| {
        p.resident_order.insert(2, vec![900]);
    });
    rejected(|p| {
        p.resident_order.insert(2, vec![3, 3]);
    });
    rejected(|p| {
        p.resident_order.insert(2, vec![3, 901]);
    });
    rejected(|p| {
        p.resident_order.insert(17, vec![1]);
    });
    rejected(|p| p.registry.copies[1].local = 3);
    rejected(|p| {
        p.registry.copies.push(moe::Replica {
            expert: 10,
            rank: 2,
            local: 901,
        })
    });
    rejected(|p| p.registry.copies[0].expert = 500);
    rejected(|p| p.registry.copies.retain(|r| r.expert != 99));
    rejected(|p| {
        p.preferred_owners.insert(10, 17);
    });
}

#[test]
fn stale_identity_bad_routes_and_dimensions_rejected() {
    rejected(|p| p.stamp.epoch += 1);
    rejected(|p| p.stamp.run_id = " ".into());
    rejected(|p| p.stamp.model_revision = "x".repeat(257));
    rejected(|p| p.stamp.plan_id.clear());
    rejected(|p| p.hidden = 0);
    rejected(|p| p.hidden = 65537);
    rejected(|p| p.experts.clear());
    rejected(|p| p.experts.push(10));
    rejected(|p| p.routes[0].clear());
    rejected(|p| {
        p.routes[0].pop();
    });
    rejected(|p| p.routes[0][1].expert = 10);
    rejected(|p| p.routes[0][1].expert = 500);
    for weight in [f64::NAN, f64::INFINITY, -1.0] {
        rejected(|p| p.routes[0][0].weight = weight);
    }
}

#[test]
fn resource_bounds_rejected_before_dispatch_allocation() {
    rejected(|p| p.routes = vec![p.routes[0].clone(); 4097]);
    rejected(|p| {
        p.hidden = 65536;
        p.routes = vec![p.routes[0].clone(); 257];
    });
    rejected(|p| p.experts = (0..1025).collect());
    rejected(|p| p.registry.copies = vec![p.registry.copies[0].clone(); 65537]);
    rejected(|p| p.routes = vec![vec![p.routes[0][0].clone(); 65]]);
    rejected(|p| {
        for rank in 100..164 {
            p.resident_order.insert(rank, vec![]);
        }
    });
}

#[test]
fn boundary_dimensions_empty_workers_and_topk_are_accepted() {
    let mut p = problem();
    p.routes.clear();
    p.hidden = 65536;
    for rank in 100..161 {
        p.resident_order.insert(rank, vec![]);
    }
    assert_eq!(dispatch(&p).unwrap().ranks.len(), 64);
    p.hidden = 4096;
    p.routes = vec![
        vec![moe::Choice {
            expert: 10,
            weight: 1.0
        }];
        4096
    ];
    assert_eq!(dispatch(&p).unwrap().slots.len(), 4096);
}

#[test]
fn serde_defaults_and_unknown_fields() {
    let p = problem();
    let mut json = serde_json::to_value(&p).unwrap();
    json.as_object_mut().unwrap().remove("preferred_owners");
    let parsed: RoutingProblem = serde_json::from_value(json.clone()).unwrap();
    assert!(parsed.preferred_owners.is_empty());
    dispatch(&parsed).unwrap();
    json["dense_weights"] = serde_json::json!([]);
    assert!(serde_json::from_value::<RoutingProblem>(json).is_err());
}

#[test]
fn dense_conversion_matches_oracle_on_fixtures_and_permutations() {
    for fixture in [
        include_str!("../fixtures/moe_seed_0.json"),
        include_str!("../fixtures/moe_seed_1.json"),
        include_str!("../fixtures/moe_seed_8.json"),
    ] {
        let mut p: moe::Problem = serde_json::from_str(fixture).unwrap();
        for _ in 0..2 {
            let routed = from_dense(&p);
            assert_eq!(dispatch(&routed).unwrap().slots, moe::dispatch(&p).unwrap());
            for residents in routed.resident_order.values() {
                assert!(residents.windows(2).all(|x| x[0] < x[1]));
            }
            p.model.experts.reverse();
            p.registry.copies.reverse();
            for row in &mut p.batch.routes {
                row.reverse();
            }
        }
    }
}

#[test]
fn dense_conversion_only_reads_metadata_and_routes() {
    let mut p: moe::Problem =
        serde_json::from_str(include_str!("../fixtures/moe_seed_0.json")).unwrap();
    for expert in &mut p.model.experts {
        expert.gate.clear();
        expert.up.clear();
        expert.down.clear();
    }
    p.batch.inputs.clear();
    assert!(moe::dispatch(&p).is_err());
    let r = from_dense(&p);
    assert_eq!(r.routes.len(), p.batch.routes.len());
    dispatch(&r).unwrap();
}

// A40 guard sweep: `rejected` above checks only the INVALID_INPUT code, which every
// clause shares, and several of its inputs break two clauses at once. Each case here
// breaks exactly one; with that clause disabled it was accepted or refused later.
fn refused_for(name: &str, change: impl FnOnce(&mut RoutingProblem), reason: &str) {
    let mut p = problem();
    change(&mut p);
    let error = validate(&p).expect_err(name);
    assert!(
        error.message.contains(reason),
        "{name}: expected {reason:?}, got {error}"
    );
}
fn replica(expert: u32, rank: u32, local: u32) -> moe::Replica {
    moe::Replica {
        expert,
        rank,
        local,
    }
}
#[test]
fn dimension_and_batch_clauses_refuse_for_their_own_reason() {
    let dims = "routing dimensions outside bounded contract";
    refused_for(
        "no experts",
        |p| {
            p.experts.clear();
            p.routes.clear();
            p.registry.copies.clear();
            p.resident_order.clear();
        },
        dims,
    );
    refused_for("1025 experts", |p| p.experts = (0..1025).collect(), dims);
    refused_for(
        "top-k 0",
        |p| {
            for r in &mut p.routes {
                r.clear();
            }
        },
        "batch length or top-k outside routing contract",
    );
    refused_for(
        "unknown routed expert",
        |p| p.routes[0][1].expert = 500,
        "unknown/duplicate expert or invalid routing weight",
    );
}
#[test]
fn registry_clauses_refuse_for_their_own_reason() {
    refused_for(
        "65537 replicas",
        |p| p.registry.copies = (0..65537).map(|rank| replica(10, rank, 0)).collect(),
        "replica or worker limit exceeded",
    );
    let replica_reason = "invalid replica or rank/local collision";
    refused_for(
        "replica of unknown expert",
        |p| p.registry.copies.push(replica(500, 2, 77)),
        replica_reason,
    );
    refused_for(
        "two experts on one rank-local id",
        |p| p.registry.copies.push(replica(99, 2, 3)),
        replica_reason,
    );
    refused_for(
        "second copy on the same rank",
        |p| p.registry.copies.push(replica(10, 2, 77)),
        replica_reason,
    );
    refused_for(
        "expert without any replica",
        |p| p.experts.push(500),
        "unowned expert or worker limit exceeded",
    );
    refused_for(
        "preferred owner without a replica",
        |p| {
            p.preferred_owners.insert(10, 17);
        },
        "preferred owner has no resident replica",
    );
}
#[test]
fn physical_order_clauses_refuse_for_their_own_reason() {
    refused_for(
        "resident rank without an order",
        |p| {
            p.resident_order.remove(&4);
        },
        "resident rank missing from physical order",
    );
    refused_for(
        "foreign local id in an order",
        |p| {
            p.resident_order.insert(2, vec![900, 77]);
        },
        "duplicate or foreign local ID in physical order",
    );
}
