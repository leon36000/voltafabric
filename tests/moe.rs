mod common;
use common::*;
use voltafabric::moe::*;
fn near(a: &[Vec<f64>], b: &[Vec<f64>]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        for (p, q) in x.iter().zip(y) {
            assert!((p - q).abs() < 1e-12, "{p} != {q}");
        }
    }
}
#[test]
fn parallel_matches_reference() {
    let p = problem();
    near(&reference(&p).unwrap(), &parallel(&p).unwrap());
}
#[test]
fn weights_are_not_normalized() {
    let mut p = problem();
    let a = reference(&p).unwrap();
    for row in &mut p.batch.routes {
        for c in row {
            c.weight *= 2.0;
        }
    }
    let b = reference(&p).unwrap();
    for (i, r) in a.iter().enumerate() {
        for (j, v) in r.iter().enumerate() {
            assert!((b[i][j] - 2.0 * v).abs() < 1e-12);
        }
    }
}
#[test]
fn global_ids_are_compacted_per_rank() {
    let p = problem();
    let s = dispatch(&p).unwrap();
    assert_eq!(s.len(), 4);
    assert_eq!((s[1].expert, s[1].rank, s[1].local), (99, 2, 1));
}
#[test]
fn empty_batch_is_valid() {
    let mut p = problem();
    p.batch.inputs.clear();
    p.batch.routes.clear();
    assert!(parallel(&p).unwrap().is_empty());
}
#[test]
fn unused_experts_are_not_slots() {
    let mut p = problem();
    for row in &mut p.batch.routes {
        *row = vec![Choice {
            expert: 10,
            weight: 1.0,
        }];
    }
    assert!(dispatch(&p).unwrap().iter().all(|x| x.expert == 10));
}
#[test]
fn replica_selection_is_explicit() {
    let mut p = problem();
    p.registry.copies.push(Replica {
        expert: 10,
        rank: 17,
        local: 0,
    });
    p.preferred_owners.insert(10, 17);
    let s = dispatch(&p).unwrap();
    assert!(s.iter().filter(|x| x.expert == 10).all(|x| x.rank == 17));
    near(&reference(&p).unwrap(), &parallel(&p).unwrap());
}
#[test]
fn unknown_preferred_replica_rejected() {
    let mut p = problem();
    p.preferred_owners.insert(10, 200);
    assert!(validate(&p).is_err());
}
#[test]
fn duplicate_expert_within_token_rejected() {
    let mut p = problem();
    p.batch.routes[0][1].expert = 10;
    assert!(validate(&p).is_err());
}
#[test]
fn missing_expert_owner_rejected() {
    let mut p = problem();
    p.registry.copies.retain(|r| r.expert != 99);
    assert!(validate(&p).is_err());
}
#[test]
fn rank_local_collision_rejected() {
    let mut p = problem();
    p.registry.copies[2].local = 0;
    assert!(validate(&p).is_err());
}
#[test]
fn invalid_dimensions_rejected() {
    let mut p = problem();
    p.model.experts[0].up.pop();
    assert!(validate(&p).is_err());
}
#[test]
fn inconsistent_topk_rejected() {
    let mut p = problem();
    p.batch.routes[0].pop();
    assert!(validate(&p).is_err());
}
#[test]
fn nonfinite_and_negative_weights_rejected() {
    for w in [f64::NAN, f64::INFINITY, -0.1] {
        let mut p = problem();
        p.batch.routes[0][0].weight = w;
        assert!(validate(&p).is_err());
    }
}
#[test]
fn nonfinite_model_rejected() {
    let mut p = problem();
    p.model.experts[0].gate[0] = f64::NAN;
    assert!(validate(&p).is_err());
}
#[test]
fn stale_registry_rejected() {
    let mut p = problem();
    p.stamp.epoch += 1;
    assert!(validate(&p).is_err());
}
#[test]
fn zero_weights_produce_zero() {
    let mut p = problem();
    for r in &mut p.batch.routes {
        for c in r {
            c.weight = 0.0;
        }
    }
    assert!(parallel(&p).unwrap().iter().flatten().all(|&x| x == 0.0));
}
#[test]
fn reference_has_independent_scalar_expected_value() {
    let mut p = problem();
    p.model.hidden = 1;
    p.model.intermediate = 1;
    p.model.experts = vec![Expert {
        id: 10,
        gate: vec![2.0],
        up: vec![3.0],
        down: vec![4.0],
    }];
    p.registry.copies.retain(|r| r.expert == 10);
    p.batch.inputs = vec![vec![0.5]];
    p.batch.routes = vec![vec![Choice {
        expert: 10,
        weight: 0.25,
    }]];
    let y = reference(&p).unwrap();
    let expected = (1.0 / (1.0 + (-1.0_f64).exp())) * 1.5 * 4.0 * 0.25;
    assert!((y[0][0] - expected).abs() < 1e-14);
}
#[test]
fn collector_accepts_out_of_order() {
    let p = problem();
    let mut c = Collector::new(&p).unwrap();
    let mut s = dispatch(&p).unwrap();
    s.reverse();
    for x in s {
        c.accept(compute_slot(&p, &x).unwrap()).unwrap();
    }
    near(&reference(&p).unwrap(), &c.finish().unwrap());
}
#[test]
fn collector_rejects_duplicate_and_stays_failed() {
    let p = problem();
    let mut c = Collector::new(&p).unwrap();
    let s = dispatch(&p).unwrap();
    let r = compute_slot(&p, &s[0]).unwrap();
    c.accept(r.clone()).unwrap();
    assert!(c.accept(r).is_err());
    assert!(c.finish().is_err());
}
#[test]
fn collector_cannot_return_partial_output() {
    let p = problem();
    let c = Collector::new(&p).unwrap();
    assert!(c.finish().is_err());
}
#[test]
fn collector_rejects_foreign_batch() {
    let p = problem();
    let mut c = Collector::new(&p).unwrap();
    let s = dispatch(&p).unwrap();
    let mut r = compute_slot(&p, &s[0]).unwrap();
    r.stamp.batch += 1;
    assert!(c.accept(r).is_err());
}
#[test]
fn collector_rejects_wrong_local_id() {
    let p = problem();
    let mut c = Collector::new(&p).unwrap();
    let s = dispatch(&p).unwrap();
    let mut r = compute_slot(&p, &s[0]).unwrap();
    r.local += 1;
    assert!(c.accept(r).is_err());
}
#[test]
fn collector_rejects_bad_width_and_nan() {
    for bad in [vec![1.0], vec![f64::NAN, 0.0]] {
        let p = problem();
        let mut c = Collector::new(&p).unwrap();
        let s = dispatch(&p).unwrap();
        let mut r = compute_slot(&p, &s[0]).unwrap();
        r.value = bad;
        assert!(c.accept(r).is_err());
    }
}
#[test]
fn expiration_is_terminal() {
    let p = problem();
    let mut c = Collector::new(&p).unwrap();
    c.expire();
    let s = dispatch(&p).unwrap();
    assert!(c.accept(compute_slot(&p, &s[0]).unwrap()).is_err());
    assert!(c.finish().is_err());
}

// A40 guard sweep: each case violates exactly one validation clause. With that
// clause alone disabled, most of these were ACCEPTED by validate(); the rest were
// refused only by a later, unrelated check. Each must fail for its own reason.
fn refused(name: &str, edit: impl FnOnce(&mut Problem), reason: &str) {
    let mut p = problem();
    edit(&mut p);
    let error = validate(&p).expect_err(name);
    assert!(
        error.message.contains(reason),
        "{name}: expected {reason:?}, got {error}"
    );
}
fn clear_matrices(p: &mut Problem) {
    for e in &mut p.model.experts {
        e.gate.clear();
        e.up.clear();
        e.down.clear();
    }
}
#[test]
fn model_contract_clauses_refuse_for_their_own_reason() {
    let dims = "CPU model dimensions outside bounded probe contract";
    refused(
        "hidden 0",
        |p| {
            p.model.hidden = 0;
            clear_matrices(p);
            for x in &mut p.batch.inputs {
                x.clear();
            }
        },
        dims,
    );
    refused(
        "intermediate 0",
        |p| {
            p.model.intermediate = 0;
            clear_matrices(p);
        },
        dims,
    );
    refused(
        "matrix over 1M",
        |p| {
            p.model.hidden = 1025;
            p.model.intermediate = 1024;
        },
        dims,
    );
    refused(
        "no experts",
        |p| {
            p.model.experts.clear();
            p.batch.inputs.clear();
            p.batch.routes.clear();
            p.registry.copies.clear();
            p.preferred_owners.clear();
        },
        dims,
    );
    refused(
        "1025 experts",
        |p| {
            let e = p.model.experts[0].clone();
            p.model.experts = (0..1025).map(|id| Expert { id, ..e.clone() }).collect();
        },
        dims,
    );
    refused(
        "weights over 16M",
        |p| {
            p.model.hidden = 1024;
            p.model.intermediate = 1024;
            let e = p.model.experts[0].clone();
            p.model.experts = (0..6).map(|id| Expert { id, ..e.clone() }).collect();
            clear_matrices(p);
        },
        "CPU probe weight limit",
    );
    let shape = "duplicate global expert or invalid matrix shape";
    refused(
        "duplicate expert id",
        |p| {
            let mut e = p.model.experts[0].clone();
            e.gate[0] = 9.0;
            p.model.experts.push(e);
        },
        shape,
    );
    refused(
        "short gate",
        |p| {
            p.model.experts[0].gate.pop();
        },
        shape,
    );
    refused(
        "short down",
        |p| {
            p.model.experts[0].down.pop();
        },
        shape,
    );
}
#[test]
fn batch_contract_clauses_refuse_for_their_own_reason() {
    let batch = "batch lengths or top-k outside CPU contract";
    refused(
        "inputs != routes",
        |p| {
            p.batch.inputs.pop();
        },
        batch,
    );
    refused(
        "4097 inputs",
        |p| {
            p.batch.inputs = vec![p.batch.inputs[0].clone(); 4097];
            p.batch.routes = vec![p.batch.routes[0].clone(); 4097];
        },
        batch,
    );
    refused(
        "top-k 0",
        |p| {
            for r in &mut p.batch.routes {
                r.clear();
            }
        },
        batch,
    );
    refused(
        "top-k 65",
        |p| {
            let e = p.model.experts[0].clone();
            p.model.experts = (0..65).map(|id| Expert { id, ..e.clone() }).collect();
            p.registry.copies = (0..65)
                .map(|i| Replica {
                    expert: i,
                    rank: 2,
                    local: i,
                })
                .collect();
            p.preferred_owners.clear();
            for r in &mut p.batch.routes {
                *r = (0..65)
                    .map(|expert| Choice {
                        expert,
                        weight: 0.1,
                    })
                    .collect();
            }
        },
        batch,
    );
    refused(
        "work over 1e8",
        |p| {
            p.model.hidden = 1024;
            p.model.intermediate = 1024;
            p.model.experts = vec![Expert {
                id: 10,
                gate: vec![0.0; 1 << 20],
                up: vec![0.0; 1 << 20],
                down: vec![0.0; 1 << 20],
            }];
            p.registry.copies.retain(|r| r.expert == 10);
            p.preferred_owners.clear();
            p.batch.inputs = vec![vec![]; 33];
            p.batch.routes = vec![
                vec![Choice {
                    expert: 10,
                    weight: 1.0,
                }];
                33
            ];
        },
        "CPU work budget exceeded",
    );
    let input = "input width/finite value or inconsistent top-k";
    refused("wide input", |p| p.batch.inputs[0].push(1.0), input);
    refused("NaN input", |p| p.batch.inputs[0][0] = f64::NAN, input);
    refused(
        "unknown routed expert",
        |p| p.batch.routes[0][0].expert = 4242,
        "unknown/duplicate expert or invalid routing weight",
    );
}
#[test]
fn identity_and_registry_clauses_refuse_for_their_own_reason() {
    let identity = "epoch mismatch or empty/oversized execution identity";
    refused("blank run id", |p| p.stamp.run_id = " ".into(), identity);
    refused(
        "257-byte plan id",
        |p| p.stamp.plan_id = "p".repeat(257),
        identity,
    );
    refused(
        "65537 replicas",
        |p| {
            p.registry.copies = (0..65537)
                .map(|rank| Replica {
                    expert: 10,
                    rank,
                    local: 0,
                })
                .collect();
        },
        "too many replicas",
    );
    let replica = "invalid replica or rank/local collision";
    refused(
        "replica of unknown expert",
        |p| {
            p.registry.copies.push(Replica {
                expert: 4242,
                rank: 2,
                local: 77,
            })
        },
        replica,
    );
    refused(
        "second copy of an expert on the same rank",
        |p| {
            let mut r = p.registry.copies[0].clone();
            r.local += 50;
            p.registry.copies.push(r);
        },
        replica,
    );
    refused(
        "65 ranks",
        |p| {
            for i in 0..65 {
                p.registry.copies.push(Replica {
                    expert: 10,
                    rank: 1000 + i,
                    local: 0,
                });
            }
        },
        "unowned expert or CPU worker limit exceeded",
    );
}
#[test]
fn forged_slot_is_never_computed() {
    let p = problem();
    let mut forged = dispatch(&p).unwrap()[0].clone();
    forged.local += 1;
    let error = compute_slot(&p, &forged).unwrap_err();
    assert!(
        error
            .message
            .contains("slot does not belong to the current dispatch"),
        "{error}"
    );
}
#[test]
fn finish_refuses_an_aborted_batch_even_when_every_slot_arrived() {
    // A duplicate after the last result aborts the batch without removing anything,
    // so only the failed flag stands between it and a successful finish.
    let p = problem();
    let mut c = Collector::new(&p).unwrap();
    let mut first = None;
    for s in dispatch(&p).unwrap() {
        let x = compute_slot(&p, &s).unwrap();
        first.get_or_insert_with(|| x.clone());
        c.accept(x).unwrap();
    }
    assert!(c.accept(first.clone().unwrap()).is_err());
    let error = c.finish().unwrap_err();
    assert_eq!(error.message, "aborted or missing results");
    let mut c = Collector::new(&p).unwrap();
    c.accept(first.unwrap()).unwrap();
    assert_eq!(
        c.finish().unwrap_err().message,
        "aborted or missing results"
    );
}

#[test]
fn matrix_size_overflow_is_refused_by_name() {
    // A41 reason sweep: hidden * intermediate overflows before any bound applies.
    refused(
        "hidden * intermediate overflows",
        |p| {
            p.model.hidden = usize::MAX / 2;
            p.model.intermediate = 3;
        },
        "matrix size overflow",
    );
}
