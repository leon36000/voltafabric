mod common;

use voltafabric::{moe, preweighted, routing};

fn problem() -> routing::RoutingProblem {
    let mut p = routing::from_dense(&common::problem());
    // All coefficients have exact binary32 representations, as required by the
    // preweighted wire contract.
    for (index, choice) in p.routes.iter_mut().flatten().enumerate() {
        choice.weight = [0.375, 0.5, 1.0, 2.0][index];
    }
    // Expert 10 is deliberately selected from a non-default physical replica.
    p.registry.copies.push(moe::Replica {
        expert: 10,
        rank: 9,
        local: 900,
    });
    p.resident_order.get_mut(&9).unwrap().push(900);
    p.preferred_owners.insert(10, 9);
    p.resident_order.get_mut(&2).unwrap().reverse();
    p
}

fn completion(
    contract: &preweighted::Contract,
    stamp: &moe::Stamp,
    slot: &moe::Slot,
    values: &[f32],
) -> preweighted::Completion {
    preweighted::Completion {
        contract: contract.clone(),
        stamp: stamp.clone(),
        token: slot.token,
        slot: slot.slot,
        expert: slot.expert,
        rank: slot.rank,
        local: slot.local,
        weight_bits: (slot.weight as f32).to_bits(),
        value_bits: values.iter().map(|value| value.to_bits()).collect(),
    }
}

fn all_completions(
    c: &preweighted::Collector,
    p: &routing::RoutingProblem,
    value: impl Fn(usize, usize) -> f32,
) -> Vec<preweighted::Completion> {
    routing::dispatch(p)
        .unwrap()
        .slots
        .iter()
        .map(|slot| {
            completion(
                c.contract(),
                &p.stamp,
                slot,
                &vec![value(slot.token, slot.slot); p.hidden],
            )
        })
        .collect()
}

#[test]
fn selected_replica_order_and_prewighted_rows_are_preserved_without_reweighting() {
    let p = problem();
    let selected = routing::dispatch(&p).unwrap();
    let expert10 = selected.slots.iter().find(|s| s.expert == 10).unwrap();
    assert_eq!((expert10.rank, expert10.local), (9, 900));

    let original_coefficients: Vec<_> = p
        .routes
        .iter()
        .flat_map(|route| route.iter().map(|choice| choice.weight.to_bits()))
        .collect();
    let mut c = preweighted::Collector::new(&p).unwrap();
    let mut rows = all_completions(&c, &p, |token, slot| {
        // Values are already weighted and intentionally unrelated to the route
        // coefficients. Applying the coefficient again changes these answers.
        10.0 + token as f32 * 4.0 + slot as f32
    });
    rows.reverse();
    for row in rows {
        c.accept(row).unwrap();
    }
    let got = c.finish().unwrap();
    assert_eq!(got, vec![vec![21.0, 21.0], vec![29.0, 29.0]]);
    assert_eq!(
        p.routes
            .iter()
            .flat_map(|route| route.iter().map(|choice| choice.weight.to_bits()))
            .collect::<Vec<_>>(),
        original_coefficients
    );
}

#[test]
fn f64_coefficient_must_equal_exact_f32_roundtrip_and_coefficient_mutation_is_bound() {
    let mut p = problem();
    p.routes[0][0].weight = 0.1;
    assert!(preweighted::Collector::new(&p).is_err());

    p.routes[0][0].weight = 0.1f32 as f64;
    let dispatch = routing::dispatch(&p).unwrap();
    let mut c = preweighted::Collector::new(&p).unwrap();
    for slot in dispatch.slots.iter().rev() {
        let row = completion(c.contract(), &p.stamp, slot, &[1.0; 2]);
        c.accept(row).unwrap();
    }
    // Changing coefficient metadata while keeping the execution Stamp fixed changes
    // the canonical contract digest; old rows cannot be replayed into it.
    let mut changed = p.clone();
    changed.routes[0][0].weight = 0.125;
    let mut changed_collector = preweighted::Collector::new(&changed).unwrap();
    assert_ne!(c.contract().sha256, changed_collector.contract().sha256);
    let old_contract_row = completion(c.contract(), &p.stamp, &dispatch.slots[0], &[1.0; 2]);
    assert!(changed_collector.accept(old_contract_row).is_err());
    assert!(changed_collector.finish().is_err());
    assert_eq!(c.finish().unwrap(), vec![vec![2.0, 2.0], vec![2.0, 2.0]]);
}

#[test]
fn zero_coefficient_accepts_signed_zero_values_and_rejects_nonzero_values() {
    let mut p = problem();
    p.routes[0][1].weight = 0.0;
    let dispatch = routing::dispatch(&p).unwrap();
    let zero_slot = dispatch
        .slots
        .iter()
        .find(|slot| slot.token == 0 && slot.slot == 1)
        .unwrap();

    let mut valid = preweighted::Collector::new(&p).unwrap();
    for slot in &dispatch.slots {
        let values = if slot.token == 0 && slot.slot == 1 {
            [-0.0, 0.0]
        } else {
            [0.0, 0.0]
        };
        valid
            .accept(completion(valid.contract(), &p.stamp, slot, &values))
            .unwrap();
    }
    assert_eq!(
        valid.finish().unwrap(),
        vec![vec![0.0, 0.0], vec![0.0, 0.0]]
    );

    let mut invalid = preweighted::Collector::new(&p).unwrap();
    let contradictory = completion(
        invalid.contract(),
        &p.stamp,
        zero_slot,
        &[f32::from_bits(1), 0.0],
    );
    assert!(invalid.accept(contradictory).is_err());
    assert!(invalid.finish().is_err());
}

#[test]
fn every_stamp_field_mismatch_poisoned_the_batch() {
    let p = problem();
    for field in 0..6 {
        let mut c = preweighted::Collector::new(&p).unwrap();
        let slot = routing::dispatch(&p).unwrap().slots[0].clone();
        let mut row = completion(c.contract(), &p.stamp, &slot, &[1.0; 2]);
        match field {
            0 => row.stamp.run_id.push('x'),
            1 => row.stamp.model_revision.push('x'),
            2 => row.stamp.plan_id.push('x'),
            3 => row.stamp.layer += 1,
            4 => row.stamp.batch += 1,
            _ => row.stamp.epoch += 1,
        }
        assert!(c.accept(row).is_err(), "Stamp field {field}");
        assert!(c.finish().is_err(), "Stamp field {field}");
    }
}

#[test]
fn slot_replica_weight_and_fp32_wire_corruption_poison_the_batch() {
    let p = problem();
    let slot = routing::dispatch(&p).unwrap().slots[0].clone();
    let mut c = preweighted::Collector::new(&p).unwrap();
    let template = completion(c.contract(), &p.stamp, &slot, &[1.0; 2]);

    let mut wrong_rank = template.clone();
    wrong_rank.rank += 1;
    assert!(c.accept(wrong_rank).is_err());
    assert!(c.finish().is_err());

    for mutate in [0, 1, 2, 3, 4, 5, 6] {
        let mut c = preweighted::Collector::new(&p).unwrap();
        let mut bad = template.clone();
        match mutate {
            0 => bad.token += 1,
            1 => bad.slot += 1,
            2 => bad.expert += 1,
            3 => bad.local += 1,
            4 => bad.weight_bits ^= 1,
            5 => {
                bad.value_bits.pop();
            }
            _ => bad.value_bits.push(0),
        }
        assert!(c.accept(bad).is_err(), "mutation {mutate}");
        assert!(c.finish().is_err(), "mutation {mutate}");
    }

    for bad_bits in [
        f32::INFINITY.to_bits(),
        f32::NEG_INFINITY.to_bits(),
        0x7fc0_0001,
        0x7f80_0001,
    ] {
        let mut c = preweighted::Collector::new(&p).unwrap();
        let mut bad = completion(c.contract(), &p.stamp, &slot, &[1.0; 2]);
        bad.value_bits[1] = bad_bits;
        assert!(c.accept(bad).is_err(), "bits {bad_bits:#010x}");
        assert!(c.finish().is_err(), "bits {bad_bits:#010x}");
    }
}

#[test]
fn identity_mismatch_after_an_accepted_slot_poisoned_the_entire_batch() {
    let p = problem();
    let seed_collector = preweighted::Collector::new(&p).unwrap();
    let rows = all_completions(&seed_collector, &p, |_, _| 1.0);

    for mutation in 0..7 {
        let mut c = preweighted::Collector::new(&p).unwrap();
        c.accept(rows[0].clone()).unwrap();
        let mut bad = rows[1].clone();
        match mutation {
            0 => bad.stamp.run_id.push('x'),
            1 => bad.token += 10,
            2 => bad.slot += 10,
            3 => bad.expert += 10,
            4 => bad.rank += 10,
            5 => bad.local += 10,
            _ => bad.weight_bits ^= 1,
        }
        assert!(c.accept(bad).is_err(), "identity mutation {mutation}");
        assert!(c.finish().is_err(), "identity mutation {mutation}");
    }
}

#[test]
fn accepted_prefix_duplicate_missing_expiry_and_empty_routing_never_return_partial_output() {
    let p = problem();
    let mut c = preweighted::Collector::new(&p).unwrap();
    let rows = all_completions(&c, &p, |_, _| 1.0);
    c.accept(rows[0].clone()).unwrap();
    assert!(c.accept(rows[0].clone()).is_err());
    assert!(c.finish().is_err());

    let mut missing = preweighted::Collector::new(&p).unwrap();
    missing.accept(rows[0].clone()).unwrap();
    assert!(missing.finish().is_err());

    let mut expired = preweighted::Collector::new(&p).unwrap();
    expired.expire();
    assert!(expired.accept(rows[0].clone()).is_err());
    assert!(expired.finish().is_err());

    let mut empty_problem = p;
    empty_problem.routes.clear();
    let empty = preweighted::Collector::new(&empty_problem).unwrap();
    assert_eq!(empty.finish().unwrap(), Vec::<Vec<f32>>::new());
}

#[test]
fn fp32_slot_sum_rounding_is_distinct_from_legacy_f64_collector() {
    let mut p = problem();
    p.hidden = 1;
    p.experts = vec![10, 25, 99];
    p.routes = vec![vec![
        moe::Choice {
            expert: 10,
            weight: 1.0,
        },
        moe::Choice {
            expert: 25,
            weight: 1.0,
        },
        moe::Choice {
            expert: 99,
            weight: 1.0,
        },
    ]];
    let dispatch = routing::dispatch(&p).unwrap();
    let values = [16_777_216.0f32, 1.0, -16_777_216.0];

    let mut fp32 = preweighted::Collector::new(&p).unwrap();
    for slot in &dispatch.slots {
        fp32.accept(completion(
            fp32.contract(),
            &p.stamp,
            slot,
            &[values[slot.slot]],
        ))
        .unwrap();
    }
    assert_eq!(fp32.finish().unwrap(), vec![vec![0.0]]);

    let mut legacy = moe::Collector::from_routing(&p).unwrap();
    for slot in &dispatch.slots {
        legacy
            .accept(moe::Completion {
                stamp: p.stamp.clone(),
                token: slot.token,
                slot: slot.slot,
                expert: slot.expert,
                rank: slot.rank,
                local: slot.local,
                value: vec![f64::from(values[slot.slot])],
            })
            .unwrap();
    }
    assert_eq!(legacy.finish().unwrap(), vec![vec![1.0]]);
}

#[test]
fn fp32_overflow_at_finish_returns_no_partial_token_output() {
    let mut p = problem();
    p.hidden = 1;
    p.routes = vec![
        vec![
            moe::Choice {
                expert: 10,
                weight: 1.0,
            },
            moe::Choice {
                expert: 25,
                weight: 1.0,
            },
        ],
        vec![
            moe::Choice {
                expert: 25,
                weight: 1.0,
            },
            moe::Choice {
                expert: 99,
                weight: 1.0,
            },
        ],
    ];
    let dispatch = routing::dispatch(&p).unwrap();
    let mut c = preweighted::Collector::new(&p).unwrap();
    for slot in &dispatch.slots {
        let value = if slot.token == 1 { f32::MAX } else { 7.0 };
        c.accept(completion(c.contract(), &p.stamp, slot, &[value]))
            .unwrap();
    }
    assert!(c.finish().is_err());
}

#[test]
fn format_and_hash_are_not_silently_inferred_or_ignored() {
    let p = problem();
    let mut c = preweighted::Collector::new(&p).unwrap();
    let mut rows = all_completions(&c, &p, |_, _| 1.0);
    let valid = rows[0].clone();
    rows[0].contract.sha256.replace_range(..1, "x");
    assert!(c.accept(rows.remove(0)).is_err());
    assert!(c.accept(valid.clone()).is_err());
    assert!(c.finish().is_err());
    let mut json = serde_json::to_value(valid).unwrap();
    json["contract"]["format"] = serde_json::json!("unweighted_fp16_v1");
    assert!(serde_json::from_value::<preweighted::Completion>(json.clone()).is_err());
    json.as_object_mut().unwrap().remove("contract");
    assert!(serde_json::from_value::<preweighted::Completion>(json).is_err());
}

#[test]
fn fingerprint_canonicalizes_metadata_but_binds_physical_order_and_full_stamp() {
    let p = problem();
    let c = preweighted::Collector::new(&p).unwrap();
    let mut reordered = p.clone();
    reordered.experts.reverse();
    reordered.registry.copies.reverse();
    assert_eq!(
        c.contract(),
        preweighted::Collector::new(&reordered).unwrap().contract()
    );
    reordered.resident_order.get_mut(&9).unwrap().reverse();
    assert_ne!(
        c.contract(),
        preweighted::Collector::new(&reordered).unwrap().contract()
    );
    for field in 0..6 {
        let mut changed = p.clone();
        match field {
            0 => changed.stamp.run_id.push('x'),
            1 => changed.stamp.model_revision.push('x'),
            2 => changed.stamp.plan_id.push('x'),
            3 => changed.stamp.layer += 1,
            4 => changed.stamp.batch += 1,
            _ => {
                changed.stamp.epoch += 1;
                changed.registry.epoch += 1;
            }
        }
        assert_ne!(
            c.contract(),
            preweighted::Collector::new(&changed).unwrap().contract()
        );
    }
}

#[test]
fn original_routing_mutation_does_not_mutate_a_constructed_collector() {
    let mut p = problem();
    let mut c = preweighted::Collector::new(&p).unwrap();
    let rows = all_completions(&c, &p, |_, _| 1.0);
    let fingerprint = c.contract().clone();
    p.routes[0][0].weight = 0.25;
    p.stamp.epoch += 1;
    assert_eq!(c.contract(), &fingerprint);
    for row in rows {
        c.accept(row).unwrap();
    }
    assert_eq!(c.finish().unwrap(), vec![vec![2.0; 2]; 2]);
}

#[test]
fn preweighted_collector_preserves_existing_allocation_admission_limit() {
    let mut p = problem();
    p.hidden = 65536;
    let route = p.routes[0].clone();
    p.routes = vec![route.clone(); 128];
    // 128 tokens * 2 slots * 65536 = the existing 16,777,216-value cap.
    assert!(preweighted::Collector::new(&p).is_ok());
    p.routes.push(route);
    assert!(preweighted::Collector::new(&p).is_err());
}
