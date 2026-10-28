use std::collections::BTreeMap;
use voltafabric::{moe, qpn2, quant, resident::*, routing};

fn matrix(rows: u64, columns: u64, seed: u8, factor: f64) -> quant::PackedRows {
    quant::PackedRows {
        rows,
        columns,
        codes: (0..rows * columns / 2)
            .map(|i| (i as u8).wrapping_mul(73).wrapping_add(seed))
            .collect(),
        scales: (0..rows * columns / 16)
            .map(|i| ((i + u64::from(seed)) % 127) as u8)
            .collect(),
        global_dequant: factor,
    }
}
fn expert(id: u32, local: u32, seed: u8) -> ResidentExpert {
    ResidentExpert {
        id,
        local,
        gate: matrix(128, 64, seed, 0.5),
        up: matrix(128, 64, seed + 1, 2.0),
        down: matrix(64, 128, seed + 2, 4.0),
    }
}
fn input() -> ResidentInput {
    ResidentInput {
        model_revision: "model-A".into(),
        layer: 3,
        rank: 17,
        registry: moe::Registry {
            epoch: 12,
            copies: vec![
                moe::Replica {
                    expert: 23,
                    rank: 17,
                    local: 3,
                },
                moe::Replica {
                    expert: 99,
                    rank: 17,
                    local: 900,
                },
                moe::Replica {
                    expert: 777,
                    rank: 2,
                    local: 0,
                },
            ],
        },
        resident_order: vec![900, 3],
        experts: vec![expert(23, 3, 4), expert(99, 900, 9)],
    }
}
fn dispatch(p: &ResidentInput) -> routing::RankDispatch {
    let problem = routing::RoutingProblem {
        hidden: 64,
        experts: vec![23, 99, 777],
        routes: vec![
            vec![
                moe::Choice {
                    expert: 23,
                    weight: 2.5,
                },
                moe::Choice {
                    expert: 99,
                    weight: 0.0,
                },
            ],
            vec![
                moe::Choice {
                    expert: 99,
                    weight: 4.0,
                },
                moe::Choice {
                    expert: 23,
                    weight: 0.25,
                },
            ],
        ],
        registry: p.registry.clone(),
        stamp: moe::Stamp {
            run_id: "run-A".into(),
            model_revision: p.model_revision.clone(),
            plan_id: "plan-A".into(),
            layer: p.layer,
            batch: 4,
            epoch: p.registry.epoch,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(17, p.resident_order.clone()), (2, vec![0])]),
    };
    routing::dispatch(&problem)
        .unwrap()
        .ranks
        .into_iter()
        .find(|r| r.rank == 17)
        .unwrap()
}
fn same_packed(a: &quant::PackedRows, b: &quant::PackedRows) {
    assert_eq!((a.rows, a.columns), (b.rows, b.columns));
    assert_eq!(a.codes, b.codes);
    assert_eq!(a.scales, b.scales);
    assert_eq!(a.global_dequant.to_bits(), b.global_dequant.to_bits());
}
fn unpack_plane(stack: &ResidentStack, physical: usize, plane: usize) -> quant::PackedRows {
    let (_, n13, k13, n2, k2) = stack.shape();
    let (rows, columns, codes, scales, factor) = if plane < 2 {
        let rows = n13 / 2;
        let code_count = (rows * k13 / 2) as usize;
        let scale_count = (rows * k13 / 16) as usize;
        let code_start = (physical * 2 + plane) * code_count;
        let scale_start = (physical * 2 + plane) * scale_count;
        let tile_start = (physical * 2 + plane) * (rows / 32) as usize;
        (
            rows,
            k13,
            stack.codes13()[code_start..code_start + code_count].to_vec(),
            stack.scales13()[scale_start..scale_start + scale_count].to_vec(),
            f64::from(stack.gscales13()[tile_start]),
        )
    } else {
        let code_count = (n2 * k2 / 2) as usize;
        let scale_count = (n2 * k2 / 16) as usize;
        (
            n2,
            k2,
            stack.codes2()[physical * code_count..(physical + 1) * code_count].to_vec(),
            stack.scales2()[physical * scale_count..(physical + 1) * scale_count].to_vec(),
            f64::from(stack.gscales2()[physical * (n2 / 32) as usize]),
        )
    };
    qpn2::unpack(&qpn2::FragmentRows {
        rows,
        columns,
        codes,
        scales,
        global_dequant: factor,
    })
    .unwrap()
}

#[test]
fn reversed_sparse_order_and_three_planes_roundtrip_exactly() {
    let p = input();
    let stack = pack(&p).unwrap();
    assert_eq!(stack.shape(), (2, 256, 64, 64, 128));
    assert_eq!(stack.resident_order(), &[900, 3]);
    assert_eq!(
        stack
            .physical_experts()
            .iter()
            .map(|e| (e.id(), e.local()))
            .collect::<Vec<_>>(),
        vec![(99, 900), (23, 3)]
    );
    assert_eq!(
        (
            stack.model_revision(),
            stack.layer(),
            stack.rank(),
            stack.epoch()
        ),
        ("model-A", 3, 17, 12)
    );
    for (physical, &local) in stack.resident_order().iter().enumerate() {
        let expert = p.experts.iter().find(|e| e.local == local).unwrap();
        same_packed(&unpack_plane(&stack, physical, 0), &expert.gate);
        same_packed(&unpack_plane(&stack, physical, 1), &expert.up);
        same_packed(&unpack_plane(&stack, physical, 2), &expert.down);
    }
    let actual_bytes = stack.codes13().len()
        + stack.scales13().len()
        + stack.codes2().len()
        + stack.scales2().len()
        + 4 * (stack.gscales13().len() + stack.gscales2().len());
    assert_eq!(stack.payload_bytes(), actual_bytes as u64);
}

#[test]
fn gate_up_distinct_factors_are_repeated_per_tile() {
    let mut p = input();
    p.experts[0].gate.global_dequant = 1.0;
    p.experts[0].up.global_dequant = 8.0;
    p.experts[0].down.global_dequant = 16.0;
    let stack = pack(&p).unwrap();
    assert_eq!(
        stack.gscales13(),
        &[0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 8.0, 8.0, 8.0, 8.0]
    );
    assert_eq!(stack.gscales2(), &[4.0, 4.0, 16.0, 16.0]);
}

#[test]
fn buffers_are_owned_after_inputs_are_mutated_and_dropped() {
    let mut p = input();
    let stack = pack(&p).unwrap();
    let expected = stack.codes13().to_vec();
    for expert in &mut p.experts {
        expert.gate.codes.fill(0);
        expert.up.codes.fill(0);
    }
    p.model_revision.clear();
    p.resident_order.clear();
    drop(p);
    assert_eq!(stack.codes13(), expected);
    assert_eq!(stack.model_revision(), "model-A");
    let serialized = serde_json::to_value(&stack).unwrap();
    assert_eq!(
        serialized["shape"],
        serde_json::json!([2, 256, 64, 64, 128])
    );
    assert_eq!(serialized["physical_experts"][0]["id"], 99);
}

#[test]
fn binding_preserves_permanent_weights_across_batches_runs_and_plans() {
    let p = input();
    let stack = pack(&p).unwrap();
    let mut d = dispatch(&p);
    stack.bind_dispatch(&d).unwrap();
    d.stamp.run_id = "another-run".into();
    d.stamp.plan_id = "another-plan".into();
    d.stamp.batch = 99;
    for slot in &mut d.slots {
        slot.slot.weight *= 3.0;
    }
    stack.bind_dispatch(&d).unwrap();
    assert_eq!(stack.gscales13()[0], 0.5);
}

#[test]
fn stale_static_identity_hidden_or_order_is_rejected() {
    let p = input();
    let stack = pack(&p).unwrap();
    let good = dispatch(&p);
    for field in 0..6 {
        let mut d = good.clone();
        match field {
            0 => d.stamp.model_revision = "stale-model".into(),
            1 => d.stamp.layer += 1,
            2 => d.stamp.epoch += 1,
            3 => d.rank += 1,
            4 => d.hidden += 64,
            _ => d.resident_order.reverse(),
        }
        assert!(stack.bind_dispatch(&d).is_err(), "field={field}");
    }
}

#[test]
fn foreign_physical_or_logical_slots_and_bad_tokens_are_rejected() {
    let p = input();
    let stack = pack(&p).unwrap();
    let good = dispatch(&p);
    for field in 0..10 {
        let mut d = good.clone();
        match field {
            0 => d.slots[0].physical_expert = 2,
            1 => d.slots[0].physical_expert ^= 1,
            2 => d.slots[0].slot.local = 777,
            3 => d.slots[0].slot.expert = 777,
            4 => d.slots[0].slot.rank = 2,
            5 => d.slots[0].slot.token = d.tokens,
            6 => d.slots[0].slot.slot = 64,
            7 => d.slots[0].slot.weight = f64::NAN,
            8 => d.slots[0].slot.weight = -1.0,
            _ => d.slots.push(d.slots[0].clone()),
        }
        assert!(stack.bind_dispatch(&d).is_err(), "field={field}");
    }
}

#[test]
fn two_resident_slots_sharing_a_token_and_slot_index_are_rejected() {
    // The adjacent (token, expert) guard cannot see this: the collision is on
    // (token, slot) while the experts stay distinct, so the slot set is the only
    // thing that stops two routed slots claiming the same output position.
    let p = input();
    let stack = pack(&p).unwrap();
    let good = dispatch(&p);
    let (first, second) = (0..good.slots.len())
        .flat_map(|i| (0..good.slots.len()).map(move |j| (i, j)))
        .find(|&(i, j)| {
            i != j
                && good.slots[i].slot.token == good.slots[j].slot.token
                && good.slots[i].slot.expert != good.slots[j].slot.expert
                && good.slots[i].slot.slot != good.slots[j].slot.slot
        })
        .expect("fixture must route one token to two different experts");
    let mut d = good.clone();
    d.slots[second].slot.slot = good.slots[first].slot.slot;
    assert_eq!(
        d.slots[second].slot.token, d.slots[first].slot.token,
        "collision must stay on one token"
    );
    assert_ne!(
        d.slots[second].slot.expert, d.slots[first].slot.expert,
        "experts must stay distinct so only the slot guard can reject"
    );
    assert!(stack.bind_dispatch(&d).is_err());
}

#[test]
fn invalid_registry_expert_and_order_collisions_are_rejected() {
    for field in 0..8 {
        let mut p = input();
        match field {
            0 => p.experts[1].id = p.experts[0].id,
            1 => p.experts[1].local = p.experts[0].local,
            2 => p.registry.copies.push(p.registry.copies[0].clone()),
            3 => p.registry.copies[0].local = 900,
            4 => p.registry.copies[0].expert = 999,
            5 => p.resident_order = vec![900, 900],
            6 => {
                p.registry.copies.remove(0);
            }
            _ => p.registry.copies.push(moe::Replica {
                expert: 11,
                rank: 17,
                local: 5,
            }),
        }
        assert!(pack(&p).is_err(), "field={field}");
    }
    let mut p = input();
    p.registry.copies.push(moe::Replica {
        expert: 888,
        rank: 2,
        local: 0,
    });
    assert!(
        pack(&p).is_err(),
        "foreign registry collision is still malformed metadata"
    );
}

#[test]
fn zero_scale_is_valid_signed_and_nan_scale_encodings_are_rejected() {
    let mut p = input();
    p.experts[0].gate.scales.fill(0);
    assert!(pack(&p).is_ok());
    for byte in [127, 128, 129, 254, 255] {
        let mut p = input();
        p.experts[1].down.scales[0] = byte;
        assert!(pack(&p).is_err(), "scale={byte}");
    }
}

#[test]
fn global_factors_must_be_positive_finite_and_exact_f32() {
    for factor in [
        0.0,
        -0.0,
        -1.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        f64::MIN_POSITIVE,
        1.00000001,
    ] {
        let mut p = input();
        p.experts[1].up.global_dequant = factor;
        assert!(pack(&p).is_err(), "factor={factor}");
    }
    for factor in [f64::from(f32::from_bits(1)), f64::from(f32::MAX), 1.5] {
        let mut p = input();
        p.experts[1].up.global_dequant = factor;
        let stack = pack(&p).unwrap();
        assert_eq!(f64::from(stack.gscales13()[4]).to_bits(), factor.to_bits());
    }
}

#[test]
fn shapes_and_plane_lengths_must_match_before_any_permutation() {
    for field in 0..7 {
        let mut p = input();
        match field {
            0 => p.experts[0].gate.rows = 32,
            1 => p.experts[0].up.rows = 64,
            2 => p.experts[0].down.rows = 128,
            3 => {
                p.experts[0].gate.codes.pop();
            }
            4 => p.experts[0].up.scales.push(0),
            5 => p.experts[0].gate.columns = 65,
            _ => p.experts[0].down.columns = 0,
        }
        assert!(pack(&p).is_err(), "field={field}");
    }
    let mut p = input();
    p.experts[1] = ResidentExpert {
        id: 99,
        local: 900,
        gate: matrix(64, 64, 1, 1.0),
        up: matrix(64, 64, 2, 1.0),
        down: matrix(64, 64, 3, 1.0),
    };
    assert!(pack(&p).is_err(), "ragged expert stack is forbidden");
}

#[test]
fn empty_rank_has_no_inferred_weight_shape_and_binds_only_empty_slots() {
    let mut p = input();
    p.rank = 44;
    p.experts.clear();
    p.resident_order.clear();
    let stack = pack(&p).unwrap();
    assert!(stack.is_empty());
    assert_eq!(stack.shape(), (0, 0, 0, 0, 0));
    assert_eq!(stack.payload_bytes(), 0);
    assert!(stack.codes13().is_empty() && stack.gscales2().is_empty());
    let mut d = dispatch(&input());
    d.rank = 44;
    d.resident_order.clear();
    d.slots.clear();
    stack.bind_dispatch(&d).unwrap();
    d.hidden = 128;
    stack.bind_dispatch(&d).unwrap(); // No H exists on an empty rank.
    d.hidden = 0;
    assert!(stack.bind_dispatch(&d).is_err());
    d.hidden = 64;
    d.slots = dispatch(&input()).slots;
    assert!(stack.bind_dispatch(&d).is_err());
    p.registry.copies.push(moe::Replica {
        expert: 23,
        rank: 44,
        local: 0,
    });
    assert!(pack(&p).is_err());
}

#[test]
fn oversized_payload_is_rejected_from_metadata_without_giant_allocations() {
    let empty = |rows, columns| quant::PackedRows {
        rows,
        columns,
        codes: vec![],
        scales: vec![],
        global_dequant: 1.0,
    };
    let mut p = ResidentInput {
        model_revision: "model".into(),
        layer: 0,
        rank: 0,
        registry: moe::Registry {
            epoch: 1,
            copies: vec![],
        },
        resident_order: vec![],
        experts: vec![],
    };
    for id in 0..925 {
        p.registry.copies.push(moe::Replica {
            expert: id,
            rank: 0,
            local: id,
        });
        p.resident_order.push(id);
        p.experts.push(ResidentExpert {
            id,
            local: id,
            gate: empty(192, 448),
            up: empty(192, 448),
            down: empty(448, 192),
        });
    }
    assert_eq!(pack(&p).unwrap_err().code, "CAPACITY_EXCEEDED");
    // The adjacent 924-expert shape is below the cap; its empty planes are then rejected.
    p.experts.pop();
    p.resident_order.pop();
    p.registry.copies.pop();
    assert_eq!(pack(&p).unwrap_err().code, "INVALID_INPUT");
}

#[test]
fn matrix_work_limit_overflow_and_expert_count_are_rejected() {
    let mut p = input();
    p.experts[0].gate.rows = 2048;
    p.experts[0].gate.columns = 1024;
    assert!(pack(&p).is_err());
    p.experts[0].gate.rows = u64::MAX - 31;
    p.experts[0].gate.columns = 64;
    assert_eq!(pack(&p).unwrap_err().code, "OVERFLOW");
    let mut p = input();
    p.experts = vec![p.experts[0].clone(); 1025];
    // A39: clones are also duplicates, so is_err() alone held with the cap widened (M1).
    assert!(pack(&p)
        .unwrap_err()
        .message
        .contains("exceeds bounded contract"));
}

#[test]
fn actual_full_plane_cpu_limit_is_accepted_without_dequantization() {
    let mut p = input();
    p.experts = vec![ResidentExpert {
        id: 23,
        local: 3,
        gate: matrix(512, 2048, 17, 0.5),
        up: matrix(512, 2048, 21, 2.0),
        down: matrix(2048, 512, 30, 4.0),
    }];
    p.resident_order = vec![3];
    p.registry
        .copies
        .retain(|replica| replica.rank != 17 || replica.expert == 23);
    let stack = pack(&p).unwrap();
    assert_eq!(stack.shape(), (1, 1024, 2048, 2048, 512));
    assert_eq!(stack.codes13().len(), 1_048_576);
    assert_eq!(stack.scales13().len(), 131_072);
    assert_eq!(stack.codes2().len(), 524_288);
    assert_eq!(stack.scales2().len(), 65_536);
    assert_eq!(stack.gscales13().len(), 32);
    assert_eq!(stack.gscales2().len(), 64);
    assert_eq!(stack.payload_bytes(), 1_769_856);
}

#[test]
fn repeated_expert_and_empty_rank_dispatch_bounds_are_rejected() {
    let p = input();
    let stack = pack(&p).unwrap();
    let mut d = dispatch(&p);
    let mut repeated = d.slots[0].clone();
    repeated.slot.slot = 63; // Distinct token-slot, same token-expert.
    d.slots.push(repeated);
    assert!(stack.bind_dispatch(&d).is_err());
    let mut empty = p.clone();
    empty.rank = 44;
    empty.experts.clear();
    empty.resident_order.clear();
    let stack = pack(&empty).unwrap();
    let mut good = dispatch(&p);
    good.rank = 44;
    good.resident_order.clear();
    good.slots.clear();
    for field in 0..4 {
        let mut bad = good.clone();
        match field {
            0 => bad.tokens = 4097,
            1 => bad.hidden = 65537,
            2 => {
                bad.tokens = 4096;
                bad.hidden = 65536;
            }
            _ => bad.stamp.run_id = " ".into(),
        }
        assert!(stack.bind_dispatch(&bad).is_err(), "field={field}");
    }
    // A39 (M3): an oversized slot list must be refused by the cheap length bound,
    // before any per-slot set is built; later checks would also refuse it.
    let mut oversized = good.clone();
    oversized.slots = vec![dispatch(&p).slots[0].clone(); 262_145];
    assert!(stack
        .bind_dispatch(&oversized)
        .unwrap_err()
        .message
        .contains("outside routing bounds"));
}

// A40 guard sweep: one clause per case. With the clause disabled pack() ACCEPTED
// non-SwiGLU shapes, a duplicated physical order entry (one more physical expert
// than experts; the A39 Q07 bound relies on this clause), blank or oversized model
// revisions and 65537 copies; the rest were refused only by a later check.
fn pack_refused(name: &str, edit: impl FnOnce(&mut ResidentInput), reason: &str) {
    let mut p = input();
    edit(&mut p);
    let error = pack(&p).expect_err(name);
    assert!(
        error.message.contains(reason),
        "{name}: expected {reason:?}, got {error}"
    );
}
#[test]
fn plane_shapes_and_sizes_are_refused_for_their_own_reason() {
    pack_refused(
        "columns 96",
        |p| {
            for e in &mut p.experts {
                e.gate = matrix(128, 96, 1, 0.5);
                e.up = matrix(128, 96, 2, 0.5);
                e.down = matrix(96, 128, 3, 0.5);
            }
        },
        "resident QPN2 planes require positive rows/32 and columns/64 alignment",
    );
    pack_refused(
        "plane over 1M elements",
        |p| {
            for e in &mut p.experts {
                e.gate = matrix(2048, 1024, 1, 0.5);
                e.up = matrix(2048, 1024, 2, 0.5);
                e.down = matrix(1024, 2048, 3, 0.5);
            }
        },
        "resident plane exceeds bounded QPN2 CPU work limit",
    );
    let swiglu = "gate/up/down or inter-expert shapes do not form a uniform SwiGLU stack";
    pack_refused(
        "up differs from gate",
        |p| {
            for e in &mut p.experts {
                e.up = matrix(96, 64, 2, 0.5);
            }
        },
        swiglu,
    );
    pack_refused(
        "down rows differ from gate columns",
        |p| {
            for e in &mut p.experts {
                e.down = matrix(128, 128, 3, 0.5);
            }
        },
        swiglu,
    );
    pack_refused(
        "down columns differ from gate rows",
        |p| {
            for e in &mut p.experts {
                e.down = matrix(64, 64, 3, 0.5);
            }
        },
        swiglu,
    );
    let length = "resident packed plane length mismatch";
    pack_refused(
        "extra code byte",
        |p| p.experts[0].gate.codes.push(0),
        length,
    );
    pack_refused(
        "extra scale byte",
        |p| p.experts[0].gate.scales.push(0),
        length,
    );
}
#[test]
fn identity_and_registry_are_refused_for_their_own_reason() {
    let identity = "resident identity or metadata exceeds bounded contract";
    pack_refused(
        "blank revision",
        |p| p.model_revision = " ".into(),
        identity,
    );
    pack_refused(
        "257-byte revision",
        |p| p.model_revision = "r".repeat(257),
        identity,
    );
    pack_refused(
        "65537 copies",
        |p| {
            p.registry.copies.extend((0..65535).map(|i| moe::Replica {
                expert: 5000 + i,
                rank: 1000 + i % 64,
                local: i,
            }))
        },
        identity,
    );
    let duplicate = "duplicate resident global expert or rank-local ID";
    pack_refused(
        "duplicate global id",
        |p| p.experts[1].id = p.experts[0].id,
        duplicate,
    );
    pack_refused(
        "duplicate local id",
        |p| p.experts[1].local = p.experts[0].local,
        duplicate,
    );
    pack_refused(
        "second copy on the same rank",
        |p| {
            let mut r = p.registry.copies[0].clone();
            r.local = 4242;
            p.registry.copies.push(r);
        },
        "registry has duplicate replica or rank-local collision",
    );
    pack_refused(
        "repeated physical order entry",
        |p| {
            let x = p.resident_order[0];
            p.resident_order.push(x);
        },
        "explicit physical order does not cover resident local IDs exactly",
    );
}

#[test]
fn a_slot_pointing_past_the_physical_stack_is_refused_by_name() {
    // A41 reason sweep: reached before, but only through is_err().
    let p = input();
    let stack = pack(&p).unwrap();
    let mut d = dispatch(&p);
    d.slots[0].physical_expert = 99;
    assert_eq!(
        stack.bind_dispatch(&d).unwrap_err().message,
        "physical expert outside resident stack"
    );
}
