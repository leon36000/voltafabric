use voltafabric::{launch, moe, quant, resident, routing};
fn plane(rows: u64, columns: u64) -> quant::PackedRows {
    quant::PackedRows {
        rows,
        columns,
        codes: vec![0x22; (rows * columns / 2) as usize],
        scales: vec![56; (rows * columns / 16) as usize],
        global_dequant: 1.0,
    }
}
fn weights() -> resident::ResidentInput {
    resident::ResidentInput {
        model_revision: "m".into(),
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
            ],
        },
        resident_order: vec![900, 3],
        experts: vec![
            resident::ResidentExpert {
                id: 23,
                local: 3,
                gate: plane(128, 64),
                up: plane(128, 64),
                down: plane(64, 128),
            },
            resident::ResidentExpert {
                id: 99,
                local: 900,
                gate: plane(128, 64),
                up: plane(128, 64),
                down: plane(64, 128),
            },
        ],
    }
}
fn dispatch() -> routing::RankDispatch {
    let stamp = moe::Stamp {
        run_id: "r".into(),
        model_revision: "m".into(),
        plan_id: "p".into(),
        layer: 3,
        batch: 4,
        epoch: 12,
    };
    let mut slots = vec![];
    for (token, slot, expert, local, physical_expert, weight) in [
        (0, 0, 23, 3, 1, 0.25),
        (0, 1, 99, 900, 0, 0.0),
        (1, 0, 99, 900, 0, 1.0),
        (1, 1, 23, 3, 1, 2.0),
    ] {
        slots.push(routing::RankSlot {
            slot: moe::Slot {
                token,
                slot,
                expert,
                rank: 17,
                local,
                weight,
            },
            physical_expert,
        });
    }
    routing::RankDispatch {
        stamp,
        rank: 17,
        hidden: 64,
        tokens: 2,
        resident_order: vec![900, 3],
        slots,
    }
}
fn options() -> launch::Options {
    launch::Options {
        splitk13: 4,
        splitk2: 8,
        max_workspace_bytes: 128 * 1024 * 1024,
    }
}
#[test]
fn compact_rows_preserve_original_slots_and_zero_weights() {
    let w = resident::pack(&weights()).unwrap();
    let d = dispatch();
    let p = launch::prepare(&w, &d, &d.stamp, &options()).unwrap();
    assert_eq!(p.source_tokens(), &[0, 0, 1, 1]);
    assert_eq!(
        p.completion_slots(),
        &d.slots.iter().map(|r| r.slot.clone()).collect::<Vec<_>>()
    );
    assert_eq!(p.block_expert(), &[0, 1]);
    assert_eq!(
        p.sorted_slots(),
        &[1, 2, 4, 4, 4, 4, 4, 4, 0, 3, 4, 4, 4, 4, 4, 4]
    );
    assert_eq!(p.ntpp(), 16);
    assert!(!p.skip());
    let j = serde_json::to_value(&p).unwrap();
    assert_eq!(j["projection13"]["n"], 256);
    assert_eq!(j["projection13"]["k"], 64);
    assert_eq!(j["projection2"]["k"], 128);
    assert_eq!(j["projection13"]["in_div"], 1);
    assert_eq!(j["projection2"]["nacc"], 1);
    assert_eq!(p.workspace_bytes(), 4172);
    assert!(std::ptr::eq(p.weights(), &w));
}
#[test]
fn arbitrary_arrival_order_has_identical_metadata_and_coefficients() {
    let w = resident::pack(&weights()).unwrap();
    let d = dispatch();
    let expected =
        serde_json::to_value(launch::prepare(&w, &d, &d.stamp, &options()).unwrap()).unwrap();
    for rotate in 0..4 {
        let mut r = d.clone();
        r.slots.rotate_left(rotate);
        r.slots.reverse();
        let actual =
            serde_json::to_value(launch::prepare(&w, &r, &d.stamp, &options()).unwrap()).unwrap();
        assert_eq!(actual, expected);
    }
}
#[test]
fn full_current_stamp_is_required_for_every_batch() {
    let w = resident::pack(&weights()).unwrap();
    let d = dispatch();
    for f in 0..6 {
        let mut stamp = d.stamp.clone();
        match f {
            0 => stamp.run_id.push('x'),
            1 => stamp.model_revision.push('x'),
            2 => stamp.plan_id.push('x'),
            3 => stamp.layer += 1,
            4 => stamp.batch += 1,
            _ => stamp.epoch += 1,
        }
        assert!(launch::prepare(&w, &d, &stamp, &options()).is_err());
    }
    let mut next = d.clone();
    next.stamp.batch += 1;
    assert!(launch::prepare(&w, &next, &next.stamp, &options()).is_ok());
}
#[test]
fn workspace_limit_includes_all_fp16_planes_and_device_metadata() {
    let w = resident::pack(&weights()).unwrap();
    let d = dispatch();
    let mut o = options();
    o.max_workspace_bytes = 4172;
    assert!(launch::prepare(&w, &d, &d.stamp, &o).is_ok());
    o.max_workspace_bytes = 4171;
    assert_eq!(
        launch::prepare(&w, &d, &d.stamp, &o).unwrap_err().code,
        "CAPACITY_EXCEEDED"
    );
    for cap in [0, 128 * 1024 * 1024 + 1, u64::MAX] {
        o.max_workspace_bytes = cap;
        assert!(launch::prepare(&w, &d, &d.stamp, &o).is_err());
    }
}
#[test]
fn splitk_configurations_must_exist_and_divide_each_projection() {
    let w = resident::pack(&weights()).unwrap();
    let d = dispatch();
    for (s13, s2, ok) in [
        (2, 2, true),
        (4, 4, true),
        (4, 8, true),
        (8, 8, false),
        (4, 16, false),
        (1, 8, false),
        (4, 3, false),
        (0, 0, false),
    ] {
        let mut o = options();
        o.splitk13 = s13;
        o.splitk2 = s2;
        assert_eq!(launch::prepare(&w, &d, &d.stamp, &o).is_ok(), ok);
    }
}
#[test]
fn foreign_duplicate_and_out_of_range_slots_never_produce_a_plan() {
    let w = resident::pack(&weights()).unwrap();
    let d = dispatch();
    for f in 0..6 {
        let mut bad = d.clone();
        match f {
            0 => bad.slots[0].physical_expert = 0,
            1 => bad.slots[0].slot.token = 2,
            2 => bad.slots.push(bad.slots[0].clone()),
            3 => bad.slots[0].slot.weight = f64::NAN,
            4 => bad.resident_order.reverse(),
            _ => bad.rank += 1,
        }
        assert!(launch::prepare(&w, &bad, &d.stamp, &options()).is_err());
    }
}
#[test]
fn resident_without_local_work_skips_all_launches() {
    let w = resident::pack(&weights()).unwrap();
    let mut d = dispatch();
    d.slots.clear();
    let p = launch::prepare(&w, &d, &d.stamp, &options()).unwrap();
    assert!(p.skip());
    assert_eq!(p.ntpp(), 0);
    assert_eq!(p.workspace_bytes(), 0);
    let j = serde_json::to_value(&p).unwrap();
    assert!(j["projection13"].is_null());
    assert!(j["projection2"].is_null());
    assert!(p.sorted_slots().is_empty() && p.block_expert().is_empty());
    let mut input = weights();
    input.rank = 44;
    input.resident_order.clear();
    input.experts.clear();
    let empty = resident::pack(&input).unwrap();
    d.rank = 44;
    d.resident_order.clear();
    assert!(launch::prepare(&empty, &d, &d.stamp, &options())
        .unwrap()
        .skip());
}
#[test]
fn block_bound_and_large_batch_are_checked_without_activation_allocation() {
    let w = resident::pack(&weights()).unwrap();
    let mut d = dispatch();
    d.tokens = 4096;
    d.slots.clear();
    for t in 0..4096 {
        for &(slot, id, local, physical) in &[(0, 99, 900, 0), (1, 23, 3, 1)] {
            d.slots.push(routing::RankSlot {
                slot: moe::Slot {
                    token: t,
                    slot,
                    expert: id,
                    rank: 17,
                    local,
                    weight: 1.0,
                },
                physical_expert: physical,
            });
        }
    }
    let p = launch::prepare(&w, &d, &d.stamp, &options()).unwrap();
    assert_eq!(p.ntpp(), 8192);
    assert_eq!(p.block_expert().len(), 1024);
    assert_eq!(p.sorted_slots().iter().filter(|&&s| s < 8192).count(), 8192);
    assert_eq!(p.source_tokens().len(), 8192);
}

#[test]
fn workspace_reserves_metadata_for_exactly_the_blocks_the_kernel_launches() {
    // A38: the block count feeding the workspace size was guarded only by the
    // Python launch oracle; no Rust test noticed it computed with the wrong block
    // size. Both plans below have 8 tokens and 8 slots, so their activation
    // buffers are identical and only the metadata part of the workspace differs.
    fn plan_with(physicals: &[usize]) -> (u64, usize) {
        let w = resident::pack(&weights()).unwrap();
        let mut d = dispatch();
        d.tokens = physicals.len();
        d.slots.clear();
        for (t, &physical) in physicals.iter().enumerate() {
            let (expert, local) = if physical == 0 { (99, 900) } else { (23, 3) };
            d.slots.push(routing::RankSlot {
                slot: moe::Slot {
                    token: t,
                    slot: 0,
                    expert,
                    rank: 17,
                    local,
                    weight: 1.0,
                },
                physical_expert: physical,
            });
        }
        let p = launch::prepare(&w, &d, &d.stamp, &options()).unwrap();
        (p.workspace_bytes(), p.block_expert().len())
    }
    // Eight slots on one expert fill one block of eight; four and four fill two
    // partial blocks.
    let (concentrated, concentrated_blocks) = plan_with(&[0; 8]);
    let (split, split_blocks) = plan_with(&[0, 1, 0, 1, 0, 1, 0, 1]);
    assert_eq!(concentrated_blocks, 1);
    assert_eq!(split_blocks, 2);
    assert_eq!(
        concentrated - 36 * concentrated_blocks as u64,
        split - 36 * split_blocks as u64,
        "workspace must reserve 36 metadata bytes per launched block"
    );
}

#[test]
fn a_dispatch_needing_more_blocks_than_the_ffi_grid_never_yields_a_plan() {
    // A39: Q07 (the launch guard `blocks > 65535`) is unreachable. pack() caps the
    // physical stack at 1024 experts and bind_dispatch caps a dispatch at 4096 tokens
    // and 262144 slots, so blocks <= (262144 + 7 * 1024) / 8 = 33664. A dispatch that
    // would need 65536 blocks must therefore die at the routing bounds, before planning.
    let w = resident::pack(&weights()).unwrap();
    let mut d = dispatch();
    let tokens = 262_144;
    d.tokens = tokens;
    d.slots.clear();
    d.slots.reserve(2 * tokens);
    for t in 0..tokens {
        for &(slot, id, local, physical) in &[(0, 99, 900, 0), (1, 23, 3, 1)] {
            d.slots.push(routing::RankSlot {
                slot: moe::Slot {
                    token: t,
                    slot,
                    expert: id,
                    rank: 17,
                    local,
                    weight: 1.0,
                },
                physical_expert: physical,
            });
        }
    }
    let error = match launch::prepare(&w, &d, &d.stamp, &options()) {
        Ok(_) => panic!("a 65536-block dispatch produced a launch plan"),
        Err(error) => error,
    };
    assert!(
        error.message.contains("outside routing bounds"),
        "refused for another reason first: {} {}",
        error.code,
        error.message
    );
}

#[test]
fn launch_options_are_refused_by_the_option_contract_itself() {
    // A40 guard sweep: splitk2 = 3 and a zero cap were already refused, but by the
    // divisibility and workspace checks. splitk2 = 1 divides every projection and an
    // empty dispatch needs no workspace, so only the option contract refuses these.
    let w = resident::pack(&weights()).unwrap();
    let d = dispatch();
    let mut empty = dispatch();
    empty.slots.clear();
    let reason = "launch options outside bounded compiled contract";
    let mut o = options();
    o.splitk2 = 1;
    let error = launch::prepare(&w, &d, &d.stamp, &o).unwrap_err();
    assert!(error.message.contains(reason), "{error}");
    let mut o = options();
    o.max_workspace_bytes = 0;
    let error = launch::prepare(&w, &empty, &empty.stamp, &o).unwrap_err();
    assert!(error.message.contains(reason), "{error}");
}
