use voltafabric::{completion, launch, moe, quant, resident, routing};
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

fn raw(d: &routing::RankDispatch) -> completion::RawOutput {
    completion::RawOutput {
        stamp: d.stamp.clone(),
        rank: d.rank,
        rows: d.slots.len(),
        columns: 64,
        fp16_bits: vec![0x3c00; d.slots.len() * 64],
    }
}

fn routing_problem() -> routing::RoutingProblem {
    let d = dispatch();
    routing::RoutingProblem {
        hidden: d.hidden,
        experts: vec![23, 99],
        routes: vec![
            vec![
                moe::Choice {
                    expert: 23,
                    weight: 0.25,
                },
                moe::Choice {
                    expert: 99,
                    weight: 0.0,
                },
            ],
            vec![
                moe::Choice {
                    expert: 99,
                    weight: 1.0,
                },
                moe::Choice {
                    expert: 23,
                    weight: 2.0,
                },
            ],
        ],
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
        stamp: d.stamp.clone(),
        preferred_owners: Default::default(),
        resident_order: [(17, vec![900, 3])].into_iter().collect(),
    }
}

#[test]
fn fp16_output_admission_reuses_resident_and_matches_f64_collector() {
    let stack = resident::pack(&weights()).unwrap();
    let codes_ptr = stack.codes13().as_ptr();
    let mut d = dispatch();
    d.slots.reverse();
    let plan = launch::prepare(&stack, &d, &d.stamp, &options()).unwrap();
    let mut output = raw(&d);
    for (row, bits) in [0x3c00, 0xc000, 0x4200, 0x4400].into_iter().enumerate() {
        output.fp16_bits[row * 64..(row + 1) * 64].fill(bits);
    }

    let problem = routing_problem();
    let mut reference = moe::Collector::from_routing(&problem).unwrap();
    for receipt in completion::from_output(&plan, &output).unwrap() {
        reference.accept(receipt).unwrap();
    }
    let expected = reference.finish().unwrap();
    assert_eq!(expected, vec![vec![0.25; 64], vec![11.0; 64]]);

    let mut candidate = moe::Collector::from_routing(&problem).unwrap();
    assert_eq!(
        completion::accept_output(&plan, output, &mut candidate).unwrap(),
        4
    );
    assert_eq!(candidate.finish().unwrap(), expected);
    assert_eq!(stack.codes13().as_ptr(), codes_ptr);
}

#[test]
fn fp16_output_admission_rejects_stale_or_bad_packet_before_collector_mutation() {
    let stack = resident::pack(&weights()).unwrap();
    let d = dispatch();
    let plan = launch::prepare(&stack, &d, &d.stamp, &options()).unwrap();
    let problem = routing_problem();
    let mut collector = moe::Collector::from_routing(&problem).unwrap();

    let mut stale = raw(&d);
    stale.stamp.epoch += 1;
    assert_eq!(
        completion::accept_output(&plan, stale, &mut collector)
            .unwrap_err()
            .code,
        "STALE_EXECUTION"
    );

    let mut malformed = raw(&d);
    malformed.rows -= 1;
    assert!(completion::accept_output(&plan, malformed, &mut collector).is_err());

    let mut nonfinite = raw(&d);
    *nonfinite.fp16_bits.last_mut().unwrap() = 0x7c00;
    assert_eq!(
        completion::accept_output(&plan, nonfinite, &mut collector)
            .unwrap_err()
            .code,
        "NON_FINITE_RESULT"
    );

    let valid = raw(&d);
    assert_eq!(
        completion::accept_output(&plan, valid, &mut collector).unwrap(),
        4
    );
    assert_eq!(
        collector.finish().unwrap(),
        vec![vec![0.25; 64], vec![3.0; 64]]
    );
}

#[test]
fn empty_fp16_output_batch_is_a_valid_noop_for_an_empty_rank() {
    let stack = resident::pack(&weights()).unwrap();
    let mut d = dispatch();
    d.slots.clear();
    d.tokens = 0;
    let plan = launch::prepare(&stack, &d, &d.stamp, &options()).unwrap();
    let mut problem = routing_problem();
    problem.routes.clear();
    let mut collector = moe::Collector::from_routing(&problem).unwrap();
    assert_eq!(
        completion::accept_output(&plan, raw(&d), &mut collector).unwrap(),
        0
    );
    assert!(collector.finish().unwrap().is_empty());
}

#[test]
fn duplicate_fp16_output_batch_keeps_collector_terminal() {
    let stack = resident::pack(&weights()).unwrap();
    let d = dispatch();
    let plan = launch::prepare(&stack, &d, &d.stamp, &options()).unwrap();
    let mut collector = moe::Collector::from_routing(&routing_problem()).unwrap();
    assert_eq!(
        completion::accept_output(&plan, raw(&d), &mut collector).unwrap(),
        4
    );
    assert_eq!(
        completion::accept_output(&plan, raw(&d), &mut collector)
            .unwrap_err()
            .code,
        "INVALID_COMPLETION"
    );
    assert_eq!(collector.finish().unwrap_err().code, "BATCH_INCOMPLETE");
}
#[test]
fn raw_rows_map_to_original_identity_and_remain_unweighted() {
    let w = resident::pack(&weights()).unwrap();
    let mut d = dispatch();
    d.slots.reverse();
    let p = launch::prepare(&w, &d, &d.stamp, &options()).unwrap();
    let mut output = raw(&d);
    for (row, bits) in [0x3c00, 0xc000, 0x4200, 0x4400].iter().enumerate() {
        output.fp16_bits[row * 64..(row + 1) * 64].fill(*bits);
    }
    let receipts = completion::from_output(&p, &output).unwrap();
    for (c, s) in receipts.iter().zip(p.completion_slots()) {
        assert_eq!(
            (c.token, c.slot, c.expert, c.rank, c.local),
            (s.token, s.slot, s.expert, s.rank, s.local)
        );
        assert_eq!(c.stamp, d.stamp);
    }
    assert_eq!(
        receipts.iter().map(|c| c.value[0]).collect::<Vec<_>>(),
        vec![1.0, -2.0, 3.0, 4.0]
    );
    // Slot weight zero still has a real raw receipt.
    assert_eq!(receipts[1].value, vec![-2.0; 64]);
}

#[test]
fn routed_fp16_rows_collect_by_identity_in_any_order_and_apply_weights_once() {
    let problem = routing_problem();
    let mut d = dispatch();
    d.slots.reverse();
    let row_values = [
        ((1usize, 1usize), 0x4400u16), // 4.0, expert 23, coefficient 2.0
        ((1, 0), 0x4200),              // 3.0, expert 99, coefficient 1.0
        ((0, 1), 0xc000),              // -2.0, expert 99, coefficient 0.0
        ((0, 0), 0x3c00),              // 1.0, expert 23, coefficient 0.25
    ];
    let by_identity = row_values
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();
    let output = completion::RoutedRawOutput {
        stamp: d.stamp.clone(),
        rank: d.rank,
        attempt: 1,
        rows: d.slots.len(),
        columns: d.hidden,
        slots: d
            .slots
            .iter()
            .map(|row| completion::OutputSlot {
                token: row.slot.token,
                slot: row.slot.slot,
                expert: row.slot.expert,
                local: row.slot.local,
            })
            .collect(),
        fp16_bits: d
            .slots
            .iter()
            .flat_map(|row| {
                std::iter::repeat_n(by_identity[&(row.slot.token, row.slot.slot)], d.hidden)
            })
            .collect(),
    };
    let payload_bytes = output.fp16_bits.len() * std::mem::size_of::<u16>();
    let mut collector = moe::Collector::from_routing(&problem).unwrap();
    assert_eq!(
        completion::accept_routed_output(1, output, &mut collector).unwrap(),
        4
    );
    assert_eq!(payload_bytes, 4 * 64 * 2);
    assert_eq!(
        collector.finish().unwrap(),
        vec![vec![0.25; 64], vec![11.0; 64]]
    );
}

#[test]
fn malformed_routed_fp16_shape_is_rejected_before_collector_admission() {
    let problem = routing_problem();
    let d = dispatch();
    let mut collector = moe::Collector::from_routing(&problem).unwrap();
    let output = completion::RoutedRawOutput {
        stamp: d.stamp.clone(),
        rank: d.rank,
        attempt: 1,
        rows: d.slots.len() - 1,
        columns: d.hidden,
        slots: d
            .slots
            .iter()
            .map(|row| completion::OutputSlot {
                token: row.slot.token,
                slot: row.slot.slot,
                expert: row.slot.expert,
                local: row.slot.local,
            })
            .collect(),
        fp16_bits: vec![0x3c00; (d.slots.len() - 1) * d.hidden],
    };
    assert_eq!(
        completion::accept_routed_output(1, output, &mut collector)
            .unwrap_err()
            .code,
        "INVALID_INPUT"
    );
    // Structural rejection occurs before changing the shared Collector.
    let valid = completion::RoutedRawOutput {
        stamp: d.stamp.clone(),
        rank: d.rank,
        attempt: 1,
        rows: d.slots.len(),
        columns: d.hidden,
        slots: d
            .slots
            .iter()
            .map(|row| completion::OutputSlot {
                token: row.slot.token,
                slot: row.slot.slot,
                expert: row.slot.expert,
                local: row.slot.local,
            })
            .collect(),
        fp16_bits: vec![0x3c00; d.slots.len() * d.hidden],
    };
    assert_eq!(
        completion::accept_routed_output(1, valid, &mut collector).unwrap(),
        d.slots.len()
    );
}

#[test]
fn routed_output_attempt_is_bound_before_collector_mutation() {
    let problem = routing_problem();
    let d = dispatch();
    let output = |attempt| completion::RoutedRawOutput {
        stamp: d.stamp.clone(),
        rank: d.rank,
        attempt,
        rows: d.slots.len(),
        columns: d.hidden,
        slots: d
            .slots
            .iter()
            .map(|row| completion::OutputSlot {
                token: row.slot.token,
                slot: row.slot.slot,
                expert: row.slot.expert,
                local: row.slot.local,
            })
            .collect(),
        fp16_bits: vec![0x3c00; d.slots.len() * d.hidden],
    };
    let mut collector = moe::Collector::from_routing(&problem).unwrap();
    assert_eq!(
        completion::accept_routed_output(17, output(18), &mut collector)
            .unwrap_err()
            .code,
        "STALE_EXECUTION"
    );
    assert_eq!(
        completion::accept_routed_output(17, output(17), &mut collector).unwrap(),
        d.slots.len()
    );
}

#[test]
fn all_nonfinite_half_encodings_are_rejected() {
    for bits in 0u16..=u16::MAX {
        if bits & 0x7c00 == 0x7c00 {
            assert_eq!(
                completion::decode_fp16(bits).unwrap_err().code,
                "NON_FINITE_RESULT"
            );
        } else {
            assert!(completion::decode_fp16(bits).unwrap().is_finite());
        }
    }
}
#[test]
fn exact_half_extremes_and_signed_zero() {
    for (bits, value) in [
        (0, 0.0),
        (0x8000, -0.0),
        (1, 2.0f64.powi(-24)),
        (0x8001, -2.0f64.powi(-24)),
        (0x03ff, 1023.0 * 2.0f64.powi(-24)),
        (0x0400, 2.0f64.powi(-14)),
        (0x7bff, 65504.0),
        (0xfbff, -65504.0),
        (0x3e00, 1.5),
    ] {
        assert_eq!(
            completion::decode_fp16(bits).unwrap().to_bits(),
            value.to_bits()
        );
    }
}
#[test]
fn each_full_stamp_field_and_rank_is_checked() {
    let w = resident::pack(&weights()).unwrap();
    let d = dispatch();
    let p = launch::prepare(&w, &d, &d.stamp, &options()).unwrap();
    for field in 0..7 {
        let mut r = raw(&d);
        match field {
            0 => r.stamp.run_id.push('x'),
            1 => r.stamp.model_revision.push('x'),
            2 => r.stamp.plan_id.push('x'),
            3 => r.stamp.layer += 1,
            4 => r.stamp.batch += 1,
            5 => r.stamp.epoch += 1,
            _ => r.rank += 1,
        }
        assert_eq!(
            completion::from_output(&p, &r).unwrap_err().code,
            "STALE_EXECUTION"
        );
    }
}
#[test]
fn shape_length_padding_and_final_nonfinite_value_abort_whole_packet() {
    let w = resident::pack(&weights()).unwrap();
    let d = dispatch();
    let p = launch::prepare(&w, &d, &d.stamp, &options()).unwrap();
    for field in 0..7 {
        let mut r = raw(&d);
        match field {
            0 => r.rows -= 1,
            1 => r.rows += 1,
            2 => r.columns = 63,
            3 => {
                r.fp16_bits.pop();
            }
            4 => r.fp16_bits.push(0),
            5 => {
                let n = r.fp16_bits.len();
                r.fp16_bits[n - 1] = 0x7c00;
            }
            _ => {
                let n = r.fp16_bits.len();
                r.fp16_bits[n - 1] = 0x7fff;
            }
        }
        assert!(completion::from_output(&p, &r).is_err());
    }
    assert!(completion::from_output(&p, &raw(&d)).is_ok());
}
#[test]
fn empty_rank_requires_exact_empty_shape_and_identity() {
    let w = resident::pack(&weights()).unwrap();
    let mut d = dispatch();
    d.slots.clear();
    let p = launch::prepare(&w, &d, &d.stamp, &options()).unwrap();
    assert!(completion::from_output(&p, &raw(&d)).unwrap().is_empty());
    let mut bad = raw(&d);
    bad.columns = 0;
    assert!(completion::from_output(&p, &bad).is_err());
    bad = raw(&d);
    bad.fp16_bits.push(0);
    assert!(completion::from_output(&p, &bad).is_err());
}

#[test]
fn row_count_must_match_the_plan_even_with_a_consistent_buffer() {
    // A40 guard sweep: rows +/- 1 with the original buffer was refused by the length
    // check; with the buffer resized to match, only the row check can refuse it.
    let w = resident::pack(&weights()).unwrap();
    let d = dispatch();
    let p = launch::prepare(&w, &d, &d.stamp, &options()).unwrap();
    for rows in [d.slots.len() - 1, d.slots.len() + 1] {
        let mut r = raw(&d);
        r.rows = rows;
        r.fp16_bits = vec![0x3c00; rows * 64];
        let error = completion::from_output(&p, &r).unwrap_err();
        assert!(
            error.message.contains("compact output dimensions differ"),
            "{error}"
        );
    }
}
