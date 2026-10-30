use std::collections::BTreeMap;
use voltafabric::{moe, routing, strata_ingress};

fn template() -> routing::RoutingProblem {
    routing::RoutingProblem {
        hidden: 4,
        experts: vec![10, 20],
        routes: vec![],
        registry: moe::Registry {
            epoch: 4,
            copies: vec![
                moe::Replica {
                    expert: 10,
                    rank: 0,
                    local: 3,
                },
                moe::Replica {
                    expert: 20,
                    rank: 1,
                    local: 7,
                },
            ],
        },
        stamp: moe::Stamp {
            run_id: "run".into(),
            model_revision: "model".into(),
            plan_id: "plan".into(),
            layer: 3,
            batch: 6,
            epoch: 4,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(0, vec![3]), (1, vec![7])]),
    }
}

#[test]
fn exact_pool_batch_preserves_identity_routes_coefficients_and_bits() {
    let p = template();
    let input = [1.0, -0.0, 0.5, -2.0, 0.25, 2.0, 4.0, 8.0];
    let result = strata_ingress::admit_pool_batch(
        &p,
        &p.stamp,
        3,
        2,
        2,
        &input,
        &[10, 20, 20, 10],
        &[0.25, 0.75, 0.125, 0.875],
    )
    .unwrap();
    assert_eq!(
        result.inputs_fp16_bits,
        vec![
            vec![0x3c00, 0x8000, 0x3800, 0xc000],
            vec![0x3400, 0x4000, 0x4400, 0x4800]
        ]
    );
    assert_eq!(result.routing.stamp, p.stamp);
    let dispatch = routing::dispatch(&result.routing).unwrap();
    assert_eq!(
        dispatch
            .slots
            .iter()
            .map(|s| (s.token, s.slot, s.expert, s.rank, s.weight))
            .collect::<Vec<_>>(),
        vec![
            (0, 0, 10, 0, 0.25),
            (0, 1, 20, 1, 0.75),
            (1, 0, 20, 1, 0.125),
            (1, 1, 10, 0, 0.875)
        ]
    );
    assert!(p.routes.is_empty());
}

#[test]
fn inexact_nonfinite_and_overflow_are_refused_without_a_batch() {
    let p = template();
    for value in [
        1.0001_f32,
        f32::NAN,
        f32::INFINITY,
        70000.0,
        f32::from_bits(1),
    ] {
        assert!(strata_ingress::admit_pool_batch(
            &p,
            &p.stamp,
            3,
            1,
            1,
            &[1.0, value, 2.0, 3.0],
            &[10],
            &[1.0]
        )
        .is_err());
    }
}

#[test]
fn every_finite_fp16_bit_pattern_roundtrips_exactly() {
    let p = template();
    let all: Vec<u16> = (0..=u16::MAX)
        .filter(|bits| bits & 0x7c00 != 0x7c00)
        .collect();
    assert_eq!(all.len(), 63_488);
    for chunk in all.chunks(4096 * p.hidden) {
        let tokens = chunk.len() / p.hidden;
        let input: Vec<f32> = chunk
            .iter()
            .map(|&bits| voltafabric::completion::decode_fp16(bits).unwrap() as f32)
            .collect();
        let admitted = strata_ingress::admit_pool_batch(
            &p,
            &p.stamp,
            3,
            tokens,
            1,
            &input,
            &vec![10; tokens],
            &vec![1.0; tokens],
        )
        .unwrap();
        assert_eq!(admitted.inputs_fp16_bits.concat(), chunk);
    }
}

#[test]
fn bad_shape_identity_routes_and_coefficients_fail_closed() {
    let p = template();
    let mut stale = p.stamp.clone();
    stale.batch += 1;
    assert!(
        strata_ingress::admit_pool_batch(&p, &stale, 3, 1, 1, &[1.0; 4], &[10], &[1.0]).is_err()
    );
    assert!(
        strata_ingress::admit_pool_batch(&p, &p.stamp, 4, 1, 1, &[1.0; 4], &[10], &[1.0]).is_err()
    );
    assert!(
        strata_ingress::admit_pool_batch(&p, &p.stamp, 3, 1, 1, &[1.0; 3], &[10], &[1.0]).is_err()
    );
    assert!(
        strata_ingress::admit_pool_batch(&p, &p.stamp, 3, 1, 1, &[1.0; 4], &[-1], &[1.0]).is_err()
    );
    assert!(
        strata_ingress::admit_pool_batch(&p, &p.stamp, 3, 1, 1, &[1.0; 4], &[999], &[1.0]).is_err()
    );
    assert!(
        strata_ingress::admit_pool_batch(&p, &p.stamp, 3, 1, 1, &[1.0; 4], &[10], &[]).is_err()
    );
    assert!(strata_ingress::admit_pool_batch(
        &p,
        &p.stamp,
        3,
        1,
        2,
        &[1.0; 4],
        &[10, 10],
        &[0.5, 0.5]
    )
    .is_err());
    for weight in [f32::NAN, f32::INFINITY, -1.0] {
        assert!(strata_ingress::admit_pool_batch(
            &p,
            &p.stamp,
            3,
            1,
            1,
            &[1.0; 4],
            &[10],
            &[weight]
        )
        .is_err());
    }
    assert!(
        strata_ingress::admit_pool_batch(&p, &p.stamp, 3, usize::MAX, 2, &[], &[], &[]).is_err()
    );
}

#[test]
fn f64_transport_preserves_every_finite_f32_without_fp16_rounding() {
    let p = template();
    let values = [1.0001_f32, f32::from_bits(1), f32::MAX, -0.0];
    let admitted = strata_ingress::admit_pool_batch_f64(
        &p,
        &p.stamp,
        3,
        1,
        2,
        &values,
        &[10, 20],
        &[0.1, 0.9],
    )
    .unwrap();
    assert_eq!(admitted.routing.stamp, p.stamp);
    for (actual, source) in admitted.inputs[0].iter().zip(values) {
        assert_eq!(actual.to_bits(), f64::from(source).to_bits());
    }
    let d = routing::dispatch(&admitted.routing).unwrap();
    assert_eq!(d.slots[0].weight.to_bits(), f64::from(0.1_f32).to_bits());
    assert_eq!(d.slots[1].weight.to_bits(), f64::from(0.9_f32).to_bits());
    assert!(strata_ingress::admit_pool_batch(
        &p,
        &p.stamp,
        3,
        1,
        2,
        &values,
        &[10, 20],
        &[0.1, 0.9]
    )
    .is_err());
}

#[test]
fn f64_transport_refuses_nonfinite_stale_and_invalid_routes() {
    let p = template();
    for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(strata_ingress::admit_pool_batch_f64(
            &p,
            &p.stamp,
            3,
            1,
            1,
            &[x, 0.0, 0.0, 0.0],
            &[10],
            &[1.0]
        )
        .is_err());
    }
    let mut stale = p.stamp.clone();
    stale.epoch += 1;
    assert!(
        strata_ingress::admit_pool_batch_f64(&p, &stale, 3, 1, 1, &[0.0; 4], &[10], &[1.0])
            .is_err()
    );
    assert!(
        strata_ingress::admit_pool_batch_f64(&p, &p.stamp, 3, 1, 1, &[0.0; 4], &[-1], &[1.0])
            .is_err()
    );
    assert!(strata_ingress::admit_pool_batch_f64(
        &p,
        &p.stamp,
        3,
        1,
        2,
        &[0.0; 4],
        &[10, 10],
        &[0.5, 0.5]
    )
    .is_err());
}

#[test]
fn collector_exposes_only_complete_unweighted_slots_in_original_order() {
    let p = template();
    let admitted = strata_ingress::admit_pool_batch_f64(
        &p,
        &p.stamp,
        3,
        2,
        2,
        &[1.0; 8],
        &[10, 20, 20, 10],
        &[0.25, 0.75, 0.125, 0.875],
    )
    .unwrap();
    let dispatch = routing::dispatch(&admitted.routing).unwrap();
    let mut collector = moe::Collector::from_routing(&admitted.routing).unwrap();
    for slot in dispatch.slots.iter().rev() {
        let value = vec![(slot.token * 10 + slot.slot + 1) as f64; 4];
        collector
            .accept(moe::Completion {
                stamp: p.stamp.clone(),
                token: slot.token,
                slot: slot.slot,
                expert: slot.expert,
                rank: slot.rank,
                local: slot.local,
                value,
            })
            .unwrap();
    }
    let rows = collector.finish_unweighted_slots().unwrap();
    assert_eq!(
        rows,
        vec![
            vec![vec![1.0; 4], vec![2.0; 4]],
            vec![vec![11.0; 4], vec![12.0; 4]]
        ]
    );
    let mut partial = moe::Collector::from_routing(&admitted.routing).unwrap();
    let slot = &dispatch.slots[0];
    let c = moe::Completion {
        stamp: p.stamp.clone(),
        token: slot.token,
        slot: slot.slot,
        expert: slot.expert,
        rank: slot.rank,
        local: slot.local,
        value: vec![1.0; 4],
    };
    partial.accept(c.clone()).unwrap();
    assert!(partial.finish_unweighted_slots().is_err());
    let mut duplicate = moe::Collector::from_routing(&admitted.routing).unwrap();
    duplicate.accept(c.clone()).unwrap();
    assert!(duplicate.accept(c).is_err());
    assert!(duplicate.finish_unweighted_slots().is_err());
}

#[test]
fn slot_output_conversion_is_ordered_rounded_and_atomic() {
    let p = template();
    let admitted = strata_ingress::admit_pool_batch_f64(
        &p,
        &p.stamp,
        3,
        1,
        2,
        &[0.0; 4],
        &[10, 20],
        &[0.25, 0.75],
    )
    .unwrap();
    let half_ulp = f64::from(f32::EPSILON) / 2.0;
    let rows = vec![vec![
        vec![
            1.0 + half_ulp,
            -0.0,
            1.0 + 3.0 * half_ulp,
            f64::from(f32::MAX),
        ],
        vec![0.5, -2.0, 0.0, 4.0],
    ]];
    let mut out = vec![99.0_f32; 8];
    strata_ingress::write_unweighted_f32(&admitted.routing, &rows, &mut out).unwrap();
    assert_eq!(
        out.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        vec![
            1.0_f32.to_bits(),
            (-0.0_f32).to_bits(),
            (1.0_f32 + 2.0 * f32::EPSILON).to_bits(),
            f32::MAX.to_bits(),
            0.5_f32.to_bits(),
            (-2.0_f32).to_bits(),
            0.0_f32.to_bits(),
            4.0_f32.to_bits()
        ]
    );
    for bad in [f64::NAN, f64::INFINITY, f64::MAX] {
        let mut broken = rows.clone();
        broken[0][1][3] = bad;
        let mut target = vec![99.0; 8];
        assert!(
            strata_ingress::write_unweighted_f32(&admitted.routing, &broken, &mut target).is_err()
        );
        assert_eq!(target, vec![99.0; 8]);
    }
    let mut target = vec![99.0; 8];
    assert!(
        strata_ingress::write_unweighted_f32(&admitted.routing, &rows[..0], &mut target).is_err()
    );
    assert_eq!(target, vec![99.0; 8]);
    assert!(
        strata_ingress::write_unweighted_f32(&admitted.routing, &rows, &mut target[..7]).is_err()
    );
    assert_eq!(target, vec![99.0; 8]);
}

#[test]
fn mixed_pool_plan_preserves_slots_and_excludes_externally_owned_experts() {
    let p = template();
    let inputs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let plan = strata_ingress::admit_mixed_pool_f64(
        &p,
        &p.stamp,
        3,
        2,
        2,
        &inputs,
        &[10, 20, 20, 10],
        &[0.25, 0.75, 0.125, 0.875],
        &[true, false, false, true],
    )
    .unwrap();
    let misses = plan.misses().unwrap();
    assert_eq!(
        misses.inputs,
        vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]]
    );
    let d = routing::dispatch(&misses.routing).unwrap();
    assert_eq!(
        d.slots
            .iter()
            .map(|s| (s.token, s.slot, s.expert))
            .collect::<Vec<_>>(),
        vec![(0, 0, 20), (1, 0, 20)]
    );
    let mut out = vec![77.0_f32; 16];
    plan.write_misses(&[vec![vec![2.0; 4]], vec![vec![3.0; 4]]], &mut out)
        .unwrap();
    assert_eq!(
        out,
        [vec![77.0; 4], vec![2.0; 4], vec![3.0; 4], vec![77.0; 4]].concat()
    );
    let before = out.clone();
    assert!(plan
        .write_misses(&[vec![vec![2.0; 4]], vec![vec![f64::MAX; 4]]], &mut out)
        .is_err());
    assert_eq!(out, before);
    let all = strata_ingress::admit_mixed_pool_f64(
        &p,
        &p.stamp,
        3,
        1,
        2,
        &inputs[..4],
        &[10, 20],
        &[0.25, 0.75],
        &[true, true],
    )
    .unwrap();
    assert!(all.misses().is_none());
    let mut untouched = vec![77.0; 8];
    all.write_misses(&[], &mut untouched).unwrap();
    assert_eq!(untouched, vec![77.0; 8]);
}

#[test]
fn mixed_pool_rejects_bad_mask_identity_and_invalid_routes() {
    let p = template();
    let invoke = |stamp: &moe::Stamp, ids: &[i32], mask: &[bool]| {
        strata_ingress::admit_mixed_pool_f64(&p, stamp, 3, 1, 2, &[1.0; 4], ids, &[0.2, 0.8], mask)
    };
    assert!(invoke(&p.stamp, &[10, 20], &[false]).is_err());
    assert!(invoke(&p.stamp, &[-1, 20], &[true, false]).is_err());
    assert!(invoke(&p.stamp, &[10, 10], &[true, false]).is_err());
    let mut stale = p.stamp.clone();
    stale.epoch += 1;
    assert!(invoke(&stale, &[10, 20], &[true, false]).is_err());
}
