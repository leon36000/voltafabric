use std::collections::BTreeMap;
use voltafabric::planner::*;
use voltafabric::topology::*;
use voltafabric::traffic::*;

const POLICIES: [WirePolicy; 3] = [
    WirePolicy::Baseline,
    WirePolicy::Grouped,
    WirePolicy::Adaptive,
];

fn device(rank: u32, capacity: u64) -> Device {
    Device {
        rank,
        host: format!("h{rank}"),
        capacity_bytes: capacity,
        reserved_bytes: 0,
    }
}

fn link(id: &str, src: u32, dst: u32) -> Link {
    Link {
        id: id.into(),
        src,
        dst,
        bandwidth_bytes_per_s: 1e6,
        latency_us: 1.0,
        enabled: true,
        transport: Transport::Synthetic,
        pcie: None,
        resources: vec![],
        cost_provenance: Default::default(),
    }
}

fn expert(id: u32, compute_us: f64) -> ExpertProfile {
    ExpertProfile {
        id,
        resident_bytes: 10,
        compute_us,
        cost_provenance: Default::default(),
    }
}

/// Source rank 0 cannot host an expert; `big` (capacity 20) takes two experts, `small` (capacity 10) one.
fn scenario(
    big: u32,
    small: u32,
    ids: [u32; 3],
    policy: WirePolicy,
    iterations: usize,
) -> PlacementProblem {
    PlacementProblem {
        topology: Topology {
            schema_version: 1,
            origin: Origin::Synthetic,
            source_note: "synthetic test only".into(),
            devices: vec![device(0, 1), device(big, 20), device(small, 10)],
            links: vec![
                link("0-big", 0, big),
                link("big-0", big, 0),
                link("0-small", 0, small),
                link("small-0", small, 0),
            ],
        },
        experts: ids.iter().map(|&i| expert(i, 1.0)).collect(),
        demands: vec![Demand {
            source_rank: 0,
            experts: ids.to_vec(),
            frequency: 1.0,
        }],
        hidden: 64,
        wire: WireFormat::default(),
        max_iterations: iterations,
        wire_policy: policy,
        max_work_units: 1_000_000,
    }
}

type Triple = (u64, u64, u64);
fn triple(c: &VariantCost) -> Triple {
    (
        c.modeled_latency_us.to_bits(),
        c.wire_bytes,
        c.wire_messages,
    )
}
fn triples(d: &DemandComparison) -> [Triple; 3] {
    [
        triple(&d.baseline),
        triple(&d.grouped),
        triple(&d.byte_adaptive),
    ]
}

fn expected_owners(big: u32, small: u32, ids: [u32; 3]) -> BTreeMap<u32, u32> {
    BTreeMap::from([(ids[0], small), (ids[1], big), (ids[2], big)])
}

fn assert_hand_costs(d: &DemandComparison) {
    // 1 us latency + 1 us/byte; 128-byte inputs and outputs, 256-byte grouped FP32 partials.
    assert!((d.baseline.modeled_latency_us - 388.0).abs() < 1e-6);
    assert!((d.grouped.modeled_latency_us - 388.0).abs() < 1e-6);
    assert!((d.byte_adaptive.modeled_latency_us - 388.0).abs() < 1e-6);
    assert_eq!((d.baseline.wire_bytes, d.baseline.wire_messages), (768, 6));
    assert_eq!((d.grouped.wire_bytes, d.grouped.wire_messages), (768, 4));
    assert_eq!(
        (d.byte_adaptive.wire_bytes, d.byte_adaptive.wire_messages),
        (640, 4)
    );
}

#[test]
fn rank_expert_id_collision_plans_under_every_wire_policy() {
    let ids = [1, 10, 11];
    for policy in POLICIES {
        for iterations in [0, 32] {
            let p = scenario(1, 2, ids, policy, iterations);
            let r = plan(&p).unwrap_or_else(|e| panic!("{policy:?}/{iterations}: {e:?}"));
            if iterations == 0 {
                assert_eq!(r.owners, expected_owners(1, 2, ids));
                assert_eq!(r.iterations, 0);
            }
            assert_eq!(r.wire_policy, policy);
            let d = &r.demand_comparisons[0];
            assert!((r.modeled_weighted_latency_us - 388.0).abs() < 1e-6);
            if iterations == 0 {
                assert_hand_costs(d);
            }
            let selected = match policy {
                WirePolicy::Baseline => WireVariant::Baseline,
                WirePolicy::Grouped => WireVariant::Grouped,
                WirePolicy::Adaptive => WireVariant::Baseline,
            };
            assert_eq!(d.selected, selected);
            let scored = score(&p, &r.owners).unwrap();
            assert_eq!(scored.to_bits(), r.modeled_weighted_latency_us.to_bits());
        }
    }
}

#[test]
fn collision_costs_equal_the_rename_control_bit_for_bit() {
    // Same topology shape with rank labels that cannot collide with expert ids 1/10/11.
    let ids = [1, 10, 11];
    for policy in POLICIES {
        let collide = plan(&scenario(1, 2, ids, policy, 0)).unwrap();
        let control = plan(&scenario(7, 2, ids, policy, 0)).unwrap();
        assert_eq!(
            triples(&collide.demand_comparisons[0]),
            triples(&control.demand_comparisons[0]),
            "{policy:?}"
        );
        assert_eq!(
            collide.modeled_weighted_latency_us.to_bits(),
            control.modeled_weighted_latency_us.to_bits()
        );
        assert_eq!(
            collide.demand_comparisons[0].selected,
            control.demand_comparisons[0].selected
        );
    }
}

#[test]
fn rename_controls_succeed_and_agree_with_hand_costs() {
    for (big, small) in [(7, 2), (5, 50), (100, 2), (3, 9)] {
        let ids = [1, 10, 11];
        let r = plan(&scenario(big, small, ids, WirePolicy::Adaptive, 0)).unwrap();
        assert_eq!(r.owners, expected_owners(big, small, ids));
        assert_hand_costs(&r.demand_comparisons[0]);
    }
}

#[test]
fn sparse_and_prefix_collisions_for_input_and_output_jobs() {
    // (big rank, small rank, ids): rows 1-4 collide (grouped rank == lone expert id), rows 5-6 only
    // share decimal prefixes.
    for (big, small, ids) in [
        (1, 2, [1, 10, 11]),
        (10, 3, [10, 11, 12]),
        (100, 2, [100, 101, 102]),
        (
            4_000_000_000,
            7,
            [4_000_000_000, 4_000_000_001, 4_000_000_002],
        ),
        (3, 30, [30, 300, 301]),
        (30, 3, [3, 30, 300]),
    ] {
        let p = scenario(big, small, ids, WirePolicy::Adaptive, 0);
        let r = plan(&p).unwrap_or_else(|e| panic!("{big}/{small}/{ids:?}: {e:?}"));
        assert_eq!(r.owners.len(), 3);
        let d = &r.demand_comparisons[0];
        assert_eq!(triples(d)[0].1, 768, "{big}/{small}");
        assert_eq!(d.byte_adaptive.wire_bytes, 640);
        assert!((d.byte_adaptive.modeled_latency_us - 388.0).abs() < 1e-6);
        for policy in POLICIES {
            plan(&scenario(big, small, ids, policy, 8)).unwrap();
        }
    }
}

#[test]
fn score_accepts_a_collision_placement_for_every_policy() {
    let ids = [1, 10, 11];
    for policy in POLICIES {
        let p = scenario(1, 2, ids, policy, 0);
        let v = score(&p, &expected_owners(1, 2, ids)).unwrap();
        assert!((v - 388.0).abs() < 1e-6, "{policy:?} {v}");
    }
}

#[test]
fn per_expert_input_order_keeps_numeric_string_prefix_order() {
    // Lone destination, baseline wire: 'in:1' < 'in:10' < 'in:100' decides the shared-link order.
    // Hand schedule (1 us latency, 1 us/byte, 129 us transfers): in:1, in:10, in:100 gives 1759;
    // the reverse order would give 1887.
    let mut p = scenario(5, 9, [1, 10, 100], WirePolicy::Baseline, 0);
    p.topology.devices[2].capacity_bytes = 1;
    p.topology.devices[1].capacity_bytes = 30;
    p.experts[0].compute_us = 1000.0;
    p.experts[1].compute_us = 1.0;
    p.experts[2].compute_us = 500.0;
    let owners = BTreeMap::from([(1, 5), (10, 5), (100, 5)]);
    let v = score(&p, &owners).unwrap();
    assert!((v - 1759.0).abs() < 1e-6, "{v}");
    let r = plan(&p).unwrap();
    assert_eq!(r.owners, owners);
    assert!((r.modeled_weighted_latency_us - 1759.0).abs() < 1e-6);
}

#[test]
fn mixed_group_and_expert_ids_without_collision_keep_their_order() {
    // Group id 'in:5' vs expert id 'in:50' / 'in:10' etc. never collided before; costs must not move.
    let ids = [50, 51, 52];
    let r = plan(&scenario(5, 500, ids, WirePolicy::Adaptive, 0)).unwrap();
    assert_hand_costs(&r.demand_comparisons[0]);
    let r = plan(&scenario(500, 5, ids, WirePolicy::Adaptive, 0)).unwrap();
    assert_hand_costs(&r.demand_comparisons[0]);
}

#[test]
fn work_budget_is_exactly_the_reported_use_for_a_collision_plan() {
    let ids = [1, 10, 11];
    for policy in POLICIES {
        let base = scenario(1, 2, ids, policy, 4);
        let used = plan(&base).unwrap().work_units_used;
        let mut exact = base.clone();
        exact.max_work_units = used;
        assert_eq!(plan(&exact).unwrap().work_units_used, used);
        let mut short = base.clone();
        short.max_work_units = used - 1;
        assert_eq!(plan(&short).unwrap_err().code, "WORK_BUDGET_EXCEEDED");
    }
}

#[test]
fn id_suffix_ablation_slash_preserves_old_string_order_and_alternatives_do_not() {
    fn old(prefix: &str, n: u32) -> String {
        format!("{prefix}:{n}")
    }
    fn new(prefix: &str, n: u32, role: &str, sep: char) -> String {
        format!("{prefix}:{n}{sep}{role}")
    }
    let mut values: Vec<u32> = (0..=1200).collect();
    values.extend([
        4_294_967_295,
        4_294_967_294,
        429_496_729,
        42_949_672,
        4_294_967,
        4_000_000_000,
        400_000_000,
    ]);
    let mut flipped_colon = false;
    let mut flipped_underscore = false;
    for &a in &values {
        for &b in &values {
            if a == b {
                continue;
            }
            for prefix in ["in", "out"] {
                for (ra, rb) in [
                    ("rank", "rank"),
                    ("rank", "expert"),
                    ("expert", "rank"),
                    ("expert", "expert"),
                ] {
                    let expected = old(prefix, a) < old(prefix, b);
                    assert_eq!(expected, new(prefix, a, ra, '/') < new(prefix, b, rb, '/'));
                    flipped_colon |=
                        expected != (new(prefix, a, ra, ':') < new(prefix, b, rb, ':'));
                    flipped_underscore |=
                        expected != (new(prefix, a, ra, '_') < new(prefix, b, rb, '_'));
                }
            }
        }
    }
    assert!(flipped_colon && flipped_underscore);
    assert_ne!(new("in", 1, "rank", '/'), new("in", 1, "expert", '/'));
}

#[test]
fn mixed_group_expert_order_on_shared_bus_has_a_sensitive_production_ablation() {
    // ByteAdaptive groups rank5 (experts51/52) while expert50 runs at rank2.
    // Every transfer uses one shared bus: changing in:5 vs in:50 order delays
    // the long grouped compute. Correct old relative order costs1386us;
    // replacing the group '/' by '_' costs1515us in the isolated mutant.
    let mut p = scenario(5, 2, [50, 51, 52], WirePolicy::Adaptive, 0);
    for link in &mut p.topology.links {
        link.resources = vec!["shared-bus".into()];
    }
    p.experts[1].compute_us = 500.0;
    p.experts[2].compute_us = 500.0;
    let result = plan(&p).unwrap();
    let mixed = &result.demand_comparisons[0].byte_adaptive;
    // Only rounding in three scalar link calculations, far below1e-9us;
    // the rejected mutation changes the modeled result by129us.
    assert!(
        (mixed.modeled_latency_us - 1386.0).abs() < 1e-9,
        "{mixed:?}"
    );
}
