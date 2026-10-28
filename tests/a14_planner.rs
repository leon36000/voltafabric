use voltafabric::{
    cost::CostProvenance,
    planner::{plan, score, Demand, ExpertProfile, PlacementProblem},
    topology::{Device, Link, Origin, Topology, Transport},
    traffic::WireFormat,
};

fn link(id: &str, src: u32, dst: u32, bandwidth: f64, latency: f64) -> Link {
    Link {
        id: id.into(),
        src,
        dst,
        bandwidth_bytes_per_s: bandwidth,
        latency_us: latency,
        enabled: true,
        transport: Transport::Synthetic,
        pcie: None,
        resources: vec![],
        cost_provenance: CostProvenance::default(),
    }
}

fn topology(links: Vec<Link>) -> Topology {
    Topology {
        schema_version: 1,
        origin: Origin::Synthetic,
        source_note: "A14 synthetic numerical regression; no hardware measurement".into(),
        devices: vec![
            Device {
                rank: 0,
                host: "synthetic".into(),
                capacity_bytes: 1 << 20,
                reserved_bytes: 0,
            },
            Device {
                rank: 1,
                host: "synthetic".into(),
                capacity_bytes: 1 << 20,
                reserved_bytes: 0,
            },
        ],
        links,
    }
}

#[test]
fn finite_parallel_link_wins_in_either_input_order() {
    let slow = link("overflow", 0, 1, 1e-320, 0.0);
    let fast = link("finite", 0, 1, 1e12, 0.0);
    for links in [vec![slow.clone(), fast.clone()], vec![fast, slow]] {
        let (cost, resources) = topology(links).transfer(0, 1, 1_000_004).unwrap();
        assert!((cost - 1.000004).abs() < 1e-12, "{cost}");
        assert!(resources.iter().any(|r| r == "vf:link:finite"));
        assert!(!resources.iter().any(|r| r == "vf:link:overflow"));
    }
}

#[test]
fn overflow_only_is_distinct_from_missing_enabled_path() {
    let overflow = topology(vec![link("overflow", 0, 1, 1e-320, 0.0)])
        .transfer(0, 1, 1_000_004)
        .unwrap_err();
    assert_eq!(overflow.code, "TRANSFER_TIME_OVERFLOW");

    let missing = topology(vec![]).transfer(0, 1, 1_000_004).unwrap_err();
    assert_eq!(missing.code, "NO_PATH");

    let mut disabled = link("disabled", 0, 1, 1e12, 0.0);
    disabled.enabled = false;
    let disabled = topology(vec![disabled])
        .transfer(0, 1, 1_000_004)
        .unwrap_err();
    assert_eq!(disabled.code, "NO_PATH");
}

fn local_problem(costs: &[f64], frequencies: &[f64]) -> PlacementProblem {
    assert_eq!(costs.len(), frequencies.len());
    PlacementProblem {
        topology: Topology {
            schema_version: 1,
            origin: Origin::Synthetic,
            source_note: "A14 synthetic local score regression; no hardware measurement".into(),
            devices: vec![Device {
                rank: 0,
                host: "synthetic".into(),
                capacity_bytes: 1 << 20,
                reserved_bytes: 0,
            }],
            links: vec![],
        },
        experts: costs
            .iter()
            .enumerate()
            .map(|(id, &compute_us)| ExpertProfile {
                id: id as u32,
                resident_bytes: 1,
                compute_us,
                cost_provenance: CostProvenance::default(),
            })
            .collect(),
        demands: frequencies
            .iter()
            .enumerate()
            .map(|(id, &frequency)| Demand {
                source_rank: 0,
                experts: vec![id as u32],
                frequency,
            })
            .collect(),
        hidden: 1,
        wire: WireFormat::default(),
        max_iterations: 0,
        wire_policy: Default::default(),
        max_work_units: 100_000,
    }
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= expected.abs().max(1.0) * 1e-14,
        "{actual:?} != {expected:?}"
    );
}

#[test]
fn single_subnormal_frequency_preserves_its_positive_cost() {
    let p = local_problem(&[0.1], &[f64::from_bits(1)]);
    close(score(&p, &[(0, 0)].into_iter().collect()).unwrap(), 0.1);

    let result = plan(&p).unwrap();
    close(result.modeled_weighted_latency_us, 0.1);
    close(
        result.demand_comparisons[0].baseline.modeled_latency_us,
        0.1,
    );
    assert!(result.work_units_used > 0);
}

#[test]
fn common_subnormal_frequency_scale_preserves_two_demand_mean() {
    let min_subnormal = f64::from_bits(1);
    let tiny = local_problem(&[0.1, 0.5], &[min_subnormal, f64::from_bits(2)]);
    let ordinary = local_problem(&[0.1, 0.5], &[0.5, 1.0]);

    let expected = (0.1 * 0.5 + 0.5) / 1.5;
    close(
        score(&tiny, &[(0, 0), (1, 0)].into_iter().collect()).unwrap(),
        expected,
    );
    close(
        score(&ordinary, &[(0, 0), (1, 0)].into_iter().collect()).unwrap(),
        expected,
    );
    close(plan(&tiny).unwrap().modeled_weighted_latency_us, expected);
    close(
        plan(&ordinary).unwrap().modeled_weighted_latency_us,
        expected,
    );
}

#[test]
fn zero_cost_and_large_finite_weights_and_costs_remain_valid() {
    let zero = local_problem(&[0.0], &[f64::from_bits(1)]);
    assert_eq!(score(&zero, &[(0, 0)].into_iter().collect()).unwrap(), 0.0);

    let large = local_problem(&[f64::MAX / 2.0, f64::MAX / 2.0], &[f64::MAX, f64::MAX]);
    close(
        score(&large, &[(0, 0), (1, 0)].into_iter().collect()).unwrap(),
        f64::MAX / 2.0,
    );
    close(
        plan(&large).unwrap().modeled_weighted_latency_us,
        f64::MAX / 2.0,
    );
}

#[test]
fn raw_extreme_frequency_ratio_has_correctly_rounded_score() {
    let min_subnormal = f64::from_bits(1);
    let costs = [f64::MAX, 0.0];
    let frequencies = [min_subnormal, f64::MAX];
    let owners = [(0, 0), (1, 0)].into_iter().collect();
    for (ordered_costs, ordered_frequencies) in [
        (costs, frequencies),
        ([costs[1], costs[0]], [frequencies[1], frequencies[0]]),
    ] {
        let p = local_problem(&ordered_costs, &ordered_frequencies);
        assert_eq!(
            score(&p, &owners).unwrap().to_bits(),
            min_subnormal.to_bits()
        );
        assert_eq!(
            plan(&p).unwrap().modeled_weighted_latency_us.to_bits(),
            min_subnormal.to_bits()
        );
    }
}

#[test]
fn exact_big_weights_and_cost_products_do_not_overflow_intermediates() {
    let p = local_problem(&[f64::MAX, 0.0], &[f64::MAX, f64::MAX]);
    let owners = [(0, 0), (1, 0)].into_iter().collect();
    assert_eq!(
        score(&p, &owners).unwrap().to_bits(),
        (f64::MAX / 2.0).to_bits()
    );
}

#[test]
fn zero_frequency_is_ignored_but_demands_are_still_validated() {
    let p = local_problem(&[f64::MAX, 3.0], &[0.0, 1.0]);
    let owners = [(0, 0), (1, 0)].into_iter().collect();
    assert_eq!(score(&p, &owners).unwrap(), 3.0);
    assert_eq!(plan(&p).unwrap().modeled_weighted_latency_us, 3.0);
    let permuted = local_problem(&[3.0, f64::MAX], &[1.0, 0.0]);
    assert_eq!(score(&permuted, &owners).unwrap(), 3.0);
    assert_eq!(plan(&permuted).unwrap().modeled_weighted_latency_us, 3.0);

    let all_zero = local_problem(&[1.0], &[0.0]);
    let error = score(&all_zero, &[(0, 0)].into_iter().collect()).unwrap_err();
    assert_eq!(error.code, "INVALID_INPUT");
    assert_eq!(plan(&all_zero).unwrap_err().code, "INVALID_INPUT");

    let mut no_demands = local_problem(&[1.0], &[1.0]);
    no_demands.demands.clear();
    assert_eq!(
        score(&no_demands, &[(0, 0)].into_iter().collect()).unwrap(),
        0.0
    );
    assert_eq!(plan(&no_demands).unwrap().modeled_weighted_latency_us, 0.0);

    for invalid_frequency in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let invalid = local_problem(&[1.0], &[invalid_frequency]);
        assert_eq!(
            score(&invalid, &[(0, 0)].into_iter().collect())
                .unwrap_err()
                .code,
            "INVALID_INPUT"
        );
    }
}

/// Ranks 0 and 1 with the given usable bytes, linked both ways at the given bandwidths.
fn two_ranks(capacity: [u64; 2], bandwidth: [f64; 2]) -> Topology {
    Topology {
        schema_version: 1,
        origin: Origin::Synthetic,
        source_note: "A53-R32 synthetic local-search overflow regression".into(),
        devices: (0..2)
            .map(|rank| Device {
                rank,
                host: "synthetic".into(),
                capacity_bytes: capacity[rank as usize],
                reserved_bytes: 0,
            })
            .collect(),
        links: vec![
            link("up", 0, 1, bandwidth[0], 0.0),
            link("down", 1, 0, bandwidth[1], 0.0),
        ],
    }
}

fn expert(id: u32, resident_bytes: u64, compute_us: f64) -> ExpertProfile {
    ExpertProfile {
        id,
        resident_bytes,
        compute_us,
        cost_provenance: CostProvenance::default(),
    }
}

#[test]
fn an_overflowing_local_search_alternative_is_discarded_not_fatal() {
    // A53-R32: local search skipped an infeasible alternative (CAPACITY_EXCEEDED, NO_PATH)
    // but aborted `plan` on one whose modelled time overflowed, though an infinite cost can
    // never replace the finite best-fit placement: allowing local search turned a valid
    // placement into a refusal of valid input.
    let mut schedule_case = PlacementProblem {
        // Best fit puts expert 0 on rank 1 and expert 1 on rank 0 (1e308 us). Moving
        // expert 0 to rank 0 runs both 1e308 us computations there in sequence: inf.
        topology: two_ranks([3, 1], [1e12, 1e12]),
        experts: vec![expert(0, 1, 1e308), expert(1, 1, 1e308)],
        demands: vec![Demand {
            source_rank: 0,
            experts: vec![0, 1],
            frequency: 1.0,
        }],
        hidden: 1,
        wire: WireFormat::default(),
        max_iterations: 0,
        wire_policy: Default::default(),
        max_work_units: 100_000,
    };
    let mut transfer_case = PlacementProblem {
        // Best fit keeps expert 0 on rank 0. Moving it to rank 1 sends its 2000-byte output
        // back at 1e-300 B/s: 2e309 us, an overflow on the only path. The 1-byte reach
        // probe of rank 1 stays finite (1e306 us), so best fit itself is unaffected.
        topology: two_ranks([2, 2], [1e12, 1e-300]),
        experts: vec![expert(0, 2, 1.0)],
        demands: vec![Demand {
            source_rank: 0,
            experts: vec![0],
            frequency: 1.0,
        }],
        hidden: 1000,
        ..schedule_case.clone()
    };
    for (name, p, alternative) in [
        ("schedule", &mut schedule_case, [(0, 0), (1, 0)]),
        ("transfer", &mut transfer_case, [(0, 1), (0, 1)]),
    ] {
        p.max_iterations = 0;
        let best_fit = plan(p).unwrap();
        assert!(best_fit.modeled_weighted_latency_us.is_finite(), "{name}");
        p.max_iterations = 8;
        let searched = plan(p).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(searched.owners, best_fit.owners, "{name}");
        assert_eq!(
            searched.modeled_weighted_latency_us, best_fit.modeled_weighted_latency_us,
            "{name}"
        );
        // Scoring that alternative directly still reports its overflow.
        let owners = alternative.into_iter().collect();
        assert!(score(p, &owners).is_err(), "{name}");
    }
}
