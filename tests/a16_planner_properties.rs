use voltafabric::{
    cost::CostProvenance,
    planner::{score, Demand, ExpertProfile, PlacementProblem},
    topology::{Device, Origin, Topology},
    traffic::WireFormat,
};

fn local_problem(costs: &[f64], frequencies: &[f64]) -> PlacementProblem {
    assert_eq!(costs.len(), frequencies.len());
    PlacementProblem {
        topology: Topology {
            schema_version: 1,
            origin: Origin::Synthetic,
            source_note: "A16 generated numerical property; no hardware measurement".into(),
            devices: vec![Device {
                rank: 0,
                host: "synthetic".into(),
                capacity_bytes: u64::MAX,
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

#[test]
fn representable_subnormal_mean_is_preserved_across_demand_permutations() {
    let costs = [1e-300, 0.0];
    let frequencies = [1e-20, 1.0];
    let original = local_problem(&costs, &frequencies);
    let reversed = local_problem(&[costs[1], costs[0]], &[frequencies[1], frequencies[0]]);
    let owners = [(0, 0), (1, 0)].into_iter().collect();
    let forward = score(&original, &owners).unwrap();
    let backward = score(&reversed, &owners).unwrap();

    // The exact dyadic quotient rounds to a nonzero f64. A generous ULP-scale
    // check is in the independent Fraction oracle; this assertion detects zero loss.
    assert!(
        forward > 0.0,
        "small-first permutation lost a representable mean"
    );
    assert!(
        backward > 0.0,
        "large-first permutation lost a representable mean"
    );
    assert_eq!(forward.to_bits(), backward.to_bits());
}

#[test]
fn tiny_frequency_times_maximum_finite_cost_is_not_lost() {
    let subnormal = f64::from_bits(1);
    let costs = [f64::MAX / 3.0, 0.0];
    let frequencies = [subnormal, 1.0];
    let owners = [(0, 0), (1, 0)].into_iter().collect();
    let first = score(&local_problem(&costs, &frequencies), &owners).unwrap();
    let second = score(
        &local_problem(&[costs[1], costs[0]], &[frequencies[1], frequencies[0]]),
        &owners,
    )
    .unwrap();

    assert!(
        first > 0.0,
        "small-first permutation lost a representable term"
    );
    assert_eq!(first.to_bits(), second.to_bits());
}

#[test]
fn tiny_weight_with_many_unit_weights_keeps_a_representable_term() {
    let mut costs = vec![0.0; 256];
    let mut frequencies = vec![1.0; 256];
    costs[255] = f64::MAX;
    frequencies[255] = f64::from_bits(1);
    let owners: std::collections::BTreeMap<_, _> = (0..256).map(|id| (id, 0)).collect();
    let result = score(&local_problem(&costs, &frequencies), &owners).unwrap();
    assert!(result > 0.0, "a representable frequency-cost term was lost");
}

#[test]
fn subnormal_halfway_tie_rounds_to_even_in_every_demand_order() {
    let minsub = f64::from_bits(1);
    let owners = vec![(0, 0), (1, 0)].into_iter().collect();

    for costs in [[3.0 * minsub, 0.0], [0.0, 3.0 * minsub]] {
        let frequencies = [1.0, 1.0];
        let problem = local_problem(&costs, &frequencies);
        let actual = score(&problem, &owners).unwrap();
        assert_eq!(actual.to_bits(), (2.0 * minsub).to_bits());
    }
}

#[test]
fn subnormal_mean_rounds_once_after_many_positive_terms() {
    let minsub = f64::from_bits(1);
    let costs = [3.0 * minsub, 3.0 * minsub, 3.0 * minsub, 0.0];
    let frequencies = [1.0; 4];
    let owners = (0..4).map(|id| (id, 0)).collect();
    let problem = local_problem(&costs, &frequencies);
    let actual = score(&problem, &owners).unwrap();
    assert_eq!(actual.to_bits(), (2.0 * minsub).to_bits());
}

#[test]
fn non_dyadic_weight_ratio_keeps_exact_subnormal_tie_rounding() {
    let minsub = f64::from_bits(1);
    let costs = [5.0 * minsub, 0.0];
    let frequencies = [3.0, 7.0];
    let owners = vec![(0, 0), (1, 0)].into_iter().collect();
    let problem = local_problem(&costs, &frequencies);
    let actual = score(&problem, &owners).unwrap();
    // 5 * (3 / (3 + 7)) = 1.5; ties-to-even chooses 2 minsub.
    assert_eq!(actual.to_bits(), (2.0 * minsub).to_bits());
}

#[test]
fn non_dyadic_weight_ratio_tie_rounding_is_order_independent() {
    let minsub = f64::from_bits(1);
    let owners = vec![(0, 0), (1, 0)].into_iter().collect();
    for (costs, frequencies) in [
        ([3.0 * minsub, 0.0], [5.0, 1.0]),
        ([0.0, 3.0 * minsub], [1.0, 5.0]),
    ] {
        let actual = score(&local_problem(&costs, &frequencies), &owners).unwrap();
        // 3 * 5 / (5 + 1) = 2.5 minsub; ties-to-even gives 2.
        assert_eq!(actual.to_bits(), (2.0 * minsub).to_bits());
    }
}

#[test]
fn near_min_normal_exact_rounding_survives_split_weight_permutations() {
    let high = f64::MIN_POSITIVE;
    let half = (1_u64 << 52) as f64 - 1.0;
    let costs = [high, high, 0.0];
    let frequencies = [half, half, 1.0];
    let expected = f64::from_bits(high.to_bits() - 1);
    let owners = vec![(0, 0), (1, 0), (2, 0)].into_iter().collect();
    for order in [[0, 1, 2], [2, 0, 1], [1, 2, 0]] {
        let permuted_costs = order.map(|i| costs[i]);
        let permuted_frequencies = order.map(|i| frequencies[i]);
        let actual = score(
            &local_problem(&permuted_costs, &permuted_frequencies),
            &owners,
        )
        .unwrap();
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn minnormal_boundary_exact_rounding_survives_demand_permutations() {
    let high = f64::MIN_POSITIVE;
    let large_weight = ((1_u64 << 53) - 2) as f64;
    let costs = [high, 0.0];
    let frequencies = [large_weight, 1.0];
    let expected = f64::from_bits(high.to_bits() - 1);
    let owners = vec![(0, 0), (1, 0)].into_iter().collect();
    for order in [[0, 1], [1, 0]] {
        let actual = score(
            &local_problem(&order.map(|i| costs[i]), &order.map(|i| frequencies[i])),
            &owners,
        )
        .unwrap();
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn raw_frequency_underflow_and_f64_max_products_remain_finite() {
    let minsub = f64::from_bits(1);
    let owners = vec![(0, 0), (1, 0)].into_iter().collect();
    for (costs, frequencies) in [
        ([f64::MAX, 0.0], [minsub, f64::MAX]),
        ([0.0, f64::MAX], [f64::MAX, minsub]),
    ] {
        assert_eq!(
            score(&local_problem(&costs, &frequencies), &owners)
                .unwrap()
                .to_bits(),
            minsub.to_bits()
        );
    }

    let exact_max_over_two = score(
        &local_problem(&[f64::MAX, 0.0], &[f64::MAX, f64::MAX]),
        &owners,
    )
    .unwrap();
    assert_eq!(exact_max_over_two.to_bits(), (f64::MAX / 2.0).to_bits());
}
