use std::collections::BTreeMap;

use voltafabric::{
    cost::CostProvenance,
    planner::{
        plan, plan_with_concurrent_scenarios, score_with_concurrent_scenarios,
        ConcurrentDemandScenario, Demand, ExpertProfile, PlacementProblem,
    },
    topology::{Device, Link, Origin, Topology, Transport},
    traffic::WireFormat,
};

fn link(id: &str, src: u32, dst: u32) -> Link {
    Link {
        id: id.into(),
        src,
        dst,
        bandwidth_bytes_per_s: 1_000_000_000.0,
        latency_us: 10.0,
        enabled: true,
        transport: Transport::Synthetic,
        pcie: None,
        resources: vec![],
        cost_provenance: CostProvenance::default(),
    }
}

fn problem(expert_capacity: u64) -> PlacementProblem {
    PlacementProblem {
        topology: Topology {
            schema_version: 1,
            origin: Origin::Synthetic,
            source_note: "R743 synthetic concurrency schedule; no hardware claim".into(),
            devices: vec![
                Device {
                    rank: 0,
                    host: "synthetic".into(),
                    capacity_bytes: expert_capacity,
                    reserved_bytes: 0,
                },
                Device {
                    rank: 1,
                    host: "synthetic".into(),
                    capacity_bytes: expert_capacity,
                    reserved_bytes: 0,
                },
            ],
            links: vec![link("0-1", 0, 1), link("1-0", 1, 0)],
        },
        experts: vec![
            ExpertProfile {
                id: 0,
                resident_bytes: 1,
                compute_us: 5.0,
                cost_provenance: CostProvenance::default(),
            },
            ExpertProfile {
                id: 1,
                resident_bytes: 1,
                compute_us: 5.0,
                cost_provenance: CostProvenance::default(),
            },
        ],
        demands: vec![
            Demand {
                source_rank: 0,
                experts: vec![0],
                frequency: 0.0,
            },
            Demand {
                source_rank: 1,
                experts: vec![1],
                frequency: 0.0,
            },
        ],
        hidden: 1,
        wire: WireFormat::default(),
        max_iterations: 2,
        wire_policy: Default::default(),
        max_work_units: 100_000,
    }
}

fn scenario(indices: Vec<usize>) -> ConcurrentDemandScenario {
    ConcurrentDemandScenario {
        demand_indices: indices,
        frequency: 1.0,
    }
}

#[test]
fn scenario_cost_exposes_shared_gpu_serialization_and_independent_gpu_overlap() {
    let p = problem(64);
    let same_rank: BTreeMap<_, _> = [(0, 0), (1, 0)].into_iter().collect();
    let split_ranks: BTreeMap<_, _> = [(0, 0), (1, 1)].into_iter().collect();

    let serialized =
        score_with_concurrent_scenarios(&p, &same_rank, &[scenario(vec![1, 0])]).unwrap();
    let overlapped =
        score_with_concurrent_scenarios(&p, &split_ranks, &[scenario(vec![0, 1])]).unwrap();
    assert!(serialized > overlapped, "{serialized} <= {overlapped}");
    assert_eq!(overlapped, 5.0);
    assert_eq!(
        serialized.to_bits(),
        score_with_concurrent_scenarios(&p, &same_rank, &[scenario(vec![0, 1])])
            .unwrap()
            .to_bits()
    );
}

#[test]
fn planner_moves_resident_experts_to_reduce_concurrent_completion_cost() {
    let p = problem(64);
    let original_routes = p.demands.clone();
    let planned = plan_with_concurrent_scenarios(&p, &[scenario(vec![1, 0])]).unwrap();

    assert_eq!(planned.owners.get(&0), Some(&0));
    assert_eq!(planned.owners.get(&1), Some(&1));
    assert!(planned.initial_latency_us > planned.modeled_weighted_latency_us);
    assert_eq!(planned.modeled_weighted_latency_us, 5.0);
    assert_eq!(
        planned.concurrent_scenario_comparisons[0].demand_indices,
        vec![0, 1]
    );
    assert_eq!(p.demands.len(), original_routes.len());
    assert_eq!(p.demands[0].experts, original_routes[0].experts);
    assert_eq!(p.demands[1].experts, original_routes[1].experts);
    assert!(planned.result_kind.contains("not_gpu_measurement"));
}

#[test]
fn resident_weight_capacity_guard_still_rejects_scenarios_atomically() {
    let p = problem(1);
    let same_rank: BTreeMap<_, _> = [(0, 0), (1, 0)].into_iter().collect();
    let error =
        score_with_concurrent_scenarios(&p, &same_rank, &[scenario(vec![0, 1])]).unwrap_err();
    assert_eq!(error.code, "CAPACITY_EXCEEDED");
}

#[test]
fn malformed_scenario_is_rejected_and_legacy_plan_shape_stays_unchanged() {
    let p = problem(64);
    let owners: BTreeMap<_, _> = [(0, 0), (1, 1)].into_iter().collect();
    for indices in [vec![0, 0], vec![0, 9]] {
        let error = score_with_concurrent_scenarios(&p, &owners, &[scenario(indices)]).unwrap_err();
        assert_eq!(error.code, "INVALID_INPUT");
    }

    let mut legacy = p.clone();
    legacy.demands[0].frequency = 1.0;
    legacy.demands[1].frequency = 1.0;
    let value = serde_json::to_value(plan(&legacy).unwrap()).unwrap();
    assert!(value.get("concurrent_scenario_comparisons").is_none());
}

#[test]
fn work_budget_exhaustion_returns_no_partial_plan() {
    let mut p = problem(64);
    p.max_work_units = 1;
    let error = plan_with_concurrent_scenarios(&p, &[scenario(vec![0, 1])]).unwrap_err();
    assert_eq!(error.code, "WORK_BUDGET_EXCEEDED");
}
