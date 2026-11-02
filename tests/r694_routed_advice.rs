use std::collections::BTreeMap;
use voltafabric_r694_exact_source_check::{
    cost::{CostOrigin, CostProvenance},
    moe::{Registry, Replica, Stamp},
    planner::{
        plan_qwen2moe_advisory, plan_qwen2moe_advisory_with_memory, ExpertProfile,
        PerRankMemoryEnvelope, PlacementProblem, WirePolicy,
    },
    routing::{self, Qwen2MoEDonorSelection, Qwen2MoERouteBatch},
    topology::{Device, Link, Origin, Topology, Transport},
    traffic::WireFormat,
};
fn routed() -> Qwen2MoERouteBatch {
    routing::qwen2moe_routes_from_donor(
        Qwen2MoEDonorSelection {
            stamp: Stamp {
                run_id: "r694-run-syn".into(),
                model_revision: "qwen2moe-synthetic-r1".into(),
                plan_id: "r694-plan-syn".into(),
                layer: 6,
                batch: 88,
                epoch: 4,
            },
            attempt: 901,
            tokens: 2,
            top_k: 2,
            selected_experts: vec![40, 3, 3, 40],
            final_weights: vec![0.25, 0.5, 0.0, 0.125],
        },
        32,
        vec![40, 3],
        Registry {
            epoch: 4,
            copies: vec![
                Replica {
                    expert: 40,
                    rank: 1,
                    local: 0,
                },
                Replica {
                    expert: 3,
                    rank: 1,
                    local: 1,
                },
            ],
        },
        BTreeMap::new(),
        BTreeMap::from([(1, vec![0, 1])]),
    )
    .unwrap()
}
fn prov() -> CostProvenance {
    CostProvenance {
        origin: CostOrigin::Synthetic,
        source: Some("R694 synthetic only".into()),
    }
}
fn link(id: &str, src: u32, dst: u32) -> Link {
    Link {
        id: id.into(),
        src,
        dst,
        bandwidth_bytes_per_s: 1_000_000_000.0,
        latency_us: 2.0,
        enabled: true,
        transport: Transport::Synthetic,
        pcie: None,
        resources: vec![],
        cost_provenance: prov(),
    }
}
fn problem() -> PlacementProblem {
    PlacementProblem {
        topology: Topology {
            schema_version: 1,
            origin: Origin::Synthetic,
            source_note: "No hardware measurement; deterministic synthetic fixture".into(),
            devices: vec![
                Device {
                    rank: 0,
                    host: "source".into(),
                    capacity_bytes: 1,
                    reserved_bytes: 0,
                },
                Device {
                    rank: 1,
                    host: "a".into(),
                    capacity_bytes: 8,
                    reserved_bytes: 0,
                },
                Device {
                    rank: 2,
                    host: "b".into(),
                    capacity_bytes: 8,
                    reserved_bytes: 0,
                },
            ],
            links: vec![
                link("0-1", 0, 1),
                link("1-0", 1, 0),
                link("0-2", 0, 2),
                link("2-0", 2, 0),
            ],
        },
        experts: vec![
            ExpertProfile {
                id: 40,
                resident_bytes: 8,
                compute_us: 5.0,
                cost_provenance: prov(),
            },
            ExpertProfile {
                id: 3,
                resident_bytes: 8,
                compute_us: 7.0,
                cost_provenance: prov(),
            },
        ],
        demands: vec![],
        hidden: 32,
        wire: WireFormat::default(),
        max_iterations: 0,
        wire_policy: WirePolicy::Adaptive,
        max_work_units: 1_000_000,
    }
}
type RouteSnapshot = (Stamp, Vec<Replica>, Vec<Vec<(u32, u64)>>);

fn snapshot(b: &Qwen2MoERouteBatch) -> RouteSnapshot {
    let p = b.problem();
    (
        p.stamp.clone(),
        p.registry.copies.clone(),
        p.routes
            .iter()
            .map(|r| r.iter().map(|c| (c.expert, c.weight.to_bits())).collect())
            .collect(),
    )
}
#[test]
fn route_bound_advice_keeps_order_weight_bits_and_registry() {
    let b = routed();
    let before = snapshot(&b);
    let d0 = routing::dispatch(b.problem()).unwrap();
    let a = plan_qwen2moe_advisory(&problem(), &b, 0).unwrap();
    let d1 = routing::dispatch(b.problem()).unwrap();
    assert_eq!(a.stamp, before.0);
    assert_eq!(a.attempt, 901);
    assert_eq!(a.source_rank, 0);
    assert_eq!(a.route_experts_by_token, vec![vec![40, 3], vec![3, 40]]);
    assert_eq!(
        a.route_weight_bits_hex_by_token,
        vec![
            vec![
                format!("{:016x}", f64::from(0.25f32).to_bits()),
                format!("{:016x}", f64::from(0.5f32).to_bits())
            ],
            vec![
                format!("{:016x}", f64::from(0.0f32).to_bits()),
                format!("{:016x}", f64::from(0.125f32).to_bits())
            ]
        ]
    );
    assert_eq!(d0.slots, d1.slots);
    assert_eq!(snapshot(&b), before);
    assert_eq!(a.placement.owners.len(), 2);
    assert_eq!(
        a.placement.result_kind,
        "analytical_resident_placement_not_gpu_measurement"
    );
}
#[test]
fn route_and_profile_expert_sets_must_match() {
    let b = routed();
    let mut p = problem();
    p.experts[0].id = 41;
    assert!(plan_qwen2moe_advisory(&p, &b, 0).is_err());
    assert_eq!(b.problem().routes[0][0].expert, 40);
}
#[test]
fn missing_path_returns_error_without_an_advice() {
    let b = routed();
    let mut p = problem();
    p.topology.links.clear();
    assert!(plan_qwen2moe_advisory(&p, &b, 0).is_err());
}

fn envelope(rank: u32, kv: u64, load_temporary: u64) -> PerRankMemoryEnvelope {
    PerRankMemoryEnvelope {
        rank,
        kv_bytes: kv,
        recurrent_bytes: 0,
        workspace_bytes: 0,
        graph_bytes: 0,
        communication_bytes: 0,
        safety_margin_bytes: 0,
        load_temporary_bytes: load_temporary,
    }
}

#[test]
fn per_rank_kv_and_loading_budgets_are_bound_to_route_advice() {
    let b = routed();
    let before = snapshot(&b);
    let mut p = problem();
    p.topology.devices[1].capacity_bytes = 32;
    p.topology.devices[2].capacity_bytes = 32;
    let advice = plan_qwen2moe_advisory_with_memory(
        &p,
        &b,
        0,
        &[envelope(0, 0, 0), envelope(1, 8, 2), envelope(2, 4, 1)],
    )
    .unwrap();

    assert_eq!(advice.routed.stamp, before.0);
    assert_eq!(advice.routed.attempt, 901);
    assert_eq!(
        advice.routed.route_experts_by_token,
        vec![vec![40, 3], vec![3, 40]]
    );
    assert_eq!(snapshot(&b), before);
    for rank in advice.ranks {
        assert!(rank.loading_peak_bytes <= rank.capacity_bytes);
        assert_eq!(
            rank.loading_peak_bytes,
            rank.steady_bytes + rank.load_temporary_bytes
        );
    }
}

#[test]
fn insufficient_kv_or_temporary_headroom_fails_without_partial_advice() {
    let b = routed();
    let before = snapshot(&b);
    let mut p = problem();
    p.topology.devices[1].capacity_bytes = 24;
    p.topology.devices[2].capacity_bytes = 24;
    let err = plan_qwen2moe_advisory_with_memory(
        &p,
        &b,
        0,
        &[envelope(0, 0, 0), envelope(1, 8, 9), envelope(2, 8, 9)],
    )
    .unwrap_err();
    assert_eq!(err.code, "CAPACITY_EXCEEDED");
    assert_eq!(snapshot(&b), before);
}

#[test]
fn memory_envelopes_must_cover_each_rank_once_and_reject_overflow() {
    let b = routed();
    let p = problem();
    let incomplete = [envelope(0, 0, 0), envelope(1, 0, 0)];
    assert!(plan_qwen2moe_advisory_with_memory(&p, &b, 0, &incomplete).is_err());

    let duplicate = [envelope(0, 0, 0), envelope(1, 0, 0), envelope(1, 0, 0)];
    assert!(plan_qwen2moe_advisory_with_memory(&p, &b, 0, &duplicate).is_err());

    let overflow = [
        envelope(0, 0, 0),
        PerRankMemoryEnvelope {
            kv_bytes: u64::MAX,
            workspace_bytes: 1,
            ..envelope(1, 0, 0)
        },
        envelope(2, 0, 0),
    ];
    assert!(plan_qwen2moe_advisory_with_memory(&p, &b, 0, &overflow).is_err());
}
