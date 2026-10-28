mod common;
use common::*;
use voltafabric::{pipeline::*, planner::*, schedule::*, traffic::*};
#[test]
fn grouping_can_reduce_payload() {
    let t = wire_bytes(4096, &[4, 4], WireFormat::default()).unwrap();
    assert_eq!(t.baseline_bytes, 131072);
    assert_eq!(t.grouped_bytes, 49152);
    assert_eq!(t.adaptive_bytes, 49152);
}
#[test]
fn grouping_can_increase_payload() {
    let t = wire_bytes(4096, &[1; 8], WireFormat::default()).unwrap();
    assert_eq!(t.grouped_bytes, 196608);
    assert_eq!(t.adaptive_bytes, 131072);
}
#[test]
fn adaptive_is_per_destination() {
    let t = wire_bytes(4096, &[1, 4], WireFormat::default()).unwrap();
    assert_eq!(t.adaptive_bytes, 16384 + 24576);
}
#[test]
fn zero_counts_do_not_send_messages() {
    let t = wire_bytes(4, &[0, 0], WireFormat::default()).unwrap();
    assert_eq!(t.adaptive_messages, 0);
    assert_eq!(t.baseline_bytes, 0);
}
#[test]
fn headers_alignment_and_metadata_are_counted() {
    let f = WireFormat {
        header_bytes: 10,
        slot_metadata_bytes: 4,
        alignment: 64,
        ..WireFormat::default()
    };
    let t = wire_bytes(4, &[2], f).unwrap();
    assert_eq!(t.baseline_bytes, 256);
    assert_eq!(t.grouped_bytes, 128);
}
#[test]
fn traffic_overflow_is_rejected() {
    assert!(wire_bytes(u64::MAX, &[8], WireFormat::default()).is_err());
}
#[test]
fn link_directions_differ() {
    let t = topology();
    t.validate().unwrap();
    assert_eq!(t.transfer(2, 9, 1000).unwrap().0, 6.0);
    assert_eq!(t.transfer(9, 2, 1000).unwrap().0, 9.0);
}
#[test]
fn no_fabricated_reverse_link() {
    let mut t = topology();
    t.links.pop();
    assert!(t.transfer(9, 2, 10).is_err());
}
#[test]
fn disabled_link_is_not_available() {
    let mut t = topology();
    t.links[0].enabled = false;
    assert!(t.transfer(2, 9, 10).is_err());
}
#[test]
fn topology_rejects_nonfinite() {
    let mut t = topology();
    t.links[0].latency_us = f64::NAN;
    assert!(t.validate().is_err());
}
fn job(id: &str, d: f64, deps: &[&str], res: &[&str]) -> Job {
    Job {
        id: id.into(),
        duration_us: d,
        depends_on: deps.iter().map(|s| s.to_string()).collect(),
        resources: res.iter().map(|s| s.to_string()).collect(),
    }
}
#[test]
fn independent_work_overlaps() {
    let s = schedule(&[
        job("a", 10.0, &[], &["gpu:0"]),
        job("b", 7.0, &[], &["gpu:1"]),
    ])
    .unwrap();
    assert_eq!(s.total_us, 10.0);
}
#[test]
fn same_uplink_serializes() {
    let s = schedule(&[
        job("a", 10.0, &[], &["uplink"]),
        job("b", 7.0, &[], &["uplink"]),
    ])
    .unwrap();
    assert_eq!(s.total_us, 17.0);
}
#[test]
fn dependencies_serialize_different_resources() {
    let s = schedule(&[
        job("a", 10.0, &[], &["dma"]),
        job("b", 7.0, &["a"], &["gpu"]),
    ])
    .unwrap();
    assert_eq!(s.total_us, 17.0);
}
#[test]
fn cycles_rejected() {
    assert!(schedule(&[job("a", 1.0, &["b"], &[]), job("b", 1.0, &["a"], &[])]).is_err());
}
#[test]
fn missing_dependency_rejected() {
    assert!(schedule(&[job("a", 1.0, &["z"], &[])]).is_err());
}
#[test]
fn empty_schedule_valid() {
    assert_eq!(schedule(&[]).unwrap().total_us, 0.0);
}
#[test]
fn planner_fits_and_improves_or_equals_initial() {
    let p = placement();
    let a = plan(&p).unwrap();
    assert_eq!(a.owners.len(), 3);
    assert!(a.modeled_weighted_latency_us <= a.initial_latency_us + 1e-9);
    for d in &p.topology.devices {
        assert!(a.bytes_by_rank[&d.rank] + d.reserved_bytes <= d.capacity_bytes);
    }
    assert!((score(&p, &a.owners).unwrap() - a.modeled_weighted_latency_us).abs() < 1e-9);
}
#[test]
fn infeasible_individual_expert_rejected() {
    let mut p = placement();
    p.experts[0].resident_bytes = 91;
    assert!(plan(&p).is_err());
}
#[test]
fn infeasible_aggregate_rejected() {
    let mut p = placement();
    for e in &mut p.experts {
        e.resident_bytes = 70;
    }
    assert!(plan(&p).is_err());
}
#[test]
fn planner_rejects_unknown_source() {
    let mut p = placement();
    p.demands[0].source_rank = 77;
    assert!(plan(&p).is_err());
}
fn pipeline() -> PipelineProblem {
    PipelineProblem {
        layers: (0..4)
            .map(|_| Layer {
                bytes: 10,
                compute_us_by_stage: vec![2.0, 2.0],
            })
            .collect(),
        stages: vec![
            Stage {
                rank: 2,
                available_bytes: 30,
                boundary_after_us: 1.0,
            },
            Stage {
                rank: 9,
                available_bytes: 30,
                boundary_after_us: 0.0,
            },
        ],
    }
}
#[test]
fn pipeline_balances_contiguous_layers() {
    let r = partition(&pipeline()).unwrap();
    assert_eq!(r.segments[0].end_layer, 2);
    assert_eq!(r.bottleneck_proxy_us, 5.0);
    assert_eq!(r.serial_latency_us, 9.0);
}
#[test]
fn pipeline_capacity_can_force_unbalanced_cut() {
    let mut p = pipeline();
    p.stages[0].available_bytes = 10;
    let r = partition(&p).unwrap();
    assert_eq!(r.segments[0].end_layer, 1);
}
#[test]
fn pipeline_impossible_capacity_rejected() {
    let mut p = pipeline();
    p.stages[0].available_bytes = 10;
    p.stages[1].available_bytes = 10;
    assert!(partition(&p).is_err());
}
#[test]
fn pipeline_does_not_allow_empty_stage() {
    let mut p = pipeline();
    p.layers.truncate(1);
    assert!(partition(&p).is_err());
}

// A40 guard sweep: each case violates one planner clause. With that clause alone
// disabled, several were ACCEPTED (empty problem, 4097 iterations, zero-byte expert,
// empty demand), one PANICKED (unknown demand expert), one returned 0.0 for a positive
// mean, and the rest were refused only by an unrelated later check.
fn plan_refused(name: &str, edit: impl FnOnce(&mut PlacementProblem), reason: &str) {
    let mut p = placement();
    edit(&mut p);
    let error = plan(&p).expect_err(name);
    assert!(
        error.message.contains(reason),
        "{name}: expected {reason:?}, got {error}"
    );
}
#[test]
fn planner_bounds_refuse_for_their_own_reason() {
    let e0 = placement().experts[0].clone();
    let limits = "bounded planner limits exceeded";
    plan_refused(
        "no experts",
        |p| {
            p.experts.clear();
            p.demands.clear();
        },
        limits,
    );
    plan_refused(
        "257 experts",
        |p| {
            p.experts = (0..257)
                .map(|id| ExpertProfile { id, ..e0.clone() })
                .collect()
        },
        limits,
    );
    plan_refused(
        "1025 demands",
        |p| p.demands = vec![p.demands[0].clone(); 1025],
        limits,
    );
    plan_refused("4097 iterations", |p| p.max_iterations = 4097, limits);
}
#[test]
fn expert_profiles_refuse_for_their_own_reason() {
    let profile = "invalid expert profile";
    plan_refused(
        "duplicate expert",
        |p| {
            let mut e = p.experts[0].clone();
            e.compute_us = 3.0;
            p.experts.push(e);
        },
        profile,
    );
    plan_refused("zero bytes", |p| p.experts[0].resident_bytes = 0, profile);
    plan_refused(
        "NaN compute",
        |p| p.experts[0].compute_us = f64::NAN,
        profile,
    );
    plan_refused(
        "negative compute",
        |p| p.experts[0].compute_us = -1.0,
        profile,
    );
}
#[test]
fn demands_refuse_for_their_own_reason_and_never_panic() {
    let e0 = placement().experts[0].clone();
    let demand = "invalid exact demand or nonnegative frequency";
    plan_refused(
        "unknown source",
        |p| p.demands[0].source_rank = 4242,
        demand,
    );
    plan_refused("empty demand", |p| p.demands[0].experts.clear(), demand);
    plan_refused(
        "65 experts in one demand",
        |p| {
            p.experts = (0..100)
                .map(|id| ExpertProfile { id, ..e0.clone() })
                .collect();
            p.demands[0].experts = (0..65).collect();
        },
        demand,
    );
    let expert = "invalid demand expert";
    plan_refused(
        "repeated expert",
        |p| p.demands[0].experts = vec![10, 10],
        expert,
    );
    plan_refused(
        "unknown expert",
        |p| p.demands[0].experts = vec![4242],
        expert,
    );
}
#[test]
fn score_requires_one_owner_per_expert() {
    let p = placement();
    let mut owners = plan(&p).unwrap().owners;
    owners.pop_first();
    let error = score(&p, &owners).unwrap_err();
    assert_eq!(error.message, "one owner per expert required");
}
#[test]
fn positive_mean_below_the_smallest_subnormal_is_refused_not_zero() {
    // Exact mean is 5e-324 / (1e300 + 1): positive, but rounds to zero in binary64.
    let mut p = placement();
    p.experts[0].compute_us = 0.0;
    p.experts[2].compute_us = 5e-324;
    p.demands = vec![
        Demand {
            source_rank: 2,
            experts: vec![10],
            frequency: 1e300,
        },
        Demand {
            source_rank: 9,
            experts: vec![99],
            frequency: 1.0,
        },
    ];
    let owners = [(10, 2), (25, 2), (99, 9)].into_iter().collect();
    let error = score(&p, &owners).unwrap_err();
    assert!(
        error.message.contains("below representable f64 range"),
        "{error}"
    );
}
#[test]
fn aggregate_memory_check_precedes_the_greedy_search() {
    plan_refused(
        "resident bytes exceed all memory",
        |p| {
            for e in &mut p.experts {
                e.resident_bytes = u64::MAX / 8;
            }
        },
        "aggregate resident bytes exceed aggregate usable memory",
    );
}

// A40 guard sweep: schedule, topology, wire-size and work-budget clauses, one per
// case. With the clause disabled each input was accepted or refused only later.
mod own_reason {
    use super::*;
    use voltafabric::topology::*;

    fn job(id: &str, duration_us: f64, depends_on: &[&str], resources: &[&str]) -> Job {
        Job {
            id: id.into(),
            duration_us,
            depends_on: depends_on.iter().map(|s| s.to_string()).collect(),
            resources: resources.iter().map(|s| s.to_string()).collect(),
        }
    }
    fn schedule_refused(jobs: &[Job], reason: &str) {
        let error = schedule(jobs).unwrap_err();
        assert!(
            error.message.contains(reason),
            "expected {reason:?}, got {error}"
        );
    }
    #[test]
    fn schedule_clauses() {
        let jobs: Vec<_> = (0..4097)
            .map(|i| job(&format!("j{i}"), 1.0, &[], &[]))
            .collect();
        schedule_refused(&jobs, "at most 4096 jobs");
        let identity = "duplicate/empty/oversized job ID or invalid duration";
        schedule_refused(&[job("", 1.0, &[], &[])], identity);
        schedule_refused(&[job(&"j".repeat(257), 1.0, &[], &[])], identity);
        schedule_refused(
            &[job("a", 1.0, &[], &[]), job("a", 2.0, &[], &[])],
            identity,
        );
        schedule_refused(&[job("a", f64::NAN, &[], &[])], identity);
        schedule_refused(&[job("a", -1.0, &[], &[])], identity);
        let lists = "empty/oversized/duplicate dependency or resource";
        schedule_refused(&[job("a", 1.0, &[], &[""])], lists);
        schedule_refused(&[job("a", 1.0, &[], &[&"r".repeat(257)])], lists);
        schedule_refused(&[job("a", 1.0, &[], &["r", "r"])], lists);
        schedule_refused(&[job("a", 1.0, &["ghost"], &[])], "missing dependency");
        // Each duration is finite; serialized on one resource they overflow.
        schedule_refused(
            &[
                job("a", 1.7e308, &[], &["r"]),
                job("b", 1.7e308, &[], &["r"]),
            ],
            "schedule time overflow",
        );
    }
    fn topology_refused(change: impl FnOnce(&mut Topology), reason: &str) {
        let mut t = common::topology();
        change(&mut t);
        let error = t.validate().unwrap_err();
        assert!(
            error.message.contains(reason),
            "expected {reason:?}, got {error}"
        );
    }
    #[test]
    fn topology_clauses() {
        let header = "topology version, provenance, or bounded device/link count";
        topology_refused(|t| t.schema_version = 2, header);
        topology_refused(|t| t.source_note = "n".repeat(4097), header);
        topology_refused(|t| t.source_note = " ".into(), header);
        topology_refused(
            |t| {
                t.devices.clear();
                t.links.clear();
            },
            header,
        );
        topology_refused(
            |t| {
                let d = t.devices[0].clone();
                t.devices = (0..257).map(|rank| Device { rank, ..d.clone() }).collect();
                t.links.clear();
            },
            header,
        );
        topology_refused(
            |t| {
                let l = t.links[0].clone();
                t.links = (0..65537)
                    .map(|i| Link {
                        id: format!("l{i}"),
                        ..l.clone()
                    })
                    .collect();
            },
            header,
        );
        let device = "duplicate rank or invalid device capacity/host";
        topology_refused(|t| t.devices[1].rank = 2, device);
        topology_refused(|t| t.devices[0].host = "h".repeat(257), device);
        topology_refused(|t| t.devices[0].host = " ".into(), device);
        topology_refused(
            |t| {
                t.devices[0].capacity_bytes = 0;
                t.devices[0].reserved_bytes = 0;
            },
            device,
        );
        let link = "invalid directed link, timing or identity";
        topology_refused(|t| t.links[0].id = String::new(), link);
        topology_refused(|t| t.links[0].id = "l".repeat(129), link);
        topology_refused(|t| t.links[1].id = t.links[0].id.clone(), link);
        topology_refused(|t| t.links[0].dst = t.links[0].src, link);
        topology_refused(|t| t.links[0].src = 4242, link);
        topology_refused(|t| t.links[0].dst = 4242, link);
        topology_refused(|t| t.links[0].bandwidth_bytes_per_s = f64::INFINITY, link);
        topology_refused(|t| t.links[0].bandwidth_bytes_per_s = 0.0, link);
        topology_refused(|t| t.links[0].latency_us = -1.0, link);
        let resource = "invalid resource identity (vf: reserved)";
        topology_refused(|t| t.links[0].resources = vec![String::new()], resource);
        topology_refused(|t| t.links[0].resources = vec!["r".repeat(257)], resource);
        topology_refused(|t| t.links[0].resources = vec!["vf:mine".into()], resource);
        topology_refused(
            |t| t.links[0].resources = vec!["r".into(), "r".into()],
            resource,
        );
        for (src, dst) in [(4242, 9), (2, 4242)] {
            let error = common::topology().transfer(src, dst, 64).unwrap_err();
            assert_eq!(error.message, "unknown transfer endpoint", "{src}->{dst}");
        }
    }
    #[test]
    fn wire_sizes_and_work_budget_clauses() {
        let sizes = "positive hidden and element sizes required";
        let f = WireFormat::default();
        for (hidden, f) in [
            (0, f),
            (
                4096,
                WireFormat {
                    input_bytes: 0,
                    ..f
                },
            ),
            (
                4096,
                WireFormat {
                    expert_output_bytes: 0,
                    ..f
                },
            ),
        ] {
            let error = message_sizes(hidden, 1, f, false).unwrap_err();
            assert_eq!(error.message, sizes);
        }
        let mut p = placement();
        p.max_work_units = 0;
        assert_eq!(
            plan(&p).unwrap_err().message,
            "work limit must be within 1..=1000000"
        );
    }
}

#[test]
fn score_names_an_owner_map_with_a_foreign_expert_or_rank() {
    // A41 reason sweep: the owner count matches, so only the lookups can refuse.
    let p = placement();
    let foreign_expert = [(10, 2), (25, 2), (4242, 9)].into_iter().collect();
    assert_eq!(
        score(&p, &foreign_expert).unwrap_err().message,
        "missing owner"
    );
    let foreign_rank = [(10, 2), (25, 2), (99, 4242)].into_iter().collect();
    assert_eq!(
        score(&p, &foreign_rank).unwrap_err().message,
        "unknown owner rank"
    );
}
#[test]
fn every_work_limit_either_plans_or_reports_budget_exhaustion() {
    // A41 reason sweep: errors raised inside the search are propagated through
    // match arms; whatever the limit, the caller must see the budget error itself.
    let mut p = placement();
    for limit in 1..=1_000_000 {
        p.max_work_units = limit;
        match plan(&p) {
            Ok(_) => return,
            Err(e) => assert_eq!(e.code, "WORK_BUDGET_EXCEEDED", "limit {limit}: {e}"),
        }
    }
    panic!("the fixture never planned within the standard work limit");
}
#[test]
fn a_dependency_cycle_is_reported_as_such() {
    let jobs = [
        Job {
            id: "a".into(),
            duration_us: 1.0,
            depends_on: vec!["b".into()],
            resources: vec![],
        },
        Job {
            id: "b".into(),
            duration_us: 1.0,
            depends_on: vec!["a".into()],
            resources: vec![],
        },
    ];
    let error = schedule(&jobs).unwrap_err();
    assert_eq!(
        (error.code.as_str(), error.message.as_str()),
        ("DEPENDENCY_CYCLE", "no ready job")
    );
}
