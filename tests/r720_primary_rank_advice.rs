use voltafabric::cost::{CostOrigin, CostProvenance};
use voltafabric::planner::recommend_primary_rank_advisory;
use voltafabric::topology::{Device, Link, Origin, Topology, Transport};

fn measured_topology(rate1: f64, rate2: f64) -> Topology {
    Topology {
        schema_version: 1,
        origin: Origin::ImportedMeasurement,
        source_note: "R720 fixture only; synthetic measurements, not hardware evidence".into(),
        devices: vec![
            Device {
                rank: 0,
                host: "pc6".into(),
                capacity_bytes: 1,
                reserved_bytes: 0,
            },
            Device {
                rank: 1,
                host: "pc6".into(),
                capacity_bytes: 1024,
                reserved_bytes: 768,
            },
            Device {
                rank: 2,
                host: "pc6".into(),
                capacity_bytes: 2048,
                reserved_bytes: 1536,
            },
        ],
        links: vec![
            link(1, rate1),
            link(2, rate2),
            reverse_link(1),
            reverse_link(2),
        ],
    }
}

fn link(rank: u32, rate: f64) -> Link {
    Link {
        id: format!("host-to-gpu-{rank}"),
        src: 0,
        dst: rank,
        bandwidth_bytes_per_s: rate,
        latency_us: 1.0,
        enabled: true,
        transport: Transport::HostStaged,
        pcie: None,
        resources: vec![format!("dma-{rank}")],
        cost_provenance: CostProvenance {
            origin: CostOrigin::ImportedMeasurement,
            source: Some(format!("fixture://R720/host-to-gpu-{rank}")),
        },
    }
}

fn reverse_link(rank: u32) -> Link {
    Link {
        id: format!("gpu-to-host-{rank}"),
        src: rank,
        dst: 0,
        bandwidth_bytes_per_s: 1.0,
        latency_us: 1.0,
        enabled: true,
        transport: Transport::HostStaged,
        pcie: None,
        resources: vec![format!("dma-{rank}")],
        cost_provenance: CostProvenance {
            origin: CostOrigin::ImportedMeasurement,
            source: Some(format!("fixture://R720/gpu-to-host-{rank}")),
        },
    }
}

#[test]
fn bandwidth_wins_outside_explicit_measurement_tie_band() {
    let advice = recommend_primary_rank_advisory(&measured_topology(100.0, 87.0), 0, &[1, 2], 0.0)
        .expect("two imported link profiles");
    assert_eq!(advice.primary_rank, 1);
    assert_eq!(advice.primary_link_bytes_per_s, 100.0);
    assert_eq!(advice.available_memory_bytes, 256);
    assert_eq!(advice.ranks_within_bandwidth_tie, vec![1]);
    assert_eq!(
        advice.result_kind,
        "imported_link_role_advice_not_executor_placement"
    );
}

#[test]
fn explicit_overlap_band_prefers_more_unreserved_memory_deterministically() {
    let advice = recommend_primary_rank_advisory(&measured_topology(100.0, 87.0), 0, &[1, 2], 0.15)
        .expect("both links fall inside caller-specified tie band");
    assert_eq!(advice.primary_rank, 2);
    assert_eq!(advice.primary_link_bytes_per_s, 87.0);
    assert_eq!(advice.available_memory_bytes, 512);
    assert_eq!(advice.ranks_within_bandwidth_tie, vec![1, 2]);
    assert_eq!(
        advice.evidence_sources_by_rank[&2],
        "fixture://R720/host-to-gpu-2"
    );
}

#[test]
fn synthetic_topology_and_unmeasured_candidate_links_fail_closed() {
    let mut synthetic = measured_topology(100.0, 87.0);
    synthetic.origin = Origin::Synthetic;
    assert_eq!(
        recommend_primary_rank_advisory(&synthetic, 0, &[1, 2], 0.0)
            .unwrap_err()
            .code,
        "UNMEASURED_TOPOLOGY"
    );

    let mut mixed = measured_topology(100.0, 87.0);
    mixed.links[1].cost_provenance.origin = CostOrigin::Synthetic;
    mixed.links[1].cost_provenance.source = Some("fixture://R720/marked-synthetic".into());
    assert_eq!(
        recommend_primary_rank_advisory(&mixed, 0, &[1, 2], 0.0)
            .unwrap_err()
            .code,
        "UNMEASURED_HOST_LINK"
    );
}

#[test]
fn invalid_tolerance_duplicate_rank_and_cross_host_candidate_are_rejected() {
    let topology = measured_topology(100.0, 87.0);
    for tolerance in [f64::NAN, -0.01, 1.0] {
        assert!(recommend_primary_rank_advisory(&topology, 0, &[1, 2], tolerance).is_err());
    }
    assert!(recommend_primary_rank_advisory(&topology, 0, &[1, 1], 0.0).is_err());
    assert!(recommend_primary_rank_advisory(&topology, 0, &[9], 0.0).is_err());

    let mut remote = measured_topology(100.0, 87.0);
    remote.devices[2].host = "pc4".into();
    assert!(recommend_primary_rank_advisory(&remote, 0, &[1, 2], 0.0).is_err());
}
