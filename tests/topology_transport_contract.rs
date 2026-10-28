use serde_json::json;
use voltafabric::topology::Topology;

fn topology(src_host: &str, dst_host: &str, transport: &str) -> Topology {
    serde_json::from_value(json!({
        "schema_version": 1,
        "origin": "synthetic",
        "source_note": "transport contract fixture",
        "devices": [
            {"rank": 0, "host": src_host, "capacity_bytes": 1, "reserved_bytes": 0},
            {"rank": 1, "host": dst_host, "capacity_bytes": 1, "reserved_bytes": 0}
        ],
        "links": [{
            "id": "link-0-1", "src": 0, "dst": 1,
            "bandwidth_bytes_per_s": 1_000_000_000.0,
            "latency_us": 1.0, "enabled": true,
            "transport": transport, "resources": []
        }]
    }))
    .unwrap()
}

fn topology_with_pcie(
    generation: u8,
    lane_width: u8,
    bandwidth_bytes_per_s: f64,
    transport: &str,
) -> serde_json::Result<Topology> {
    serde_json::from_value(json!({
        "schema_version": 1,
        "origin": "synthetic",
        "source_note": "PCIe capacity contract fixture",
        "devices": [
            {"rank": 0, "host": "pc3", "capacity_bytes": 1, "reserved_bytes": 0},
            {"rank": 1, "host": "pc3", "capacity_bytes": 1, "reserved_bytes": 0}
        ],
        "links": [{
            "id": "link-0-1", "src": 0, "dst": 1,
            "bandwidth_bytes_per_s": bandwidth_bytes_per_s,
            "latency_us": 1.0, "enabled": true,
            "transport": transport, "resources": [],
            "pcie": {"generation": generation, "lane_width": lane_width}
        }]
    }))
}

#[test]
fn pcie_p2p_cannot_connect_devices_on_different_hosts() {
    let invalid_edge = topology("pc6", "pc3", "pcie_p2p");
    assert_eq!(invalid_edge.validate().unwrap_err().code, "INVALID_INPUT");
}

#[test]
fn pcie_p2p_within_one_host_remains_valid() {
    topology("pc3", "pc3", "pcie_p2p").validate().unwrap();
}

#[test]
fn synthetic_inter_host_edges_remain_available_for_modeling() {
    topology("pc6", "pc3", "synthetic").validate().unwrap();
}

#[test]
fn optional_gen3_x4_profile_is_accepted_for_a_bounded_link() {
    let link = topology_with_pcie(3, 4, 1_000_000_000.0, "pcie_p2p").unwrap();
    link.validate().unwrap();
    assert_eq!(
        link.links[0]
            .pcie
            .as_ref()
            .unwrap()
            .theoretical_ceiling_bytes_per_s()
            .unwrap(),
        3_938_461_538
    );
}

#[test]
fn gen3_lane_width_scales_the_encoding_ceiling_not_a_throughput_claim() {
    for (width, expected) in [(4, 3_938_461_538), (8, 7_876_923_076), (16, 15_753_846_153)] {
        let link = topology_with_pcie(3, width, 1.0, "pcie_p2p").unwrap();
        link.validate().unwrap();
        assert_eq!(
            link.links[0]
                .pcie
                .as_ref()
                .unwrap()
                .theoretical_ceiling_bytes_per_s()
                .unwrap(),
            expected
        );
    }
}

#[test]
fn generation_ceiling_matches_per_direction_pcie_data_rates() {
    for (generation, expected_x4) in [
        (1, 1_000_000_000),
        (2, 2_000_000_000),
        (3, 3_938_461_538),
        (4, 7_876_923_076),
        (5, 15_753_846_153),
        (6, 32_000_000_000),
        (7, 64_000_000_000),
    ] {
        let link = topology_with_pcie(generation, 4, 1.0, "pcie_p2p").unwrap();
        link.validate().unwrap();
        assert_eq!(
            link.links[0]
                .pcie
                .as_ref()
                .unwrap()
                .theoretical_ceiling_bytes_per_s()
                .unwrap(),
            expected_x4,
            "Gen{generation}"
        );
    }
}

#[test]
fn gen6_and_gen7_x16_ceiling_is_half_the_pcie_sig_bidirectional_bandwidth() {
    // PCI-SIG lists 256 GB/s bidirectional for Gen6 x16 and 512 GB/s for Gen7 x16.
    // This planner value is the one-direction ceiling, before FLIT/FEC overhead.
    for (generation, expected_one_way) in [(6, 128_000_000_000), (7, 256_000_000_000)] {
        let link = topology_with_pcie(generation, 16, 1.0, "pcie_p2p").unwrap();
        link.validate().unwrap();
        assert_eq!(
            link.links[0]
                .pcie
                .as_ref()
                .unwrap()
                .theoretical_ceiling_bytes_per_s()
                .unwrap(),
            expected_one_way,
            "Gen{generation} x16 one direction"
        );
    }
}

#[test]
fn bandwidth_at_gen3_x4_ceiling_is_accepted_and_one_byte_per_second_over_is_rejected() {
    let ceiling = topology_with_pcie(3, 4, 3_938_461_538.0, "pcie_p2p").unwrap();
    ceiling.validate().unwrap();

    let over = topology_with_pcie(3, 4, 3_938_461_539.0, "pcie_p2p").unwrap();
    assert_eq!(over.validate().unwrap_err().code, "INVALID_INPUT");
}

#[test]
fn pcie_profile_rejects_unknown_generation_width_and_non_pcie_transport() {
    for (generation, width, transport) in [
        (0, 4, "pcie_p2p"),
        (8, 4, "pcie_p2p"),
        (3, 3, "pcie_p2p"),
        (3, 64, "pcie_p2p"),
        (3, 4, "rdma"),
    ] {
        let link = topology_with_pcie(generation, width, 1.0, transport).unwrap();
        assert_eq!(link.validate().unwrap_err().code, "INVALID_INPUT");
    }
}
