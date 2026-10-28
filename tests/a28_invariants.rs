//! Bounded-contract rejections across topology, traffic, the work budget and
//! the artifact writer.
//!
//! The first three close real gaps found by scripts/rust_mutations.py: removing
//! each guard left the whole suite green. The artifact case was an equivalent
//! mutant, rejected by a second check, and is pinned so that stays true.
use serde_json::{json, Value};
use std::path::Path;
use voltafabric::planner::{plan, PlacementProblem};
use voltafabric::traffic::{wire_bytes, WireFormat};
use voltafabric::{moe, quant, resident, resident_artifact};

fn input_value() -> Value {
    json!({
        "topology": {
            "schema_version": 1, "origin": "synthetic",
            "source_note": "A28 probe fixture, no hardware measurements",
            "devices": [
                {"rank": 0, "host": "synthetic", "capacity_bytes": 8, "reserved_bytes": 0},
                {"rank": 1, "host": "synthetic", "capacity_bytes": 8, "reserved_bytes": 0}
            ],
            "links": [
                {"id":"0-1","src":0,"dst":1,"bandwidth_bytes_per_s":1e9,"latency_us":0.0,"enabled":true,"transport":"synthetic","resources":[]},
                {"id":"1-0","src":1,"dst":0,"bandwidth_bytes_per_s":1e9,"latency_us":0.0,"enabled":true,"transport":"synthetic","resources":[]}
            ]
        },
        "experts": [
            {"id":10,"resident_bytes":2,"compute_us":100.0},
            {"id":99,"resident_bytes":2,"compute_us":100.0}
        ],
        "demands":[{"source_rank":0,"experts":[10,99],"frequency":1.0}],
        "hidden":4096,
        "wire":{"input_bytes":2,"expert_output_bytes":2,"partial_output_bytes":4,"header_bytes":0,"slot_metadata_bytes":0,"alignment":1},
        "max_iterations":0
    })
}

fn matrix(rows: u64, columns: u64, seed: u8, factor: f64) -> quant::PackedRows {
    quant::PackedRows {
        rows,
        columns,
        codes: (0..rows * columns / 2)
            .map(|i| (i as u8).wrapping_mul(73).wrapping_add(seed))
            .collect(),
        scales: (0..rows * columns / 16)
            .map(|i| ((i + u64::from(seed)) % 127) as u8)
            .collect(),
        global_dequant: factor,
    }
}

fn resident_expert(id: u32, local: u32, seed: u8) -> resident::ResidentExpert {
    resident::ResidentExpert {
        id,
        local,
        gate: matrix(128, 64, seed, 0.5),
        up: matrix(128, 64, seed + 1, 2.0),
        down: matrix(64, 128, seed + 2, 4.0),
    }
}

#[test]
fn reserved_bytes_beyond_device_capacity_is_rejected() {
    let mut value = input_value();
    value["topology"]["devices"][0]["reserved_bytes"] = json!(9);
    let p: PlacementProblem = serde_json::from_value(value).unwrap();
    assert!(plan(&p).is_err());
}

#[test]
fn zero_partial_output_bytes_is_rejected() {
    let f = WireFormat {
        input_bytes: 2,
        expert_output_bytes: 2,
        partial_output_bytes: 0,
        header_bytes: 0,
        slot_metadata_bytes: 0,
        alignment: 1,
    };
    assert!(wire_bytes(4096, &[1, 1], f).is_err());
}

#[test]
fn work_limit_outside_its_bound_is_rejected() {
    for limit in [0u64, 2_000_000] {
        let mut value = input_value();
        value["max_work_units"] = json!(limit);
        let p: PlacementProblem = serde_json::from_value(value).unwrap();
        assert!(plan(&p).is_err(), "limit={limit}");
    }
}

#[test]
fn export_path_without_a_file_name_is_rejected() {
    let input = resident::ResidentInput {
        model_revision: "model-A".into(),
        layer: 3,
        rank: 0,
        registry: moe::Registry {
            epoch: 12,
            copies: vec![
                moe::Replica {
                    expert: 41,
                    rank: 0,
                    local: 2,
                },
                moe::Replica {
                    expert: 77,
                    rank: 0,
                    local: 99,
                },
            ],
        },
        resident_order: vec![2, 99],
        experts: vec![resident_expert(41, 2, 5), resident_expert(77, 99, 9)],
    };
    let stack = resident::pack(&input).unwrap();
    assert!(resident_artifact::export(&stack, Path::new("/")).is_err());
}
