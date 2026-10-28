//! Public-surface rejections for planner and SM70 scale-domain invariants.
//!
//! The scale-domain case closes a real gap found by scripts/rust_mutations.py:
//! removing the K bound left every test green because the existing cases all
//! use K=64. The planner cases were equivalent mutants, rejected by a second
//! guard, and are pinned here so that stays true at the public API.
use serde_json::{json, Value};
use std::collections::BTreeMap;
use voltafabric::planner::{plan, score, PlacementProblem};
use voltafabric::quant::validate_sm70_scale_domain;

fn input() -> Value {
    json!({
        "topology": {
            "schema_version": 1, "origin": "synthetic",
            "source_note": "A27 analytical fixture, no hardware measurements",
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

fn tile(scale: u8, k: u64) -> Vec<u8> {
    vec![scale; (k as usize / 16) * 32]
}

#[test]
fn invalid_scale_domain_k_is_rejected_even_with_consistent_lengths() {
    // Both cases keep scales and gscales self-consistent, so the length checks
    // cannot fire and only the K bound can reject them.
    // 96 is a multiple of 32 but not of 64.
    assert!(validate_sm70_scale_domain(&tile(0x38, 96), &[1.0], 96).is_err());
    // 65_600 is a multiple of 64 but above MAX_K.
    assert!(validate_sm70_scale_domain(&tile(0x38, 65_600), &[1.0], 65_600).is_err());
    // The valid neighbour still passes, so the bound is not simply refusing work.
    assert_eq!(
        validate_sm70_scale_domain(&tile(0x38, 64), &[1.0], 64),
        Ok(())
    );
}

#[test]
fn duplicate_expert_in_one_demand_never_produces_a_placement() {
    let mut value = input();
    value["demands"][0]["experts"] = json!([10, 10]);
    let p: PlacementProblem = serde_json::from_value(value).unwrap();
    assert!(plan(&p).is_err());
}

#[test]
fn demand_from_a_rank_outside_the_topology_never_produces_a_placement() {
    let mut value = input();
    value["demands"][0]["source_rank"] = json!(7);
    let p: PlacementProblem = serde_json::from_value(value).unwrap();
    assert!(plan(&p).is_err());
}

#[test]
fn score_rejects_an_owner_map_that_misses_an_expert() {
    let p: PlacementProblem = serde_json::from_value(input()).unwrap();
    let mut owners = BTreeMap::new();
    owners.insert(10u32, 0u32);
    assert!(score(&p, &owners).is_err());
}
