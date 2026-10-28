use serde_json::{json, Value};
use voltafabric::planner::{plan, PlacementProblem};

fn problem(shared: bool, latency: f64) -> PlacementProblem {
    serde_json::from_value(input(shared, latency)).unwrap()
}
fn input(shared: bool, latency: f64) -> Value {
    let resources = if shared { vec!["shared-bus"] } else { vec![] };
    json!({
        "topology": {
            "schema_version": 1, "origin": "synthetic",
            "source_note": "Adversarial analytical fixture, no hardware measurements",
            "devices": [
                {"rank": 0, "host": "synthetic", "capacity_bytes": 1, "reserved_bytes": 0},
                {"rank": 1, "host": "synthetic", "capacity_bytes": 4, "reserved_bytes": 0}
            ],
            "links": [
                {"id":"0-1","src":0,"dst":1,"bandwidth_bytes_per_s":1e9,"latency_us":latency,"enabled":true,"transport":"synthetic","resources":resources},
                {"id":"1-0","src":1,"dst":0,"bandwidth_bytes_per_s":1e9,"latency_us":latency,"enabled":true,"transport":"synthetic","resources":resources}
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
fn close(actual: f64, expected: f64) {
    // Analytical decimal costs have only floating-point roundoff; no numerical kernel tolerance.
    assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
}
#[test]
fn adaptive_avoids_byte_saving_latency_regression() {
    let result = plan(&problem(false, 0.0)).unwrap();
    close(result.modeled_weighted_latency_us, 216.384);
}
#[test]
fn grouped_positive_control_with_shared_transport() {
    let result = plan(&problem(true, 100.0)).unwrap();
    close(result.modeled_weighted_latency_us, 424.576);
}
#[test]
fn explicit_baseline_remains_available_for_ablation() {
    let mut value = input(true, 100.0);
    value["wire_policy"] = json!("baseline");
    let p: PlacementProblem = serde_json::from_value(value).unwrap();
    close(plan(&p).unwrap().modeled_weighted_latency_us, 432.768);
}
#[test]
fn explicit_grouped_remains_available_for_ablation() {
    let mut value = input(false, 0.0);
    value["wire_policy"] = json!("grouped");
    let p: PlacementProblem = serde_json::from_value(value).unwrap();
    close(plan(&p).unwrap().modeled_weighted_latency_us, 224.576);
}
#[test]
fn work_budget_includes_initial_placement_without_candidates() {
    let mut value = input(false, 0.0);
    value["max_work_units"] = json!(1);
    let p: PlacementProblem = serde_json::from_value(value).unwrap();
    assert_eq!(plan(&p).unwrap_err().code, "WORK_BUDGET_EXCEEDED");
}
#[test]
fn adaptive_reports_all_three_graphs_and_legacy_costs_as_assumed() {
    let output = serde_json::to_value(plan(&problem(false, 0.0)).unwrap()).unwrap();
    let comparison = &output["demand_comparisons"][0];
    assert_eq!(comparison["selected"], "baseline");
    close(
        comparison["baseline"]["modeled_latency_us"]
            .as_f64()
            .unwrap(),
        216.384,
    );
    close(
        comparison["grouped"]["modeled_latency_us"]
            .as_f64()
            .unwrap(),
        224.576,
    );
    assert_eq!(comparison["baseline"]["wire_bytes"], 32768);
    assert_eq!(comparison["grouped"]["wire_bytes"], 24576);
    assert_eq!(output["compute_profiles"]["10"]["origin"], "assumed");
    assert_eq!(output["link_profiles"]["0-1"]["origin"], "assumed");
    assert!(output["work_units_used"].as_u64().unwrap() <= 1_000_000);
}
#[test]
fn imported_measurement_requires_evidence_reference() {
    let mut value = input(false, 0.0);
    value["experts"][0]["cost_provenance"] = json!({"origin":"imported_measurement"});
    let p: PlacementProblem = serde_json::from_value(value).unwrap();
    assert_eq!(plan(&p).unwrap_err().code, "INVALID_INPUT");
}
#[test]
fn imported_link_measurement_requires_evidence_reference() {
    let mut value = input(false, 0.0);
    value["topology"]["links"][0]["cost_provenance"] = json!({"origin":"imported_measurement"});
    let p: PlacementProblem = serde_json::from_value(value).unwrap();
    assert_eq!(plan(&p).unwrap_err().code, "INVALID_INPUT");
}

#[test]
fn ties_choose_baseline() {
    let output = serde_json::to_value(plan(&problem(false, 100.0)).unwrap()).unwrap();
    assert_eq!(output["demand_comparisons"][0]["selected"], "baseline");
    close(
        output["demand_comparisons"][0]["baseline"]["modeled_latency_us"]
            .as_f64()
            .unwrap(),
        424.576,
    );
}
#[test]
fn valid_cost_references_remain_typed_without_promoting_the_score() {
    let mut value = input(false, 0.0);
    value["experts"][0]["cost_provenance"] =
        json!({"origin":"imported_measurement","source":"fixture-evidence://compute-10"});
    value["experts"][1]["cost_provenance"] =
        json!({"origin":"synthetic","source":"hand-derived fixture"});
    value["topology"]["links"][0]["cost_provenance"] =
        json!({"origin":"imported_measurement","source":"fixture-evidence://link-0-1"});
    let p: PlacementProblem = serde_json::from_value(value).unwrap();
    let output = serde_json::to_value(plan(&p).unwrap()).unwrap();
    assert_eq!(
        output["compute_profiles"]["10"]["origin"],
        "imported_measurement"
    );
    assert_eq!(output["compute_profiles"]["99"]["origin"], "synthetic");
    assert_eq!(
        output["link_profiles"]["0-1"]["source"],
        "fixture-evidence://link-0-1"
    );
    assert_eq!(
        output["result_kind"],
        "analytical_resident_placement_not_gpu_measurement"
    );
}
#[test]
fn empty_and_oversized_cost_references_are_rejected() {
    for source in [" ".to_string(), "x".repeat(4097)] {
        let mut value = input(false, 0.0);
        value["experts"][0]["cost_provenance"] =
            json!({"origin":"imported_measurement","source":source});
        let p: PlacementProblem = serde_json::from_value(value).unwrap();
        assert_eq!(plan(&p).unwrap_err().code, "INVALID_INPUT");
    }
}
#[test]
fn budget_is_shared_by_final_diagnostics_and_not_reset_per_candidate() {
    let p = problem(false, 0.0);
    let first = plan(&p).unwrap();
    assert_eq!(first.work_units_used, plan(&p).unwrap().work_units_used);
    let mut value = input(false, 0.0);
    value["max_work_units"] = json!(first.work_units_used - 1);
    let limited: PlacementProblem = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(plan(&limited).unwrap_err().code, "WORK_BUDGET_EXCEEDED");
    value["max_work_units"] = json!(first.work_units_used);
    let exact: PlacementProblem = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(plan(&exact).unwrap().work_units_used, first.work_units_used);
    value["max_iterations"] = json!(1);
    let search: PlacementProblem = serde_json::from_value(value).unwrap();
    assert_eq!(plan(&search).unwrap_err().code, "WORK_BUDGET_EXCEEDED");
}
#[test]
fn missing_reverse_link_is_an_error_not_a_partial_placement() {
    let mut p = problem(false, 0.0);
    p.topology.links.pop();
    assert_eq!(plan(&p).unwrap_err().code, "GREEDY_PLACEMENT_FAILED");
    let owners = [(10, 1), (99, 1)].into_iter().collect();
    assert_eq!(
        voltafabric::planner::score(&p, &owners).unwrap_err().code,
        "NO_PATH"
    );
}
#[test]
fn prepared_parallel_links_preserve_selection_by_message_size() {
    let mut value = input(false, 0.0);
    value["topology"]["links"][0]["latency_us"] = json!(0.0);
    value["topology"]["links"][0]["bandwidth_bytes_per_s"] = json!(1e6);
    let mut parallel = value["topology"]["links"][0].clone();
    parallel["id"] = json!("fast-bandwidth");
    parallel["latency_us"] = json!(100.0);
    parallel["bandwidth_bytes_per_s"] = json!(1e9);
    value["topology"]["links"]
        .as_array_mut()
        .unwrap()
        .push(parallel);
    value["hidden"] = json!(4096);
    value["wire_policy"] = json!("baseline");
    let large: PlacementProblem = serde_json::from_value(value.clone()).unwrap();
    close(plan(&large).unwrap().modeled_weighted_latency_us, 324.576);
    value["hidden"] = json!(1);
    let small: PlacementProblem = serde_json::from_value(value).unwrap();
    close(plan(&small).unwrap().modeled_weighted_latency_us, 202.002);
}
#[test]
fn empty_demand_set_has_zero_analytical_score() {
    let mut p = problem(false, 0.0);
    p.demands.clear();
    let result = plan(&p).unwrap();
    assert_eq!(result.modeled_weighted_latency_us, 0.0);
    assert!(result.demand_comparisons.is_empty());
}
#[test]
fn fp32_wire_and_headers_change_bytes_without_hidden_fp16_assumptions() {
    let mut p = problem(false, 0.0);
    p.wire.input_bytes = 4;
    p.wire.expert_output_bytes = 4;
    p.wire.partial_output_bytes = 4;
    p.wire.header_bytes = 10;
    p.wire.slot_metadata_bytes = 4;
    p.wire.alignment = 64;
    let output = serde_json::to_value(plan(&p).unwrap()).unwrap();
    assert_eq!(
        output["demand_comparisons"][0]["baseline"]["wire_bytes"],
        65792
    );
    assert_eq!(
        output["demand_comparisons"][0]["grouped"]["wire_bytes"],
        32896
    );
}

#[test]
fn complete_graph_comparison_keeps_the_mixed_byte_adaptive_candidate() {
    let mut value = input(true, 20.0);
    value["topology"]["devices"]
        .as_array_mut()
        .unwrap()
        .push(json!({"rank":2,"host":"synthetic","capacity_bytes":2,"reserved_bytes":0}));
    for (src, dst) in [(0, 2), (2, 0)] {
        value["topology"]["links"]
            .as_array_mut()
            .unwrap()
            .push(json!({
                "id":format!("{src}-{dst}"),"src":src,"dst":dst,"bandwidth_bytes_per_s":1e9,
                "latency_us":20.0,"enabled":true,"transport":"synthetic","resources":["shared-bus"]
            }));
    }
    value["experts"] = json!([
        {"id":10,"resident_bytes":2,"compute_us":0.0},
        {"id":50,"resident_bytes":2,"compute_us":0.0},
        {"id":99,"resident_bytes":2,"compute_us":0.0}
    ]);
    value["demands"][0]["experts"] = json!([10, 50, 99]);
    let p: PlacementProblem = serde_json::from_value(value).unwrap();
    let output = serde_json::to_value(plan(&p).unwrap()).unwrap();
    let comparison = &output["demand_comparisons"][0];
    assert_eq!(comparison["selected"], "byte_adaptive");
    close(
        comparison["baseline"]["modeled_latency_us"]
            .as_f64()
            .unwrap(),
        169.152,
    );
    close(
        comparison["grouped"]["modeled_latency_us"]
            .as_f64()
            .unwrap(),
        129.152,
    );
    close(
        comparison["byte_adaptive"]["modeled_latency_us"]
            .as_f64()
            .unwrap(),
        120.96,
    );
    assert_eq!(comparison["baseline"]["wire_bytes"], 49152);
    assert_eq!(comparison["grouped"]["wire_bytes"], 49152);
    assert_eq!(comparison["byte_adaptive"]["wire_bytes"], 40960);
}
