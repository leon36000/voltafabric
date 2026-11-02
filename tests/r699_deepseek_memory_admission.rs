//! R699 joins pinned DeepSeek-V4.1 expert slices to the persistent local CPU
//! Session and FP32 preweighted Collector. This is a one-token, two-rank expert
//! slice, not the model's router, attention, KV cache, or full generation.
#![cfg(target_os = "linux")]

use std::collections::BTreeMap;
use std::time::{Duration, Instant};
use voltafabric::cost::{CostOrigin, CostProvenance};
use voltafabric::moe::{Choice, Registry, Replica, Stamp};
use voltafabric::planner::{
    self, ExpertProfile, PerRankMemoryEnvelope, PlacementProblem, WirePolicy,
};
use voltafabric::preweighted::Collector;
use voltafabric::process::{Fault, ReceiptState};
use voltafabric::routing::RoutingProblem;
use voltafabric::session::{mxfp4, Endpoint, Session};
use voltafabric::topology::{Device, Link, Origin, Topology, Transport};
use voltafabric::traffic::WireFormat;

const REVISION: &str = "314113f8a7dadb388215396ba5fbee2c610a04d6";
const TIMEOUT: Duration = Duration::from_secs(180);
const HIDDEN: usize = 5120;
const EXPERT0_SHA256: &str = "f1bdb87fe86fcafa8ae72e4154a8b4a0e66b90eaba8366a398676cb2fbe4510b";
const EXPERT383_SHA256: &str = "9f80cd6bd1244876fb9df5c3a8571ea2c44a981767bbff039e05bab3380e1ab3";
const EXPERT0_RANK: u32 = 0;
const EXPERT383_RANK: u32 = 1;
const EXPERT0_LOCAL: u32 = 17;
const EXPERT383_LOCAL: u32 = 4;

struct Golden {
    input_bits: Vec<u32>,
    output: Vec<f32>,
    route_weight: f32,
}

fn signed_hex_f32(word: &str) -> u32 {
    let (negative, magnitude) = word
        .strip_prefix('-')
        .map_or((false, word), |magnitude| (true, magnitude));
    let bits = u32::from_str_radix(magnitude, 16).expect("pinned oracle hex word");
    f32::from_bits(if negative { bits.wrapping_neg() } else { bits }).to_bits()
}

fn golden(bytes: &[u8]) -> Golden {
    let value: serde_json::Value = serde_json::from_slice(bytes).expect("pinned expert oracle");
    assert_eq!(value["model_revision"], REVISION);
    let input_bits: Vec<_> = value["input_f32_bits"]
        .as_array()
        .expect("input bits")
        .iter()
        .map(|word| signed_hex_f32(word.as_str().expect("hex string")))
        .collect();
    let output: Vec<_> = value["output_f32"]
        .as_array()
        .expect("oracle output")
        .iter()
        .map(|number| number.as_f64().expect("finite output") as f32)
        .collect();
    let weight_bits = u32::from_str_radix(
        value["routing_weight_f32_bits"]
            .as_str()
            .expect("route weight bits"),
        16,
    )
    .expect("valid route weight word");
    Golden {
        input_bits,
        output,
        route_weight: f32::from_bits(weight_bits),
    }
}

fn endpoint(rank: u32) -> Endpoint {
    Endpoint {
        rank,
        argv: vec![env!("CARGO_BIN_EXE_vf-rank-session").into()],
    }
}

fn binding(expert: u32, local: u32, rank: u32, sha256: &str) -> mxfp4::Binding {
    mxfp4::Binding {
        artifact_sha256: sha256.to_owned(),
        model_revision: REVISION.to_owned(),
        rank,
        layer: 0,
        epoch: 1,
        experts: vec![mxfp4::Expert { expert, local }],
    }
}

fn problem(weight0: f32, weight383: f32) -> RoutingProblem {
    RoutingProblem {
        hidden: HIDDEN,
        experts: vec![0, 383],
        routes: vec![vec![
            Choice {
                expert: 0,
                weight: f64::from(weight0),
            },
            Choice {
                expert: 383,
                weight: f64::from(weight383),
            },
        ]],
        registry: Registry {
            epoch: 1,
            copies: vec![
                Replica {
                    expert: 0,
                    rank: EXPERT0_RANK,
                    local: EXPERT0_LOCAL,
                },
                Replica {
                    expert: 383,
                    rank: EXPERT383_RANK,
                    local: EXPERT383_LOCAL,
                },
            ],
        },
        stamp: Stamp {
            run_id: "r699-deepseek-memory-session-v1".to_owned(),
            model_revision: REVISION.to_owned(),
            plan_id: "r699-one-token-two-rank-cpu-v1".to_owned(),
            layer: 0,
            batch: 0,
            epoch: 1,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([
            (EXPERT0_RANK, vec![EXPERT0_LOCAL]),
            (EXPERT383_RANK, vec![EXPERT383_LOCAL]),
        ]),
    }
}

fn provenance() -> CostProvenance {
    CostProvenance {
        origin: CostOrigin::Synthetic,
        source: Some("R699 bounded memory-admission fixture; no hardware measurement".into()),
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
        cost_provenance: provenance(),
    }
}
fn memory_problem(weight0: u64, weight383: u64, fail_one_byte: bool) -> PlacementProblem {
    let capacity0 = weight0.checked_mul(2).unwrap() - u64::from(fail_one_byte);
    let capacity383 = weight383.checked_mul(2).unwrap();
    PlacementProblem {
        topology: Topology {
            schema_version: 1,
            origin: Origin::Synthetic,
            source_note: "Two local CPU session ranks; advisory topology only".into(),
            devices: vec![
                Device {
                    rank: 0,
                    host: "local-cpu-rank-0".into(),
                    capacity_bytes: capacity0,
                    reserved_bytes: 0,
                },
                Device {
                    rank: 1,
                    host: "local-cpu-rank-1".into(),
                    capacity_bytes: capacity383,
                    reserved_bytes: 0,
                },
            ],
            links: vec![link("0-1", 0, 1), link("1-0", 1, 0)],
        },
        experts: vec![
            ExpertProfile {
                id: 0,
                resident_bytes: weight0,
                compute_us: 1.0,
                cost_provenance: provenance(),
            },
            ExpertProfile {
                id: 383,
                resident_bytes: weight383,
                compute_us: 1.0,
                cost_provenance: provenance(),
            },
        ],
        demands: vec![],
        hidden: HIDDEN as u64,
        wire: WireFormat::default(),
        max_iterations: 0,
        wire_policy: WirePolicy::Adaptive,
        max_work_units: 1_000_000,
    }
}
fn envelopes(weight0: u64, weight383: u64) -> Vec<PerRankMemoryEnvelope> {
    [(EXPERT0_RANK, weight0), (EXPERT383_RANK, weight383)]
        .into_iter()
        .map(|(rank, load_temporary_bytes)| PerRankMemoryEnvelope {
            rank,
            resident_expert_bytes: 0,
            kv_bytes: 0,
            recurrent_bytes: 0,
            workspace_bytes: 0,
            graph_bytes: 0,
            communication_bytes: 0,
            safety_margin_bytes: 0,
            load_temporary_bytes,
        })
        .collect()
}

fn dispatch_one(
    worker: &mut Session,
    plan: &RoutingProblem,
    collector: &mut Collector,
    attempt: u64,
    input_bits: Vec<u32>,
) {
    let pending = worker
        .start_preweighted_fp32(
            plan,
            collector,
            attempt,
            vec![mxfp4::TokenInput {
                token: 0,
                value_bits: input_bits,
            }],
            Fault::None,
        )
        .expect("fresh identified Session dispatch");
    let receipt = worker.finish_preweighted_fp32(pending, collector, Instant::now() + TIMEOUT);
    assert_eq!(receipt.state, ReceiptState::Completed, "{receipt:?}");
    assert!(receipt.error.is_none(), "{receipt:?}");
    assert_eq!(
        receipt.attempt, attempt,
        "completion attempt must match admission"
    );
}

#[test]
fn pinned_real_experts_cross_two_persistent_sessions_into_one_collector() {
    let bytes0 = include_bytes!("../fixtures/deepseek_v41_layer0_expert0_314113f8.safetensors");
    let bytes383 = include_bytes!("../fixtures/deepseek_v41_layer0_expert383_314113f8.safetensors");
    let golden0 = golden(include_bytes!(
        "../fixtures/deepseek_v41_expert0_forward_bf16_314113f8.json"
    ));
    let golden383 = golden(include_bytes!(
        "../fixtures/deepseek_v41_expert383_forward_bf16_314113f8.json"
    ));

    assert_eq!(golden0.input_bits.len(), HIDDEN);
    assert_eq!(golden383.input_bits, golden0.input_bits);
    assert_eq!(golden0.output.len(), HIDDEN);
    assert_eq!(golden383.output.len(), HIDDEN);
    assert_eq!(golden0.route_weight.to_bits(), 0x3ec0_0000);
    assert_eq!(golden383.route_weight.to_bits(), 0x3ec0_0000);
    assert_eq!(
        voltafabric::sha256::hex(bytes0),
        EXPERT0_SHA256,
        "pin the actual official expert-0 bytes"
    );
    assert_eq!(
        voltafabric::sha256::hex(bytes383),
        EXPERT383_SHA256,
        "pin the actual official expert-383 bytes"
    );

    let artifact0 = mxfp4::Artifact::new(
        bytes0.to_vec(),
        binding(0, EXPERT0_LOCAL, EXPERT0_RANK, EXPERT0_SHA256),
    )
    .expect("admit exact layer-0 expert 0 MXFP4 K32 artifact");
    let artifact383 = mxfp4::Artifact::new(
        bytes383.to_vec(),
        binding(383, EXPERT383_LOCAL, EXPERT383_RANK, EXPERT383_SHA256),
    )
    .expect("admit exact layer-0 expert 383 MXFP4 K32 artifact");

    let plan = problem(golden0.route_weight, golden383.route_weight);
    let bytes0_u64 = u64::try_from(bytes0.len()).unwrap();
    let bytes383_u64 = u64::try_from(bytes383.len()).unwrap();
    let attempt = 903;
    let advice = planner::plan_routing_advisory_with_memory(
        &memory_problem(bytes0_u64, bytes383_u64, false),
        &plan,
        attempt,
        EXPERT0_RANK,
        &envelopes(bytes0_u64, bytes383_u64),
    )
    .expect("admit complete route before creating/loading sessions");
    assert_eq!(advice.routed.stamp.run_id, plan.stamp.run_id);
    assert_eq!(
        advice.routed.stamp.model_revision,
        plan.stamp.model_revision
    );
    assert_eq!(advice.routed.stamp.plan_id, plan.stamp.plan_id);
    assert_eq!(advice.routed.stamp.layer, plan.stamp.layer);
    assert_eq!(advice.routed.stamp.batch, plan.stamp.batch);
    assert_eq!(advice.routed.stamp.epoch, plan.stamp.epoch);
    assert_eq!(advice.routed.attempt, attempt);
    assert_eq!(advice.routed.route_experts_by_token, vec![vec![0, 383]]);
    assert_eq!(
        advice.routed.route_weight_bits_hex_by_token,
        vec![vec![
            format!("{:016x}", f64::from(golden0.route_weight).to_bits()),
            format!("{:016x}", f64::from(golden383.route_weight).to_bits()),
        ]]
    );
    assert_eq!(advice.routed.placement.owners.get(&0), Some(&EXPERT0_RANK));
    assert_eq!(
        advice.routed.placement.owners.get(&383),
        Some(&EXPERT383_RANK)
    );
    let rank0 = advice
        .ranks
        .iter()
        .find(|x| x.rank == EXPERT0_RANK)
        .unwrap();
    let rank383 = advice
        .ranks
        .iter()
        .find(|x| x.rank == EXPERT383_RANK)
        .unwrap();
    assert_eq!(rank0.expert_weight_bytes, bytes0_u64);
    assert_eq!(rank383.expert_weight_bytes, bytes383_u64);
    assert_eq!(rank0.loading_peak_bytes, bytes0_u64 * 2);
    assert_eq!(rank383.loading_peak_bytes, bytes383_u64 * 2);
    assert_eq!(rank0.headroom_bytes, 0);
    assert_eq!(rank383.headroom_bytes, 0);
    let mut collector = Collector::new(&plan).expect("complete two-expert route");
    let mut worker0 = Session::connect(&endpoint(EXPERT0_RANK), TIMEOUT).unwrap();
    let mut worker383 = Session::connect(&endpoint(EXPERT383_RANK), TIMEOUT).unwrap();
    worker0.load_mxfp4(&artifact0, 1, TIMEOUT).unwrap();
    worker383.load_mxfp4(&artifact383, 1, TIMEOUT).unwrap();

    dispatch_one(
        &mut worker0,
        &plan,
        &mut collector,
        attempt,
        golden0.input_bits.clone(),
    );
    dispatch_one(
        &mut worker383,
        &plan,
        &mut collector,
        attempt,
        golden383.input_bits.clone(),
    );

    let actual = collector.finish().expect("both rank outputs collected");
    assert_eq!(actual.len(), 1);
    assert_eq!(actual[0].len(), HIDDEN);
    // The fixture's pinned Python-oracle vectors already include each exact 0.375
    // coefficient at the pre-down boundary. The ordinary addition here mirrors the
    // selected FP32-preweighted Collector contract; no coefficient is applied twice.
    let expected: Vec<_> = golden0
        .output
        .iter()
        .zip(&golden383.output)
        .map(|(left, right)| left + right)
        .collect();
    let max_abs = actual[0]
        .iter()
        .zip(expected)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs <= 5.0e-4,
        "R9 FP32 two-rank collector differs from pinned per-expert oracle by {max_abs}"
    );

    assert_eq!(worker0.traffic.loads_transferred, 1);
    assert_eq!(worker383.traffic.loads_transferred, 1);
    assert_eq!(worker0.close(TIMEOUT).bye_execs, Some(1));
    assert_eq!(worker383.close(TIMEOUT).bye_execs, Some(1));
}

#[test]
fn one_byte_under_loading_peak_refuses_before_any_session_is_created() {
    let bytes0 = include_bytes!("../fixtures/deepseek_v41_layer0_expert0_314113f8.safetensors");
    let bytes383 = include_bytes!("../fixtures/deepseek_v41_layer0_expert383_314113f8.safetensors");
    let golden0 = golden(include_bytes!(
        "../fixtures/deepseek_v41_expert0_forward_bf16_314113f8.json"
    ));
    let golden383 = golden(include_bytes!(
        "../fixtures/deepseek_v41_expert383_forward_bf16_314113f8.json"
    ));
    let plan = problem(golden0.route_weight, golden383.route_weight);
    let bytes0_u64 = u64::try_from(bytes0.len()).unwrap();
    let bytes383_u64 = u64::try_from(bytes383.len()).unwrap();
    let refused = planner::plan_routing_advisory_with_memory(
        &memory_problem(bytes0_u64, bytes383_u64, true),
        &plan,
        904,
        EXPERT0_RANK,
        &envelopes(bytes0_u64, bytes383_u64),
    );
    assert!(
        refused.is_err(),
        "over-budget routes return no partial advice"
    );
}
