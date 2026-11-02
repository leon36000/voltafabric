//! R698 joins pinned DeepSeek-V4.1 expert slices to the persistent local CPU
//! Session and FP32 preweighted Collector. This is a one-token, two-rank expert
//! slice, not the model's router, attention, KV cache, or full generation.
#![cfg(target_os = "linux")]

use std::collections::BTreeMap;
use std::time::{Duration, Instant};
use voltafabric::moe::{Choice, Registry, Replica, Stamp};
use voltafabric::preweighted::Collector;
use voltafabric::process::{Fault, ReceiptState};
use voltafabric::routing::RoutingProblem;
use voltafabric::session::{mxfp4, Endpoint, Session};

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
            run_id: "r698-real-deepseek-session-v2".to_owned(),
            model_revision: REVISION.to_owned(),
            plan_id: "r698-one-token-two-rank-cpu-v2".to_owned(),
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

fn dispatch_one(
    worker: &mut Session,
    plan: &RoutingProblem,
    collector: &mut Collector,
    input_bits: Vec<u32>,
) {
    let pending = worker
        .start_preweighted_fp32(
            plan,
            collector,
            1,
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
    let mut collector = Collector::new(&plan).expect("complete two-expert route");
    let mut worker0 = Session::connect(&endpoint(EXPERT0_RANK), TIMEOUT).unwrap();
    let mut worker383 = Session::connect(&endpoint(EXPERT383_RANK), TIMEOUT).unwrap();
    worker0.load_mxfp4(&artifact0, 1, TIMEOUT).unwrap();
    worker383.load_mxfp4(&artifact383, 1, TIMEOUT).unwrap();

    dispatch_one(
        &mut worker0,
        &plan,
        &mut collector,
        golden0.input_bits.clone(),
    );
    dispatch_one(
        &mut worker383,
        &plan,
        &mut collector,
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
