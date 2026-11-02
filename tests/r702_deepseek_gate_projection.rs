use safetensors::tensor::{serialize, Dtype, TensorView};
use std::collections::BTreeMap;
use voltafabric::deepseek_v41_gate::{
    DeepSeekV41Gate, GateShardSpec, GateTensorHashes, TokenKind, GATE_BIAS_BYTES,
    GATE_PAYLOAD_BYTES, GATE_WEIGHT_BYTES, HIDDEN_SIZE, PINNED_MODEL_REVISION, ROUTED_EXPERTS,
    ROUTE_SCALE,
};
use voltafabric::{
    cost::{CostOrigin, CostProvenance},
    planner::{ExpertProfile, PerRankMemoryEnvelope, PlacementProblem, WirePolicy},
    topology::{Device, Origin, Topology},
    traffic::WireFormat,
};

fn fixture() -> (Vec<u8>, GateShardSpec) {
    fixture_with_weight_layout(Dtype::BF16, vec![ROUTED_EXPERTS, HIDDEN_SIZE])
}

fn fixture_with_weight_layout(
    weight_dtype: Dtype,
    weight_shape: Vec<usize>,
) -> (Vec<u8>, GateShardSpec) {
    let mut weight = vec![0u8; GATE_WEIGHT_BYTES as usize];
    for expert in 0..ROUTED_EXPERTS {
        for bit in 0..9 {
            if expert & (1 << bit) != 0 {
                let offset = (expert * HIDDEN_SIZE + bit) * 2;
                weight[offset..offset + 2].copy_from_slice(&0x3f80u16.to_le_bytes());
            }
        }
    }
    let mut text_bias = vec![0.0f32; ROUTED_EXPERTS];
    let mut image_bias = vec![0.0f32; ROUTED_EXPERTS];
    text_bias[0] = 1_000.0;
    image_bias[1] = 1_000.0;
    let text_bytes = encode_f32(&text_bias);
    let image_bytes = encode_f32(&image_bias);
    let unrelated = vec![0x5au8; 65_536];

    let tensors = vec![
        (
            "layers.0.ffn.gate.weight".to_owned(),
            TensorView::new(weight_dtype, weight_shape, &weight).unwrap(),
        ),
        (
            "layers.0.ffn.gate.bias".to_owned(),
            TensorView::new(Dtype::F32, vec![ROUTED_EXPERTS], &text_bytes).unwrap(),
        ),
        (
            "layers.0.ffn.gate.bias_vl".to_owned(),
            TensorView::new(Dtype::F32, vec![ROUTED_EXPERTS], &image_bytes).unwrap(),
        ),
        (
            "unrelated.large.tensor".to_owned(),
            TensorView::new(Dtype::U8, vec![unrelated.len()], &unrelated).unwrap(),
        ),
    ];
    let shard = serialize(tensors, None).unwrap();
    let spec = GateShardSpec {
        model_revision: PINNED_MODEL_REVISION.to_owned(),
        layer: 0,
        gate_temperature: 1.0,
        route_scale: ROUTE_SCALE,
        tensor_sha256: GateTensorHashes {
            weight: voltafabric::sha256::hex(&weight),
            text_bias: voltafabric::sha256::hex(&text_bytes),
            image_bias: voltafabric::sha256::hex(&image_bytes),
        },
        max_payload_bytes: GATE_PAYLOAD_BYTES,
    };
    (shard, spec)
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[test]
fn selected_gate_projection_routes_against_the_independent_decimal_oracle() {
    let (shard, spec) = fixture();
    let gate = DeepSeekV41Gate::load(&mut std::io::Cursor::new(&shard), &spec).unwrap();
    let receipt = gate.receipt();
    assert_eq!(receipt.model_revision, PINNED_MODEL_REVISION);
    assert_eq!(receipt.layer, 0);
    assert_eq!(receipt.file_bytes, shard.len() as u64);
    assert!(receipt.io_bytes < receipt.file_bytes);
    assert_eq!(receipt.selected_payload_bytes, GATE_PAYLOAD_BYTES);
    assert_eq!(
        receipt.io_bytes,
        receipt.header_bytes + 8 + GATE_PAYLOAD_BYTES
    );
    assert_eq!(receipt.tensors.len(), 3);
    assert!(receipt.tensors.iter().any(|tensor| {
        tensor.name == "layers.0.ffn.gate.weight"
            && tensor.bytes == GATE_WEIGHT_BYTES
            && tensor.sha256 == spec.tensor_sha256.weight
    }));
    assert!(receipt.tensors.iter().any(|tensor| {
        tensor.name == "layers.0.ffn.gate.bias"
            && tensor.bytes == GATE_BIAS_BYTES
            && tensor.sha256 == spec.tensor_sha256.text_bias
    }));

    // Binary-coded BF16 rows and a power-of-two input produce exact logits 0..383.
    let mut hidden = vec![0.0f32; HIDDEN_SIZE];
    for (bit, value) in hidden.iter_mut().take(9).enumerate() {
        *value = (1u32 << bit) as f32;
    }
    assert_eq!(
        gate.project(&hidden).unwrap(),
        (0..384).map(|x| x as f32).collect::<Vec<_>>()
    );

    let text = gate.route(&hidden, TokenKind::Text).unwrap();
    let expected = [
        (0, 0.012_687_707_376_015_25_f64),
        (383, 0.298_242_691_616_205_54_f64),
        (382, 0.297_853_086_366_122_95_f64),
        (381, 0.297_462_970_826_91_f64),
        (380, 0.297_072_342_988_230_93_f64),
        (379, 0.296_681_200_826_515_33_f64),
    ];
    assert_eq!(text.len(), expected.len());
    for (actual, (expert, coefficient)) in text.iter().zip(expected) {
        assert_eq!(actual.expert, expert);
        assert!((actual.weight - coefficient).abs() < 2.0e-7);
    }
    let image = gate.route(&hidden, TokenKind::Image).unwrap();
    assert_eq!(
        image.iter().map(|choice| choice.expert).collect::<Vec<_>>(),
        [1, 383, 382, 381, 380, 379]
    );
}

#[test]
fn gate_route_becomes_a_full_identity_bound_voltafabric_batch_without_rescaling() {
    let (shard, spec) = fixture();
    let gate = DeepSeekV41Gate::load(&mut std::io::Cursor::new(&shard), &spec).unwrap();
    let mut hidden = vec![0.0f32; HIDDEN_SIZE];
    for (bit, value) in hidden.iter_mut().take(9).enumerate() {
        *value = (1u32 << bit) as f32;
    }
    let experts = [0, 383, 382, 381, 380, 379];
    let registry = voltafabric::moe::Registry {
        epoch: 13,
        copies: experts
            .iter()
            .enumerate()
            .map(|(local, &expert)| voltafabric::moe::Replica {
                expert,
                rank: 4,
                local: local as u32,
            })
            .collect(),
    };
    let stamp = voltafabric::moe::Stamp {
        run_id: "r704-deepseek-gate-route".to_owned(),
        model_revision: PINNED_MODEL_REVISION.to_owned(),
        plan_id: "r704-one-token-cpu-route".to_owned(),
        layer: 0,
        batch: 7,
        epoch: 13,
    };
    let batch = gate
        .prepare_route_batch(
            &hidden,
            TokenKind::Text,
            stamp.clone(),
            29,
            registry,
            BTreeMap::new(),
            BTreeMap::from([(4, (0..6).collect())]),
        )
        .unwrap();
    assert_eq!(batch.attempt(), 29);
    assert_eq!(batch.problem().stamp.run_id, stamp.run_id);
    assert_eq!(batch.problem().stamp.model_revision, stamp.model_revision);
    assert_eq!(batch.problem().stamp.plan_id, stamp.plan_id);
    assert_eq!(batch.problem().stamp.layer, stamp.layer);
    assert_eq!(batch.problem().stamp.batch, stamp.batch);
    assert_eq!(batch.problem().stamp.epoch, stamp.epoch);
    assert_eq!(batch.problem().routes[0].len(), 6);
    assert_eq!(
        batch.problem().routes[0]
            .iter()
            .map(|choice| choice.expert)
            .collect::<Vec<_>>(),
        experts
    );
    let total = batch.problem().routes[0]
        .iter()
        .map(|choice| choice.weight)
        .sum::<f64>();
    assert!((total - f64::from(ROUTE_SCALE)).abs() < 2.0e-7);
    voltafabric::routing::validate(batch.problem()).unwrap();

    let mut wrong_layer = stamp.clone();
    wrong_layer.layer = 1;
    assert!(gate
        .prepare_route_batch(
            &hidden,
            TokenKind::Text,
            wrong_layer,
            29,
            batch.problem().registry.clone(),
            BTreeMap::new(),
            batch.problem().resident_order.clone(),
        )
        .is_err());
    assert!(gate
        .prepare_route_batch(
            &hidden,
            TokenKind::Text,
            stamp,
            0,
            batch.problem().registry.clone(),
            BTreeMap::new(),
            batch.problem().resident_order.clone(),
        )
        .is_err());
}

#[test]
fn projection_import_fails_closed_on_revision_hash_shape_and_budget() {
    let (shard, spec) = fixture();
    let mut bad_revision = spec.clone();
    bad_revision.model_revision = "unverified-revision".to_owned();
    assert!(DeepSeekV41Gate::load(&mut std::io::Cursor::new(&shard), &bad_revision).is_err());

    let mut bad_hash = spec.clone();
    bad_hash.tensor_sha256.text_bias = "0".repeat(64);
    assert!(DeepSeekV41Gate::load(&mut std::io::Cursor::new(&shard), &bad_hash).is_err());

    let mut missing_layer = spec.clone();
    missing_layer.layer = 1;
    assert!(DeepSeekV41Gate::load(&mut std::io::Cursor::new(&shard), &missing_layer).is_err());

    let (wrong_dtype_shard, wrong_dtype_spec) =
        fixture_with_weight_layout(Dtype::F32, vec![ROUTED_EXPERTS, HIDDEN_SIZE / 2]);
    assert!(DeepSeekV41Gate::load(
        &mut std::io::Cursor::new(&wrong_dtype_shard),
        &wrong_dtype_spec
    )
    .is_err());

    let mut short_budget = spec;
    short_budget.max_payload_bytes = GATE_PAYLOAD_BYTES - 1;
    assert!(DeepSeekV41Gate::load(&mut std::io::Cursor::new(&shard), &short_budget).is_err());
}

#[test]
fn deepseek_gate_batch_flows_through_memory_advice_without_changing_route_identity() {
    let (shard, spec) = fixture();
    let gate = DeepSeekV41Gate::load(&mut std::io::Cursor::new(&shard), &spec).unwrap();
    let mut hidden = vec![0.0f32; HIDDEN_SIZE];
    for (bit, value) in hidden.iter_mut().take(9).enumerate() {
        *value = (1u32 << bit) as f32;
    }
    let selected = [0, 383, 382, 381, 380, 379];
    let stamp = voltafabric::moe::Stamp {
        run_id: "r705-memory-advisory".into(),
        model_revision: PINNED_MODEL_REVISION.into(),
        plan_id: "r705-synthetic-plan".into(),
        layer: 0,
        batch: 8,
        epoch: 13,
    };
    let registry = voltafabric::moe::Registry {
        epoch: stamp.epoch,
        copies: selected
            .iter()
            .enumerate()
            .map(|(local, &expert)| voltafabric::moe::Replica {
                expert,
                rank: 4,
                local: local as u32,
            })
            .collect(),
    };
    let batch = gate
        .prepare_route_batch(
            &hidden,
            TokenKind::Text,
            stamp.clone(),
            31,
            registry,
            BTreeMap::new(),
            BTreeMap::from([(4, (0..6).collect())]),
        )
        .unwrap();
    let route_before = batch.problem().routes[0]
        .iter()
        .map(|choice| (choice.expert, choice.weight.to_bits()))
        .collect::<Vec<_>>();
    let registry_before = batch.problem().registry.clone();
    let weights = route_before
        .iter()
        .map(|(_, weight_bits)| format!("{weight_bits:016x}"))
        .collect::<Vec<_>>();
    let base = PlacementProblem {
        topology: Topology {
            schema_version: 1,
            origin: Origin::Synthetic,
            source_note: "R705 deterministic CPU fixture; not a hardware measurement".into(),
            devices: vec![Device {
                rank: 4,
                host: "cpu-fixture".into(),
                capacity_bytes: 1_000_000,
                reserved_bytes: 8_192,
            }],
            links: vec![],
        },
        experts: selected
            .iter()
            .map(|&id| ExpertProfile {
                id,
                resident_bytes: 2_048,
                compute_us: 1.0,
                cost_provenance: CostProvenance {
                    origin: CostOrigin::Synthetic,
                    source: Some("R705 deterministic fixture; not a hardware measurement".into()),
                },
            })
            .collect(),
        demands: vec![],
        hidden: HIDDEN_SIZE as u64,
        wire: WireFormat::default(),
        max_iterations: 0,
        wire_policy: WirePolicy::Adaptive,
        max_work_units: 1_000_000,
    };
    let envelope = PerRankMemoryEnvelope {
        rank: 4,
        resident_expert_bytes: 512,
        kv_bytes: 1_024,
        recurrent_bytes: 256,
        workspace_bytes: 512,
        graph_bytes: 1_024,
        communication_bytes: 128,
        safety_margin_bytes: 1_024,
        load_temporary_bytes: 512,
    };

    let advice = batch
        .plan_memory_advisory(&base, 4, std::slice::from_ref(&envelope))
        .unwrap();
    assert_eq!(advice.routed.stamp, stamp);
    assert_eq!(advice.routed.attempt, 31);
    assert_eq!(advice.routed.source_rank, 4);
    assert_eq!(
        advice.routed.route_experts_by_token,
        vec![selected.to_vec()]
    );
    assert_eq!(advice.routed.route_weight_bits_hex_by_token, vec![weights]);
    assert_eq!(advice.ranks.len(), 1);
    assert!(advice.ranks[0].loading_peak_bytes <= advice.ranks[0].capacity_bytes);
    assert_eq!(
        batch.problem().routes[0]
            .iter()
            .map(|choice| (choice.expert, choice.weight.to_bits()))
            .collect::<Vec<_>>(),
        route_before
    );
    assert_eq!(batch.problem().registry.epoch, registry_before.epoch);
    assert_eq!(
        batch.problem().registry.copies.len(),
        registry_before.copies.len()
    );

    let wrong_rank = PerRankMemoryEnvelope {
        rank: 5,
        ..envelope.clone()
    };
    assert!(batch.plan_memory_advisory(&base, 4, &[wrong_rank]).is_err());
    let over_budget = PerRankMemoryEnvelope {
        kv_bytes: u64::MAX,
        ..envelope
    };
    assert!(batch
        .plan_memory_advisory(&base, 4, &[over_budget])
        .is_err());
    assert_eq!(
        batch.problem().routes[0]
            .iter()
            .map(|choice| (choice.expert, choice.weight.to_bits()))
            .collect::<Vec<_>>(),
        route_before
    );
    assert_eq!(batch.problem().registry.epoch, registry_before.epoch);
}

#[test]
fn deepseek_gate_batch_enters_mxfp4_session_with_exact_attempt_and_route() {
    let (shard, spec) = fixture();
    let gate = DeepSeekV41Gate::load(&mut std::io::Cursor::new(&shard), &spec).unwrap();
    let mut hidden = vec![0.0f32; HIDDEN_SIZE];
    for (bit, value) in hidden.iter_mut().take(9).enumerate() {
        *value = (1u32 << bit) as f32;
    }
    let experts = [0, 383, 382, 381, 380, 379];
    let stamp = voltafabric::moe::Stamp {
        run_id: "r706-deepseek-session-adapter".to_owned(),
        model_revision: PINNED_MODEL_REVISION.to_owned(),
        plan_id: "r706-one-token-cpu".to_owned(),
        layer: 0,
        batch: 19,
        epoch: 23,
    };
    let route = gate
        .prepare_route_batch(
            &hidden,
            TokenKind::Text,
            stamp.clone(),
            41,
            voltafabric::moe::Registry {
                epoch: stamp.epoch,
                copies: experts
                    .iter()
                    .enumerate()
                    .map(|(local, &expert)| voltafabric::moe::Replica {
                        expert,
                        rank: 4,
                        local: local as u32,
                    })
                    .collect(),
            },
            BTreeMap::new(),
            BTreeMap::from([(4, (0..6).collect())]),
        )
        .unwrap();
    let input = voltafabric::session::mxfp4::TokenInput {
        token: 0,
        value_bits: hidden.iter().map(|value| value.to_bits()).collect(),
    };
    let session_batch = route.into_mxfp4_session_batch(vec![input.clone()]).unwrap();
    assert_eq!(session_batch.attempt, 41);
    assert_eq!(session_batch.problem.stamp, stamp);
    assert_eq!(session_batch.inputs[0].value_bits, input.value_bits);
    assert_eq!(
        session_batch.problem.routes[0]
            .iter()
            .map(|choice| (choice.expert, choice.weight.to_bits()))
            .collect::<Vec<_>>(),
        route.problem().routes[0]
            .iter()
            .map(|choice| (choice.expert, choice.weight.to_bits()))
            .collect::<Vec<_>>()
    );
    let mut inexact = input;
    inexact.value_bits[0] = 0x3f80_0001;
    assert!(route.into_mxfp4_session_batch(vec![inexact]).is_err());
    assert!(route.into_mxfp4_session_batch(Vec::new()).is_err());
}
