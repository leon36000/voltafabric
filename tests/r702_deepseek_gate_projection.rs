use safetensors::tensor::{serialize, Dtype, TensorView};
use voltafabric::deepseek_v41_gate::{
    DeepSeekV41Gate, GateShardSpec, GateTensorHashes, TokenKind, GATE_BIAS_BYTES,
    GATE_PAYLOAD_BYTES, GATE_WEIGHT_BYTES, HIDDEN_SIZE, PINNED_MODEL_REVISION, ROUTED_EXPERTS,
    ROUTE_SCALE,
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
