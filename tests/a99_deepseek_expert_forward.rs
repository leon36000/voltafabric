//! Bounded one-token CPU reference over the pinned compressed DeepSeek expert0.
use serde_json::Value;
use voltafabric::deepseek_v41::{
    cast_bf16_rne_reference, single_token_expert_forward_reference, Mxfp4Matrix,
};

fn matrix_from_fixture(
    tensors: &safetensors::SafeTensors<'_>,
    projection: &str,
    rows: u64,
    cols: u64,
) -> Mxfp4Matrix {
    let prefix = format!("layers.0.ffn.experts.0.{projection}");
    let weight = tensors.tensor(&format!("{prefix}.weight")).unwrap();
    let scale = tensors.tensor(&format!("{prefix}.scale")).unwrap();
    assert_eq!(weight.data().len() as u64, rows * cols / 2);
    assert_eq!(scale.data().len() as u64, rows * cols / 32);
    Mxfp4Matrix::new(rows, cols, weight.data().to_vec(), scale.data().to_vec()).unwrap()
}

fn input_vector() -> Vec<f32> {
    let pattern = [
        0.125f32, -0.375, 0.5, -0.25, 0.0625, -0.5, 0.0, 0.3125, 0.75, -0.125, 0.25, -0.625,
        -0.1875, 0.4375, -0.0625, 0.5625,
    ];
    (0..5120).map(|i| pattern[i % pattern.len()]).collect()
}

#[test]
fn bf16_cast_uses_ties_to_even_and_preserves_signed_underflow() {
    assert_eq!(
        cast_bf16_rne_reference(f32::from_bits(0x3f80_8000))
            .unwrap()
            .to_bits(),
        1.0f32.to_bits()
    );
    assert_eq!(
        cast_bf16_rne_reference(f32::from_bits(0x3f81_8000))
            .unwrap()
            .to_bits(),
        1.015625f32.to_bits()
    );
    assert_eq!(
        cast_bf16_rne_reference(f32::from_bits(0xbf80_8000))
            .unwrap()
            .to_bits(),
        (-1.0f32).to_bits()
    );
    assert_eq!(
        cast_bf16_rne_reference(-f32::from_bits(1))
            .unwrap()
            .to_bits(),
        (-0.0f32).to_bits()
    );
    assert_eq!(
        cast_bf16_rne_reference(f32::INFINITY).unwrap_err().code,
        "NONFINITE_INPUT"
    );
}

#[test]
fn pinned_compressed_expert0_matches_independent_single_token_torch_oracle() {
    let artifact = include_bytes!("../fixtures/deepseek_v41_layer0_expert0_314113f8.safetensors");
    let golden: Value = serde_json::from_slice(include_bytes!(
        "../fixtures/deepseek_v41_expert0_forward_bf16_314113f8.json"
    ))
    .unwrap();
    assert_eq!(
        golden["artifact_sha256"],
        "f1bdb87fe86fcafa8ae72e4154a8b4a0e66b90eaba8366a398676cb2fbe4510b"
    );
    assert_eq!(
        golden["model_revision"],
        "314113f8a7dadb388215396ba5fbee2c610a04d6"
    );
    assert_eq!(golden["input_dtype"], "BF16");
    let tensors = safetensors::SafeTensors::deserialize(artifact).unwrap();
    let w1 = matrix_from_fixture(&tensors, "w1", 2304, 5120);
    let w3 = matrix_from_fixture(&tensors, "w3", 2304, 5120);
    let w2 = matrix_from_fixture(&tensors, "w2", 5120, 2304);
    assert_eq!(w1.summary().compressed_bytes, 6_266_880);
    assert_eq!(w3.summary().compressed_bytes, 6_266_880);
    assert_eq!(w2.summary().compressed_bytes, 6_266_880);

    let output =
        single_token_expert_forward_reference(&w1, &w3, &w2, &input_vector(), Some(0.375)).unwrap();
    assert_eq!(output.len(), 5120);
    let expected = golden["output_f32"].as_array().unwrap();
    assert_eq!(expected.len(), output.len());
    let mut max_abs_error = 0.0f32;
    for (index, (actual, expected)) in output.iter().zip(expected).enumerate() {
        let expected = expected.as_f64().unwrap() as f32;
        assert!(actual.is_finite(), "nonfinite output at {index}");
        max_abs_error = max_abs_error.max((actual - expected).abs());
    }
    assert!(
        max_abs_error <= 2.0e-5,
        "CPU FP64 reference vs pinned PyTorch/NumPy oracle max abs error {max_abs_error}"
    );
}

#[test]
fn rejects_non_bf16_input_and_invalid_routing_coefficient_before_projection() {
    let zeros32 = || Mxfp4Matrix::new(32, 32, vec![0; 512], vec![127; 32]).unwrap();
    let w1 = zeros32();
    let w3 = zeros32();
    let w2 = zeros32();
    let mut input = vec![0.0; 32];
    input[0] = f32::from_bits(0x3f80_8000); // midpoint; not a BF16 value
    let error = single_token_expert_forward_reference(&w1, &w3, &w2, &input, None).unwrap_err();
    assert_eq!(error.code, "INVALID_INPUT");

    input[0] = 0.5;
    let error =
        single_token_expert_forward_reference(&w1, &w3, &w2, &input, Some(f32::NAN)).unwrap_err();
    assert_eq!(error.code, "INVALID_INPUT");
}
