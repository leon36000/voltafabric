//! A100: retain one immutable Safetensors expert artifact and view its compressed planes.
use safetensors::tensor::{serialize, Dtype, TensorView};
use serde_json::Value;
use std::path::Path;
use std::time::Instant;
use voltafabric::deepseek_v41::Mxfp4Expert;

const FIXTURE_SHA256: &str = "f1bdb87fe86fcafa8ae72e4154a8b4a0e66b90eaba8366a398676cb2fbe4510b";

fn input_vector() -> Vec<f32> {
    let pattern = [
        0.125f32, -0.375, 0.5, -0.25, 0.0625, -0.5, 0.0, 0.3125, 0.75, -0.125, 0.25, -0.625,
        -0.1875, 0.4375, -0.0625, 0.5625,
    ];
    (0..5120).map(|i| pattern[i % pattern.len()]).collect()
}

fn small_expert_artifact(
    wrong_weight_dtype: bool,
    wrong_scale_shape: bool,
    invalid_scale_code: bool,
) -> Vec<u8> {
    let mut tensors = Vec::new();
    for projection in ["w1", "w2", "w3"] {
        let dtype = if wrong_weight_dtype && projection == "w1" {
            Dtype::F32
        } else {
            Dtype::I8
        };
        let weight_bytes = if dtype == Dtype::F32 { 2048 } else { 512 };
        tensors.push((
            format!("layers.0.ffn.experts.0.{projection}.weight"),
            dtype,
            vec![32, 16],
            vec![0u8; weight_bytes],
        ));
        let bad_shape = wrong_scale_shape && projection == "w1";
        let scale_shape = if bad_shape { vec![32, 2] } else { vec![32, 1] };
        let scale_len = if bad_shape { 64 } else { 32 };
        let scale_byte = if invalid_scale_code && projection == "w1" {
            255
        } else {
            127
        };
        tensors.push((
            format!("layers.0.ffn.experts.0.{projection}.scale"),
            Dtype::F8_E8M0,
            scale_shape,
            vec![scale_byte; scale_len],
        ));
    }
    let views = tensors
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            (
                name.clone(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize(views, None).unwrap()
}

#[test]
fn shared_compressed_artifact_forward_matches_the_pinned_a99_oracle() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let bytes =
        std::fs::read(root.join("fixtures/deepseek_v41_layer0_expert0_314113f8.safetensors"))
            .unwrap();
    let artifact_bytes = bytes.len();
    let load_start = Instant::now();
    let expert =
        Mxfp4Expert::from_safetensors_expert_artifact(bytes, 0, 0, FIXTURE_SHA256).unwrap();
    let load_elapsed = load_start.elapsed();
    let golden: Value = serde_json::from_slice(
        &std::fs::read(root.join("fixtures/deepseek_v41_expert0_forward_bf16_314113f8.json"))
            .unwrap(),
    )
    .unwrap();

    assert_eq!(expert.artifact_sha256(), FIXTURE_SHA256);
    assert_eq!(expert.backing_bytes(), artifact_bytes);
    assert_eq!(expert.compressed_bytes(), 18_800_640);
    assert_eq!(expert.layer(), 0);
    assert_eq!(expert.expert(), 0);

    let forward_start = Instant::now();
    let actual = expert
        .forward_reference(&input_vector(), Some(0.375))
        .unwrap();
    let forward_elapsed = forward_start.elapsed();
    let expected = golden["output_f32"].as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected.as_f64().unwrap() as f32).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs <= 2.0e-5,
        "shared artifact path differs from A99 oracle: {max_abs}"
    );
    println!(
        "artifact_bytes={artifact_bytes} compressed_bytes={} admission_ms={:.3} forward_ms={:.3} max_abs_error={max_abs}",
        expert.compressed_bytes(),
        load_elapsed.as_secs_f64() * 1_000.0,
        forward_elapsed.as_secs_f64() * 1_000.0,
    );
}

#[test]
fn shared_artifact_loader_rejects_missing_or_wrong_tensors() {
    let bad = b"not safetensors".to_vec();
    let bad_sha = voltafabric::sha256::hex(&bad);
    assert_eq!(
        Mxfp4Expert::from_safetensors_expert_artifact(bad, 0, 0, &bad_sha)
            .unwrap_err()
            .code,
        "INVALID_INPUT"
    );

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let bytes =
        std::fs::read(root.join("fixtures/deepseek_v41_layer0_expert0_314113f8.safetensors"))
            .unwrap();
    assert_eq!(
        Mxfp4Expert::from_safetensors_expert_artifact(bytes.clone(), 0, 7, FIXTURE_SHA256)
            .unwrap_err()
            .code,
        "INVALID_INPUT"
    );
    assert_eq!(
        Mxfp4Expert::from_safetensors_expert_artifact(bytes, u32::MAX, 0, FIXTURE_SHA256)
            .unwrap_err()
            .code,
        "INVALID_INPUT"
    );
}

#[test]
fn shared_artifact_loader_validates_dtype_scale_shape_and_e8m0_domain() {
    let wrong_dtype = small_expert_artifact(true, false, false);
    let wrong_dtype_sha = voltafabric::sha256::hex(&wrong_dtype);
    assert_eq!(
        Mxfp4Expert::from_safetensors_expert_artifact(wrong_dtype, 0, 0, &wrong_dtype_sha)
            .unwrap_err()
            .message,
        "MXFP4 weight tensor dtype must be I8"
    );

    let wrong_shape = small_expert_artifact(false, true, false);
    let wrong_shape_sha = voltafabric::sha256::hex(&wrong_shape);
    assert_eq!(
        Mxfp4Expert::from_safetensors_expert_artifact(wrong_shape, 0, 0, &wrong_shape_sha)
            .unwrap_err()
            .message,
        "MXFP4 scale tensor shape must be [rows, cols/32]"
    );

    let invalid_scale = small_expert_artifact(false, false, true);
    let invalid_scale_sha = voltafabric::sha256::hex(&invalid_scale);
    assert_eq!(
        Mxfp4Expert::from_safetensors_expert_artifact(invalid_scale, 0, 0, &invalid_scale_sha)
            .unwrap_err()
            .code,
        "MXFP4_SCALE_NAN"
    );
}

#[test]
fn pinned_artifact_owner_rejects_a_content_digest_mismatch() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let bytes =
        std::fs::read(root.join("fixtures/deepseek_v41_layer0_expert0_314113f8.safetensors"))
            .unwrap();
    let error =
        Mxfp4Expert::from_safetensors_expert_artifact(bytes, 0, 0, &"0".repeat(64)).unwrap_err();
    assert_eq!(error.code, "ARTIFACT_SHA256_MISMATCH");
}
