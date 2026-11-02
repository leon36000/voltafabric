#![cfg(target_os = "linux")]

use serde_json::Value;
use std::{
    fs::{self, File},
    io::{Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};
use voltafabric::{
    deepseek_v41_gate::{
        hc::{
            DeepSeekV41HcProjection, HcProjectionBranch, HcProjectionSpec,
            HcProjectionTensorHashes, HC_INPUT_WIDTH, HC_MIX_COUNT, HC_PROJECTION_BASE_BYTES,
            HC_PROJECTION_PAYLOAD_BYTES, HC_PROJECTION_SCALE_BYTES, HC_PROJECTION_WEIGHT_BYTES,
            PINNED_HC_NORMALIZATION_EPSILON, PINNED_HC_SINKHORN_EPSILON,
            PINNED_HC_SINKHORN_ITERATIONS,
        },
        DeepSeekV41ResidualStream, PinnedGateIndexSource, HIDDEN_SIZE, PINNED_MODEL_REVISION,
    },
    moe::Stamp,
};

const HEADER_SHA256: &str = "71213f444f9c15a4af6dbcbddc36f7611542884ebb3aadef54de650071357449";
const PINNED_INDEX_SHA256: &str =
    "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8";
const SHARD_BYTES: u64 = 7_389_759_032;
static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

struct TempDirectory(PathBuf);

impl TempDirectory {
    fn new() -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(".work");
        fs::create_dir_all(&root).unwrap();
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(
            "r731-hc-projection-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn write_tensor_payload(
    shard: &mut File,
    header: &Value,
    data_start: u64,
    name: &str,
    payload: &[u8],
) {
    let offsets = header[name]["data_offsets"].as_array().unwrap();
    let start = data_start + offsets[0].as_u64().unwrap();
    let end = data_start + offsets[1].as_u64().unwrap();
    assert_eq!(end - start, payload.len() as u64, "wrong bytes for {name}");
    shard.seek(SeekFrom::Start(start)).unwrap();
    shard.write_all(payload).unwrap();
}

struct Fixture {
    checkpoint: TempDirectory,
    index: &'static [u8],
    spec: HcProjectionSpec,
    weights: Vec<f32>,
    base: Vec<f32>,
    scale: Vec<f32>,
    ffn_spec: HcProjectionSpec,
    ffn_weights: Vec<f32>,
    ffn_base: Vec<f32>,
    ffn_scale: Vec<f32>,
    residual: DeepSeekV41ResidualStream,
}

fn exact_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let tie_to_even_bias = 0x7fffu32 + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(tie_to_even_bias) & 0xffff_0000)
}

fn fixture() -> Fixture {
    let index: &'static [u8] = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
    );
    assert_eq!(voltafabric::sha256::hex(index), PINNED_INDEX_SHA256);
    let header: &'static [u8] = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model-00003.header.314113f8.json"
    );
    assert_eq!(voltafabric::sha256::hex(header), HEADER_SHA256);
    let header_json: Value = serde_json::from_slice(header).unwrap();
    let data_start = header.len() as u64 + 8;
    let checkpoint = TempDirectory::new();
    let shard_path = checkpoint.path().join("model-00003-of-00048.safetensors");
    let mut shard = File::create(&shard_path).unwrap();
    shard.set_len(SHARD_BYTES).unwrap();
    shard
        .write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    shard.write_all(header).unwrap();

    let weights = (0..HC_MIX_COUNT * HC_INPUT_WIDTH)
        .map(|index| {
            let row = index / HC_INPUT_WIDTH;
            let column = index % HC_INPUT_WIDTH;
            let code = (row * 37 + column * 13 + (row * column) % 11) % 19;
            (code as f32 - 9.0) / 8192.0
        })
        .collect::<Vec<_>>();
    let base = (0..HC_MIX_COUNT)
        .map(|index| (index as f32 - 11.5) / 16.0)
        .collect::<Vec<_>>();
    let scale = vec![0.75, -0.5, 0.25];
    let ffn_weights = weights
        .iter()
        .map(|weight| -*weight * 0.75)
        .collect::<Vec<_>>();
    let ffn_base = base.iter().map(|value| value + 0.125).collect::<Vec<_>>();
    let ffn_scale = vec![-0.25, 0.375, -0.625];
    let weight_payload = bytes(&weights);
    let base_payload = bytes(&base);
    let scale_payload = bytes(&scale);
    let ffn_weight_payload = bytes(&ffn_weights);
    let ffn_base_payload = bytes(&ffn_base);
    let ffn_scale_payload = bytes(&ffn_scale);
    assert_eq!(weight_payload.len() as u64, HC_PROJECTION_WEIGHT_BYTES);
    assert_eq!(base_payload.len() as u64, HC_PROJECTION_BASE_BYTES);
    assert_eq!(scale_payload.len() as u64, HC_PROJECTION_SCALE_BYTES);

    write_tensor_payload(
        &mut shard,
        &header_json,
        data_start,
        "layers.0.hc_attn_fn",
        &weight_payload,
    );
    write_tensor_payload(
        &mut shard,
        &header_json,
        data_start,
        "layers.0.hc_attn_base",
        &base_payload,
    );
    write_tensor_payload(
        &mut shard,
        &header_json,
        data_start,
        "layers.0.hc_attn_scale",
        &scale_payload,
    );
    write_tensor_payload(
        &mut shard,
        &header_json,
        data_start,
        "layers.0.hc_ffn_fn",
        &ffn_weight_payload,
    );
    write_tensor_payload(
        &mut shard,
        &header_json,
        data_start,
        "layers.0.hc_ffn_base",
        &ffn_base_payload,
    );
    write_tensor_payload(
        &mut shard,
        &header_json,
        data_start,
        "layers.0.hc_ffn_scale",
        &ffn_scale_payload,
    );
    drop(shard);

    let stamp = Stamp {
        run_id: "r731-hc-projection".to_owned(),
        model_revision: PINNED_MODEL_REVISION.to_owned(),
        plan_id: "r731-synthetic-hc-plan".to_owned(),
        layer: 0,
        batch: 8,
        epoch: 19,
    };
    let residual_copies = (0..4)
        .map(|copy| {
            (0..HIDDEN_SIZE)
                .map(|hidden| {
                    let code = (copy * 17 + hidden * 7 + hidden % 13) % 23;
                    exact_bf16((code as f32 - 11.0) / 64.0)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let residual = DeepSeekV41ResidualStream::new(stamp, 23, residual_copies).unwrap();
    let spec = HcProjectionSpec {
        model_revision: PINNED_MODEL_REVISION.to_owned(),
        layer: 0,
        branch: HcProjectionBranch::Attention,
        normalization_epsilon: PINNED_HC_NORMALIZATION_EPSILON,
        sinkhorn_epsilon: PINNED_HC_SINKHORN_EPSILON,
        sinkhorn_iterations: PINNED_HC_SINKHORN_ITERATIONS,
        tensor_sha256: HcProjectionTensorHashes {
            weight: voltafabric::sha256::hex(&weight_payload),
            base: voltafabric::sha256::hex(&base_payload),
            scale: voltafabric::sha256::hex(&scale_payload),
        },
        max_payload_bytes: HC_PROJECTION_PAYLOAD_BYTES,
    };
    let ffn_spec = HcProjectionSpec {
        branch: HcProjectionBranch::Ffn,
        tensor_sha256: HcProjectionTensorHashes {
            weight: voltafabric::sha256::hex(&ffn_weight_payload),
            base: voltafabric::sha256::hex(&ffn_base_payload),
            scale: voltafabric::sha256::hex(&ffn_scale_payload),
        },
        ..spec.clone()
    };

    Fixture {
        checkpoint,
        index,
        spec,
        weights,
        base,
        scale,
        ffn_spec,
        ffn_weights,
        ffn_base,
        ffn_scale,
        residual,
    }
}

fn oracle_projection(weights: &[f32], inputs: &[f32]) -> Vec<f32> {
    let mean_square = inputs
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>()
        / HC_INPUT_WIDTH as f64;
    let inverse_rms = (mean_square + f64::from(PINNED_HC_NORMALIZATION_EPSILON))
        .sqrt()
        .recip();
    (0..HC_MIX_COUNT)
        .map(|row| {
            let start = row * HC_INPUT_WIDTH;
            let dot = weights[start..start + HC_INPUT_WIDTH]
                .iter()
                .zip(inputs)
                .map(|(&weight, &value)| f64::from(weight) * f64::from(value))
                .sum::<f64>();
            (dot * inverse_rms) as f32
        })
        .collect()
}

fn oracle_split_sinkhorn(
    mixes: &[f32],
    base: &[f32],
    scale: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f64>) {
    let eps = f64::from(PINNED_HC_SINKHORN_EPSILON);
    let sigmoid = |value: f64| 1.0 / (1.0 + (-value).exp());
    let mut pre = Vec::with_capacity(4);
    let mut post = Vec::with_capacity(4);
    for copy in 0..4 {
        pre.push(
            (sigmoid(f64::from(mixes[copy]) * f64::from(scale[0]) + f64::from(base[copy])) + eps)
                as f32,
        );
        post.push(
            (2.0 * sigmoid(
                f64::from(mixes[copy + 4]) * f64::from(scale[1]) + f64::from(base[copy + 4]),
            )) as f32,
        );
    }

    let mut comb = vec![0.0f64; 16];
    for row in 0..4 {
        let start = row * 4;
        let mut maximum = f64::NEG_INFINITY;
        for column in 0..4 {
            let index = 8 + start + column;
            comb[start + column] =
                f64::from(mixes[index]) * f64::from(scale[2]) + f64::from(base[index]);
            maximum = maximum.max(comb[start + column]);
        }
        let mut denominator = 0.0f64;
        for column in 0..4 {
            let index = start + column;
            comb[index] = (comb[index] - maximum).exp();
            denominator += comb[index];
        }
        for column in 0..4 {
            let index = start + column;
            comb[index] = comb[index] / denominator + eps;
        }
    }
    normalize_columns_f64(&mut comb, eps);
    for _ in 1..PINNED_HC_SINKHORN_ITERATIONS {
        for row in 0..4 {
            let start = row * 4;
            let denominator = comb[start..start + 4].iter().sum::<f64>() + eps;
            for value in &mut comb[start..start + 4] {
                *value /= denominator;
            }
        }
        normalize_columns_f64(&mut comb, eps);
    }
    (pre, post, comb)
}

fn normalize_columns_f64(matrix: &mut [f64], epsilon: f64) {
    for column in 0..4 {
        let denominator = (0..4).map(|row| matrix[row * 4 + column]).sum::<f64>() + epsilon;
        for row in 0..4 {
            matrix[row * 4 + column] /= denominator;
        }
    }
}

#[test]
fn indexed_hc_projection_split_sinkhorn_and_pre_match_independent_oracle() {
    let fixture = fixture();
    let source = PinnedGateIndexSource {
        index_bytes: fixture.index,
        expected_header_sha256: HEADER_SHA256,
        max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
    };
    let projection =
        DeepSeekV41HcProjection::load_indexed(fixture.checkpoint.path(), source, &fixture.spec)
            .unwrap();
    let receipt = projection.receipt();
    assert_eq!(receipt.model_revision, PINNED_MODEL_REVISION);
    assert_eq!(receipt.layer, 0);
    assert_eq!(receipt.branch, HcProjectionBranch::Attention);
    assert_eq!(
        receipt.source_shard_name,
        "model-00003-of-00048.safetensors"
    );
    assert_eq!(receipt.index_sha256, PINNED_INDEX_SHA256);
    assert_eq!(receipt.header_sha256, HEADER_SHA256);
    assert_eq!(receipt.shard_bytes, SHARD_BYTES);
    assert_eq!(receipt.tensor_bytes, HC_PROJECTION_PAYLOAD_BYTES);
    assert!(receipt.io_bytes < receipt.shard_bytes);

    let actual = projection.project(&fixture.residual).unwrap();
    assert_eq!(actual.stamp(), fixture.residual.stamp());
    assert_eq!(actual.attempt(), fixture.residual.attempt());
    assert_eq!(actual.branch(), HcProjectionBranch::Attention);

    let inputs = fixture
        .residual
        .copies_bf16_f32()
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let projected = oracle_projection(&fixture.weights, &inputs);
    let (expected_pre, expected_post, expected_comb) =
        oracle_split_sinkhorn(&projected, &fixture.base, &fixture.scale);

    let max_pre_error = actual
        .pre_mix()
        .iter()
        .zip(&expected_pre)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_pre_error <= 2.0e-6,
        "pre coefficient error {max_pre_error}"
    );
    let max_post_error = actual
        .post()
        .iter()
        .zip(&expected_post)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_post_error <= 2.0e-6,
        "post coefficient error {max_post_error}"
    );
    let max_comb_error = actual
        .comb()
        .iter()
        .flatten()
        .zip(expected_comb.iter())
        .map(|(&actual, &expected)| (f64::from(actual) - expected).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_comb_error <= 2.0e-5,
        "comb coefficient error {max_comb_error}"
    );

    let actual_pre = actual.apply_pre(&fixture.residual).unwrap();
    let flattened = fixture
        .residual
        .copies_bf16_f32()
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    let delegated_pre = voltafabric::deepseek_v41::deepseek_hc_pre_reference(
        &flattened,
        actual.pre_mix(),
        HIDDEN_SIZE,
    )
    .unwrap();
    assert_eq!(actual_pre, delegated_pre);
    let expected_pre_output = (0..HIDDEN_SIZE)
        .map(|hidden| {
            let sum = (0..4)
                .map(|copy| {
                    f64::from(expected_pre[copy])
                        * f64::from(fixture.residual.copies_bf16_f32()[copy][hidden])
                })
                .sum::<f64>();
            exact_bf16(sum as f32)
        })
        .collect::<Vec<_>>();
    let max_pre_output_error = actual_pre
        .iter()
        .zip(&expected_pre_output)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_pre_output_error <= 0.004,
        "HC-pre output error {max_pre_output_error}"
    );
    assert!(actual_pre
        .iter()
        .all(
            |value| voltafabric::deepseek_v41::cast_bf16_rne_reference(*value)
                .unwrap()
                .to_bits()
                == value.to_bits()
        ));

    let wrong_batch = Stamp {
        batch: fixture.residual.stamp().batch + 1,
        ..fixture.residual.stamp().clone()
    };
    let wrong_residual = DeepSeekV41ResidualStream::new(
        wrong_batch,
        fixture.residual.attempt(),
        fixture.residual.copies_bf16_f32().to_vec(),
    )
    .unwrap();
    assert!(actual.apply_pre(&wrong_residual).is_err());

    let mut too_small = fixture.spec.clone();
    too_small.max_payload_bytes -= 1;
    assert!(
        DeepSeekV41HcProjection::load_indexed(fixture.checkpoint.path(), source, &too_small)
            .is_err()
    );

    let ffn_projection =
        DeepSeekV41HcProjection::load_indexed(fixture.checkpoint.path(), source, &fixture.ffn_spec)
            .unwrap();
    assert_eq!(ffn_projection.receipt().branch, HcProjectionBranch::Ffn);
    assert_eq!(
        ffn_projection.receipt().tensor_bytes,
        HC_PROJECTION_PAYLOAD_BYTES
    );
    let ffn_actual = ffn_projection.project(&fixture.residual).unwrap();
    assert_eq!(ffn_actual.branch(), HcProjectionBranch::Ffn);
    let ffn_projected = oracle_projection(&fixture.ffn_weights, &inputs);
    let (ffn_pre, ffn_post, ffn_comb) =
        oracle_split_sinkhorn(&ffn_projected, &fixture.ffn_base, &fixture.ffn_scale);
    let max_ffn_pre_error = ffn_actual
        .pre_mix()
        .iter()
        .zip(&ffn_pre)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(max_ffn_pre_error <= 2.0e-6);
    let max_ffn_post_error = ffn_actual
        .post()
        .iter()
        .zip(&ffn_post)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(max_ffn_post_error <= 2.0e-6);
    let max_ffn_comb_error = ffn_actual
        .comb()
        .iter()
        .flatten()
        .zip(ffn_comb.iter())
        .map(|(&actual, &expected)| (f64::from(actual) - expected).abs())
        .fold(0.0f64, f64::max);
    assert!(max_ffn_comb_error <= 2.0e-5);

    let other_layer_stamp = Stamp {
        layer: 1,
        ..fixture.residual.stamp().clone()
    };
    let other_layer_residual = DeepSeekV41ResidualStream::new(
        other_layer_stamp,
        fixture.residual.attempt(),
        fixture.residual.copies_bf16_f32().to_vec(),
    )
    .unwrap();
    assert!(projection.project(&other_layer_residual).is_err());
}
