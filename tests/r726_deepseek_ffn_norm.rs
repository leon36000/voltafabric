#![cfg(target_os = "linux")]

use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};
use voltafabric::{
    deepseek_v41_gate::{
        DeepSeekV41FfnRmsNorm, DeepSeekV41IndexedFfnTokenRequest, DeepSeekV41IndexedMoeLayer,
        FfnNormTensorSpec, GateShardSpec, GateTensorHashes, PinnedGateIndexSource,
        PinnedMxfp4PayloadSource, TokenKind, GATE_PAYLOAD_BYTES, GATE_WEIGHT_BYTES, HIDDEN_SIZE,
        PINNED_MODEL_REVISION, PINNED_RMS_NORM_EPSILON, RMS_NORM_WEIGHT_BYTES, ROUTED_EXPERTS,
        ROUTE_SCALE,
    },
    moe::{Registry, Replica, Stamp},
};

const HEADER_SHA256: &str = "71213f444f9c15a4af6dbcbddc36f7611542884ebb3aadef54de650071357449";
const SHARD_BYTES: u64 = 7_389_759_032;
const PINNED_INDEX_SHA256: &str =
    "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8";
const TOP6: [u32; 6] = [0, 383, 382, 381, 380, 379];
const ROUTE_PAYLOAD_BYTES: u64 = 112_803_840;
static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

struct TempDirectory(PathBuf);

impl TempDirectory {
    fn new() -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(".work");
        fs::create_dir_all(&root).unwrap();
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!("r726-ffn-norm-{}-{sequence}", std::process::id()));
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

struct Fixture {
    checkpoint: TempDirectory,
    index: &'static [u8],
    source: PinnedMxfp4PayloadSource<'static>,
    gate_spec: GateShardSpec,
    norm_spec: FfnNormTensorSpec,
    norm_payload: Vec<u8>,
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

fn fixture() -> Fixture {
    let index = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
    );
    assert_eq!(voltafabric::sha256::hex(index), PINNED_INDEX_SHA256);
    let header = include_bytes!(
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

    // The router payload is synthetic. Its bit-plane rows preserve the pinned
    // top-6 ordering after the ffn_norm's common positive scale factor.
    let mut gate_weight = vec![0u8; GATE_WEIGHT_BYTES as usize];
    for expert in 0..ROUTED_EXPERTS {
        for bit in 0..9 {
            if expert & (1 << bit) != 0 {
                let offset = (expert * HIDDEN_SIZE + bit) * 2;
                gate_weight[offset..offset + 2].copy_from_slice(&0x3f80u16.to_le_bytes());
            }
        }
    }
    let mut text_bias = vec![0u8; ROUTED_EXPERTS * 4];
    text_bias[..4].copy_from_slice(&1000.0f32.to_le_bytes());
    let image_bias = vec![0u8; ROUTED_EXPERTS * 4];

    // Synthetic resident gamma values (BF16 sixteen). This proves the indexed
    // load/compute boundary, not the checkpoint's real norm payload.
    let mut norm_payload = Vec::with_capacity(RMS_NORM_WEIGHT_BYTES as usize);
    for _ in 0..HIDDEN_SIZE {
        norm_payload.extend_from_slice(&0x4180u16.to_le_bytes());
    }
    write_tensor_payload(
        &mut shard,
        &header_json,
        data_start,
        "layers.0.ffn.gate.weight",
        &gate_weight,
    );
    write_tensor_payload(
        &mut shard,
        &header_json,
        data_start,
        "layers.0.ffn.gate.bias",
        &text_bias,
    );
    write_tensor_payload(
        &mut shard,
        &header_json,
        data_start,
        "layers.0.ffn.gate.bias_vl",
        &image_bias,
    );
    write_tensor_payload(
        &mut shard,
        &header_json,
        data_start,
        "layers.0.ffn_norm.weight",
        &norm_payload,
    );
    drop(shard);

    Fixture {
        checkpoint,
        index,
        source: PinnedMxfp4PayloadSource {
            index_bytes: index,
            header_bytes: header,
            expected_header_sha256: HEADER_SHA256,
            header_shard_name: "model-00003-of-00048.safetensors",
            source_shard_bytes: SHARD_BYTES,
            max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
        },
        gate_spec: GateShardSpec {
            model_revision: PINNED_MODEL_REVISION.to_owned(),
            layer: 0,
            gate_temperature: 1.0,
            route_scale: ROUTE_SCALE,
            tensor_sha256: GateTensorHashes {
                weight: voltafabric::sha256::hex(&gate_weight),
                text_bias: voltafabric::sha256::hex(&text_bias),
                image_bias: voltafabric::sha256::hex(&image_bias),
            },
            max_payload_bytes: GATE_PAYLOAD_BYTES,
        },
        norm_spec: FfnNormTensorSpec {
            layer: 0,
            epsilon: PINNED_RMS_NORM_EPSILON,
            tensor_sha256: voltafabric::sha256::hex(&norm_payload),
        },
        norm_payload,
    }
}

fn exact_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let tie_to_even_bias = 0x7fffu32 + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(tie_to_even_bias) & 0xffff_0000)
}

#[test]
fn indexed_ffn_norm_load_matches_independent_fp64_oracle_and_routes_into_moe() {
    let fixture = fixture();
    let layer = DeepSeekV41IndexedMoeLayer::load_with_ffn_norm(
        fixture.checkpoint.path(),
        PinnedGateIndexSource {
            index_bytes: fixture.index,
            expected_header_sha256: HEADER_SHA256,
            max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
        },
        fixture.source,
        &fixture.gate_spec,
        &fixture.norm_spec,
    )
    .unwrap();
    let norm_receipt = layer.ffn_norm_receipt().unwrap();
    assert_eq!(norm_receipt.tensor_name, "layers.0.ffn_norm.weight");
    assert_eq!(norm_receipt.tensor_sha256, fixture.norm_spec.tensor_sha256);
    assert_eq!(norm_receipt.tensor_bytes, RMS_NORM_WEIGHT_BYTES);
    assert_eq!(norm_receipt.shard_bytes, SHARD_BYTES);

    let mut raw = vec![0.0f32; HIDDEN_SIZE];
    for (bit, value) in raw.iter_mut().take(9).enumerate() {
        *value = (1u32 << bit) as f32;
    }
    let actual = layer.normalize_ffn_hidden(&raw).unwrap();
    let gamma: Vec<f64> = fixture
        .norm_payload
        .chunks_exact(2)
        .map(|pair| {
            f64::from(f32::from_bits(
                u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16,
            ))
        })
        .collect();
    let square_sum = raw
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>();
    let inverse_rms = (square_sum / HIDDEN_SIZE as f64 + f64::from(PINNED_RMS_NORM_EPSILON))
        .sqrt()
        .recip();
    let expected: Vec<f32> = raw
        .iter()
        .zip(gamma)
        .map(|(&value, weight)| exact_bf16(((f64::from(value) * inverse_rms) * weight) as f32))
        .collect();
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs <= 1.0e-4,
        "independent FP64 RMSNorm max abs error {max_abs}"
    );
    assert!(actual
        .iter()
        .all(|value| exact_bf16(*value).to_bits() == value.to_bits()));

    let stamp = Stamp {
        run_id: "r726-ffn-norm".to_owned(),
        model_revision: voltafabric::deepseek_v41::REVISION.to_owned(),
        plan_id: "r726-synthetic-norm-plan".to_owned(),
        layer: 0,
        batch: 1,
        epoch: 7,
    };
    let registry = Registry {
        epoch: stamp.epoch,
        copies: TOP6
            .iter()
            .enumerate()
            .map(|(index, &expert)| Replica {
                expert,
                rank: if index < 2 { 4 } else { 3 + index as u32 },
                local: if index < 2 { index as u32 } else { 0 },
            })
            .collect(),
    };
    let request = || DeepSeekV41IndexedFfnTokenRequest {
        hidden: &raw,
        token_kind: TokenKind::Text,
        stamp: stamp.clone(),
        attempt: 41,
        registry: registry.clone(),
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([
            (4, vec![0, 1]),
            (5, vec![0]),
            (6, vec![0]),
            (7, vec![0]),
            (8, vec![0]),
        ]),
        max_total_selected_payload_bytes: ROUTE_PAYLOAD_BYTES,
    };
    let routed = layer.prepare_ffn_token(request()).unwrap();
    assert_eq!(
        routed
            .activation()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "the activation consumed by routing is the one carried to the expert stage"
    );
    let normalized_route = layer
        .prepare_token(
            voltafabric::deepseek_v41_gate::DeepSeekV41IndexedTokenRequest {
                hidden: &actual,
                token_kind: TokenKind::Text,
                stamp,
                attempt: 41,
                registry,
                preferred_owners: BTreeMap::new(),
                resident_order: BTreeMap::from([
                    (4, vec![0, 1]),
                    (5, vec![0]),
                    (6, vec![0]),
                    (7, vec![0]),
                    (8, vec![0]),
                ]),
                max_total_selected_payload_bytes: ROUTE_PAYLOAD_BYTES,
            },
        )
        .unwrap();
    let route_a = &routed.batch().problem().routes[0];
    let route_b = &normalized_route.batch().problem().routes[0];
    assert_eq!(route_a.len(), 6);
    assert_eq!(
        route_a
            .iter()
            .map(|choice| (choice.expert, choice.weight.to_bits()))
            .collect::<Vec<_>>(),
        route_b
            .iter()
            .map(|choice| (choice.expert, choice.weight.to_bits()))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        route_a
            .iter()
            .map(|choice| choice.expert)
            .collect::<Vec<_>>(),
        TOP6
    );
    assert_eq!(routed.batch().attempt(), 41);
}

#[test]
fn invalid_pinned_norm_epsilon_is_refused_before_opening_a_shard() {
    let index = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
    );
    let header = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model-00003.header.314113f8.json"
    );
    let empty = TempDirectory::new();
    let error = DeepSeekV41FfnRmsNorm::load_indexed(
        empty.path(),
        PinnedMxfp4PayloadSource {
            index_bytes: index,
            header_bytes: header,
            expected_header_sha256: HEADER_SHA256,
            header_shard_name: "model-00003-of-00048.safetensors",
            source_shard_bytes: SHARD_BYTES,
            max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
        },
        &FfnNormTensorSpec {
            layer: 0,
            epsilon: 1.0e-6,
            tensor_sha256: "0".repeat(64),
        },
    )
    .unwrap_err();
    assert!(error.message.contains("epsilon"));
}
