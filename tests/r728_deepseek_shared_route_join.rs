//! Indexed pinned shared-FP8 loading and atomic routed-plus-shared CPU boundary.
#![cfg(target_os = "linux")]

use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};
use voltafabric::{
    deepseek_v41::{cast_bf16_rne_reference, PinnedMxfp4RouteExecutionReceipt},
    deepseek_v41_gate::{
        combine_pinned_moe_route_with_shared, DeepSeekV41IndexedFfnTokenRequest,
        DeepSeekV41IndexedMoeLayer, FfnNormTensorSpec, GateShardSpec, GateTensorHashes,
        PinnedGateIndexSource, PinnedMxfp4PayloadSource, SharedExpertTensorHashes,
        SharedExpertTensorSpec, TokenKind, GATE_PAYLOAD_BYTES, GATE_WEIGHT_BYTES, HIDDEN_SIZE,
        PINNED_MODEL_REVISION, PINNED_RMS_NORM_EPSILON, RMS_NORM_WEIGHT_BYTES, ROUTED_EXPERTS,
        ROUTE_SCALE, SHARED_EXPERT_INTERMEDIATE_SIZE, SHARED_EXPERT_PAYLOAD_BYTES,
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
        let path = root.join(format!("r728-shared-fp8-{}-{sequence}", std::process::id()));
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
    source: PinnedMxfp4PayloadSource<'static>,
    gate_spec: GateShardSpec,
    norm_spec: FfnNormTensorSpec,
    shared_spec: SharedExpertTensorSpec,
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
    // The sparse file reproduces the pinned 7.3 GB shard length without writing
    // unrelated checkpoint ranges; the test writes only selected tensor payloads.
    shard.set_len(SHARD_BYTES).unwrap();
    shard
        .write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    shard.write_all(header).unwrap();

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
    let mut norm_payload = Vec::with_capacity(RMS_NORM_WEIGHT_BYTES as usize);
    for _ in 0..HIDDEN_SIZE {
        norm_payload.extend_from_slice(&0x4180u16.to_le_bytes());
    }

    let weight_plane = vec![0x38; HIDDEN_SIZE * SHARED_EXPERT_INTERMEDIATE_SIZE];
    let names = ["w1", "w2", "w3"];
    let mut shared_hashes = BTreeMap::new();
    for projection in names {
        let rows = if projection == "w2" {
            HIDDEN_SIZE
        } else {
            SHARED_EXPERT_INTERMEDIATE_SIZE
        };
        let columns = if projection == "w2" {
            SHARED_EXPERT_INTERMEDIATE_SIZE
        } else {
            HIDDEN_SIZE
        };
        let scales = vec![127; (rows / 32) * (columns / 32)];
        let weight_name = format!("layers.0.ffn.shared_experts.{projection}.weight");
        let scale_name = format!("layers.0.ffn.shared_experts.{projection}.scale");
        write_tensor_payload(
            &mut shard,
            &header_json,
            data_start,
            &weight_name,
            &weight_plane,
        );
        write_tensor_payload(&mut shard, &header_json, data_start, &scale_name, &scales);
        shared_hashes.insert(weight_name, voltafabric::sha256::hex(&weight_plane));
        shared_hashes.insert(scale_name, voltafabric::sha256::hex(&scales));
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

    let hash = |projection: &str, plane: &str| {
        shared_hashes[&format!("layers.0.ffn.shared_experts.{projection}.{plane}")].clone()
    };
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
        shared_spec: SharedExpertTensorSpec {
            layer: 0,
            tensor_sha256: SharedExpertTensorHashes {
                w1_weight: hash("w1", "weight"),
                w1_scale: hash("w1", "scale"),
                w2_weight: hash("w2", "weight"),
                w2_scale: hash("w2", "scale"),
                w3_weight: hash("w3", "weight"),
                w3_scale: hash("w3", "scale"),
            },
        },
    }
}

fn stamp() -> Stamp {
    Stamp {
        run_id: "r728-indexed-shared".to_owned(),
        model_revision: voltafabric::deepseek_v41::REVISION.to_owned(),
        plan_id: "r728-synthetic-route-plan".to_owned(),
        layer: 0,
        batch: 8,
        epoch: 13,
    }
}

fn route_receipt(
    state: &str,
    stamp: &Stamp,
    attempt: u64,
    values: Option<Vec<u32>>,
) -> PinnedMxfp4RouteExecutionReceipt {
    let output_bits = values.map(|row| vec![row]);
    serde_json::from_value(json!({
        "state": state,
        "stamp": stamp,
        "attempt": attempt,
        "route_experts": TOP6,
        "route_fingerprint": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "exec_receipts": if state == "Completed" { json!([{
            "rank": 4,
            "attempt": attempt,
            "state": "completed",
            "started": true,
            "expected_slots": 6,
            "received_slots": 6,
            "request_bytes": 1,
            "error": null
        }]) } else { json!([]) },
        "output_bits": output_bits,
        "error": null
    }))
    .unwrap()
}

fn indexed_layer(fixture: &Fixture) -> DeepSeekV41IndexedMoeLayer<'_> {
    DeepSeekV41IndexedMoeLayer::load_with_ffn_norm_and_shared_expert(
        fixture.checkpoint.path(),
        PinnedGateIndexSource {
            index_bytes: fixture.index,
            expected_header_sha256: HEADER_SHA256,
            max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
        },
        fixture.source,
        &fixture.gate_spec,
        &fixture.norm_spec,
        &fixture.shared_spec,
    )
    .unwrap()
}

#[test]
fn indexed_shared_fp8_load_executes_and_joins_only_the_matching_route_once() {
    assert_eq!(SHARED_EXPERT_PAYLOAD_BYTES, 35_424_000);
    let fixture = fixture();
    let layer = indexed_layer(&fixture);
    let receipt = layer.shared_expert_receipt().unwrap();
    assert_eq!(receipt.layer, 0);
    assert_eq!(
        receipt.source_shard_name,
        "model-00003-of-00048.safetensors"
    );
    assert_eq!(receipt.index_sha256, PINNED_INDEX_SHA256);
    assert_eq!(receipt.header_sha256, HEADER_SHA256);
    assert_eq!(receipt.tensor_bytes, SHARED_EXPERT_PAYLOAD_BYTES);
    assert_eq!(receipt.shard_bytes, SHARD_BYTES);
    assert_eq!(receipt.tensor_sha256, fixture.shared_spec.tensor_sha256);

    let raw = {
        let mut values = vec![0.0f32; HIDDEN_SIZE];
        for (bit, value) in values.iter_mut().take(9).enumerate() {
            *value = (1u32 << bit) as f32;
        }
        values
    };
    let stamp = stamp();
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
    let plan = layer
        .prepare_ffn_token(DeepSeekV41IndexedFfnTokenRequest {
            hidden: &raw,
            token_kind: TokenKind::Text,
            stamp: stamp.clone(),
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
        })
        .unwrap();
    assert_eq!(plan.batch().problem().routes[0].len(), 6);
    let shared = layer.execute_shared_reference(&plan).unwrap();
    assert_eq!(shared.stamp(), &stamp);
    assert_eq!(shared.attempt(), 41);
    assert_eq!(shared.output_fp32().len(), HIDDEN_SIZE);
    assert!(shared.output_fp32().iter().all(|value| value.is_finite()));

    let routed_value = 2.0f32;
    let routed = route_receipt(
        "Completed",
        &stamp,
        41,
        Some(vec![routed_value.to_bits(); HIDDEN_SIZE]),
    );
    let combined = combine_pinned_moe_route_with_shared(&routed, &shared).unwrap();
    assert_eq!(combined.stamp(), &stamp);
    assert_eq!(combined.attempt(), 41);
    assert_eq!(combined.output_bf16_f32().len(), HIDDEN_SIZE);
    for (index, (&actual, &shared_value)) in combined
        .output_bf16_f32()
        .iter()
        .zip(shared.output_fp32())
        .enumerate()
    {
        let expected = cast_bf16_rne_reference(routed_value + shared_value).unwrap();
        assert_eq!(actual.to_bits(), expected.to_bits(), "output index {index}");
    }

    let overflowing_route = route_receipt(
        "Completed",
        &stamp,
        41,
        Some(vec![f32::MAX.to_bits(); HIDDEN_SIZE]),
    );
    assert_eq!(
        combine_pinned_moe_route_with_shared(&overflowing_route, &shared)
            .unwrap_err()
            .code,
        "NUMERICAL_OVERFLOW"
    );

    let mut bad_hash = fixture.shared_spec.clone();
    bad_hash.tensor_sha256.w1_weight = "0".repeat(64);
    let rejected = DeepSeekV41IndexedMoeLayer::load_with_ffn_norm_and_shared_expert(
        fixture.checkpoint.path(),
        PinnedGateIndexSource {
            index_bytes: fixture.index,
            expected_header_sha256: HEADER_SHA256,
            max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
        },
        fixture.source,
        &fixture.gate_spec,
        &fixture.norm_spec,
        &bad_hash,
    );
    assert!(
        rejected.is_err(),
        "shared tensor hash mismatch must withhold layer"
    );
}

#[test]
fn incomplete_unknown_or_mismatched_route_never_exposes_a_combined_row() {
    let fixture = fixture();
    let layer = indexed_layer(&fixture);
    let mut raw = vec![0.0f32; HIDDEN_SIZE];
    for (bit, value) in raw.iter_mut().take(9).enumerate() {
        *value = (1u32 << bit) as f32;
    }
    let stamp = stamp();
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
    let plan = layer
        .prepare_ffn_token(DeepSeekV41IndexedFfnTokenRequest {
            hidden: &raw,
            token_kind: TokenKind::Text,
            stamp: stamp.clone(),
            attempt: 43,
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
        })
        .unwrap();
    let shared = layer.execute_shared_reference(&plan).unwrap();

    let unknown = route_receipt("Unknown", &stamp, 43, None);
    assert_eq!(
        combine_pinned_moe_route_with_shared(&unknown, &shared)
            .unwrap_err()
            .code,
        "ROUTE_UNKNOWN"
    );
    let aborted = route_receipt("Aborted", &stamp, 43, None);
    assert_eq!(
        combine_pinned_moe_route_with_shared(&aborted, &shared)
            .unwrap_err()
            .code,
        "ROUTE_ABORTED"
    );

    let mut wrong_identity = stamp.clone();
    wrong_identity.plan_id.push_str("-other");
    let wrong_plan = route_receipt(
        "Completed",
        &wrong_identity,
        43,
        Some(vec![1.0f32.to_bits(); HIDDEN_SIZE]),
    );
    assert!(combine_pinned_moe_route_with_shared(&wrong_plan, &shared).is_err());
    let wrong_attempt = route_receipt(
        "Completed",
        &stamp,
        44,
        Some(vec![1.0f32.to_bits(); HIDDEN_SIZE]),
    );
    assert!(combine_pinned_moe_route_with_shared(&wrong_attempt, &shared).is_err());
    let partial = route_receipt(
        "Completed",
        &stamp,
        43,
        Some(vec![1.0f32.to_bits(); HIDDEN_SIZE - 1]),
    );
    assert!(combine_pinned_moe_route_with_shared(&partial, &shared).is_err());
}
