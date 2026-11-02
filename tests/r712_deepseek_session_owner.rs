//! R713/R714 preflight, LOAD and execute a synthetic six-expert DeepSeek route on CPU.
//! The sparse source weights are synthetic; no checkpoint payload is read.
#![cfg(target_os = "linux")]

use safetensors::tensor::{serialize, Dtype, TensorView};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use voltafabric::deepseek_v41::{
    execute_pinned_mxfp4_route, execute_pinned_mxfp4_route_grouped,
    group_pinned_mxfp4_route_residents, plan_pinned_mxfp4_expert_payloads,
    preflight_pinned_mxfp4_route_residents, read_pinned_mxfp4_rank_expert,
    read_pinned_mxfp4_rank_expert_indexed, Mxfp4ExpertNamespace, PinnedExpertPayloadRequest,
    PinnedMxfp4IndexedRankExpertReadRequest, PinnedMxfp4RankExpertReadRequest,
    PinnedMxfp4RouteExecutionState, REVISION,
};
use voltafabric::deepseek_v41_gate::{
    DeepSeekV41Gate, GateShardSpec, GateTensorHashes, PinnedMxfp4PayloadSource, TokenKind,
    GATE_PAYLOAD_BYTES, GATE_WEIGHT_BYTES, HIDDEN_SIZE, PINNED_MODEL_REVISION, ROUTED_EXPERTS,
    ROUTE_SCALE,
};
use voltafabric::moe::{Choice, Registry, Replica, Stamp};
use voltafabric::routing::RoutingProblem;
use voltafabric::session::{mxfp4, Endpoint, Session};

const HEADER_SHA256: &str = "71213f444f9c15a4af6dbcbddc36f7611542884ebb3aadef54de650071357449";
const SHARD_BYTES: u64 = 7_389_759_032;
const PINNED_INDEX_SHA256: &str =
    "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8";
const TOP6: [u32; 6] = [0, 383, 382, 381, 380, 379];
const ROUTE_WEIGHTS: [f64; 6] = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125];
static NEXT_INDEXED_TEMP: AtomicU64 = AtomicU64::new(0);

struct IndexedTempDirectory(PathBuf);

impl IndexedTempDirectory {
    fn new() -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(".work");
        fs::create_dir_all(&root).unwrap();
        let sequence = NEXT_INDEXED_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(
            "r722-indexed-session-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for IndexedTempDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct SparseShard {
    file_bytes: u64,
    position: u64,
    segments: Vec<(u64, Vec<u8>)>,
}

impl Read for SparseShard {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let available = self.file_bytes.saturating_sub(self.position);
        let count = output
            .len()
            .min(usize::try_from(available).unwrap_or(usize::MAX));
        if count == 0 {
            return Ok(0);
        }
        let start = self.position;
        let end = start + count as u64;
        output[..count].fill(0);
        for (segment_start, bytes) in &self.segments {
            let segment_end = *segment_start + bytes.len() as u64;
            let overlap_start = start.max(*segment_start);
            let overlap_end = end.min(segment_end);
            if overlap_start < overlap_end {
                let output_start = usize::try_from(overlap_start - start)
                    .map_err(|error| io::Error::other(error.to_string()))?;
                let output_end = usize::try_from(overlap_end - start)
                    .map_err(|error| io::Error::other(error.to_string()))?;
                let input_start = usize::try_from(overlap_start - *segment_start)
                    .map_err(|error| io::Error::other(error.to_string()))?;
                let input_end = input_start + (output_end - output_start);
                output[output_start..output_end].copy_from_slice(&bytes[input_start..input_end]);
            }
        }
        self.position = end;
        Ok(count)
    }
}

impl Seek for SparseShard {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let next = match from {
            SeekFrom::Start(position) => i128::from(position),
            SeekFrom::Current(delta) => i128::from(self.position) + i128::from(delta),
            SeekFrom::End(delta) => i128::from(self.file_bytes) + i128::from(delta),
        };
        if !(0..=i128::from(u64::MAX)).contains(&next) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek out of range",
            ));
        }
        self.position = next as u64;
        Ok(self.position)
    }
}

fn route_problem() -> RoutingProblem {
    RoutingProblem {
        hidden: voltafabric::deepseek_v41_gate::HIDDEN_SIZE,
        experts: TOP6.to_vec(),
        routes: vec![TOP6
            .iter()
            .zip(ROUTE_WEIGHTS)
            .map(|(&expert, weight)| Choice { expert, weight })
            .collect()],
        registry: Registry {
            epoch: 13,
            copies: TOP6
                .iter()
                .enumerate()
                .map(|(local, &expert)| Replica {
                    expert,
                    rank: 4,
                    local: local as u32,
                })
                .collect(),
        },
        stamp: Stamp {
            run_id: "r712-session-owner".to_owned(),
            model_revision: REVISION.to_owned(),
            plan_id: "r712-route-rank-plan".to_owned(),
            layer: 0,
            batch: 7,
            epoch: 13,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(4, (0..TOP6.len() as u32).collect())]),
    }
}

#[test]
fn colocated_top6_dispatch_requires_all_six_owner_receipts_before_a_worker_exists() {
    let route = route_problem();
    let error = preflight_pinned_mxfp4_route_residents(&route, 29, &[], 1).unwrap_err();
    assert!(error.message.contains("exactly six owners"));
    let dispatch = voltafabric::routing::dispatch(&route).unwrap();
    assert_eq!(dispatch.slots.len(), TOP6.len());
    assert_eq!(dispatch.ranks.len(), 1);
    assert_eq!(dispatch.ranks[0].rank, 4);
    assert_eq!(
        dispatch.ranks[0].resident_order,
        (0..TOP6.len() as u32).collect::<Vec<_>>()
    );
}

fn distributed_route_problem() -> RoutingProblem {
    let mut route = route_problem();
    route.registry.copies = TOP6
        .iter()
        .enumerate()
        .map(|(index, &expert)| Replica {
            expert,
            rank: 4 + index as u32,
            local: 0,
        })
        .collect();
    route.resident_order = (0..TOP6.len())
        .map(|index| (4 + index as u32, vec![0]))
        .collect();
    route
}

fn grouped_route_problem() -> RoutingProblem {
    let mut route = distributed_route_problem();
    route.registry.copies = TOP6
        .iter()
        .enumerate()
        .map(|(index, &expert)| Replica {
            expert,
            rank: if index < 2 { 4 } else { 3 + index as u32 },
            local: if index < 2 { index as u32 } else { 0 },
        })
        .collect();
    route.resident_order = BTreeMap::from([
        (4, vec![0, 1]),
        (5, vec![0]),
        (6, vec![0]),
        (7, vec![0]),
        (8, vec![0]),
    ]);
    route
}

fn gated_grouped_route_batch() -> voltafabric::deepseek_v41_gate::DeepSeekV41RouteBatch {
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
    text_bias[0] = 1_000.0;
    let image_bias = vec![0.0f32; ROUTED_EXPERTS];
    let encode = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let text_bytes = encode(&text_bias);
    let image_bytes = encode(&image_bias);
    let shard = serialize(
        vec![
            (
                "layers.0.ffn.gate.weight".to_owned(),
                TensorView::new(Dtype::BF16, vec![ROUTED_EXPERTS, HIDDEN_SIZE], &weight).unwrap(),
            ),
            (
                "layers.0.ffn.gate.bias".to_owned(),
                TensorView::new(Dtype::F32, vec![ROUTED_EXPERTS], &text_bytes).unwrap(),
            ),
            (
                "layers.0.ffn.gate.bias_vl".to_owned(),
                TensorView::new(Dtype::F32, vec![ROUTED_EXPERTS], &image_bytes).unwrap(),
            ),
        ],
        None,
    )
    .unwrap();
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
    let gate = DeepSeekV41Gate::load(&mut io::Cursor::new(&shard), &spec).unwrap();
    let mut hidden = vec![0.0f32; HIDDEN_SIZE];
    for (bit, value) in hidden.iter_mut().take(9).enumerate() {
        *value = (1u32 << bit) as f32;
    }
    let stamp = Stamp {
        run_id: "r716-gate-to-grouped-session".to_owned(),
        model_revision: REVISION.to_owned(),
        plan_id: "r716-synthetic-rank-plan".to_owned(),
        layer: 0,
        batch: 9,
        epoch: 13,
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
    gate.prepare_route_batch(
        &hidden,
        TokenKind::Text,
        stamp,
        29,
        registry,
        BTreeMap::new(),
        BTreeMap::from([
            (4, vec![0, 1]),
            (5, vec![0]),
            (6, vec![0]),
            (7, vec![0]),
            (8, vec![0]),
        ]),
    )
    .unwrap()
}

fn synthetic_route_payload_shard(
    plan: &voltafabric::deepseek_v41::Mxfp4ExpertPayloadPlan,
) -> (SparseShard, BTreeMap<String, String>, Vec<u8>) {
    let header = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model-00003.header.314113f8.json"
    )
    .to_vec();
    let mut prefix = (header.len() as u64).to_le_bytes().to_vec();
    prefix.extend_from_slice(&header);
    let mut segments = vec![(0, prefix)];
    let mut hashes = BTreeMap::new();
    for tensor in &plan.tensors {
        let length = usize::try_from(tensor.file_offset_end - tensor.file_offset_start).unwrap();
        let byte = if tensor.name.ends_with(".scale") {
            0x6f
        } else {
            0x22
        };
        let payload = vec![byte; length];
        hashes.insert(tensor.name.clone(), voltafabric::sha256::hex(&payload));
        segments.push((tensor.file_offset_start, payload));
    }
    (
        SparseShard {
            file_bytes: SHARD_BYTES,
            position: 0,
            segments,
        },
        hashes,
        header,
    )
}

fn resident_for_expert(
    route: &RoutingProblem,
    expert: u32,
    rank: u32,
) -> voltafabric::deepseek_v41::PinnedMxfp4RankResidentExpert {
    let index = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
    );
    let header = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model-00003.header.314113f8.json"
    );
    assert_eq!(voltafabric::sha256::hex(index), PINNED_INDEX_SHA256);

    let plan = plan_pinned_mxfp4_expert_payloads(PinnedExpertPayloadRequest {
        index_bytes: index,
        header_bytes: header,
        expected_header_sha256: HEADER_SHA256,
        header_shard_name: "model-00003-of-00048.safetensors",
        source_shard_bytes: SHARD_BYTES,
        max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
        namespace: Mxfp4ExpertNamespace::Backbone,
        layer: 0,
        experts: &[expert],
    })
    .unwrap();
    let mut prefix = (header.len() as u64).to_le_bytes().to_vec();
    prefix.extend_from_slice(header);
    let mut segments = vec![(0, prefix)];
    let mut hashes = BTreeMap::new();
    for (slot, tensor) in plan.tensors.iter().enumerate() {
        let length = usize::try_from(tensor.file_offset_end - tensor.file_offset_start).unwrap();
        let payload = vec![u8::try_from(slot + 1).unwrap(); length];
        hashes.insert(tensor.name.clone(), voltafabric::sha256::hex(&payload));
        segments.push((tensor.file_offset_start, payload));
    }
    let mut source = SparseShard {
        file_bytes: SHARD_BYTES,
        position: 0,
        segments,
    };
    let read = read_pinned_mxfp4_rank_expert(
        &mut source,
        PinnedMxfp4RankExpertReadRequest {
            route,
            attempt: 29,
            expert,
            rank,
            index_bytes: index,
            header_shard_name: "model-00003-of-00048.safetensors",
            expected_header_sha256: HEADER_SHA256,
            expected_payload_sha256: &hashes,
            max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
            max_payload_bytes: voltafabric::checkpoint::MAX_NAMED_TENSOR_PAYLOAD_BYTES,
        },
    )
    .unwrap();
    let peak = plan.selected_payload_bytes * 2 + voltafabric::checkpoint::MAX_HEADER_BYTES + 8;
    read.into_session_resident(peak).unwrap()
}

fn indexed_residents_for_route(
    route: &RoutingProblem,
) -> (
    IndexedTempDirectory,
    Vec<voltafabric::deepseek_v41::PinnedMxfp4RankResidentExpert>,
) {
    let index = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
    );
    let header = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model-00003.header.314113f8.json"
    );
    let plan = plan_pinned_mxfp4_expert_payloads(PinnedExpertPayloadRequest {
        index_bytes: index,
        header_bytes: header,
        expected_header_sha256: HEADER_SHA256,
        header_shard_name: "model-00003-of-00048.safetensors",
        source_shard_bytes: SHARD_BYTES,
        max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
        namespace: Mxfp4ExpertNamespace::Backbone,
        layer: 0,
        experts: &TOP6,
    })
    .unwrap();
    let temp = IndexedTempDirectory::new();
    let shard_path = temp.path().join(&plan.source_shard);
    let mut shard = File::create(&shard_path).unwrap();
    shard.set_len(SHARD_BYTES).unwrap();
    shard
        .write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    shard.write_all(header).unwrap();

    let mut hashes_by_expert = TOP6
        .iter()
        .map(|&expert| (expert, BTreeMap::new()))
        .collect::<BTreeMap<_, _>>();
    for tensor in &plan.tensors {
        let length = usize::try_from(tensor.file_offset_end - tensor.file_offset_start).unwrap();
        let value = if tensor.name.ends_with(".scale") {
            0x6f
        } else {
            0x22
        };
        let payload = vec![value; length];
        let expert = tensor.expert;
        hashes_by_expert
            .get_mut(&expert)
            .unwrap()
            .insert(tensor.name.clone(), voltafabric::sha256::hex(&payload));
        shard
            .seek(SeekFrom::Start(tensor.file_offset_start))
            .unwrap();
        shard.write_all(&payload).unwrap();
    }
    drop(shard);

    let residents = TOP6
        .iter()
        .map(|&expert| {
            let rank = route
                .registry
                .copies
                .iter()
                .find(|copy| copy.expert == expert)
                .unwrap()
                .rank;
            let read = read_pinned_mxfp4_rank_expert_indexed(
                temp.path(),
                PinnedMxfp4IndexedRankExpertReadRequest {
                    route,
                    attempt: 29,
                    expert,
                    rank,
                    index_bytes: index,
                    expected_header_sha256: HEADER_SHA256,
                    expected_payload_sha256: &hashes_by_expert[&expert],
                    max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
                    max_payload_bytes: voltafabric::checkpoint::MAX_NAMED_TENSOR_PAYLOAD_BYTES,
                },
            )
            .unwrap();
            assert_eq!(read.plan().source_shard, plan.source_shard);
            assert_eq!(read.plan().experts, vec![expert]);
            let peak = read.plan().selected_payload_bytes * 2
                + voltafabric::checkpoint::MAX_HEADER_BYTES
                + 8;
            read.into_session_resident(peak).unwrap()
        })
        .collect();
    (temp, residents)
}

fn indexed_gate_route_payloads() -> (
    IndexedTempDirectory,
    voltafabric::deepseek_v41_gate::DeepSeekV41RouteBatch,
    voltafabric::deepseek_v41_gate::DeepSeekV41RoutePayloadReads,
) {
    let index = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
    );
    let header = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model-00003.header.314113f8.json"
    );
    let header_value: serde_json::Value = serde_json::from_slice(header).unwrap();
    let data_start = header.len() as u64 + 8;
    let temp = IndexedTempDirectory::new();
    let shard_path = temp.path().join("model-00003-of-00048.safetensors");
    let mut shard = File::create(&shard_path).unwrap();
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
    let mut text_bias_values = vec![0.0f32; ROUTED_EXPERTS];
    text_bias_values[0] = 1_000.0;
    let image_bias_values = vec![0.0f32; ROUTED_EXPERTS];
    let encode = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let text_bias = encode(&text_bias_values);
    let image_bias = encode(&image_bias_values);
    let gate_payloads = [
        ("layers.0.ffn.gate.weight", gate_weight.as_slice()),
        ("layers.0.ffn.gate.bias", text_bias.as_slice()),
        ("layers.0.ffn.gate.bias_vl", image_bias.as_slice()),
    ];
    for (name, payload) in gate_payloads {
        let offsets = header_value[name]["data_offsets"].as_array().unwrap();
        let start = data_start + offsets[0].as_u64().unwrap();
        let end = data_start + offsets[1].as_u64().unwrap();
        assert_eq!(end - start, payload.len() as u64);
        shard.seek(SeekFrom::Start(start)).unwrap();
        shard.write_all(payload).unwrap();
    }
    drop(shard);

    let gate_spec = GateShardSpec {
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
    };
    let gate = DeepSeekV41Gate::load_indexed(
        temp.path(),
        voltafabric::deepseek_v41_gate::PinnedGateIndexSource {
            index_bytes: index,
            expected_header_sha256: HEADER_SHA256,
            max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
        },
        &gate_spec,
    )
    .unwrap();
    let mut hidden = vec![0.0f32; HIDDEN_SIZE];
    for (bit, value) in hidden.iter_mut().take(9).enumerate() {
        *value = (1u32 << bit) as f32;
    }
    let stamp = Stamp {
        run_id: "r716-gate-to-grouped-session".to_owned(),
        model_revision: REVISION.to_owned(),
        plan_id: "r716-synthetic-rank-plan".to_owned(),
        layer: 0,
        batch: 9,
        epoch: 13,
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
    let batch = gate
        .prepare_route_batch(
            &hidden,
            TokenKind::Text,
            stamp,
            29,
            registry,
            BTreeMap::new(),
            BTreeMap::from([
                (4, vec![0, 1]),
                (5, vec![0]),
                (6, vec![0]),
                (7, vec![0]),
                (8, vec![0]),
            ]),
        )
        .unwrap();
    let reference_batch = gated_grouped_route_batch();
    assert_eq!(batch.problem().experts, TOP6);
    assert_eq!(
        batch.problem().routes[0]
            .iter()
            .map(|choice| choice.weight.to_bits())
            .collect::<Vec<_>>(),
        reference_batch.problem().routes[0]
            .iter()
            .map(|choice| choice.weight.to_bits())
            .collect::<Vec<_>>()
    );

    let source = PinnedMxfp4PayloadSource {
        index_bytes: index,
        header_bytes: header,
        expected_header_sha256: HEADER_SHA256,
        header_shard_name: "model-00003-of-00048.safetensors",
        source_shard_bytes: SHARD_BYTES,
        max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
    };
    const ROUTE_PAYLOAD_BYTES: u64 = 112_803_840;
    let plan = batch
        .plan_selected_mxfp4_payloads(source, ROUTE_PAYLOAD_BYTES)
        .unwrap();
    assert_eq!(
        plan.payload_plan().source_shard,
        "model-00003-of-00048.safetensors"
    );
    let mut shard = std::fs::OpenOptions::new()
        .write(true)
        .open(&shard_path)
        .unwrap();
    let mut hashes = BTreeMap::new();
    for tensor in &plan.payload_plan().tensors {
        let length = usize::try_from(tensor.file_offset_end - tensor.file_offset_start).unwrap();
        let byte = if tensor.name.ends_with(".scale") {
            0x6f
        } else {
            0x22
        };
        let payload = vec![byte; length];
        hashes.insert(tensor.name.clone(), voltafabric::sha256::hex(&payload));
        shard
            .seek(SeekFrom::Start(tensor.file_offset_start))
            .unwrap();
        shard.write_all(&payload).unwrap();
    }
    drop(shard);
    let reads = batch
        .read_selected_mxfp4_indexed_payloads(
            temp.path(),
            &plan,
            source,
            &hashes,
            ROUTE_PAYLOAD_BYTES,
        )
        .unwrap();
    assert_eq!(reads.stamp(), &batch.problem().stamp);
    assert_eq!(reads.attempt(), batch.attempt());
    assert_eq!(reads.route_fingerprint(), plan.route_fingerprint());
    assert_eq!(reads.reads().len(), TOP6.len());
    (temp, batch, reads)
}

#[test]
fn complete_top6_is_preflighted_then_loaded_on_six_cpu_sessions_without_exec() {
    let route = distributed_route_problem();
    let residents = TOP6
        .iter()
        .enumerate()
        .map(|(index, &expert)| resident_for_expert(&route, expert, 4 + index as u32))
        .collect::<Vec<_>>();
    let total = residents
        .iter()
        .map(|resident| resident.packed_artifact_bytes())
        .sum::<u64>();

    // Every rejection is checked while no worker handle exists. The valid proof is
    // the only value that crosses the subsequent connection/LOAD boundary.
    assert!(preflight_pinned_mxfp4_route_residents(&route, 29, &residents[..5], total).is_err());
    let mut changed_coefficients = route.clone();
    changed_coefficients.routes[0][2].weight =
        f64::from_bits(changed_coefficients.routes[0][2].weight.to_bits() + 1);
    assert!(
        preflight_pinned_mxfp4_route_residents(&changed_coefficients, 29, &residents, total,)
            .is_err()
    );
    assert!(preflight_pinned_mxfp4_route_residents(&route, 29, &residents, total - 1).is_err());
    let preflight = preflight_pinned_mxfp4_route_residents(&route, 29, &residents, total).unwrap();
    assert_eq!(preflight.route_experts(), TOP6);
    assert_eq!(preflight.total_artifact_bytes(), total);

    let mut unconsumed = residents
        .into_iter()
        .map(|owner| (owner.expert(), owner))
        .collect::<BTreeMap<_, _>>();
    let mut sessions = Vec::with_capacity(preflight.owners().len());
    let mut receipts = BTreeMap::new();
    for expected in preflight.owners() {
        let owner = unconsumed.remove(&expected.expert).unwrap();
        assert_eq!(
            (owner.rank(), owner.local()),
            (expected.rank, expected.local)
        );
        let endpoint = Endpoint {
            rank: expected.rank,
            argv: vec![env!("CARGO_BIN_EXE_vf-rank-session").into()],
        };
        let mut session = Session::connect_mxfp4(
            &endpoint,
            Duration::from_secs(10),
            &mxfp4::Producer::default(),
        )
        .unwrap();
        let receipt = owner
            .load_into_session(&mut session, &preflight, Duration::from_secs(30))
            .unwrap();
        assert_eq!(receipt.route_fingerprint(), preflight.route_fingerprint());
        assert_eq!(receipt.route_experts(), TOP6);
        assert_eq!(receipt.expert(), expected.expert);
        assert_eq!(receipt.rank(), expected.rank);
        assert_eq!(receipt.local(), expected.local);
        assert_eq!(receipt.session_load().rank, expected.rank);
        assert_eq!(receipt.session_load().epoch, route.stamp.epoch);
        assert!(receipt.session_load().transferred);
        receipts.insert(expected.expert, receipt);
        sessions.push(session);
    }
    assert!(unconsumed.is_empty());
    assert_eq!(receipts.len(), TOP6.len());
    for session in sessions {
        let close = session.close(Duration::from_secs(10));
        assert_eq!(close.bye_execs, Some(0));
        assert!(close.error.is_none(), "{close:?}");
    }
}

#[test]
fn complete_top6_exec_uses_one_route_weight_and_matches_fp64_oracle() {
    let route = distributed_route_problem();
    let (_indexed_source, residents) = indexed_residents_for_route(&route);
    let input: Vec<f32> = (0..route.hidden)
        .map(|index| ((index % 17) as i32 - 8) as f32 / 8.0)
        .collect();
    let input = input
        .into_iter()
        .inspect(|value| {
            assert_eq!(
                voltafabric::deepseek_v41::cast_bf16_rne_reference(*value).unwrap(),
                *value
            );
        })
        .collect::<Vec<_>>();

    // Independent format path: the resident's FP64 MXFP4 matvec oracle performs
    // quantization/dequantization and SwiGLU separately from the Session's R9 FP32
    // warp producer. The route coefficient enters once, before each oracle W2.
    let mut expected = vec![0.0_f32; route.hidden];
    for choice in &route.routes[0] {
        let resident = residents
            .iter()
            .find(|resident| resident.expert() == choice.expert)
            .unwrap();
        let expert = resident.owner().expert(choice.expert).unwrap();
        let row = expert
            .forward_reference(&input, Some(choice.weight as f32))
            .unwrap();
        for (sum, value) in expected.iter_mut().zip(row) {
            *sum += value;
        }
    }

    let total = residents
        .iter()
        .map(|resident| resident.packed_artifact_bytes())
        .sum::<u64>();
    let preflight = preflight_pinned_mxfp4_route_residents(&route, 29, &residents, total).unwrap();
    let mut unconsumed = residents
        .into_iter()
        .map(|owner| (owner.expert(), owner))
        .collect::<BTreeMap<_, _>>();
    let mut sessions = BTreeMap::new();
    let mut loads = Vec::with_capacity(TOP6.len());
    for expected_owner in preflight.owners() {
        let owner = unconsumed.remove(&expected_owner.expert).unwrap();
        let endpoint = Endpoint {
            rank: expected_owner.rank,
            argv: vec![env!("CARGO_BIN_EXE_vf-rank-session").into()],
        };
        let mut session = Session::connect_mxfp4(
            &endpoint,
            Duration::from_secs(10),
            &mxfp4::Producer::default(),
        )
        .unwrap();
        loads.push(
            owner
                .load_into_session(&mut session, &preflight, Duration::from_secs(30))
                .unwrap(),
        );
        assert!(sessions.insert(expected_owner.rank, session).is_none());
    }
    assert!(unconsumed.is_empty());
    assert!(sessions.values().all(|session| session.traffic.execs == 0));

    let receipt = execute_pinned_mxfp4_route(
        preflight,
        &route,
        29,
        vec![mxfp4::TokenInput {
            token: 0,
            value_bits: input.iter().map(|value| value.to_bits()).collect(),
        }],
        loads,
        &mut sessions,
        Duration::from_secs(90),
    )
    .unwrap();
    assert_eq!(receipt.state(), PinnedMxfp4RouteExecutionState::Completed);
    assert_eq!(receipt.stamp(), &route.stamp);
    assert_eq!(receipt.attempt(), 29);
    assert_eq!(receipt.route_experts(), TOP6);
    assert_eq!(receipt.exec_receipts().len(), TOP6.len());
    assert!(receipt.error().is_none());
    for exec in receipt.exec_receipts() {
        assert_eq!(exec.state, voltafabric::process::ReceiptState::Completed);
        assert!(exec.started);
        assert_eq!(exec.expected_slots, 1);
        assert_eq!(exec.received_slots, 1);
    }
    let actual = receipt
        .output_bits()
        .unwrap()
        .first()
        .unwrap()
        .iter()
        .map(|bits| f32::from_bits(*bits))
        .collect::<Vec<_>>();
    assert_eq!(actual.len(), route.hidden);
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs <= 5.0e-4,
        "FP64 MXFP4 oracle max abs error {max_abs}"
    );

    for (_, session) in sessions {
        assert_eq!(session.traffic.execs, 1);
        let close = session.close(Duration::from_secs(10));
        assert_eq!(close.bye_execs, Some(1));
        assert!(close.error.is_none(), "{close:?}");
    }
}

#[test]
fn colocated_top6_groups_two_selected_experts_and_executes_against_fp64_oracle() {
    let route = grouped_route_problem();
    let residents = TOP6
        .iter()
        .enumerate()
        .map(|(index, &expert)| {
            let rank = if index < 2 { 4 } else { 3 + index as u32 };
            resident_for_expert(&route, expert, rank)
        })
        .collect::<Vec<_>>();
    let input: Vec<f32> = (0..route.hidden)
        .map(|index| ((index % 17) as i32 - 8) as f32 / 8.0)
        .collect();
    for value in &input {
        assert_eq!(
            voltafabric::deepseek_v41::cast_bf16_rne_reference(*value).unwrap(),
            *value
        );
    }

    let expected = route.routes[0]
        .iter()
        .try_fold(
            vec![0.0_f32; route.hidden],
            |mut sum, choice| -> Result<_, String> {
                let resident = residents
                    .iter()
                    .find(|resident| resident.expert() == choice.expert)
                    .ok_or_else(|| "oracle resident missing".to_owned())?;
                let expert = resident
                    .owner()
                    .expert(choice.expert)
                    .map_err(|error| error.to_string())?;
                let row = expert
                    .forward_reference(&input, Some(choice.weight as f32))
                    .map_err(|error| error.to_string())?;
                for (target, value) in sum.iter_mut().zip(row) {
                    *target += value;
                }
                Ok(sum)
            },
        )
        .unwrap();

    let total = residents
        .iter()
        .map(|resident| resident.packed_artifact_bytes())
        .sum::<u64>();
    let preflight = preflight_pinned_mxfp4_route_residents(&route, 29, &residents, total).unwrap();
    assert!(
        group_pinned_mxfp4_route_residents(&route, preflight, residents, 1).is_err(),
        "group packing must honor its peak-memory admission before creating any Session"
    );

    // Rebuild residents after the deliberately refused ownership move. A separate
    // `UNKNOWN` or partial route is never retried; this is a fresh CPU test fixture.
    let residents = TOP6
        .iter()
        .enumerate()
        .map(|(index, &expert)| {
            let rank = if index < 2 { 4 } else { 3 + index as u32 };
            resident_for_expert(&route, expert, rank)
        })
        .collect::<Vec<_>>();
    let total = residents
        .iter()
        .map(|resident| resident.packed_artifact_bytes())
        .sum::<u64>();
    let preflight = preflight_pinned_mxfp4_route_residents(&route, 29, &residents, total).unwrap();
    let (preflight, groups) =
        group_pinned_mxfp4_route_residents(&route, preflight, residents, 256 * 1024 * 1024)
            .unwrap();
    assert_eq!(groups.len(), 5);
    let colocated = groups.iter().find(|group| group.rank() == 4).unwrap();
    assert_eq!(colocated.members().len(), 2);
    assert!(
        colocated.packed_artifact_bytes()
            <= voltafabric::deepseek_v41::MAX_MXFP4_EXPERT_ARTIFACT_BYTES
    );
    assert!(colocated.packing_peak_bound_bytes() <= 256 * 1024 * 1024);

    let mut sessions = BTreeMap::new();
    let mut loads = Vec::new();
    for group in groups {
        let rank = group.rank();
        let endpoint = Endpoint {
            rank,
            argv: vec![env!("CARGO_BIN_EXE_vf-rank-session").into()],
        };
        let mut session = Session::connect_mxfp4(
            &endpoint,
            Duration::from_secs(10),
            &mxfp4::Producer::default(),
        )
        .unwrap();
        loads.push(
            group
                .load_into_session(&mut session, &preflight, Duration::from_secs(30))
                .unwrap(),
        );
        assert!(sessions.insert(rank, session).is_none());
    }
    assert_eq!(sessions.len(), 5);
    assert!(sessions.values().all(|session| session.traffic.execs == 0));

    let receipt = execute_pinned_mxfp4_route_grouped(
        preflight,
        &route,
        29,
        vec![mxfp4::TokenInput {
            token: 0,
            value_bits: input.iter().map(|value| value.to_bits()).collect(),
        }],
        loads,
        &mut sessions,
        Duration::from_secs(90),
    )
    .unwrap();
    assert_eq!(receipt.state(), PinnedMxfp4RouteExecutionState::Completed);
    assert_eq!(receipt.stamp(), &route.stamp);
    assert_eq!(receipt.attempt(), 29);
    assert_eq!(receipt.route_experts(), TOP6);
    assert_eq!(receipt.exec_receipts().len(), 5);
    assert!(receipt.error().is_none());
    assert_eq!(
        receipt
            .exec_receipts()
            .iter()
            .map(|row| row.expected_slots)
            .sum::<usize>(),
        6
    );
    for exec in receipt.exec_receipts() {
        assert_eq!(exec.state, voltafabric::process::ReceiptState::Completed);
        assert!(exec.started);
        assert_eq!(exec.expected_slots, exec.received_slots);
    }
    let actual = receipt
        .output_bits()
        .unwrap()
        .first()
        .unwrap()
        .iter()
        .map(|bits| f32::from_bits(*bits))
        .collect::<Vec<_>>();
    assert_eq!(actual.len(), route.hidden);
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs <= 5.0e-4,
        "FP64 MXFP4 oracle max abs error {max_abs}"
    );

    for (_, session) in sessions {
        assert_eq!(session.traffic.execs, 1);
        let close = session.close(Duration::from_secs(10));
        assert_eq!(close.bye_execs, Some(1));
        assert!(close.error.is_none(), "{close:?}");
    }
}

#[test]
fn gated_top6_is_preflighted_grouped_loaded_and_executed_without_restating_route() {
    let (_checkpoint, batch, reads) = indexed_gate_route_payloads();
    let route = batch.problem();
    assert_eq!(route.experts, TOP6);
    assert_eq!(batch.attempt(), 29);
    let residents = reads.into_session_residents(64 * 1024 * 1024).unwrap();
    let input: Vec<f32> = (0..route.hidden)
        .map(|index| ((index % 17) as i32 - 8) as f32 / 8.0)
        .collect();
    for value in &input {
        assert_eq!(
            voltafabric::deepseek_v41::cast_bf16_rne_reference(*value).unwrap(),
            *value
        );
    }
    let expected = route.routes[0]
        .iter()
        .try_fold(
            vec![0.0_f32; route.hidden],
            |mut sum, choice| -> Result<_, String> {
                let resident = residents
                    .iter()
                    .find(|resident| resident.expert() == choice.expert)
                    .ok_or_else(|| "oracle resident missing".to_owned())?;
                let expert = resident
                    .owner()
                    .expert(choice.expert)
                    .map_err(|error| error.to_string())?;
                let row = expert
                    .forward_reference(&input, Some(choice.weight as f32))
                    .map_err(|error| error.to_string())?;
                for (target, value) in sum.iter_mut().zip(row) {
                    *target += value;
                }
                Ok(sum)
            },
        )
        .unwrap();
    let total = residents
        .iter()
        .map(|resident| resident.packed_artifact_bytes())
        .sum::<u64>();
    let preflight = batch.preflight_mxfp4_residents(&residents, total).unwrap();
    let (preflight, groups) = batch
        .group_mxfp4_residents(preflight, residents, 256 * 1024 * 1024)
        .unwrap();
    let mut sessions = BTreeMap::new();
    let mut loads = Vec::new();
    for group in groups {
        let rank = group.rank();
        let endpoint = Endpoint {
            rank,
            argv: vec![env!("CARGO_BIN_EXE_vf-rank-session").into()],
        };
        let mut session = Session::connect_mxfp4(
            &endpoint,
            Duration::from_secs(10),
            &mxfp4::Producer::default(),
        )
        .unwrap();
        loads.push(
            group
                .load_into_session(&mut session, &preflight, Duration::from_secs(30))
                .unwrap(),
        );
        assert!(sessions.insert(rank, session).is_none());
    }
    let receipt = batch
        .execute_grouped_mxfp4(
            preflight,
            vec![mxfp4::TokenInput {
                token: 0,
                value_bits: input.iter().map(|value| value.to_bits()).collect(),
            }],
            loads,
            &mut sessions,
            Duration::from_secs(90),
        )
        .unwrap();
    assert_eq!(receipt.state(), PinnedMxfp4RouteExecutionState::Completed);
    assert_eq!(receipt.route_experts(), TOP6);
    assert_eq!(receipt.exec_receipts().len(), 5);
    assert_eq!(
        receipt
            .exec_receipts()
            .iter()
            .map(|r| r.expected_slots)
            .sum::<usize>(),
        6
    );
    let actual = receipt
        .output_bits()
        .unwrap()
        .first()
        .unwrap()
        .iter()
        .map(|bits| f32::from_bits(*bits))
        .collect::<Vec<_>>();
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs <= 5.0e-4,
        "gated route oracle max abs error {max_abs}"
    );
    for (_, session) in sessions {
        assert_eq!(session.traffic.execs, 1);
        let close = session.close(Duration::from_secs(10));
        assert_eq!(close.bye_execs, Some(1));
        assert!(close.error.is_none(), "{close:?}");
    }
}

#[test]
fn gate_selected_route_plans_exact_pinned_checkpoint_ranges_before_payload_reads() {
    let index = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
    );
    let header = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model-00003.header.314113f8.json"
    );
    let batch = gated_grouped_route_batch();
    let source = voltafabric::deepseek_v41_gate::PinnedMxfp4PayloadSource {
        index_bytes: index,
        header_bytes: header,
        expected_header_sha256: HEADER_SHA256,
        header_shard_name: "model-00003-of-00048.safetensors",
        source_shard_bytes: SHARD_BYTES,
        max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
    };
    const ROUTE_PAYLOAD_BYTES: u64 = 112_803_840;

    let plan = batch
        .plan_selected_mxfp4_payloads(source, ROUTE_PAYLOAD_BYTES)
        .unwrap();
    assert_eq!(plan.stamp(), &batch.problem().stamp);
    assert_eq!(plan.attempt(), batch.attempt());
    assert_eq!(plan.route_experts(), TOP6);
    assert!(!plan.route_fingerprint().is_empty());
    assert_eq!(
        plan.payload_plan().source_shard,
        "model-00003-of-00048.safetensors"
    );
    assert_eq!(plan.payload_plan().source_shard_bytes, SHARD_BYTES);
    assert_eq!(plan.payload_plan().tensors.len(), 36);
    assert_eq!(
        plan.payload_plan().selected_payload_bytes,
        ROUTE_PAYLOAD_BYTES
    );

    let err = batch
        .plan_selected_mxfp4_payloads(source, ROUTE_PAYLOAD_BYTES - 1)
        .unwrap_err();
    assert_eq!(err.code, "ARTIFACT_BUDGET");
}

#[test]
fn route_payload_read_preflights_all_digests_then_returns_six_residents_atomically() {
    let index = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
    );
    let header = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model-00003.header.314113f8.json"
    );
    let batch = gated_grouped_route_batch();
    let source = PinnedMxfp4PayloadSource {
        index_bytes: index,
        header_bytes: header,
        expected_header_sha256: HEADER_SHA256,
        header_shard_name: "model-00003-of-00048.safetensors",
        source_shard_bytes: SHARD_BYTES,
        max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
    };
    const ROUTE_PAYLOAD_BYTES: u64 = 112_803_840;
    let route_plan = batch
        .plan_selected_mxfp4_payloads(source, ROUTE_PAYLOAD_BYTES)
        .unwrap();
    let (mut complete_source, complete_hashes, header_bytes) =
        synthetic_route_payload_shard(route_plan.payload_plan());
    assert_eq!(header_bytes.as_slice(), header);
    assert_eq!(complete_hashes.len(), 36);

    let mut incomplete_hashes = complete_hashes.clone();
    incomplete_hashes.pop_first();
    let mut header_prefix = (header.len() as u64).to_le_bytes().to_vec();
    header_prefix.extend_from_slice(header);
    let mut untouched_source = SparseShard {
        file_bytes: SHARD_BYTES,
        position: 0,
        segments: vec![(0, header_prefix)],
    };
    let error = batch
        .read_selected_mxfp4_payloads(
            &mut untouched_source,
            &route_plan,
            source,
            &incomplete_hashes,
            ROUTE_PAYLOAD_BYTES,
        )
        .unwrap_err();
    assert!(error.message.contains("exactly 36 selected tensors"));
    assert_eq!(
        untouched_source.position, 0,
        "no source read before full digest coverage"
    );

    let reads = batch
        .read_selected_mxfp4_payloads(
            &mut complete_source,
            &route_plan,
            source,
            &complete_hashes,
            ROUTE_PAYLOAD_BYTES,
        )
        .unwrap();
    assert_eq!(reads.stamp(), &batch.problem().stamp);
    assert_eq!(reads.attempt(), batch.attempt());
    assert_eq!(reads.route_fingerprint(), route_plan.route_fingerprint());
    assert_eq!(reads.payload_plan().tensors.len(), 36);
    assert_eq!(reads.reads().len(), TOP6.len());
    assert_eq!(
        reads
            .reads()
            .iter()
            .map(|read| read.expert())
            .collect::<Vec<_>>(),
        TOP6
    );

    let residents = reads.into_session_residents(64 * 1024 * 1024).unwrap();
    let route = batch.problem();
    assert_eq!(residents.len(), TOP6.len());
    assert_eq!(
        residents
            .iter()
            .map(voltafabric::deepseek_v41::PinnedMxfp4RankResidentExpert::expert)
            .collect::<Vec<_>>(),
        TOP6
    );
    let input = vec![0.5_f32; route.hidden];
    for value in &input {
        assert_eq!(
            voltafabric::deepseek_v41::cast_bf16_rne_reference(*value).unwrap(),
            *value
        );
    }
    let expected = route.routes[0]
        .iter()
        .try_fold(
            vec![0.0_f32; route.hidden],
            |mut sum, choice| -> Result<_, String> {
                let resident = residents
                    .iter()
                    .find(|resident| resident.expert() == choice.expert)
                    .ok_or_else(|| "oracle resident missing".to_owned())?;
                let expert = resident
                    .owner()
                    .expert(choice.expert)
                    .map_err(|error| error.to_string())?;
                let row = expert
                    .forward_reference(&input, Some(choice.weight as f32))
                    .map_err(|error| error.to_string())?;
                for (target, value) in sum.iter_mut().zip(row) {
                    *target += value;
                }
                Ok(sum)
            },
        )
        .unwrap();
    assert!(expected.iter().any(|value| *value != 0.0));
    let total_artifact_bytes = residents
        .iter()
        .map(voltafabric::deepseek_v41::PinnedMxfp4RankResidentExpert::packed_artifact_bytes)
        .sum::<u64>();
    let preflight = batch
        .preflight_mxfp4_residents(&residents, total_artifact_bytes)
        .unwrap();
    let (preflight, groups) = batch
        .group_mxfp4_residents(preflight, residents, 256 * 1024 * 1024)
        .unwrap();
    assert_eq!(groups.len(), 5);
    let colocated = groups.iter().find(|group| group.rank() == 4).unwrap();
    assert_eq!(
        colocated
            .members()
            .iter()
            .map(|member| member.expert)
            .collect::<Vec<_>>(),
        TOP6[..2]
    );

    let mut sessions = BTreeMap::new();
    let mut loads = Vec::new();
    for group in groups {
        let rank = group.rank();
        let endpoint = Endpoint {
            rank,
            argv: vec![env!("CARGO_BIN_EXE_vf-rank-session").into()],
        };
        let mut session = Session::connect_mxfp4(
            &endpoint,
            Duration::from_secs(10),
            &mxfp4::Producer::default(),
        )
        .unwrap();
        loads.push(
            group
                .load_into_session(&mut session, &preflight, Duration::from_secs(30))
                .unwrap(),
        );
        assert!(sessions.insert(rank, session).is_none());
    }
    let receipt = batch
        .execute_grouped_mxfp4(
            preflight,
            vec![mxfp4::TokenInput {
                token: 0,
                value_bits: input.iter().map(|value| value.to_bits()).collect(),
            }],
            loads,
            &mut sessions,
            Duration::from_secs(90),
        )
        .unwrap();
    assert_eq!(receipt.state(), PinnedMxfp4RouteExecutionState::Completed);
    assert_eq!(receipt.route_experts(), TOP6);
    assert_eq!(receipt.exec_receipts().len(), 5);
    assert_eq!(
        receipt
            .exec_receipts()
            .iter()
            .map(|exec| exec.expected_slots)
            .sum::<usize>(),
        TOP6.len()
    );
    let actual = receipt
        .output_bits()
        .unwrap()
        .first()
        .unwrap()
        .iter()
        .map(|bits| f32::from_bits(*bits))
        .collect::<Vec<_>>();
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs <= 5.0e-4,
        "route-read Session MXFP4 oracle max abs error {max_abs}"
    );
    assert!(actual.iter().any(|value| *value != 0.0));
    for (_, session) in sessions {
        assert_eq!(session.traffic.execs, 1);
        let close = session.close(Duration::from_secs(10));
        assert_eq!(close.bye_execs, Some(1));
        assert!(close.error.is_none(), "{close:?}");
    }
}

#[test]
fn gate_indexed_payload_read_validates_all_route_hashes_before_opening_source() {
    let index = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
    );
    let header = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model-00003.header.314113f8.json"
    );
    let batch = gated_grouped_route_batch();
    let source = PinnedMxfp4PayloadSource {
        index_bytes: index,
        header_bytes: header,
        expected_header_sha256: HEADER_SHA256,
        header_shard_name: "model-00003-of-00048.safetensors",
        source_shard_bytes: SHARD_BYTES,
        max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
    };
    const ROUTE_PAYLOAD_BYTES: u64 = 112_803_840;
    let plan = batch
        .plan_selected_mxfp4_payloads(source, ROUTE_PAYLOAD_BYTES)
        .unwrap();
    let temp = IndexedTempDirectory::new();

    // The directory deliberately contains no shard. An incomplete digest manifest
    // must be rejected before the file-backed reader attempts to open any shard.
    let error = batch
        .read_selected_mxfp4_indexed_payloads(
            temp.path(),
            &plan,
            source,
            &BTreeMap::new(),
            ROUTE_PAYLOAD_BYTES,
        )
        .unwrap_err();
    assert!(error.message.contains("exactly 36 selected tensors"));
}

#[test]
fn indexed_gate_load_rejects_a_nonpinned_index_before_opening_shard() {
    let mut index = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
    )
    .to_vec();
    index[0] ^= 1;
    let temp = IndexedTempDirectory::new();
    let spec = GateShardSpec {
        model_revision: PINNED_MODEL_REVISION.to_owned(),
        layer: 0,
        gate_temperature: 1.0,
        route_scale: ROUTE_SCALE,
        tensor_sha256: GateTensorHashes {
            weight: "0".repeat(64),
            text_bias: "0".repeat(64),
            image_bias: "0".repeat(64),
        },
        max_payload_bytes: GATE_PAYLOAD_BYTES,
    };
    let error = DeepSeekV41Gate::load_indexed(
        temp.path(),
        voltafabric::deepseek_v41_gate::PinnedGateIndexSource {
            index_bytes: &index,
            expected_header_sha256: HEADER_SHA256,
            max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
        },
        &spec,
    )
    .unwrap_err();
    assert!(error.message.contains("wrong digest"));
}

#[test]
fn indexed_gate_load_rejects_malformed_tensor_hashes_before_opening_shard() {
    let index = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
    );
    let temp = IndexedTempDirectory::new();
    let spec = GateShardSpec {
        model_revision: PINNED_MODEL_REVISION.to_owned(),
        layer: 0,
        gate_temperature: 1.0,
        route_scale: ROUTE_SCALE,
        tensor_sha256: GateTensorHashes {
            weight: "not-a-digest".to_owned(),
            text_bias: "0".repeat(64),
            image_bias: "0".repeat(64),
        },
        max_payload_bytes: GATE_PAYLOAD_BYTES,
    };
    let error = DeepSeekV41Gate::load_indexed(
        temp.path(),
        voltafabric::deepseek_v41_gate::PinnedGateIndexSource {
            index_bytes: index,
            expected_header_sha256: HEADER_SHA256,
            max_source_shard_bytes: 8 * 1024 * 1024 * 1024,
        },
        &spec,
    )
    .unwrap_err();
    assert!(error.message.contains("three complete SHA-256 digests"));
}

#[test]
fn pinned_index_colocates_each_layer_gate_and_all_expert_payloads() {
    let index = include_bytes!(
        "../evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
    );
    assert_eq!(voltafabric::sha256::hex(index), PINNED_INDEX_SHA256);
    let parsed: serde_json::Value = serde_json::from_slice(index).unwrap();
    let weight_map = parsed["weight_map"].as_object().unwrap();
    for layer in 0..40 {
        let mut layer_shard = None;
        let mut record = |name: String| {
            let shard = weight_map
                .get(&name)
                .and_then(serde_json::Value::as_str)
                .unwrap_or_else(|| panic!("pinned index is missing {name}"));
            if let Some(previous) = layer_shard {
                assert_eq!(previous, shard, "layer {layer} is spread across shards");
            } else {
                layer_shard = Some(shard);
            }
        };
        for suffix in ["weight", "bias", "bias_vl"] {
            record(format!("layers.{layer}.ffn.gate.{suffix}"));
        }
        for expert in 0..ROUTED_EXPERTS {
            for projection in ["w1", "w2", "w3"] {
                for suffix in ["weight", "scale"] {
                    record(format!(
                        "layers.{layer}.ffn.experts.{expert}.{projection}.{suffix}"
                    ));
                }
            }
        }
        assert!(layer_shard.is_some());
    }
}
