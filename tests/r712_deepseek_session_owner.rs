//! R713/R714 preflight, LOAD and execute a synthetic six-expert DeepSeek route on CPU.
//! The sparse source weights are synthetic; no checkpoint payload is read.
#![cfg(target_os = "linux")]

use std::collections::BTreeMap;
use std::io::{self, Read, Seek, SeekFrom};
use std::time::Duration;
use voltafabric::deepseek_v41::{
    execute_pinned_mxfp4_route, plan_pinned_mxfp4_expert_payloads,
    preflight_pinned_mxfp4_route_residents, read_pinned_mxfp4_rank_expert, Mxfp4ExpertNamespace,
    PinnedExpertPayloadRequest, PinnedMxfp4RankExpertReadRequest, PinnedMxfp4RouteExecutionState,
    REVISION,
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
fn colocated_top6_is_refused_before_the_owner_set_or_any_worker_exists() {
    let route = route_problem();
    let error = preflight_pinned_mxfp4_route_residents(&route, 29, &[], 1).unwrap_err();
    assert!(error
        .message
        .contains("one distinct rank per selected expert"));
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
    let residents = TOP6
        .iter()
        .enumerate()
        .map(|(index, &expert)| resident_for_expert(&route, expert, 4 + index as u32))
        .collect::<Vec<_>>();
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
