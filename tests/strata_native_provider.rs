#![cfg(feature = "native-ggml-cpu")]
use std::io::{Read, Seek, SeekFrom};
use voltafabric::{
    native_ggml_cpu::{Binding, Resident},
    strata_native_layout::NativeLayout,
};
const INDEX: &str = include_str!("../fixtures/strata_sc117_native_experts_v3.txt");
const INDEX_SHA: &str = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24";
const BLOB_SHA: &str = "50eff9144c635807dfdc0cafd1c3bb978d0202d537ae892ec8ebed90dc13e172";
const SIZE: u64 = 50_292_326_400;
struct Zeros(u64);
impl Seek for Zeros {
    fn seek(&mut self, s: SeekFrom) -> std::io::Result<u64> {
        self.0 = match s {
            SeekFrom::Start(n) => n,
            SeekFrom::End(n) => (SIZE as i64 + n) as u64,
            SeekFrom::Current(n) => (self.0 as i64 + n) as u64,
        };
        Ok(self.0)
    }
}
impl Read for Zeros {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        b.fill(0);
        self.0 += b.len() as u64;
        Ok(b.len())
    }
}
fn binding() -> Binding {
    Binding {
        model_revision: "model-test".into(),
        rank: 2,
        layer: 0,
        expert: 0,
        local: 0,
        epoch: 3,
        index_sha256: INDEX_SHA.into(),
        blob_sha256: BLOB_SHA.into(),
    }
}
#[test]
fn verified_native_expert_loads_once_and_executes_twice_in_rust() {
    let layout = NativeLayout::parse_sc117_v3(INDEX, SIZE).unwrap();
    let blob = layout
        .read_verified_expert(&mut Zeros(0), 0, 0, INDEX_SHA, BLOB_SHA)
        .unwrap();
    let mut resident = Resident::load(&layout, blob, binding()).unwrap();
    assert_eq!(resident.binding().expert, 0);
    let x = vec![0.125; 2560];
    let first = resident.forward(&x).unwrap();
    let second = resident.forward(&x).unwrap();
    assert_eq!(first, vec![0.0; 2560]);
    assert_eq!(first, second);
    assert!(resident.forward(&[f32::NAN; 2560]).is_err());
}
#[test]
fn native_resident_refuses_changed_identity_or_mutated_payload() {
    let layout = NativeLayout::parse_sc117_v3(INDEX, SIZE).unwrap();
    let mut blob = layout
        .read_verified_expert(&mut Zeros(0), 0, 0, INDEX_SHA, BLOB_SHA)
        .unwrap();
    blob.bytes[0] = 1;
    assert!(Resident::load(&layout, blob, binding()).is_err());
    let blob = layout
        .read_verified_expert(&mut Zeros(0), 0, 0, INDEX_SHA, BLOB_SHA)
        .unwrap();
    let mut wrong = binding();
    wrong.expert = 1;
    assert!(Resident::load(&layout, blob, wrong).is_err());
}

struct Pattern {
    pos: u64,
    data: Vec<u8>,
}
impl Seek for Pattern {
    fn seek(&mut self, s: SeekFrom) -> std::io::Result<u64> {
        self.pos = match s {
            SeekFrom::Start(n) => n,
            SeekFrom::End(n) => (SIZE as i64 + n) as u64,
            SeekFrom::Current(n) => (self.pos as i64 + n) as u64,
        };
        Ok(self.pos)
    }
}
impl Read for Pattern {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let start = self.pos as usize;
        if start >= self.data.len() {
            return Ok(0);
        }
        let n = buf.len().min(self.data.len() - start);
        buf[..n].copy_from_slice(&self.data[start..start + n]);
        self.pos += n as u64;
        Ok(n)
    }
}
#[test]
fn native_owner_processes_nonzero_verified_synthetic_expert() {
    let layout = NativeLayout::parse_sc117_v3(INDEX, SIZE).unwrap();
    let mut data = vec![0u8; 2_176_000];
    for i in (0..1_254_400).step_by(98) {
        data[i + 1] = 0x28;
    }
    for i in (1_254_400..2_176_000).step_by(18) {
        data[i + 1] = 0x28;
    }
    let digest = "2c7dcdcf2992bb6127b86407fb43abe67f4ba202acbe005a9bc67d7df9e072b0";
    let blob = layout
        .read_verified_expert(&mut Pattern { pos: 0, data }, 0, 0, INDEX_SHA, digest)
        .unwrap();
    let mut identity = binding();
    identity.blob_sha256 = digest.into();
    let mut owner = Resident::load(&layout, blob, identity).unwrap();
    let out = owner.forward(&vec![0.125; 2560]).unwrap();
    assert!(out.iter().all(|x| x.is_finite()));
    assert!(out.iter().any(|x| x.abs() > 1e-8));
    assert_eq!(out, owner.forward(&vec![0.125; 2560]).unwrap());
    use voltafabric::{
        completion,
        moe::{Slot, Stamp},
        native_ggml_cpu::RankWorker,
        routing::{RankDispatch, RankSlot},
    };
    let reference = owner.forward(&vec![0.001; 2560]).unwrap();
    let mut worker = RankWorker::new(vec![owner]).unwrap();
    let mut dispatch = RankDispatch {
        stamp: Stamp {
            run_id: "r382-overflow".into(),
            model_revision: "model-test".into(),
            plan_id: "plan".into(),
            layer: 0,
            batch: 0,
            epoch: 3,
        },
        rank: 2,
        hidden: 2560,
        tokens: 1,
        resident_order: vec![0],
        slots: vec![RankSlot {
            slot: Slot {
                token: 0,
                slot: 0,
                expert: 0,
                rank: 2,
                local: 0,
                weight: 0.5,
            },
            physical_expert: 0,
        }],
    };
    let safe = vec![0.001_f32; 2560];
    let raw = worker.execute(&dispatch, &[(0, safe)], 1).unwrap();
    assert!(raw.fp16_bits.iter().any(|&v| v != 0));
    assert_eq!(
        raw.fp16_bits,
        reference
            .into_iter()
            .map(|v| completion::encode_fp16(f64::from(v)).unwrap())
            .collect::<Vec<_>>()
    );
    assert!(raw
        .fp16_bits
        .iter()
        .all(|&v| completion::decode_fp16(v).is_ok()));
    dispatch.stamp.batch = 1;
    assert_eq!(
        worker
            .execute(&dispatch, &[(0, vec![0.125; 2560])], 2)
            .unwrap_err()
            .code,
        "FP16_OVERFLOW"
    );
    dispatch.stamp.batch = 2;
    assert!(worker
        .execute(&dispatch, &[(0, vec![0.001; 2560])], 3)
        .is_err());
}

#[test]
fn native_fleet_routes_two_residents_into_fp16_collector_without_replay() {
    use std::collections::BTreeMap;
    use voltafabric::{
        completion,
        moe::{Choice, Registry, Replica, Stamp},
        native_ggml_cpu::Fleet,
        routing,
    };
    let layout = NativeLayout::parse_sc117_v3(INDEX, SIZE).unwrap();
    let mut data = vec![0u8; 2_176_000];
    for i in (0..1_254_400).step_by(98) {
        data[i + 1] = 0x28;
    }
    for i in (1_254_400..2_176_000).step_by(18) {
        data[i + 1] = 0x28;
    }
    let digest = "2c7dcdcf2992bb6127b86407fb43abe67f4ba202acbe005a9bc67d7df9e072b0";
    // Each synthetic span returns the same bytes, while retaining distinct expert identity.
    struct Repeat {
        pos: u64,
        bytes: Vec<u8>,
    }
    impl Seek for Repeat {
        fn seek(&mut self, s: SeekFrom) -> std::io::Result<u64> {
            self.pos = match s {
                SeekFrom::Start(n) => n,
                SeekFrom::End(n) => (SIZE as i64 + n) as u64,
                SeekFrom::Current(n) => (self.pos as i64 + n) as u64,
            };
            Ok(self.pos)
        }
    }
    impl Read for Repeat {
        fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
            let offset = (self.pos % 2_176_000) as usize;
            let n = b.len().min(self.bytes.len() - offset);
            b[..n].copy_from_slice(&self.bytes[offset..offset + n]);
            self.pos += n as u64;
            Ok(n)
        }
    }
    let mut owners = Vec::new();
    for e in 0..2 {
        let blob = layout
            .read_verified_expert(
                &mut Repeat {
                    pos: 0,
                    bytes: data.clone(),
                },
                0,
                e,
                INDEX_SHA,
                digest,
            )
            .unwrap();
        let mut b = binding();
        b.expert = e as u32;
        b.rank = 2 + e as u32;
        b.blob_sha256 = digest.into();
        owners.push(Resident::load(&layout, blob, b).unwrap());
    }
    let x = vec![0.001_f32; 2560];
    let reference = owners[0].forward(&x).unwrap();
    let p = routing::RoutingProblem {
        hidden: 2560,
        experts: vec![0, 1],
        routes: vec![vec![
            Choice {
                expert: 0,
                weight: 0.25,
            },
            Choice {
                expert: 1,
                weight: 0.75,
            },
        ]],
        registry: Registry {
            epoch: 3,
            copies: vec![
                Replica {
                    expert: 0,
                    rank: 2,
                    local: 0,
                },
                Replica {
                    expert: 1,
                    rank: 3,
                    local: 0,
                },
            ],
        },
        stamp: Stamp {
            run_id: "r381-native-fleet".into(),
            model_revision: "model-test".into(),
            plan_id: "plan".into(),
            layer: 0,
            batch: 0,
            epoch: 3,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(2, vec![0]), (3, vec![0])]),
    };
    let mut fleet = Fleet::new(owners).unwrap();
    let mut stale = p.clone();
    stale.stamp.epoch += 1;
    assert!(fleet
        .run_fp16_collected(&stale, std::slice::from_ref(&x), 1)
        .is_err());
    let result = fleet
        .run_fp16_collected(&p, std::slice::from_ref(&x), 1)
        .unwrap();
    assert_eq!(result.attempt, 1);
    assert_eq!(result.stamp, p.stamp);
    assert_eq!(result.completed_slots, 2);
    for (actual, value) in result.output[0].iter().zip(reference) {
        let decoded =
            completion::decode_fp16(completion::encode_fp16(f64::from(value)).unwrap()).unwrap();
        assert_eq!(
            actual.to_bits(),
            (0.25 * decoded + 0.75 * decoded).to_bits()
        );
    }
    assert!(fleet
        .run_fp16_collected(&p, std::slice::from_ref(&x), 2)
        .is_err());
    let mut next = p.clone();
    next.stamp.batch += 1;
    assert!(fleet
        .run_fp16_collected(&next, std::slice::from_ref(&x), 1)
        .is_err());
    assert!(fleet.run_fp16_collected(&next, &[x], 2).is_ok());

    let mut overflow_owners = Vec::new();
    for e in 0..2 {
        let blob = layout
            .read_verified_expert(
                &mut Repeat {
                    pos: 0,
                    bytes: data.clone(),
                },
                0,
                e,
                INDEX_SHA,
                digest,
            )
            .unwrap();
        let mut b = binding();
        b.expert = e as u32;
        b.rank = 2 + e as u32;
        b.blob_sha256 = digest.into();
        overflow_owners.push(Resident::load(&layout, blob, b).unwrap());
    }
    let mut overflow = Fleet::new(overflow_owners).unwrap();
    let mut p2 = p.clone();
    p2.stamp.run_id = "r381-native-overflow".into();
    assert_eq!(
        overflow
            .run_fp16_collected(&p2, &[vec![0.125; 2560]], 1)
            .unwrap_err()
            .code,
        "FP16_OVERFLOW"
    );
    p2.stamp.batch += 1;
    assert!(overflow
        .run_fp16_collected(&p2, &[vec![0.001; 2560]], 2)
        .is_err());
}

#[test]
fn native_rank_workers_produce_atomic_compact_rows_for_existing_collector() {
    use std::collections::BTreeMap;
    use voltafabric::{
        completion,
        moe::{self, Choice, Registry, Replica, Stamp},
        native_ggml_cpu::RankWorker,
        routing,
    };
    let layout = NativeLayout::parse_sc117_v3(INDEX, SIZE).unwrap();
    let mut workers = Vec::new();
    for expert in 0..2 {
        let blob = layout
            .read_verified_expert(&mut Zeros(0), 0, expert, INDEX_SHA, BLOB_SHA)
            .unwrap();
        let mut b = binding();
        b.expert = expert as u32;
        b.rank = 2 + expert as u32;
        workers.push(RankWorker::new(vec![Resident::load(&layout, blob, b).unwrap()]).unwrap());
    }
    let p = routing::RoutingProblem {
        hidden: 2560,
        experts: vec![0, 1],
        routes: vec![vec![
            Choice {
                expert: 0,
                weight: 0.25,
            },
            Choice {
                expert: 1,
                weight: 0.75,
            },
        ]],
        registry: Registry {
            epoch: 3,
            copies: vec![
                Replica {
                    expert: 0,
                    rank: 2,
                    local: 0,
                },
                Replica {
                    expert: 1,
                    rank: 3,
                    local: 0,
                },
            ],
        },
        stamp: Stamp {
            run_id: "r382-rank".into(),
            model_revision: "model-test".into(),
            plan_id: "plan".into(),
            layer: 0,
            batch: 0,
            epoch: 3,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(2, vec![0]), (3, vec![0])]),
    };
    let dispatch = routing::dispatch(&p).unwrap();
    let inputs = vec![(0, vec![0.001; 2560])];
    let mut wrong = dispatch.ranks[0].clone();
    wrong.slots[0].physical_expert = 1;
    assert!(workers[0].execute(&wrong, &inputs, 1).is_err());
    let mut collector = moe::Collector::from_routing(&p).unwrap();
    let first = workers[0].execute(&dispatch.ranks[0], &inputs, 1).unwrap();
    assert!(completion::accept_routed_output(2, first.clone(), &mut collector).is_err());
    completion::accept_routed_output(1, first, &mut collector).unwrap();
    assert!(workers[0].execute(&dispatch.ranks[0], &inputs, 2).is_err());
    let second = workers[1].execute(&dispatch.ranks[1], &inputs, 1).unwrap();
    completion::accept_routed_output(1, second, &mut collector).unwrap();
    assert_eq!(collector.finish().unwrap(), vec![vec![0.0; 2560]]);
    let mut other = dispatch.ranks[0].clone();
    other.stamp.batch += 1;
    other.stamp.plan_id = "other".into();
    assert!(workers[0].execute(&other, &inputs, 2).is_err());
}
