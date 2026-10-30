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

#[test]
fn native_ggml_sessions_execute_two_actual_child_processes() {
    use std::{collections::BTreeMap, time::Duration};
    use voltafabric::{
        completion,
        moe::{Choice, Registry, Replica, Stamp},
        routing,
        session::{Endpoint, Session},
    };
    let timeout = Duration::from_secs(30);
    let ledger_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".runs")
        .join(format!(
            "r384-ledger-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    std::fs::create_dir(&ledger_path).unwrap();
    let ledger = voltafabric::native_ledger::NativeBatchLedger::open(&ledger_path).unwrap();
    let mut data = vec![0u8; 2_176_000];
    for i in (0..1_254_400).step_by(98) {
        data[i + 1] = 0x28;
    }
    for i in (1_254_400..2_176_000).step_by(18) {
        data[i + 1] = 0x28;
    }
    let digest = "2c7dcdcf2992bb6127b86407fb43abe67f4ba202acbe005a9bc67d7df9e072b0";
    let layout = NativeLayout::parse_sc117_v3(INDEX, SIZE).unwrap();
    let mut b = binding();
    b.blob_sha256 = digest.into();
    let blob = layout
        .read_verified_expert(
            &mut Pattern {
                pos: 0,
                data: data.clone(),
            },
            0,
            0,
            INDEX_SHA,
            digest,
        )
        .unwrap();
    let mut reference = Resident::load(&layout, blob, b.clone()).unwrap();
    let x = vec![0.001_f32; 2560];
    let raw_reference = reference.forward(&x).unwrap();
    let expected = raw_reference
        .iter()
        .map(|&v| completion::decode_fp16(completion::encode_fp16(f64::from(v)).unwrap()).unwrap())
        .collect::<Vec<_>>();
    let mut sessions = Vec::new();
    for expert in 0..2 {
        let rank = 2 + expert;
        let endpoint = Endpoint {
            rank,
            argv: vec![env!("CARGO_BIN_EXE_vf-rank-session").to_owned()],
        };
        let mut session = Session::connect(&endpoint, timeout).unwrap();
        let mut binding = b.clone();
        binding.rank = rank;
        binding.expert = expert;
        session
            .load_native_ggml(&binding, INDEX, SIZE, &data, timeout)
            .unwrap();
        session.seal_native_ggml(1, timeout).unwrap();
        sessions.push(session);
    }
    assert_ne!(sessions[0].pid(), sessions[1].pid());
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
            run_id: format!(
                "r482-two-processes-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ),
            model_revision: "model-test".into(),
            plan_id: "plan".into(),
            layer: 0,
            batch: 0,
            epoch: 3,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(2, vec![0]), (3, vec![0])]),
    };
    let mut consumed = p.clone();
    consumed.stamp.batch = 99;
    ledger.reserve(&consumed.stamp, 123).unwrap();
    let sent: Vec<_> = sessions.iter().map(|s| s.traffic.bytes_sent).collect();
    assert_eq!(
        Session::exec_native_ggml_batch(
            &mut sessions,
            &consumed,
            std::slice::from_ref(&x),
            &ledger,
            timeout
        )
        .unwrap_err()
        .code,
        "LEDGER_RESERVATION"
    );
    assert_eq!(
        sent,
        sessions
            .iter()
            .map(|s| s.traffic.bytes_sent)
            .collect::<Vec<_>>()
    );
    let output = Session::exec_native_ggml_batch(
        &mut sessions,
        &p,
        std::slice::from_ref(&x),
        &ledger,
        timeout,
    )
    .unwrap();
    for (actual, value) in output[0].iter().zip(&expected) {
        assert_eq!(actual.to_bits(), (0.25 * value + 0.75 * value).to_bits());
    }
    assert_eq!(
        sessions
            .iter()
            .map(|s| s.traffic.native_ggml_raw_execs)
            .collect::<Vec<_>>(),
        vec![1, 1]
    );
    // Simulate an older rank: the next request must use the legacy JSON path.
    let raw_capability = voltafabric::session::CAPABILITY_NATIVE_GGML_RAW_F32.to_owned();
    assert!(sessions
        .iter()
        .all(|s| s.hello.capabilities.contains(&raw_capability)));
    sessions[1]
        .hello
        .capabilities
        .retain(|c| c != &raw_capability);
    let mut reserved = p.clone();
    reserved.stamp.batch = 101;
    let ticket_attempt = u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    )
    .unwrap();
    let ticket = ledger
        .reserve_ticket(&reserved.stamp, ticket_attempt)
        .unwrap();
    let collected = Session::exec_native_ggml_reserved(
        &mut sessions,
        &reserved,
        std::slice::from_ref(&x),
        &ledger,
        ticket,
        timeout,
    )
    .unwrap();
    assert_eq!(collected.stamp(), &reserved.stamp);
    assert_eq!(collected.attempt(), ticket_attempt);
    sessions[1].hello.capabilities.push(raw_capability);
    assert_eq!(
        sessions
            .iter()
            .map(|s| s.traffic.native_ggml_raw_execs)
            .collect::<Vec<_>>(),
        vec![2, 1]
    );
    assert_eq!(
        sessions.iter().map(|s| s.traffic.execs).collect::<Vec<_>>(),
        vec![2, 2]
    );
    let mut one_rank = p.clone();
    one_rank.stamp.batch = 103;
    one_rank.routes = vec![vec![Choice {
        expert: 0,
        weight: 1.0,
    }]];
    let sparse_output = Session::exec_native_ggml_batch(
        &mut sessions,
        &one_rank,
        std::slice::from_ref(&x),
        &ledger,
        timeout,
    )
    .unwrap();
    assert_eq!(sparse_output[0], expected);
    assert_eq!(
        sessions.iter().map(|s| s.traffic.execs).collect::<Vec<_>>(),
        vec![3, 2]
    );
    assert_eq!(
        sessions
            .iter()
            .map(|s| s.traffic.native_ggml_raw_execs)
            .collect::<Vec<_>>(),
        vec![3, 1]
    );
    for (actual, value) in collected.rows()[0].iter().zip(&expected) {
        assert_eq!(actual.to_bits(), (0.25 * value + 0.75 * value).to_bits());
    }
    assert!(ledger
        .reserve_ticket(&reserved.stamp, ticket_attempt + 1)
        .is_err());
    let other_path = ledger_path.with_extension("other");
    std::fs::create_dir(&other_path).unwrap();
    let other = voltafabric::native_ledger::NativeBatchLedger::open(&other_path).unwrap();
    let mut wrong_ledger = p.clone();
    wrong_ledger.stamp.batch = 102;
    let ticket = other
        .reserve_ticket(&wrong_ledger.stamp, ticket_attempt + 2)
        .unwrap();
    let sent: Vec<_> = sessions.iter().map(|s| s.traffic.bytes_sent).collect();
    assert!(Session::exec_native_ggml_reserved(
        &mut sessions,
        &wrong_ledger,
        std::slice::from_ref(&x),
        &ledger,
        ticket,
        timeout
    )
    .is_err());
    assert_eq!(
        sent,
        sessions
            .iter()
            .map(|s| s.traffic.bytes_sent)
            .collect::<Vec<_>>()
    );
    assert!(Session::exec_native_ggml_batch(
        &mut sessions,
        &p,
        std::slice::from_ref(&x),
        &ledger,
        timeout
    )
    .is_err());
    let mut routed = p.clone();
    routed.stamp.batch = 3;
    let make_plan = || {
        voltafabric::strata_ingress::admit_strata_unweighted_f64(
            &routed,
            &routed.stamp,
            0,
            1,
            3,
            &x,
            &[0, 2, 1],
            &[-1, 0, -1],
            &[0, 1, 2],
        )
        .unwrap()
    };
    let mut short = vec![9.0; 1];
    assert!(make_plan()
        .execute_native(&mut sessions, &ledger, timeout, &mut short)
        .is_err());
    let mut destination = vec![9.0; 3 * 2560];
    make_plan()
        .execute_native(&mut sessions, &ledger, timeout, &mut destination)
        .unwrap();
    for i in 0..2560 {
        assert_eq!(destination[i].to_bits(), (expected[i] as f32).to_bits());
        assert_eq!(destination[2560 + i], 9.0);
        assert_eq!(
            destination[5120 + i].to_bits(),
            (expected[i] as f32).to_bits()
        );
    }
    let mut f32_plan = p.clone();
    f32_plan.stamp.batch = 4;
    let admitted = voltafabric::strata_ingress::admit_strata_unweighted_f64(
        &f32_plan,
        &f32_plan.stamp,
        0,
        1,
        3,
        &x,
        &[0, 2, 1],
        &[-1, 0, -1],
        &[0, 1, 2],
    )
    .unwrap();
    let capability = voltafabric::session::CAPABILITY_NATIVE_GGML_UNWEIGHTED_F32.to_owned();
    sessions[1].hello.capabilities.retain(|x| x != &capability);
    let before: Vec<_> = sessions.iter().map(|s| s.traffic.bytes_sent).collect();
    let mut refused = vec![9.0; 3 * 2560];
    assert!(admitted
        .execute_native_fp32(&mut sessions, &ledger, timeout, &mut refused)
        .is_err());
    assert!(refused.iter().all(|v| *v == 9.0));
    assert_eq!(
        before,
        sessions
            .iter()
            .map(|s| s.traffic.bytes_sent)
            .collect::<Vec<_>>()
    );
    sessions[1].hello.capabilities.push(capability);
    let admitted = voltafabric::strata_ingress::admit_strata_unweighted_f64(
        &f32_plan,
        &f32_plan.stamp,
        0,
        1,
        3,
        &x,
        &[0, 2, 1],
        &[-1, 0, -1],
        &[0, 1, 2],
    )
    .unwrap();
    let mut exact = vec![9.0_f32; 3 * 2560];
    admitted
        .execute_native_fp32(&mut sessions, &ledger, timeout, &mut exact)
        .unwrap();
    let mut fp16_difference = false;
    for i in 0..2560 {
        assert_eq!(exact[i].to_bits(), raw_reference[i].to_bits());
        assert_eq!(exact[2560 + i], 9.0);
        assert_eq!(exact[5120 + i].to_bits(), raw_reference[i].to_bits());
        fp16_difference |= exact[i].to_bits() != (expected[i] as f32).to_bits();
    }
    assert!(
        fp16_difference,
        "test must distinguish FP32 from lossy FP16"
    );
    let large = vec![0.125_f32; 2560];
    let large_reference = reference.forward(&large).unwrap();
    let mut f32_large = p.clone();
    f32_large.stamp.batch = 5;
    let plan = voltafabric::strata_ingress::admit_strata_unweighted_f64(
        &f32_large,
        &f32_large.stamp,
        0,
        1,
        2,
        &large,
        &[0, 1],
        &[-1, -1],
        &[0, 1],
    )
    .unwrap();
    let mut large_output = vec![0.0; 2 * 2560];
    plan.execute_native_fp32(&mut sessions, &ledger, timeout, &mut large_output)
        .unwrap();
    assert!(large_reference.iter().any(|v| v.abs() > 65504.0));
    for i in 0..2560 {
        assert_eq!(large_output[i].to_bits(), large_reference[i].to_bits());
        assert_eq!(
            large_output[2560 + i].to_bits(),
            large_reference[i].to_bits()
        );
    }
    let mut overflow = p.clone();
    overflow.stamp.batch = 1;
    let mut values = x.clone();
    values.extend(vec![0.125; 2560]);
    let plan = voltafabric::strata_ingress::admit_strata_unweighted_f64(
        &overflow,
        &overflow.stamp,
        0,
        2,
        1,
        &values,
        &[0, 1],
        &[-1, -1],
        &[0, 1],
    )
    .unwrap();
    let mut unchanged = vec![9.0; 2 * 2560];
    assert_eq!(
        plan.execute_native(&mut sessions, &ledger, timeout, &mut unchanged)
            .unwrap_err()
            .code,
        "FP16_OVERFLOW"
    );
    assert!(unchanged.iter().all(|v| *v == 9.0));
    assert!(sessions.iter().all(|s| !s.is_usable()));
    let mut after = p.clone();
    after.stamp.batch = 2;
    assert!(
        Session::exec_native_ggml_batch(&mut sessions, &after, &[x], &ledger, timeout).is_err()
    );
    for session in sessions {
        let _ = session.close(timeout);
    }
}

#[test]
fn persistent_native_bridge_loads_sparse_pack_and_routes_real_child_sessions() {
    use std::{
        collections::BTreeMap,
        io::{BufRead, BufReader, Read, Seek, SeekFrom, Write},
        process::{Child, ChildStdin, Command, Stdio},
        time::{SystemTime, UNIX_EPOCH},
    };
    use voltafabric::{
        moe::{Choice, Registry, Replica, Stamp},
        routing,
        session::Endpoint,
        strata_native_bridge::{Init, Reply, Request},
    };
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".runs")
        .join(format!(
            "r388-bridge-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    std::fs::create_dir(&base).unwrap();
    let ledger = base.join("ledger");
    std::fs::create_dir(&ledger).unwrap();
    let pack_path = base.join("synthetic-sparse-pack.bin");
    let mut pack = std::fs::File::create(&pack_path).unwrap();
    pack.set_len(SIZE).unwrap();
    let mut data = vec![0u8; 2_176_000];
    for i in (0..1_254_400).step_by(98) {
        data[i + 1] = 0x28;
    }
    for i in (1_254_400..2_176_000).step_by(18) {
        data[i + 1] = 0x28;
    }
    pack.seek(SeekFrom::Start(0)).unwrap();
    pack.write_all(&data).unwrap();
    pack.write_all(&data).unwrap();
    pack.sync_all().unwrap();
    let digest = "2c7dcdcf2992bb6127b86407fb43abe67f4ba202acbe005a9bc67d7df9e072b0";
    let mut b = binding();
    b.blob_sha256 = digest.into();
    let layout = NativeLayout::parse_sc117_v3(INDEX, SIZE).unwrap();
    let blob = layout
        .read_verified_expert(
            &mut Pattern {
                pos: 0,
                data: data.clone(),
            },
            0,
            0,
            INDEX_SHA,
            digest,
        )
        .unwrap();
    let mut reference = Resident::load(&layout, blob, b.clone()).unwrap();
    let x = vec![0.001_f32; 2560];
    let expected = reference.forward(&x).unwrap();
    let mut bindings = Vec::new();
    let mut endpoints = Vec::new();
    for expert in 0..2 {
        let mut item = b.clone();
        item.expert = expert;
        item.rank = 2 + expert;
        bindings.push(item);
        endpoints.push(Endpoint {
            rank: 2 + expert,
            argv: vec![env!("CARGO_BIN_EXE_vf-rank-session").to_owned()],
        });
    }
    let template = routing::RoutingProblem {
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
            run_id: "r388-local-bridge".into(),
            model_revision: "model-test".into(),
            plan_id: "plan".into(),
            layer: 0,
            batch: 0,
            epoch: 3,
        },
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(2, vec![0]), (3, vec![0])]),
    };
    let init = Init {
        template,
        model_experts: vec![0, 1, 2],
        index: INDEX.into(),
        pack_path: pack_path.to_str().unwrap().into(),
        ledger_path: ledger.to_str().unwrap().into(),
        endpoints,
        bindings,
        resident_payload_limit: None,
        timeout_ms: 30_000,
    };
    fn send(w: &mut ChildStdin, value: &impl serde::Serialize) {
        serde_json::to_writer(&mut *w, value).unwrap();
        w.write_all(b"\n").unwrap();
        w.flush().unwrap();
    }
    fn recv(r: &mut BufReader<std::process::ChildStdout>) -> Reply {
        let mut line = String::new();
        assert!(r.read_line(&mut line).unwrap() > 0);
        serde_json::from_str(&line).unwrap()
    }
    fn start(init: &Init) -> (Child, ChildStdin, BufReader<std::process::ChildStdout>) {
        let mut p = Command::new(env!("CARGO_BIN_EXE_vf-strata-native-bridge"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut input = p.stdin.take().unwrap();
        let mut output = BufReader::new(p.stdout.take().unwrap());
        send(&mut input, init);
        assert!(matches!(recv(&mut output),Reply::Ready {ranks} if ranks==vec![2,3]));
        (p, input, output)
    }
    let request = |batch| Request::Batch {
        batch,
        tokens: 1,
        top_k: 3,
        inputs: x.clone(),
        ids: vec![0, 2, 1],
        kinds: vec![-1, 0, -1],
    };
    let (mut child, mut input, mut output) = start(&init);
    for batch in [12, 13] {
        send(&mut input, &request(batch));
        match recv(&mut output) {
            Reply::Completed {
                batch: got,
                cpu_rows,
            } => {
                assert_eq!(got, batch);
                assert_eq!(cpu_rows.len(), 2);
                assert_eq!(cpu_rows[0].position, 0);
                assert_eq!(cpu_rows[1].position, 2);
                for row in cpu_rows {
                    assert_eq!(
                        row.value.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    send(&mut input, &request(12));
    assert!(matches!(recv(&mut output), Reply::Failed { .. }));
    drop(input);
    assert!(!child.wait().unwrap().success());
    let (mut child, mut input, mut output) = start(&init);
    send(&mut input, &request(12));
    assert!(
        matches!(recv(&mut output),Reply::Failed { error } if error.code=="LEDGER_RESERVATION")
    );
    drop(input);
    assert!(!child.wait().unwrap().success());
    let (mut child, mut input, mut output) = start(&init);
    send(&mut input, &Request::Shutdown);
    assert!(matches!(recv(&mut output), Reply::Closed));
    drop(input);
    assert!(child.wait().unwrap().success());
    let (mut child, mut input, mut output) = start(&init);
    send(&mut input, &Request::EnterBinary);
    assert!(matches!(recv(&mut output), Reply::BinaryReady));
    let mut request = Vec::new();
    request.extend_from_slice(b"VFB1");
    request.extend_from_slice(&1_u32.to_le_bytes());
    request.extend_from_slice(&14_u64.to_le_bytes());
    request.extend_from_slice(&1_u32.to_le_bytes());
    request.extend_from_slice(&3_u32.to_le_bytes());
    for &v in &x {
        request.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    for v in [0_i32, 2, 1, -1, 0, -1] {
        request.extend_from_slice(&v.to_le_bytes());
    }
    input.write_all(&request).unwrap();
    input.flush().unwrap();
    let mut header = [0_u8; 24];
    output.read_exact(&mut header).unwrap();
    assert_eq!(&header[..4], b"VFR1");
    assert_eq!(u32::from_le_bytes(header[4..8].try_into().unwrap()), 0);
    assert_eq!(u64::from_le_bytes(header[8..16].try_into().unwrap()), 14);
    assert_eq!(u32::from_le_bytes(header[16..20].try_into().unwrap()), 2);
    for pos in [0_u32, 2] {
        let mut b = [0u8; 4];
        output.read_exact(&mut b).unwrap();
        assert_eq!(u32::from_le_bytes(b), pos);
        for v in &expected {
            output.read_exact(&mut b).unwrap();
            assert_eq!(u32::from_le_bytes(b), v.to_bits());
        }
    }
    let mut shutdown = Vec::from(&b"VFB1"[..]);
    shutdown.extend_from_slice(&2_u32.to_le_bytes());
    shutdown.extend_from_slice(&[0; 16]);
    input.write_all(&shutdown).unwrap();
    input.flush().unwrap();
    output.read_exact(&mut header).unwrap();
    assert_eq!(u32::from_le_bytes(header[4..8].try_into().unwrap()), 2);
    drop(input);
    assert!(child.wait().unwrap().success());
    // A miswired client must be rejected before reserving or executing a batch.
    let (mut child, mut input, mut output) = start(&init);
    send(&mut input, &Request::EnterBinary);
    assert!(matches!(recv(&mut output), Reply::BinaryReady));
    let mut wrong_layer = Vec::from(&b"VFB2"[..]);
    wrong_layer.extend_from_slice(&1_u32.to_le_bytes());
    wrong_layer.extend_from_slice(&90_u64.to_le_bytes());
    wrong_layer.extend_from_slice(&1_u32.to_le_bytes());
    wrong_layer.extend_from_slice(&3_u32.to_le_bytes());
    wrong_layer.extend_from_slice(&7_u32.to_le_bytes());
    input.write_all(&wrong_layer).unwrap();
    input.flush().unwrap();
    let mut rejected = [0_u8; 28];
    output.read_exact(&mut rejected).unwrap();
    assert_eq!(&rejected[..4], b"VFR2");
    assert_eq!(u32::from_le_bytes(rejected[4..8].try_into().unwrap()), 1);
    assert_eq!(u32::from_le_bytes(rejected[24..28].try_into().unwrap()), 0);
    drop(input);
    assert!(!child.wait().unwrap().success());
    let (mut child, mut input, mut output) = start(&init);
    send(
        &mut input,
        &Request::Batch {
            batch: 90,
            tokens: 1,
            top_k: 3,
            inputs: x.clone(),
            ids: vec![0, 2, 1],
            kinds: vec![-1, 0, -1],
        },
    );
    assert!(matches!(
        recv(&mut output),
        Reply::Completed { batch: 90, .. }
    ));
    send(&mut input, &Request::Shutdown);
    assert!(matches!(recv(&mut output), Reply::Closed));
    drop(input);
    assert!(child.wait().unwrap().success());
    // Five resident experts in one real worker, with an explicit byte budget.
    let mut expanded: Init = serde_json::from_str(&serde_json::to_string(&init).unwrap()).unwrap();
    expanded.model_experts = (0..5).collect();
    expanded.template.experts = (0..5).collect();
    expanded.template.registry.copies = (0..5)
        .map(|expert| Replica {
            expert,
            rank: 2,
            local: expert,
        })
        .collect();
    expanded.template.resident_order = BTreeMap::from([(2, (0..5).collect())]);
    expanded.endpoints.retain(|endpoint| endpoint.rank == 2);
    expanded.bindings = (0..5)
        .map(|expert| {
            let mut item = b.clone();
            item.expert = expert;
            item.local = expert;
            if expert >= 2 {
                item.blob_sha256 = BLOB_SHA.into();
            }
            item
        })
        .collect();
    expanded.resident_payload_limit = Some(5 * 2_176_000);
    let mut child = Command::new(env!("CARGO_BIN_EXE_vf-strata-native-bridge"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap());
    send(&mut input, &expanded);
    assert!(matches!(recv(&mut output), Reply::Ready { ranks } if ranks == vec![2]));
    send(
        &mut input,
        &Request::Batch {
            batch: 101,
            tokens: 1,
            top_k: 2,
            inputs: x.clone(),
            ids: vec![0, 4],
            kinds: vec![-1, -1],
        },
    );
    match recv(&mut output) {
        Reply::Completed {
            batch: 101,
            cpu_rows,
        } => {
            assert_eq!(cpu_rows.len(), 2);
            assert_eq!(
                cpu_rows[0]
                    .value
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert!(cpu_rows[1].value.iter().all(|v| *v == 0.0));
        }
        other => panic!("five-expert batch failed: {other:?}"),
    }
    send(&mut input, &Request::Shutdown);
    assert!(matches!(recv(&mut output), Reply::Closed));
    drop(input);
    assert!(child.wait().unwrap().success());
    expanded.resident_payload_limit = Some(5 * 2_176_000 - 1);
    let mut child = Command::new(env!("CARGO_BIN_EXE_vf-strata-native-bridge"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap());
    send(&mut input, &expanded);
    assert!(matches!(recv(&mut output), Reply::Failed { .. }));
    drop(input);
    assert!(!child.wait().unwrap().success());
    // A late digest failure happens after a worker has loaded earlier experts.
    // The coordinator must fail admission and reap that worker before replying.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let wrapper = base.join("record-worker.sh");
        let pid_file = base.join("record-worker.pid");
        std::fs::write(
            &wrapper,
            "#!/bin/sh\nprintf '%s\\n' \"$$\" > \"$VF_TEST_PID_FILE\"\nexec \"$VF_TEST_WORKER\"\n",
        )
        .unwrap();
        std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700)).unwrap();
        expanded.resident_payload_limit = Some(5 * 2_176_000);
        expanded.bindings[4].blob_sha256 = "0".repeat(64);
        expanded.endpoints[0].argv = vec![wrapper.to_str().unwrap().into()];
        let mut child = Command::new(env!("CARGO_BIN_EXE_vf-strata-native-bridge"))
            .env("VF_TEST_PID_FILE", &pid_file)
            .env("VF_TEST_WORKER", env!("CARGO_BIN_EXE_vf-rank-session"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut input = child.stdin.take().unwrap();
        let mut output = BufReader::new(child.stdout.take().unwrap());
        send(&mut input, &expanded);
        assert!(matches!(recv(&mut output), Reply::Failed { .. }));
        let pid: u32 = std::fs::read_to_string(pid_file)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert!(
            !std::path::Path::new(&format!("/proc/{pid}")).exists(),
            "worker survived failed admission"
        );
        drop(input);
        assert!(!child.wait().unwrap().success());
    }
    #[cfg(unix)]
    for client_path in [
        std::env::var_os("VF_STRATA_CPP_CLIENT"),
        std::env::var_os("VF_STRATA_CPP_POOL"),
        std::env::var_os("VF_STRATA_CPP_DONOR_BRANCH"),
        std::env::var_os("VF_STRATA_CPP_DONOR_FULL"),
    ]
    .into_iter()
    .flatten()
    {
        use std::os::{fd::OwnedFd, unix::net::UnixStream};
        let expected_path = base.join("expected-f32.bin");
        let mut expected_file = std::fs::File::create(&expected_path).unwrap();
        for v in &expected {
            expected_file.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(expected_file);
        let (mut parent, remote) = UnixStream::pair().unwrap();
        let remote_out = remote.try_clone().unwrap();
        let mut bridge = Command::new(env!("CARGO_BIN_EXE_vf-strata-native-bridge"))
            .stdin(Stdio::from(OwnedFd::from(remote)))
            .stdout(Stdio::from(OwnedFd::from(remote_out)))
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        fn line(socket: &mut UnixStream) -> String {
            let mut bytes = Vec::new();
            loop {
                let mut b = [0_u8; 1];
                socket.read_exact(&mut b).unwrap();
                if b[0] == b'\n' {
                    break;
                }
                bytes.push(b[0]);
                assert!(bytes.len() < 65536);
            }
            String::from_utf8(bytes).unwrap()
        }
        serde_json::to_writer(&mut parent, &init).unwrap();
        parent.write_all(b"\n").unwrap();
        assert!(matches!(
            serde_json::from_str::<Reply>(&line(&mut parent)).unwrap(),
            Reply::Ready { .. }
        ));
        serde_json::to_writer(&mut parent, &Request::EnterBinary).unwrap();
        parent.write_all(b"\n").unwrap();
        assert!(matches!(
            serde_json::from_str::<Reply>(&line(&mut parent)).unwrap(),
            Reply::BinaryReady
        ));
        let parent_out = parent.try_clone().unwrap();
        let mut client = Command::new(client_path)
            .arg(&expected_path)
            .stdin(Stdio::from(OwnedFd::from(parent)))
            .stdout(Stdio::from(OwnedFd::from(parent_out)))
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        assert!(client.wait().unwrap().success(), "C++ binary client failed");
        assert!(
            bridge.wait().unwrap().success(),
            "binary coordinator failed"
        );
    }
    #[cfg(unix)]
    if let Some(client_path) = std::env::var_os("VF_STRATA_CPP_FLEET") {
        use std::os::{fd::OwnedFd, unix::net::UnixStream};
        let mut init7: Init = serde_json::from_str(&serde_json::to_string(&init).unwrap()).unwrap();
        init7.template.stamp.layer = 7;
        for binding in &mut init7.bindings {
            binding.layer = 7;
            binding.blob_sha256 = BLOB_SHA.into();
        }
        fn fleet_open(init: &Init) -> (Child, UnixStream) {
            let (mut parent, remote) = UnixStream::pair().unwrap();
            let remote_out = remote.try_clone().unwrap();
            let bridge = Command::new(env!("CARGO_BIN_EXE_vf-strata-native-bridge"))
                .stdin(Stdio::from(OwnedFd::from(remote)))
                .stdout(Stdio::from(OwnedFd::from(remote_out)))
                .stderr(Stdio::null())
                .spawn()
                .unwrap();
            fn response(socket: &mut UnixStream) -> Reply {
                let mut bytes = Vec::new();
                loop {
                    let mut b = [0_u8; 1];
                    socket.read_exact(&mut b).unwrap();
                    if b[0] == b'\n' {
                        break;
                    }
                    bytes.push(b[0]);
                    assert!(bytes.len() < 65536);
                }
                serde_json::from_slice(&bytes).unwrap()
            }
            serde_json::to_writer(&mut parent, init).unwrap();
            parent.write_all(b"\n").unwrap();
            assert!(matches!(response(&mut parent), Reply::Ready { .. }));
            serde_json::to_writer(&mut parent, &Request::EnterBinary).unwrap();
            parent.write_all(b"\n").unwrap();
            assert!(matches!(response(&mut parent), Reply::BinaryReady));
            (bridge, parent)
        }
        let expected_path = base.join("fleet-expected-f32.bin");
        let mut expected_file = std::fs::File::create(&expected_path).unwrap();
        for v in &expected {
            expected_file.write_all(&v.to_le_bytes()).unwrap();
        }
        drop(expected_file);
        let (mut bridge0, parent0) = fleet_open(&init);
        let (mut bridge7, parent7) = fleet_open(&init7);
        let parent0out = parent0.try_clone().unwrap();
        let mut client = Command::new(client_path)
            .arg(&expected_path)
            .stdin(Stdio::from(OwnedFd::from(parent0)))
            .stdout(Stdio::from(OwnedFd::from(parent0out)))
            .stderr(Stdio::from(OwnedFd::from(parent7)))
            .spawn()
            .unwrap();
        assert!(
            client.wait().unwrap().success(),
            "two-layer C++ fleet failed"
        );
        assert!(bridge0.wait().unwrap().success());
        assert!(bridge7.wait().unwrap().success());
    }
    let mut bad: Init = serde_json::from_str(&serde_json::to_string(&init).unwrap()).unwrap();
    bad.bindings[0].blob_sha256 = "0".repeat(64);
    let mut child = Command::new(env!("CARGO_BIN_EXE_vf-strata-native-bridge"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap());
    send(&mut input, &bad);
    assert!(matches!(recv(&mut output), Reply::Failed { .. }));
    drop(input);
    assert!(!child.wait().unwrap().success());
    let fifo = base.join("not-a-pack-fifo");
    assert!(Command::new("mkfifo")
        .arg(&fifo)
        .status()
        .unwrap()
        .success());
    let mut bad_fifo: Init = serde_json::from_str(&serde_json::to_string(&init).unwrap()).unwrap();
    bad_fifo.pack_path = fifo.to_str().unwrap().into();
    let mut child = Command::new(env!("CARGO_BIN_EXE_vf-strata-native-bridge"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap());
    send(&mut input, &bad_fifo);
    assert!(matches!(recv(&mut output), Reply::Failed { .. }));
    drop(input);
    assert!(!child.wait().unwrap().success());
}
