#![cfg(feature = "native-ggml-cpu")]
use std::{
    collections::BTreeMap,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use voltafabric::{
    moe::{Choice, Registry, Replica, Stamp},
    native_ggml_cpu::Binding,
    native_ledger::NativeBatchLedger,
    routing::RoutingProblem,
    session::{Endpoint, NativeEviction, Session},
};
const INDEX: &str = include_str!("../fixtures/strata_sc117_native_experts_v3.txt");
const INDEX_SHA: &str = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24";
const DIGEST: &str = "2c7dcdcf2992bb6127b86407fb43abe67f4ba202acbe005a9bc67d7df9e072b0";
const SIZE: u64 = 50_292_326_400;

#[test]
fn evict_rebuilds_physical_order_preserves_collector_and_refuses_replay() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".runs")
        .join(format!(
            "r463-evict-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    std::fs::create_dir(&dir).unwrap();
    let ledger = NativeBatchLedger::open(&dir).unwrap();
    let mut bytes = vec![0u8; 2_176_000];
    for i in (0..1_254_400).step_by(98) {
        bytes[i + 1] = 0x28;
    }
    for i in (1_254_400..2_176_000).step_by(18) {
        bytes[i + 1] = 0x28;
    }
    let mut other_bytes = bytes.clone();
    for i in (0..1_254_400).step_by(98) {
        other_bytes[i + 1] = 0x30;
    }
    for i in (1_254_400..2_176_000).step_by(18) {
        other_bytes[i + 1] = 0x30;
    }
    let other_digest = voltafabric::sha256::hex(&other_bytes);
    let bind = |expert, local| Binding {
        model_revision: "model-test".into(),
        rank: 2,
        layer: 0,
        expert,
        local,
        epoch: 3,
        index_sha256: INDEX_SHA.into(),
        blob_sha256: if expert >= 2 {
            other_digest.clone()
        } else {
            DIGEST.into()
        },
    };
    let stamp = |batch| Stamp {
        run_id: "r463-fresh".into(),
        model_revision: "model-test".into(),
        plan_id: "plan".into(),
        layer: 0,
        batch,
        epoch: 3,
    };
    let problem = |batch, expert, locals: Vec<u32>| RoutingProblem {
        hidden: 2560,
        experts: locals.clone(),
        routes: vec![vec![Choice {
            expert,
            weight: 1.0,
        }]],
        registry: Registry {
            epoch: 3,
            copies: locals
                .iter()
                .map(|&local| Replica {
                    expert: local,
                    rank: 2,
                    local,
                })
                .collect(),
        },
        stamp: stamp(batch),
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(2, locals)]),
    };
    let timeout = Duration::from_secs(30);
    let mut session = Session::connect(
        &Endpoint {
            rank: 2,
            argv: vec![env!("CARGO_BIN_EXE_vf-rank-session").into()],
        },
        timeout,
    )
    .unwrap();
    for i in 0..3 {
        session
            .load_native_ggml(
                &bind(i, i),
                INDEX,
                SIZE,
                if i >= 2 { &other_bytes } else { &bytes },
                timeout,
            )
            .unwrap();
    }
    session.seal_native_ggml(3, timeout).unwrap();
    let input = vec![vec![0.001; 2560]];
    let old = problem(0, 0, vec![0, 1, 2]);
    let different = Session::exec_native_ggml_reserved(
        std::slice::from_mut(&mut session),
        &old,
        &input,
        &ledger,
        ledger.reserve_ticket(&old.stamp, 1).unwrap(),
        timeout,
    )
    .unwrap();
    let first = problem(1, 2, vec![0, 1, 2]);
    let baseline = Session::exec_native_ggml_reserved(
        std::slice::from_mut(&mut session),
        &first,
        &input,
        &ledger,
        ledger.reserve_ticket(&first.stamp, 2).unwrap(),
        timeout,
    )
    .unwrap();
    assert_ne!(baseline.rows(), different.rows());
    let op = "a".repeat(64);
    session
        .evict_native_ggml(
            &NativeEviction {
                stamp: &first.stamp,
                attempt: 2,
                operation_id: &op,
                binding: &bind(1, 1),
                expected_bytes: 2_176_000,
                expected_remaining: 2,
            },
            timeout,
        )
        .unwrap();
    let after = problem(2, 2, vec![0, 2]);
    let out = Session::exec_native_ggml_reserved(
        std::slice::from_mut(&mut session),
        &after,
        &input,
        &ledger,
        ledger.reserve_ticket(&after.stamp, 3).unwrap(),
        timeout,
    )
    .unwrap();
    assert_eq!(baseline.rows(), out.rows());
    session
        .extend_native_ggml(&bind(3, 3), INDEX, SIZE, &other_bytes, timeout)
        .unwrap();
    let appended = problem(3, 3, vec![0, 2, 3]);
    let out = Session::exec_native_ggml_reserved(
        std::slice::from_mut(&mut session),
        &appended,
        &input,
        &ledger,
        ledger.reserve_ticket(&appended.stamp, 4).unwrap(),
        timeout,
    )
    .unwrap();
    assert_eq!(baseline.rows(), out.rows());
    assert!(session
        .evict_native_ggml(
            &NativeEviction {
                stamp: &appended.stamp,
                attempt: 4,
                operation_id: &op,
                binding: &bind(0, 0),
                expected_bytes: 2_176_000,
                expected_remaining: 2
            },
            timeout
        )
        .is_err());
    assert_eq!(
        session
            .send(&voltafabric::session::Request::Shutdown)
            .unwrap_err()
            .code,
        "SESSION_UNUSABLE"
    );
    let receipt = session.close(timeout);
    assert_eq!(receipt.bye_execs, None);
}

#[test]
fn eviction_requires_completed_stamp_exact_binding_and_nonempty_owner() {
    use voltafabric::{
        moe::Slot,
        native_ggml_cpu::{RankWorker, Resident},
        routing::{RankDispatch, RankSlot},
        strata_native_layout::{NativeLayout, VerifiedNativeExpert},
    };
    let layout = NativeLayout::parse_sc117_v3(INDEX, SIZE).unwrap();
    let mut bytes = vec![0u8; 2_176_000];
    for i in (0..1_254_400).step_by(98) {
        bytes[i + 1] = 0x28;
    }
    for i in (1_254_400..2_176_000).step_by(18) {
        bytes[i + 1] = 0x28;
    }
    let bind = |n| Binding {
        model_revision: "model-test".into(),
        rank: 2,
        layer: 0,
        expert: n,
        local: n,
        epoch: 3,
        index_sha256: INDEX_SHA.into(),
        blob_sha256: DIGEST.into(),
    };
    let owner = |n: u32| {
        Resident::load(
            &layout,
            VerifiedNativeExpert {
                layer: 0,
                expert: n.into(),
                span: layout.expert(0, n.into()).unwrap(),
                bytes: bytes.clone(),
            },
            bind(n),
        )
        .unwrap()
    };
    let mut worker = RankWorker::new(vec![owner(0), owner(1)]).unwrap();
    let stamp = Stamp {
        run_id: "run".into(),
        model_revision: "model-test".into(),
        plan_id: "plan".into(),
        layer: 0,
        batch: 7,
        epoch: 3,
    };
    let op = "b".repeat(64);
    assert!(worker.evict_owner(&stamp, 8, &op, &bind(1)).is_err());
    let dispatch = RankDispatch {
        stamp: stamp.clone(),
        rank: 2,
        hidden: 2560,
        tokens: 1,
        resident_order: vec![0, 1],
        slots: vec![RankSlot {
            slot: Slot {
                token: 0,
                slot: 0,
                expert: 0,
                rank: 2,
                local: 0,
                weight: 1.0,
            },
            physical_expert: 0,
        }],
    };
    worker
        .execute(&dispatch, &[(0, vec![0.001; 2560])], 8)
        .unwrap();
    let mut wrong_stamp = stamp.clone();
    wrong_stamp.batch += 1;
    assert!(worker.evict_owner(&wrong_stamp, 8, &op, &bind(1)).is_err());
    assert!(worker.evict_owner(&stamp, 9, &op, &bind(1)).is_err());
    assert!(worker.evict_owner(&stamp, 8, "bad", &bind(1)).is_err());
    let mut wrong_binding = bind(1);
    wrong_binding.blob_sha256 = "0".repeat(64);
    assert!(worker.evict_owner(&stamp, 8, &op, &wrong_binding).is_err());
    assert_eq!(worker.owner_count(), 2);
    assert_eq!(
        worker.evict_owner(&stamp, 8, &op, &bind(1)).unwrap(),
        (2_176_000, 1)
    );
    assert!(worker
        .evict_owner(&stamp, 8, &"c".repeat(64), &bind(0))
        .is_err());
    assert_eq!(worker.owner_count(), 1);
}

#[test]
fn lost_or_wrong_eviction_acknowledgement_makes_client_unusable() {
    let hello = r#"{"reply":"hello","hello":{"protocol":1,"rank":2,"arch":"x","os":"y","endian":"little","build":"0","capabilities":["cpu_f64_dense_moe_v1","cpu_strata_sc117_evict_resident_v1"],"max_message_bytes":16777216,"pid":1}}"#;
    let stamp = Stamp {
        run_id: "run".into(),
        model_revision: "model-test".into(),
        plan_id: "plan".into(),
        layer: 0,
        batch: 1,
        epoch: 3,
    };
    let binding = Binding {
        model_revision: "model-test".into(),
        rank: 2,
        layer: 0,
        expert: 1,
        local: 1,
        epoch: 3,
        index_sha256: INDEX_SHA.into(),
        blob_sha256: DIGEST.into(),
    };
    for suffix in ["", "printf '%s\\n' '{\"reply\":\"bye\",\"execs\":0}'"] {
        let script = format!("read l; printf '%s\\n' '{hello}'; read l; {suffix}");
        let endpoint = Endpoint {
            rank: 2,
            argv: vec!["/bin/sh".into(), "-c".into(), script],
        };
        let mut s = Session::connect(&endpoint, Duration::from_secs(5)).unwrap();
        assert!(s
            .evict_native_ggml(
                &NativeEviction {
                    stamp: &stamp,
                    attempt: 1,
                    operation_id: &"e".repeat(64),
                    binding: &binding,
                    expected_bytes: 2_176_000,
                    expected_remaining: 1
                },
                Duration::from_secs(5)
            )
            .is_err());
        assert_eq!(
            s.send(&voltafabric::session::Request::Shutdown)
                .unwrap_err()
                .code,
            "SESSION_UNUSABLE"
        );
        let _ = s.close(Duration::from_secs(5));
    }
}
