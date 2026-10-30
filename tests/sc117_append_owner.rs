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
    session::{Endpoint, Session},
};
const INDEX: &str = include_str!("../fixtures/strata_sc117_native_experts_v3.txt");
const INDEX_SHA: &str = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24";
const DIGEST: &str = "2c7dcdcf2992bb6127b86407fb43abe67f4ba202acbe005a9bc67d7df9e072b0";
const SIZE: u64 = 50_292_326_400;

#[test]
fn appended_owner_preserves_residency_scope_attempts_and_collector() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".runs")
        .join(format!(
            "r439-append-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    std::fs::create_dir(&path).unwrap();
    let ledger = NativeBatchLedger::open(&path).unwrap();
    let mut bytes = vec![0u8; 2_176_000];
    for i in (0..1_254_400).step_by(98) {
        bytes[i + 1] = 0x28;
    }
    for i in (1_254_400..2_176_000).step_by(18) {
        bytes[i + 1] = 0x28;
    }
    let bind = |expert, local| Binding {
        model_revision: "model-test".into(),
        rank: 2,
        layer: 0,
        expert,
        local,
        epoch: 3,
        index_sha256: INDEX_SHA.into(),
        blob_sha256: DIGEST.into(),
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
    session
        .load_native_ggml(&bind(0, 0), INDEX, SIZE, &bytes, timeout)
        .unwrap();
    session.seal_native_ggml(1, timeout).unwrap();
    let stamp = |batch| Stamp {
        run_id: "r439-fresh".into(),
        model_revision: "model-test".into(),
        plan_id: "plan".into(),
        layer: 0,
        batch,
        epoch: 3,
    };
    let problem = |batch, expert, copies, locals: Vec<u32>| RoutingProblem {
        hidden: 2560,
        experts: locals.clone(),
        routes: vec![vec![Choice {
            expert,
            weight: 1.0,
        }]],
        registry: Registry { epoch: 3, copies },
        stamp: stamp(batch),
        preferred_owners: BTreeMap::new(),
        resident_order: BTreeMap::from([(2, locals)]),
    };
    let a = problem(
        0,
        0,
        vec![Replica {
            expert: 0,
            rank: 2,
            local: 0,
        }],
        vec![0],
    );
    let x = vec![vec![0.001; 2560]];
    let first = Session::exec_native_ggml_reserved(
        std::slice::from_mut(&mut session),
        &a,
        &x,
        &ledger,
        ledger.reserve_ticket(&a.stamp, 1).unwrap(),
        timeout,
    )
    .unwrap();
    assert!(first.rows()[0].iter().any(|x| x.abs() > 1e-8));
    session
        .extend_native_ggml(&bind(1, 1), INDEX, SIZE, &bytes, timeout)
        .unwrap();
    let b = problem(
        1,
        1,
        vec![
            Replica {
                expert: 0,
                rank: 2,
                local: 0,
            },
            Replica {
                expert: 1,
                rank: 2,
                local: 1,
            },
        ],
        vec![0, 1],
    );
    let second = Session::exec_native_ggml_reserved(
        std::slice::from_mut(&mut session),
        &b,
        &x,
        &ledger,
        ledger.reserve_ticket(&b.stamp, 2).unwrap(),
        timeout,
    )
    .unwrap();
    assert_eq!(first.rows(), second.rows());
    assert!(ledger.reserve_ticket(&a.stamp, 3).is_err());
    let receipt = session.close(timeout);
    assert!(!receipt.killed && receipt.exit_code == Some(0) && receipt.bye_execs == Some(2));
}

#[test]
fn rejected_post_seal_duplicate_makes_client_terminal() {
    let mut bytes = vec![0u8; 2_176_000];
    for i in (0..1_254_400).step_by(98) {
        bytes[i + 1] = 0x28;
    }
    for i in (1_254_400..2_176_000).step_by(18) {
        bytes[i + 1] = 0x28;
    }
    let binding = Binding {
        model_revision: "model-test".into(),
        rank: 2,
        layer: 0,
        expert: 0,
        local: 0,
        epoch: 3,
        index_sha256: INDEX_SHA.into(),
        blob_sha256: DIGEST.into(),
    };
    let timeout = Duration::from_secs(30);
    let mut s = Session::connect(
        &Endpoint {
            rank: 2,
            argv: vec![env!("CARGO_BIN_EXE_vf-rank-session").into()],
        },
        timeout,
    )
    .unwrap();
    s.load_native_ggml(&binding, INDEX, SIZE, &bytes, timeout)
        .unwrap();
    s.seal_native_ggml(1, timeout).unwrap();
    assert!(s
        .extend_native_ggml(&binding, INDEX, SIZE, &bytes, timeout)
        .is_err());
    let err = s
        .send(&voltafabric::session::Request::Shutdown)
        .unwrap_err();
    assert_eq!(err.code, "SESSION_UNUSABLE");
    let receipt = s.close(timeout);
    assert_eq!(receipt.bye_execs, None);
}

#[test]
fn append_preserves_worker_scope_attempt_history_and_existing_positions() {
    use voltafabric::{
        moe::Slot,
        native_ggml_cpu::{RankWorker, Resident},
        routing::{RankDispatch, RankSlot},
        strata_native_layout::{NativeLayout, VerifiedNativeExpert},
    };
    let mut bytes = vec![0u8; 2_176_000];
    for i in (0..1_254_400).step_by(98) {
        bytes[i + 1] = 0x28;
    }
    for i in (1_254_400..2_176_000).step_by(18) {
        bytes[i + 1] = 0x28;
    }
    let layout = NativeLayout::parse_sc117_v3(INDEX, SIZE).unwrap();
    let owner = |expert, local, epoch| {
        let binding = Binding {
            model_revision: "model-test".into(),
            rank: 2,
            layer: 0,
            expert,
            local,
            epoch,
            index_sha256: INDEX_SHA.into(),
            blob_sha256: DIGEST.into(),
        };
        let blob = VerifiedNativeExpert {
            layer: 0,
            expert: expert.into(),
            span: layout.expert(0, expert.into()).unwrap(),
            bytes: bytes.clone(),
        };
        Resident::load(&layout, blob, binding).unwrap()
    };
    let mut worker = RankWorker::new(vec![owner(0, 0, 3)]).unwrap();
    let mut dispatch = RankDispatch {
        stamp: Stamp {
            run_id: "run".into(),
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
                weight: 1.0,
            },
            physical_expert: 0,
        }],
    };
    let input = vec![(0, vec![0.001; 2560])];
    let before = worker.execute(&dispatch, &input, 7).unwrap();
    worker.add_owner(owner(1, 1, 3)).unwrap();
    assert!(worker.add_owner(owner(1, 2, 3)).is_err());
    assert!(worker.add_owner(owner(2, 2, 4)).is_err());
    assert_eq!(worker.owner_count(), 2);
    dispatch.resident_order.push(1);
    dispatch.stamp.batch = 1;
    assert!(worker.execute(&dispatch, &input, 7).is_err());
    let after = worker.execute(&dispatch, &input, 8).unwrap();
    assert_eq!(before.fp16_bits, after.fp16_bits);
    dispatch.stamp.batch = 2;
    dispatch.stamp.plan_id = "other".into();
    assert!(worker.execute(&dispatch, &input, 9).is_err());
}
