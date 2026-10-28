//! A52 targeted mutation: each refusal of src/session.rs, driven by an input that
//! violates it alone, with its exact code and message. Protocol violations a correct
//! worker never commits come from `sh -c` fakes or a `sed` filter around the real
//! worker; no script file is written, so no ETXTBSY race with sibling tests.
#![cfg(target_os = "linux")]
mod common;

use std::collections::BTreeMap;
use std::time::Duration;
use voltafabric::moe::{self, Problem};
use voltafabric::process::{Fault, ReceiptState};
use voltafabric::session::{self, Endpoint, Reply, Request, SeriesInput, Session};
use voltafabric::{routing, Error};

const T: Duration = Duration::from_secs(10);
const BIN: &str = env!("CARGO_BIN_EXE_vf-rank-session");
const HELLO_2: &str = r#"{"reply":"hello","hello":{"protocol":1,"rank":2,"arch":"x","os":"y","endian":"little","build":"0","capabilities":["cpu_f64_dense_moe_v1"],"max_message_bytes":16777216,"pid":1}}"#;

fn endpoint(rank: u32) -> Endpoint {
    Endpoint {
        rank,
        argv: vec![BIN.into()],
    }
}
/// A worker played by a shell script given inline.
fn fake(rank: u32, script: &str) -> Endpoint {
    Endpoint {
        rank,
        argv: vec!["/bin/sh".into(), "-c".into(), script.into()],
    }
}
/// The real worker with its replies rewritten by `sed -u <program>`.
fn filtered(rank: u32, program: &str) -> Endpoint {
    Endpoint {
        rank,
        argv: vec![
            "/bin/sh".into(),
            "-c".into(),
            "\"$1\" | sed -u \"$2\"".into(),
            "sh".into(),
            BIN.into(),
            program.into(),
        ],
    }
}
fn problem(run_id: &str) -> Problem {
    let mut p = common::problem();
    p.stamp.run_id = run_id.into();
    p
}
fn rank_dispatch(p: &Problem, rank: u32) -> routing::RankDispatch {
    routing::dispatch(&routing::from_dense(p))
        .unwrap()
        .ranks
        .into_iter()
        .find(|r| r.rank == rank)
        .unwrap()
}
fn assert_reason(error: &Error, code: &str, message: &str) {
    assert_eq!(
        (error.code.as_str(), error.message.as_str()),
        (code, message)
    );
}
fn refused(reply: Reply) -> Error {
    match reply {
        Reply::Refused { error } => error,
        other => panic!("expected a refusal, got {other:?}"),
    }
}
fn expert(id: u32, n: usize) -> moe::Expert {
    moe::Expert {
        id,
        gate: vec![0.25; n],
        up: vec![0.5; n],
        down: vec![0.125; n],
    }
}
fn model(hidden: usize, intermediate: usize, ids: &[u32]) -> moe::Model {
    let n = hidden.saturating_mul(intermediate);
    moe::Model {
        hidden,
        intermediate,
        experts: ids.iter().map(|&id| expert(id, n)).collect(),
    }
}

// ------------------------------------------------------------- worker: LOAD

/// Send one LOAD carrying `model` (with its true digest) and return the refusal.
fn load_refusal(model: moe::Model, order: Vec<u32>, revision: &str) -> Error {
    let mut s = Session::connect(&endpoint(2), T).unwrap();
    s.send(&Request::Load {
        artifact_sha256: session::artifact_digest(&model).unwrap(),
        model_revision: revision.into(),
        layer: 0,
        epoch: 1,
        resident_order: order,
        model: Some(model),
    })
    .unwrap();
    let error = refused(s.reply(T).unwrap());
    assert_eq!(s.close(T).exit_code, Some(0));
    error
}

#[test]
fn load_refuses_each_malformed_resident_artifact() {
    let outside = "resident artifact identity or dimensions outside the session contract";
    let ids: Vec<u32> = (0..1025).collect();
    let cases: Vec<(&str, moe::Model, Vec<u32>, String)> = vec![
        ("no experts", model(2, 2, &[]), vec![], "r".into()),
        ("1025 experts", model(1, 1, &ids), ids.clone(), "r".into()),
        ("order shorter", model(2, 2, &[10, 99]), vec![0], "r".into()),
        (
            "order repeats",
            model(2, 2, &[10, 99]),
            vec![0, 0],
            "r".into(),
        ),
        ("ids repeat", model(2, 2, &[10, 10]), vec![0, 1], "r".into()),
        ("hidden 0", model(0, 2, &[10]), vec![0], "r".into()),
        ("hidden 65537", model(65537, 1, &[10]), vec![0], "r".into()),
        ("intermediate 0", model(2, 0, &[10]), vec![0], "r".into()),
        ("blank revision", model(2, 2, &[10]), vec![0], "  ".into()),
        (
            "long revision",
            model(2, 2, &[10]),
            vec![0],
            "r".repeat(257),
        ),
    ];
    for (name, m, order, revision) in cases {
        assert_reason(&load_refusal(m, order, &revision), "INVALID_INPUT", outside);
        eprintln!("{name}: refused");
    }
    // hidden x intermediate overflows usize; no weight can then have that length.
    let overflow = moe::Model {
        hidden: 2,
        intermediate: usize::MAX,
        experts: vec![expert(10, 0)],
    };
    assert_reason(
        &load_refusal(overflow, vec![0], "r"),
        "INVALID_INPUT",
        outside,
    );
    let mut short = model(2, 2, &[10]);
    short.experts[0].down.pop();
    assert_reason(
        &load_refusal(short, vec![0], "r"),
        "INVALID_INPUT",
        "resident expert weights are malformed or nonfinite",
    );
}

#[test]
fn session_load_reports_the_worker_refusal() {
    let p = problem("a52r-load-refusal");
    let mut s = Session::connect(&endpoint(2), T).unwrap();
    let d = rank_dispatch(&p, 2);
    let m = session::resident_model(&p, &d).unwrap();
    s.load(&m, "rev", 0, 5, &d.resident_order, T).unwrap();
    let error = s.load(&m, "rev", 0, 4, &d.resident_order, T).unwrap_err();
    assert_reason(
        &error,
        "EPOCH_ROLLBACK",
        "ownership epoch is older than the loaded one",
    );
    assert!(s.is_usable());
    assert_eq!(s.close(T).exit_code, Some(0));
}

// ---------------------------------------------------- worker: session framing

fn stderr_after(requests: &[Request]) -> (Vec<Error>, session::CloseReceipt) {
    let mut s = Session::spawn(&endpoint(2)).unwrap();
    let mut refusals = vec![];
    for request in requests {
        s.send(request).unwrap();
        match s.reply(T).unwrap() {
            Reply::Refused { error } => refusals.push(error),
            Reply::Hello { .. } => {}
            other => panic!("unexpected {other:?}"),
        }
    }
    (refusals, s.close_input(T))
}

#[test]
fn hello_errors_close_the_session_with_their_reason_on_stderr() {
    let hello = |protocol| Request::Hello { protocol, rank: 2 };
    let cases = [
        (
            vec![hello(session::PROTOCOL + 1)],
            ("PROTOCOL_MISMATCH", "session protocol version differs"),
        ),
        (
            vec![hello(session::PROTOCOL), hello(session::PROTOCOL)],
            ("SESSION_PROTOCOL", "duplicate HELLO"),
        ),
        (
            vec![Request::Shutdown],
            ("SESSION_PROTOCOL", "HELLO must open the session"),
        ),
    ];
    for (requests, (code, message)) in cases {
        let (refusals, receipt) = stderr_after(&requests);
        assert_eq!(refusals.len(), 1);
        assert_reason(&refusals[0], code, message);
        assert_eq!(receipt.exit_code, Some(2), "{receipt:?}");
        assert_eq!(receipt.stderr, format!("{code}: {message}\n"));
    }
}

#[test]
fn a_stream_closed_without_shutdown_is_abandoned() {
    let s = Session::connect(&endpoint(2), T).unwrap();
    let receipt = s.close_input(T);
    assert_eq!(receipt.exit_code, Some(2), "{receipt:?}");
    assert_eq!(
        receipt.stderr,
        "SESSION_ABANDONED: stream closed without SHUTDOWN\n"
    );
}

// ------------------------------------------------- coordinator: HELLO and LOAD

#[test]
fn connect_reports_a_refused_or_foreign_hello_and_a_malformed_frame() {
    let refuse = fake(
        2,
        r#"read l; printf '%s\n' '{"reply":"refused","error":{"code":"FAKE_REFUSAL","message":"no"}}'"#,
    );
    assert_reason(
        &Session::connect(&refuse, T).unwrap_err(),
        "FAKE_REFUSAL",
        "no",
    );
    let foreign = fake(9, &format!("read l; printf '%s\\n' '{HELLO_2}'"));
    assert_reason(
        &Session::connect(&foreign, T).unwrap_err(),
        "SESSION_PROTOCOL",
        "HELLO answer differs from this coordinator's protocol, rank or bounds",
    );
    let malformed = fake(2, "read l; echo '{}'");
    let error = Session::connect(&malformed, T).unwrap_err();
    assert_reason(
        &error,
        "WORKER_PROTOCOL",
        "missing field `reply` at line 1 column 2",
    );
}

#[test]
fn unexpected_replies_to_load_and_evict_make_the_session_unusable() {
    let p = problem("a52r-unexpected");
    let d = rank_dispatch(&p, 2);
    let m = session::resident_model(&p, &d).unwrap();
    let bye = r#"{"reply":"bye","execs":0}"#;
    let mut s = Session::connect(
        &fake(
            2,
            &format!("read l; printf '%s\\n' '{HELLO_2}'; read l; printf '%s\\n' '{bye}'"),
        ),
        T,
    )
    .unwrap();
    assert_reason(
        &s.load(&m, "rev", 0, 1, &d.resident_order, T).unwrap_err(),
        "SESSION_PROTOCOL",
        "unexpected reply to LOAD",
    );
    assert!(!s.is_usable());
    drop(s);
    // A fake that loads by echoing the digest and epoch, then answers EVICT with BYE.
    let echo = format!(
        r#"read l; printf '%s\n' '{HELLO_2}'; read l
sha=$(printf '%s' "$l" | sed -n 's/.*"artifact_sha256":"\([0-9a-f]*\)".*/\1/p')
ep=$(printf '%s' "$l" | sed -n 's/.*"epoch":\([0-9]*\).*/\1/p')
printf '{{"reply":"loaded","artifact_sha256":"%s","epoch":%s,"experts":2,"transferred":true}}\n' "$sha" "$ep"
read l; printf '%s\n' '{bye}'"#
    );
    let mut s = Session::connect(&fake(2, &echo), T).unwrap();
    s.load(&m, "rev", 0, 1, &d.resident_order, T).unwrap();
    assert_reason(
        &s.evict(T).unwrap_err(),
        "SESSION_PROTOCOL",
        "unexpected reply to EVICT",
    );
    assert!(!s.is_usable());
}

#[test]
fn an_empty_argv_is_refused() {
    let empty = Endpoint {
        rank: 2,
        argv: vec![],
    };
    assert_reason(
        &Session::spawn(&empty).unwrap_err(),
        "INVALID_INPUT",
        "session endpoint has an empty argv",
    );
}

// -------------------------------------------------------- coordinator: EXEC

#[test]
fn exec_without_an_artifact_or_on_an_unusable_session_is_refused_locally() {
    let p = problem("a52r-local-exec");
    let d = rank_dispatch(&p, 2);
    let mut collector = moe::Collector::new(&p).unwrap();
    let mut s = Session::connect(&endpoint(2), T).unwrap();
    let error = s
        .exec(&d, 1, vec![], Fault::None, &mut collector, T)
        .unwrap_err();
    assert_reason(
        &error,
        "INVALID_INPUT",
        "no artifact is loaded in this session",
    );
    assert_eq!(s.traffic.execs, 0);
    // A reply that never comes makes the session unusable.
    assert_eq!(
        s.reply(Duration::from_millis(20)).unwrap_err().code,
        "SESSION_TIMEOUT"
    );
    let error = s
        .exec(&d, 2, vec![], Fault::None, &mut collector, T)
        .unwrap_err();
    assert_reason(
        &error,
        "SESSION_UNUSABLE",
        "session must be reopened after a failure",
    );
}

/// One batch where rank 9's replies pass through `sed -u program`.
fn forged_batch(run_id: &str, program: &str) -> (session::BatchReport, Vec<Session>) {
    let p = problem(run_id);
    let mut sessions = session::prepare(&[endpoint(2), filtered(9, program)], &p, T).unwrap();
    let report = session::run_batch(&mut sessions, &p, &BTreeMap::new(), false, T).unwrap();
    (report, sessions)
}
fn rank9(report: &session::BatchReport) -> &session::ExecReceipt {
    report.receipts.iter().find(|r| r.rank == 9).unwrap()
}

#[test]
fn forged_worker_events_are_refused_and_poison_the_session() {
    let cases = [
        (
            r#"/"kind":"started"/d"#,
            ("WORKER_PROTOCOL", "invalid worker event lifecycle"),
        ),
        (
            r#"/"kind":"started"/c\{"reply":"bye","execs":0}"#,
            ("SESSION_PROTOCOL", "unexpected reply to EXEC"),
        ),
        (
            r#"s/"attempt":[0-9]*/"attempt":1/"#,
            ("STALE_ATTEMPT", "worker event belongs to another attempt"),
        ),
        (
            r#"s/"plan_id":"plan-1"/"plan_id":"forged"/"#,
            (
                "INVALID_COMPLETION",
                "worker event has stale stamp or foreign rank",
            ),
        ),
        (
            r#"s/"rank":9,"local"/"rank":4242,"local"/"#,
            ("INVALID_COMPLETION", "completion has foreign rank"),
        ),
    ];
    for (n, (program, (code, message))) in cases.into_iter().enumerate() {
        let (report, sessions) = forged_batch(&format!("a52r-forged-{n}"), program);
        assert!(!report.ok, "{program}");
        assert!(report.output.is_none());
        let r = rank9(&report);
        assert_eq!(r.state, ReceiptState::Unknown, "{program}: {r:?}");
        assert_reason(r.error.as_ref().unwrap(), code, message);
        assert!(!sessions[1].is_usable(), "{program}");
        assert!(sessions[0].is_usable(), "{program}");
    }
}

#[test]
fn a_rank_that_finishes_without_all_results_is_refused() {
    let p = problem("a52r-missing");
    let mut sessions = session::prepare(&[endpoint(2), endpoint(9)], &p, T).unwrap();
    let faults = BTreeMap::from([(2, Fault::Missing)]);
    let report = session::run_batch(&mut sessions, &p, &faults, false, T).unwrap();
    assert!(!report.ok);
    let r = report.receipts.iter().find(|r| r.rank == 2).unwrap();
    assert_reason(
        r.error.as_ref().unwrap(),
        "BATCH_INCOMPLETE",
        "rank finished with missing results",
    );
}

#[test]
fn ranks_drained_after_a_failure_are_still_checked() {
    // Rank 2 refuses first; rank 9 is drained without a collector and duplicates.
    let p = problem("a52r-drain");
    let mut sessions = session::prepare(&[endpoint(2), endpoint(9)], &p, T).unwrap();
    let faults = BTreeMap::from([(2, Fault::RejectBeforeCompute), (9, Fault::Duplicate)]);
    let report = session::run_batch(&mut sessions, &p, &faults, false, T).unwrap();
    assert!(!report.ok);
    assert_reason(
        report.error.as_ref().unwrap(),
        "WORKER_REJECTED",
        "injected rejection before computation",
    );
    assert_reason(
        rank9(&report).error.as_ref().unwrap(),
        "INVALID_COMPLETION",
        "duplicate completion while draining",
    );
    assert!(sessions[0].is_usable());
    assert!(!sessions[1].is_usable());
}

// ------------------------------------------------- coordinator: batch set-up

#[test]
fn run_batch_refuses_a_missing_unusable_or_fault_mismatched_session() {
    let p = problem("a52r-batch-setup");
    let mut only2 = session::prepare(&[endpoint(2)], &p, T).unwrap();
    assert_reason(
        &session::run_batch(&mut only2, &p, &BTreeMap::new(), false, T).unwrap_err(),
        "INVALID_INPUT",
        "no session serves a participating rank",
    );
    let mut sessions = session::prepare(&[endpoint(2), endpoint(9)], &p, T).unwrap();
    let faults = BTreeMap::from([(5, Fault::None)]);
    assert_reason(
        &session::run_batch(&mut sessions, &p, &faults, false, T).unwrap_err(),
        "INVALID_INPUT",
        "fault refers to a nonparticipating rank",
    );
    let _ = sessions[1].reply(Duration::from_millis(20)).unwrap_err();
    assert_reason(
        &session::run_batch(&mut sessions, &p, &BTreeMap::new(), false, T).unwrap_err(),
        "SESSION_UNUSABLE",
        "session must be reopened after a failure",
    );
}

#[test]
fn prepare_refuses_bad_endpoint_sets_and_ranks_without_residents() {
    let p = problem("a52r-prepare");
    let message = "session endpoints must name 1..64 distinct ranks";
    let many: Vec<_> = (0..65).map(endpoint).collect();
    for endpoints in [vec![], many, vec![endpoint(2), endpoint(2)]] {
        assert_reason(
            &session::prepare(&endpoints, &p, T).unwrap_err(),
            "INVALID_INPUT",
            message,
        );
    }
    assert_reason(
        &session::prepare(&[endpoint(7)], &p, T).unwrap_err(),
        "INVALID_INPUT",
        "endpoint rank owns no resident experts",
    );
}

#[test]
fn resident_model_refuses_an_inconsistent_physical_order_or_registry() {
    let p = problem("a52r-order");
    let d = rank_dispatch(&p, 2);
    let message = "physical order does not cover this rank's residents";
    let mut repeated = d.clone();
    repeated.resident_order.push(d.resident_order[1]);
    let mut foreign = d.clone();
    foreign.resident_order[1] = 7;
    for order in [repeated, foreign] {
        assert_reason(
            &session::resident_model(&p, &order).unwrap_err(),
            "INVALID_INPUT",
            message,
        );
    }
    let mut unknown = p.clone();
    unknown.registry.copies.push(moe::Replica {
        expert: 777,
        rank: 2,
        local: 5,
    });
    let mut with5 = d.clone();
    with5.resident_order.push(5);
    assert_reason(
        &session::resident_model(&unknown, &with5).unwrap_err(),
        "INVALID_INPUT",
        "resident expert missing from model",
    );
}

#[test]
fn series_bounds_are_refused_before_any_worker_starts() {
    let base = SeriesInput {
        problem: problem("a52r-series"),
        // A53-R15: a series' endpoints must serve every participating rank (2 and 9).
        workers: [2, 9]
            .map(|rank| Endpoint {
                rank,
                argv: vec!["/definitely-missing-vf-rank-session".into()],
            })
            .into(),
        batches: 10,
        timeout_ms: 1000,
        epoch_every: 0,
        warmup: 0,
        tolerance_abs: 0.0,
        faults: vec![],
    };
    let fault = session::SeriesFault {
        batch: 0,
        rank: 2,
        fault: Fault::None,
    };
    type Change = Box<dyn Fn(&mut SeriesInput)>;
    let cases: Vec<Change> = vec![
        Box::new(|s| s.batches = 0),
        Box::new(|s| s.batches = 100_001),
        Box::new(|s| s.timeout_ms = 0),
        Box::new(|s| s.timeout_ms = 60_001),
        Box::new(|s| s.warmup = 10),
        Box::new(|s| s.tolerance_abs = f64::NAN),
        Box::new(|s| s.tolerance_abs = -1.0),
        Box::new(move |s| s.faults = vec![fault.clone(); 65]),
    ];
    for change in cases {
        let mut input = base.clone();
        change(&mut input);
        // A missing worker would fail with WORKER_IO: the bound refuses first.
        assert_reason(
            &session::run_series(&input).unwrap_err(),
            "INVALID_INPUT",
            "series needs 1..100000 batches, a 1..60000 ms timeout, warmup below batches and a finite tolerance",
        );
    }
    assert_eq!(session::run_series(&base).unwrap_err().code, "WORKER_IO");
}

#[test]
fn a_refused_evict_is_reported() {
    let p = problem("a52r-evict-refused");
    let d = rank_dispatch(&p, 2);
    let m = session::resident_model(&p, &d).unwrap();
    // Loads by echoing the digest and epoch, then refuses EVICT.
    let script = format!(
        r#"read l; printf '%s\n' '{HELLO_2}'; read l
sha=$(printf '%s' "$l" | sed -n 's/.*"artifact_sha256":"\([0-9a-f]*\)".*/\1/p')
ep=$(printf '%s' "$l" | sed -n 's/.*"epoch":\([0-9]*\).*/\1/p')
printf '{{"reply":"loaded","artifact_sha256":"%s","epoch":%s,"experts":2,"transferred":true}}\n' "$sha" "$ep"
read l; printf '%s\n' '{{"reply":"refused","error":{{"code":"FAKE_EVICT","message":"kept"}}}}'"#
    );
    let mut s = Session::connect(&fake(2, &script), T).unwrap();
    s.load(&m, "rev", 0, 1, &d.resident_order, T).unwrap();
    assert_reason(&s.evict(T).unwrap_err(), "FAKE_EVICT", "kept");
    assert!(s.is_usable());
    assert!(s.artifact().is_some());
}

#[test]
fn a_batch_with_one_oversize_exec_is_refused_before_any_exec_is_sent() {
    // A53-R28: `exec_all` claimed the batch identity, then encoded and sent each rank's
    // EXEC in turn. A rank whose frame exceeded 16 MiB was refused only after the ranks
    // before it had been sent their work, which they computed for nothing, and the
    // identity was spent: the same batch could not run again without recovery. Rank 2
    // gets token 0 and is sent first; rank 9 gets the 4095 others. Its f64 body is
    // 16,281,720 bytes; together with the dispatch header it exceeds the 16 MiB frame
    // limit, while the all-zero retry still fits through the legacy JSON path.
    let (hidden, tokens) = (497, 4096);
    let mut p = problem("a53r28-oversize-exec");
    p.model = model(hidden, 4, &[1, 2]);
    p.registry.copies = vec![
        moe::Replica {
            expert: 1,
            rank: 2,
            local: 0,
        },
        moe::Replica {
            expert: 2,
            rank: 9,
            local: 0,
        },
    ];
    p.batch.inputs = vec![vec![0.0; hidden]; tokens];
    p.batch.routes = (0..tokens)
        .map(|t| {
            vec![moe::Choice {
                expert: if t == 0 { 1 } else { 2 },
                weight: 1.0,
            }]
        })
        .collect();
    let mut sessions = session::prepare(&[endpoint(2), endpoint(9)], &p, T).unwrap();
    let mut big = p.clone();
    for (t, row) in big.batch.inputs.iter_mut().enumerate() {
        for (i, x) in row.iter_mut().enumerate() {
            *x = 0.1 + (t * hidden + i) as f64 * 1e-7;
        }
    }
    let error = session::run_batch(&mut sessions, &big, &BTreeMap::new(), false, T).unwrap_err();
    assert_reason(
        &error,
        "MESSAGE_TOO_LARGE",
        "private-pipe JSON message exceeds 16 MiB",
    );
    assert!(sessions
        .iter()
        .all(|s| s.traffic.execs == 0 && s.is_usable()));
    // Nothing was attempted: the same batch identity still runs with activations that fit.
    let report = session::run_batch(&mut sessions, &p, &BTreeMap::new(), false, T).unwrap();
    assert!(report.ok, "{:?}", report.error);
    for s in sessions {
        assert_eq!(s.close(T).exit_code, Some(0));
    }
}
