//! A52.1: persistent rank-worker sessions. Real child processes speak the session
//! protocol over pipes, exactly as they do over `ssh -T`; weights cross the stream once
//! per session and every batch is checked against the dense CPU oracle.
#![cfg(target_os = "linux")]
mod common;

use std::collections::BTreeMap;
use std::process::Command;
use std::time::Duration;
use voltafabric::moe::{self, Problem};
use voltafabric::process::{Fault, ReceiptState, TokenInput};
use voltafabric::session::{self, Endpoint, Reply, Request, Session};
use voltafabric::{routing, Error};

const T: Duration = Duration::from_secs(10);

fn endpoint(rank: u32) -> Endpoint {
    Endpoint {
        rank,
        argv: vec![env!("CARGO_BIN_EXE_vf-rank-session").into()],
    }
}
fn problem(run_id: &str) -> Problem {
    let mut p = common::problem();
    p.stamp.run_id = run_id.into();
    p
}
/// Ranks 2 (experts 10 and 99) and 9 (expert 25), loaded for `p`.
fn sessions(p: &Problem) -> Vec<Session> {
    session::prepare(&[endpoint(2), endpoint(9)], p, T).unwrap()
}
/// Batch `b` of a run: new batch number, deterministic new activations.
fn batch(base: &Problem, b: u64) -> Problem {
    let mut p = base.clone();
    p.stamp.batch = base.stamp.batch + b;
    for (t, row) in p.batch.inputs.iter_mut().enumerate() {
        for (i, x) in row.iter_mut().enumerate() {
            *x *= 1.0 + ((b as usize * 7 + t * 3 + i) % 11) as f64 / 64.0;
        }
    }
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
fn inputs(p: &Problem, d: &routing::RankDispatch) -> Vec<TokenInput> {
    let tokens: std::collections::BTreeSet<_> = d.slots.iter().map(|s| s.slot.token).collect();
    tokens
        .into_iter()
        .map(|token| TokenInput {
            token,
            value: p.batch.inputs[token].clone(),
        })
        .collect()
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

#[test]
fn hello_reports_protocol_architecture_bounds_and_rank() {
    let s = Session::connect(&endpoint(9), T).unwrap();
    let hello = &s.hello;
    assert_eq!(hello.protocol, session::PROTOCOL);
    assert_eq!(hello.rank, 9);
    assert_eq!(hello.arch, std::env::consts::ARCH);
    assert_eq!(hello.endian, "little");
    // HELLO advertises every implemented protocol capability, including batched raw
    // completions and raw Engram rows with negotiated compact row IDs.
    assert_eq!(
        hello.capabilities,
        vec![
            session::CAPABILITY.to_string(),
            session::mxfp4::CAPABILITY.to_string(),
            session::mxfp4::BF16_ACTIVATION_CAPABILITY.to_string(),
            session::CAPABILITY_QPN2.to_string(),
            session::CAPABILITY_QPN2_RAW.to_string(),
            session::CAPABILITY_EXEC_F64_RAW.to_string(),
            session::CAPABILITY_EXEC_FP16_RAW.to_string(),
            session::CAPABILITY_COMPLETION_F64_RAW.to_string(),
            session::CAPABILITY_COMPLETION_BATCH_F64_RAW.to_string(),
            session::CAPABILITY_COMPLETION_BATCH_SESSION_F64_RAW.to_string(),
            session::CAPABILITY_COMPLETION_BATCH_FP16_RAW.to_string(),
            session::CAPABILITY_ENGRAM_ROWS_RAW.to_string(),
            session::CAPABILITY_ENGRAM_ROW_IDS_VARINT.to_string()
        ]
    );
    assert_eq!(hello.max_message_bytes, 16 * 1024 * 1024);
    assert!(hello.pid > 0);
    let receipt = s.close(T);
    assert_eq!(receipt.exit_code, Some(0), "{receipt:?}");
    assert_eq!(receipt.bye_execs, Some(0));
}

#[test]
fn persistent_sessions_serve_120_batches_with_weights_sent_once() {
    let base = problem("a52-persistent");
    let mut sessions = sessions(&base);
    let loaded: Vec<_> = sessions.iter().map(|s| s.traffic.clone()).collect();
    for b in 0..120 {
        let mut p = batch(&base, b);
        // Renew the epoch every 40 batches: rebind, never resend the weights.
        p.stamp.epoch = base.stamp.epoch + b / 40;
        p.registry.epoch = p.stamp.epoch;
        session::bind(&mut sessions, &p, T).unwrap();
        let report = session::run_batch(&mut sessions, &p, &BTreeMap::new(), false, T).unwrap();
        assert!(report.ok, "batch {b}: {:?}", report.error);
        assert_eq!(
            report.output.unwrap(),
            moe::reference(&p).unwrap(),
            "batch {b}"
        );
        assert!(report
            .receipts
            .iter()
            .all(|r| r.state == ReceiptState::Completed));
    }
    for (s, at_load) in sessions.iter().zip(&loaded) {
        assert_eq!(s.traffic.loads_transferred, 1, "rank {}", s.rank);
        assert_eq!(s.traffic.rebinds, 2, "rank {}", s.rank);
        assert_eq!(s.traffic.execs, 120, "rank {}", s.rank);
        // Every weight byte was sent before the first batch.
        assert_eq!(s.traffic.weight_bytes_sent, at_load.weight_bytes_sent);
        assert!(s.traffic.weight_bytes_sent > 0);
    }
    for s in sessions {
        let receipt = s.close(T);
        assert_eq!(receipt.exit_code, Some(0), "{receipt:?}");
        assert_eq!(receipt.bye_execs, Some(120));
        assert!(!receipt.killed);
    }
}

#[test]
fn exec_frames_carry_activations_and_slots_but_never_weights() {
    let p = problem("a52-no-weights");
    let d = rank_dispatch(&p, 2);
    let model = session::resident_model(&p, &d).unwrap();
    let request = Request::Exec {
        artifact_sha256: session::artifact_digest(&model).unwrap(),
        dispatch: d.clone(),
        attempt: 1,
        inputs: inputs(&p, &d),
        fault: Fault::None,
    };
    let text = serde_json::to_string(&request).unwrap();
    for weight_field in ["\"gate\"", "\"up\"", "\"down\"", "\"experts\""] {
        assert!(!text.contains(weight_field), "{weight_field} in {text}");
    }
}

#[test]
fn random_routes_and_activations_match_the_oracle() {
    // Bounded property check: 60 batches, fresh routes over the same residents.
    let base = problem("a52-property");
    let mut sessions = sessions(&base);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for b in 0..60 {
        let mut p = batch(&base, b);
        for (t, route) in p.batch.routes.iter_mut().enumerate() {
            let first = [10, 25, 99][(next() % 3) as usize];
            let second = [10, 25, 99]
                .into_iter()
                .filter(|e| *e != first)
                .nth((next() % 2) as usize)
                .unwrap();
            *route = vec![
                moe::Choice {
                    expert: first,
                    weight: (next() % 1000) as f64 / 1000.0,
                },
                moe::Choice {
                    expert: second,
                    weight: (next() % 1000) as f64 / 1000.0,
                },
            ];
            for x in &mut p.batch.inputs[t] {
                *x = (next() % 2001) as f64 / 1000.0 - 1.0;
            }
        }
        let report = session::run_batch(&mut sessions, &p, &BTreeMap::new(), false, T).unwrap();
        assert!(report.ok, "batch {b}: {:?}", report.error);
        assert_eq!(
            report.output.unwrap(),
            moe::reference(&p).unwrap(),
            "batch {b}"
        );
    }
    for s in sessions {
        assert_eq!(s.close(T).exit_code, Some(0));
    }
}

#[test]
fn out_of_order_completions_still_match_the_oracle() {
    let p = problem("a52-reverse");
    let mut sessions = sessions(&p);
    let faults = BTreeMap::from([(2, Fault::Reverse)]);
    let report = session::run_batch(&mut sessions, &p, &faults, false, T).unwrap();
    assert!(report.ok, "{:?}", report.error);
    assert_eq!(report.output.unwrap(), moe::reference(&p).unwrap());
}

#[test]
fn duplicate_completion_aborts_the_batch_without_output() {
    let p = problem("a52-duplicate");
    let mut sessions = sessions(&p);
    let faults = BTreeMap::from([(2, Fault::Duplicate)]);
    let report = session::run_batch(&mut sessions, &p, &faults, false, T).unwrap();
    assert!(!report.ok);
    assert!(report.output.is_none());
    assert_reason(
        report.error.as_ref().unwrap(),
        "INVALID_COMPLETION",
        "duplicate, stale, unexpected or malformed result; entire batch aborted",
    );
    // The stream position after an aborted batch is not trusted: no reuse, no retry.
    assert!(!sessions[0].is_usable());
}

#[test]
fn stale_epoch_completion_aborts_the_batch() {
    let p = problem("a52-stale-completion");
    let mut sessions = sessions(&p);
    let faults = BTreeMap::from([(9, Fault::StaleEpoch)]);
    let report = session::run_batch(&mut sessions, &p, &faults, false, T).unwrap();
    assert!(!report.ok);
    assert!(report.output.is_none());
    assert_eq!(report.error.unwrap().code, "INVALID_COMPLETION");
}

#[test]
fn replayed_or_older_attempts_are_refused_before_execution() {
    let p = problem("a52-attempts");
    let mut s = Session::connect(&endpoint(2), T).unwrap();
    let d = rank_dispatch(&p, 2);
    let model = session::resident_model(&p, &d).unwrap();
    s.load(
        &model,
        &p.stamp.model_revision,
        p.stamp.layer,
        p.stamp.epoch,
        &d.resident_order,
        T,
    )
    .unwrap();
    let mut collector = moe::Collector::new(&p).unwrap();
    let first = s
        .exec(&d, 50, inputs(&p, &d), Fault::None, &mut collector, T)
        .unwrap();
    assert_eq!(first.state, ReceiptState::Completed, "{first:?}");
    for attempt in [50, 49] {
        let mut collector = moe::Collector::new(&p).unwrap();
        let again = s
            .exec(&d, attempt, inputs(&p, &d), Fault::None, &mut collector, T)
            .unwrap();
        assert_eq!(again.state, ReceiptState::NotStarted, "{again:?}");
        assert_reason(
            again.error.as_ref().unwrap(),
            "STALE_ATTEMPT",
            "attempt is not newer than this session's last execution",
        );
        assert!(s.is_usable());
    }
    assert_eq!(s.close(T).bye_execs, Some(1));
}

#[test]
fn old_epoch_wrong_revision_and_wrong_rank_are_refused() {
    let p = problem("a52-identity");
    let mut s = Session::connect(&endpoint(2), T).unwrap();
    let d = rank_dispatch(&p, 2);
    let model = session::resident_model(&p, &d).unwrap();
    let rev = p.stamp.model_revision.clone();
    s.load(
        &model,
        &rev,
        p.stamp.layer,
        p.stamp.epoch,
        &d.resident_order,
        T,
    )
    .unwrap();
    let rebound = s
        .load(
            &model,
            &rev,
            p.stamp.layer,
            p.stamp.epoch + 1,
            &d.resident_order,
            T,
        )
        .unwrap();
    assert!(!rebound.transferred);
    let mut attempt = 10;
    let mut refuse = |d: &routing::RankDispatch, s: &mut Session, code: &str, message: &str| {
        attempt += 1;
        let mut collector = moe::Collector::new(&p).unwrap();
        let r = s
            .exec(d, attempt, inputs(&p, d), Fault::None, &mut collector, T)
            .unwrap();
        assert_eq!(r.state, ReceiptState::NotStarted, "{r:?}");
        assert_reason(r.error.as_ref().unwrap(), code, message);
    };
    // The dispatch still carries the previous epoch.
    refuse(
        &d,
        &mut s,
        "STALE_EXECUTION",
        "dispatch identity differs from the loaded artifact epoch, revision or layer",
    );
    let mut wrong = d.clone();
    wrong.stamp.epoch += 1;
    wrong.stamp.model_revision.push('x');
    refuse(
        &wrong,
        &mut s,
        "STALE_EXECUTION",
        "dispatch identity differs from the loaded artifact epoch, revision or layer",
    );
    let mut foreign = d.clone();
    foreign.stamp.epoch += 1;
    foreign.rank = 9;
    refuse(
        &foreign,
        &mut s,
        "INVALID_INPUT",
        "dispatch rank or physical order differs from this session",
    );
    assert!(s.is_usable());
    assert_eq!(s.close(T).exit_code, Some(0));
}

#[test]
fn epoch_rollback_wrong_digest_and_second_artifact_are_refused_on_load() {
    let p = problem("a52-load");
    let mut s = Session::connect(&endpoint(2), T).unwrap();
    let d = rank_dispatch(&p, 2);
    let model = session::resident_model(&p, &d).unwrap();
    let digest = session::artifact_digest(&model).unwrap();
    let rev = &p.stamp.model_revision;
    let load = |model: Option<moe::Model>, sha: &str, epoch: u64| Request::Load {
        artifact_sha256: sha.into(),
        model_revision: rev.clone(),
        layer: p.stamp.layer,
        epoch,
        resident_order: d.resident_order.clone(),
        model,
    };
    s.send(&load(Some(model.clone()), &"0".repeat(64), 7))
        .unwrap();
    assert_reason(
        &refused(s.reply(T).unwrap()),
        "ARTIFACT_DIGEST_MISMATCH",
        "artifact_sha256 differs from the SHA-256 of the received model",
    );
    s.send(&load(Some(model.clone()), &digest, 7)).unwrap();
    assert!(matches!(
        s.reply(T).unwrap(),
        Reply::Loaded {
            transferred: true,
            ..
        }
    ));
    for (epoch, code, message) in [
        (
            6,
            "EPOCH_ROLLBACK",
            "ownership epoch is older than the loaded one",
        ),
        (
            7,
            "EPOCH_NOT_NEWER",
            "rebinding requires a newer ownership epoch",
        ),
    ] {
        s.send(&load(None, &digest, epoch)).unwrap();
        assert_reason(&refused(s.reply(T).unwrap()), code, message);
    }
    let mut other = model.clone();
    other.experts[0].gate[0] += 1.0;
    let other_digest = session::artifact_digest(&other).unwrap();
    s.send(&load(Some(other), &other_digest, 8)).unwrap();
    assert_reason(
        &refused(s.reply(T).unwrap()),
        "ARTIFACT_LOADED",
        "evict the loaded artifact before loading another",
    );
    s.send(&load(None, &other_digest, 8)).unwrap();
    assert_reason(
        &refused(s.reply(T).unwrap()),
        "ARTIFACT_NOT_LOADED",
        "no loaded artifact has this SHA-256",
    );
    assert_eq!(s.close(T).exit_code, Some(0));
}

#[test]
fn exec_before_load_and_after_evict_is_refused() {
    let p = problem("a52-evict");
    let mut s = Session::connect(&endpoint(2), T).unwrap();
    let d = rank_dispatch(&p, 2);
    let model = session::resident_model(&p, &d).unwrap();
    let digest = session::artifact_digest(&model).unwrap();
    let exec = |attempt| Request::Exec {
        artifact_sha256: digest.clone(),
        dispatch: d.clone(),
        attempt,
        inputs: inputs(&p, &d),
        fault: Fault::None,
    };
    s.send(&exec(1)).unwrap();
    assert_reason(
        &refused(s.reply(T).unwrap()),
        "ARTIFACT_NOT_LOADED",
        "no loaded artifact has this SHA-256",
    );
    s.load(
        &model,
        &p.stamp.model_revision,
        p.stamp.layer,
        p.stamp.epoch,
        &d.resident_order,
        T,
    )
    .unwrap();
    s.send(&Request::Evict {
        artifact_sha256: digest.clone(),
    })
    .unwrap();
    assert!(matches!(s.reply(T).unwrap(), Reply::Evicted { .. }));
    s.send(&exec(2)).unwrap();
    assert_reason(
        &refused(s.reply(T).unwrap()),
        "ARTIFACT_NOT_LOADED",
        "no loaded artifact has this SHA-256",
    );
    assert_eq!(s.close(T).bye_execs, Some(0));
}

#[test]
fn empty_rank_dispatch_is_refused() {
    let p = problem("a52-empty");
    let mut s = Session::connect(&endpoint(2), T).unwrap();
    let mut d = rank_dispatch(&p, 2);
    let model = session::resident_model(&p, &d).unwrap();
    s.load(
        &model,
        &p.stamp.model_revision,
        p.stamp.layer,
        p.stamp.epoch,
        &d.resident_order,
        T,
    )
    .unwrap();
    d.slots.clear();
    let mut collector = moe::Collector::new(&p).unwrap();
    let r = s
        .exec(&d, 3, vec![], Fault::None, &mut collector, T)
        .unwrap();
    assert_eq!(r.state, ReceiptState::NotStarted);
    assert_reason(
        r.error.as_ref().unwrap(),
        "INVALID_INPUT",
        "invalid rank-local dimensions or physical order",
    );
    assert_eq!(s.close(T).exit_code, Some(0));
}

#[test]
fn nonfinite_expert_output_fails_after_start_without_output() {
    // Finite weights and inputs whose SwiGLU product overflows f64 on the worker.
    let mut p = problem("a52-nonfinite");
    p.batch.inputs = vec![vec![1.0, 1.0], vec![0.5, 0.5]];
    for e in &mut p.model.experts {
        for w in e.gate.iter_mut().chain(&mut e.up) {
            *w = 1e200;
        }
    }
    let mut sessions = sessions(&p);
    let report = session::run_batch(&mut sessions, &p, &BTreeMap::new(), false, T).unwrap();
    assert!(!report.ok);
    assert!(report.output.is_none());
    let failed = report.receipts.iter().find(|r| r.error.is_some()).unwrap();
    assert_eq!(failed.state, ReceiptState::Started, "{failed:?}");
    assert_reason(
        failed.error.as_ref().unwrap(),
        "NUMERICAL_OVERFLOW",
        "CPU intermediate is nonfinite",
    );
}

#[test]
fn crash_before_and_after_started_are_distinguishable_and_reaped() {
    for (fault, started, code) in [
        (Fault::DieBeforeStart, false, 71),
        (Fault::DieAfterStart, true, 72),
    ] {
        let p = problem(&format!("a52-crash-{started}"));
        let mut sessions = sessions(&p);
        let faults = BTreeMap::from([(9, fault)]);
        let report = session::run_batch(&mut sessions, &p, &faults, false, T).unwrap();
        assert!(!report.ok);
        assert!(report.output.is_none());
        let r = report.receipts.iter().find(|r| r.rank == 9).unwrap();
        assert_eq!(r.state, ReceiptState::Unknown, "{r:?}");
        assert_eq!(r.started, started, "{r:?}");
        assert_eq!(r.error.as_ref().unwrap().code, "SESSION_CLOSED");
        let crashed = sessions.pop().unwrap();
        assert!(!crashed.is_usable());
        let receipt = crashed.close(T);
        assert_eq!(receipt.exit_code, Some(code), "{receipt:?}");
        assert_eq!(receipt.bye_execs, None);
    }
}

#[test]
fn stream_cut_during_execution_yields_unknown_without_partial_output() {
    let p = problem("a52-cut");
    let mut sessions = sessions(&p);
    let pid = sessions[1].pid();
    let killer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        Command::new("kill")
            .args(["-9", &pid.to_string()])
            .status()
            .unwrap()
    });
    let faults = BTreeMap::from([(9, Fault::Delay { milliseconds: 3000 })]);
    let report = session::run_batch(&mut sessions, &p, &faults, false, T).unwrap();
    assert!(killer.join().unwrap().success());
    assert!(!report.ok);
    assert!(report.output.is_none());
    let r = report.receipts.iter().find(|r| r.rank == 9).unwrap();
    assert_eq!((r.state, r.started), (ReceiptState::Unknown, true), "{r:?}");
    let receipt = sessions.pop().unwrap().close(T);
    assert_eq!(receipt.exit_code, None, "killed by a signal: {receipt:?}");
}

#[test]
fn a_silent_worker_times_out_as_unknown_and_is_killed() {
    let p = problem("a52-timeout");
    let mut sessions = sessions(&p);
    let faults = BTreeMap::from([(9, Fault::Delay { milliseconds: 5000 })]);
    let report = session::run_batch(
        &mut sessions,
        &p,
        &faults,
        false,
        Duration::from_millis(500),
    )
    .unwrap();
    assert!(!report.ok);
    let r = report.receipts.iter().find(|r| r.rank == 9).unwrap();
    assert_eq!(r.state, ReceiptState::Unknown);
    assert_reason(
        r.error.as_ref().unwrap(),
        "SESSION_TIMEOUT",
        "no session reply before the deadline",
    );
    let receipt = sessions.pop().unwrap().close(Duration::from_millis(500));
    assert!(receipt.killed, "{receipt:?}");
}

#[test]
fn truncated_frame_closes_the_session() {
    let mut s = Session::connect(&endpoint(2), T).unwrap();
    s.send_raw(b"{\"op\":\"shutdown\"").unwrap();
    let receipt = s.close_input(T);
    assert_eq!(receipt.exit_code, Some(2), "{receipt:?}");
    assert_eq!(
        receipt.stderr, "WORKER_PROTOCOL: unterminated JSON frame\n",
        "{receipt:?}"
    );
}

#[test]
fn oversize_frame_is_refused_and_closes_the_session() {
    let mut s = Session::connect(&endpoint(2), T).unwrap();
    let mut line = vec![b' '; 16 * 1024 * 1024 + 1];
    line.push(b'\n');
    s.send_raw(&line).unwrap();
    assert_reason(
        &refused(s.reply(T).unwrap()),
        "MESSAGE_TOO_LARGE",
        "private-pipe JSON message exceeds 16 MiB",
    );
    let receipt = s.close_input(T);
    assert_eq!(receipt.exit_code, Some(2), "{receipt:?}");
}

#[test]
fn oversize_exec_is_refused_before_it_is_sent() {
    let p = problem("a52-too-large");
    let mut s = Session::connect(&endpoint(2), T).unwrap();
    let d = rank_dispatch(&p, 2);
    let model = session::resident_model(&p, &d).unwrap();
    s.load(
        &model,
        &p.stamp.model_revision,
        p.stamp.layer,
        p.stamp.epoch,
        &d.resident_order,
        T,
    )
    .unwrap();
    let before = s.traffic.bytes_sent;
    let mut big = inputs(&p, &d);
    big[0].value = vec![0.123_456_789_012_345_67; 2_000_000];
    let mut collector = moe::Collector::new(&p).unwrap();
    let error = s
        .exec(&d, 4, big, Fault::None, &mut collector, T)
        .unwrap_err();
    assert_reason(
        &error,
        "MESSAGE_TOO_LARGE",
        "private-pipe JSON message exceeds 16 MiB",
    );
    assert_eq!(s.traffic.bytes_sent, before);
    assert!(s.is_usable());
    assert_eq!(s.close(T).exit_code, Some(0));
}

#[test]
fn protocol_version_mismatch_is_refused_and_closes() {
    let mut s = Session::spawn(&endpoint(2)).unwrap();
    s.send(&Request::Hello {
        protocol: session::PROTOCOL + 1,
        rank: 2,
    })
    .unwrap();
    assert_reason(
        &refused(s.reply(T).unwrap()),
        "PROTOCOL_MISMATCH",
        "session protocol version differs",
    );
    assert_eq!(s.close_input(T).exit_code, Some(2));
}

#[test]
fn connecting_to_an_unreachable_worker_fails_closed() {
    let missing = Endpoint {
        rank: 2,
        argv: vec!["/definitely-missing-vf-rank-session".into()],
    };
    let error = Session::connect(&missing, T).unwrap_err();
    assert_reason(
        &error,
        "WORKER_IO",
        "spawn session worker: No such file or directory (os error 2)",
    );
    // A transport that starts but closes at once, as ssh does for a refused port.
    let closed = Endpoint {
        rank: 2,
        argv: vec!["/bin/sh".into(), "-c".into(), "exit 255".into()],
    };
    let error = Session::connect(&closed, T).unwrap_err();
    assert_eq!(error.code, "SESSION_CLOSED", "{error}");
}

#[test]
fn shutdown_reports_executions_and_reaps_the_worker() {
    let p = problem("a52-shutdown");
    let mut sessions = sessions(&p);
    for b in 0..3 {
        let p = batch(&p, b);
        assert!(
            session::run_batch(&mut sessions, &p, &BTreeMap::new(), false, T)
                .unwrap()
                .ok
        );
    }
    for s in sessions {
        let pid = s.pid();
        let receipt = s.close(T);
        assert_eq!((receipt.exit_code, receipt.bye_execs), (Some(0), Some(3)));
        assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists());
    }
}

fn series(run_id: &str, batches: u64, workers: Vec<Endpoint>) -> session::SeriesInput {
    session::SeriesInput {
        problem: problem(run_id),
        workers,
        batches,
        timeout_ms: 10_000,
        epoch_every: 50,
        warmup: 10,
        tolerance_abs: 0.0,
        faults: vec![],
    }
}

#[test]
fn series_of_150_batches_reuses_two_sessions_and_matches_the_oracle() {
    let report =
        session::run_series(&series("a52-series", 150, vec![endpoint(2), endpoint(9)])).unwrap();
    assert!(report.ok, "{:?} {:?}", report.error, report.failure);
    assert_eq!((report.batches_ok, report.exact_batches), (150, 150));
    assert_eq!(report.max_abs_diff, 0.0);
    assert_eq!(report.batch_latency.as_ref().unwrap().n, 140);
    assert_eq!(report.rebind_latency.as_ref().unwrap().n, 2);
    assert_eq!(report.exec_requests, 300);
    for t in &report.traffic {
        assert_eq!(
            (
                t.traffic.loads_transferred,
                t.traffic.rebinds,
                t.traffic.execs
            ),
            (1, 2, 150),
            "rank {}",
            t.rank
        );
    }
    for c in &report.close {
        assert_eq!((c.exit_code, c.bye_execs), (Some(0), Some(150)), "{c:?}");
        assert!(c.peak_rss_kib.is_some_and(|kib| kib > 0), "{c:?}");
    }
}

#[test]
fn series_with_a_missing_worker_fails_closed_before_any_batch() {
    let missing = Endpoint {
        rank: 9,
        argv: vec!["/definitely-missing-vf-rank-session".into()],
    };
    let error = session::run_series(&series(
        "a52-series-missing",
        20,
        vec![endpoint(2), missing],
    ))
    .unwrap_err();
    assert_eq!(error.code, "WORKER_IO", "{error}");
}

#[test]
fn session_run_cli_reports_a_verified_series() {
    let dir = std::env::temp_dir().join(format!("vf-a52-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("series.json");
    let body = series("a52-cli", 30, vec![endpoint(2), endpoint(9)]);
    std::fs::write(&input, serde_json::to_vec(&body).unwrap()).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_vf"))
        .args(["session-run".as_ref(), input.as_os_str()])
        .output()
        .unwrap();
    std::fs::remove_file(&input).unwrap();
    std::fs::remove_dir(&dir).unwrap();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(value["ok"], true);
    assert_eq!(value["result"]["batches_ok"], 30);
    assert_eq!(value["result"]["hellos"][1]["rank"], 9);
}

#[test]
fn latency_percentiles_use_the_nearest_rank() {
    let samples: Vec<f64> = (1..=20).map(f64::from).collect();
    let stats = session::LatencyStats::of(&samples).unwrap();
    assert_eq!(
        (
            stats.min_us,
            stats.p50_us,
            stats.p95_us,
            stats.max_us,
            stats.mean_us
        ),
        (1.0, 10.0, 19.0, 20.0, 10.5)
    );
    assert!(session::LatencyStats::of(&[]).is_none());
}

#[test]
fn series_stops_at_an_injected_fault_without_retry_or_output() {
    let mut input = series("a52-series-fault", 40, vec![endpoint(2), endpoint(9)]);
    input.faults = vec![session::SeriesFault {
        batch: 25,
        rank: 9,
        fault: Fault::DieAfterStart,
    }];
    let report = session::run_series(&input).unwrap();
    assert!(!report.ok);
    assert_eq!(report.batches_ok, 25);
    let failure = report.failure.unwrap();
    assert_eq!(failure.batch, 25);
    assert_eq!(failure.error.unwrap().code, "SESSION_CLOSED");
    let r = failure.receipts.iter().find(|r| r.rank == 9).unwrap();
    assert_eq!((r.state, r.started), (ReceiptState::Unknown, true));
    let crashed = report.close.iter().find(|c| c.rank == 9).unwrap();
    assert_eq!((crashed.exit_code, crashed.bye_execs), (Some(72), None));
    let healthy = report.close.iter().find(|c| c.rank == 2).unwrap();
    assert_eq!((healthy.exit_code, healthy.bye_execs), (Some(0), Some(26)));
}

#[test]
fn batch_whose_weights_differ_from_the_loaded_artifact_is_refused() {
    let p = problem("a52-changed-weights");
    let mut sessions = sessions(&p);
    let mut changed = p.clone();
    changed.stamp.batch += 1;
    changed.model.experts[2].down[0] += 1e-9;
    let error =
        session::run_batch(&mut sessions, &changed, &BTreeMap::new(), false, T).unwrap_err();
    assert_reason(
        &error,
        "STALE_EXECUTION",
        "session does not hold this batch's resident artifact and epoch; bind first",
    );
    // Binding transfers the new artifact; the batch then runs on it.
    session::bind(&mut sessions, &changed, T).unwrap();
    let report = session::run_batch(&mut sessions, &changed, &BTreeMap::new(), false, T).unwrap();
    assert_eq!(report.output.unwrap(), moe::reference(&changed).unwrap());
    let rank2 = sessions.iter().find(|s| s.rank == 2).unwrap();
    assert_eq!(rank2.traffic.loads_transferred, 2);
}
