//! A53-R7: a series fault must name one of the series' own batches, at most once per
//! rank. Before R7 a fault past the last batch was never injected (the series reported
//! ok) and a second fault for the same batch and rank silently replaced the first.
//! Both series kinds share the bound and refuse before any worker or artifact file.
mod common;

use voltafabric::process::Fault;
use voltafabric::session::{self, CompressedSeriesInput, Endpoint, SeriesFault, SeriesInput};
use voltafabric::{routing, Error};

const FAULTS: &str = "a series fault names one of its own batches, at most once per rank";

fn ghost(rank: u32) -> Endpoint {
    Endpoint {
        rank,
        argv: vec!["/nonexistent/vf-rank-session".into()],
    }
}

fn fault(batch: u64, rank: u32, fault: Fault) -> SeriesFault {
    SeriesFault { batch, rank, fault }
}

fn dense(faults: Vec<SeriesFault>) -> Error {
    let mut problem = common::problem();
    problem.stamp.run_id = "a53r7-dense".into();
    session::run_series(&SeriesInput {
        problem,
        workers: vec![ghost(2), ghost(9)],
        batches: 6,
        timeout_ms: 1000,
        epoch_every: 0,
        warmup: 0,
        tolerance_abs: 0.0,
        faults,
    })
    .unwrap_err()
}

fn compressed(faults: Vec<SeriesFault>) -> Error {
    session::run_compressed_series(&CompressedSeriesInput {
        artifacts: vec!["/nonexistent/a53r7.safetensors".into()],
        routing: routing::from_dense(&common::problem()),
        inputs: vec![],
        inputs_fp16_bits: None,
        workers: vec![ghost(2)],
        batches: 6,
        timeout_ms: 1000,
        epoch_every: 0,
        warmup: 0,
        tolerance_abs: 0.0,
        fp16_completions: false,
        faults,
        record: vec![],
    })
    .unwrap_err()
}

#[test]
fn a_fault_outside_the_series_or_twice_on_one_rank_is_refused_before_any_worker() {
    let refused: Vec<(&str, Vec<SeriesFault>)> = vec![
        (
            "past the last batch",
            vec![fault(6, 9, Fault::RejectBeforeCompute)],
        ),
        ("far past", vec![fault(u64::MAX, 2, Fault::DieAfterStart)]),
        (
            "same batch and rank twice",
            vec![
                fault(3, 9, Fault::DieAfterStart),
                fault(3, 9, Fault::RejectBeforeCompute),
            ],
        ),
        (
            "same fault twice",
            vec![fault(0, 2, Fault::Missing), fault(0, 2, Fault::Missing)],
        ),
    ];
    for (name, faults) in refused {
        for (kind, error) in [
            ("dense", dense(faults.clone())),
            ("compressed", compressed(faults)),
        ] {
            assert_eq!(
                (error.code.as_str(), error.message.as_str()),
                ("INVALID_INPUT", FAULTS),
                "{kind}: {name}"
            );
        }
    }
    // Still admitted: the last batch, and one fault per rank in the same batch. The
    // missing worker or artifact file then fails, as before.
    let admitted = vec![
        fault(5, 9, Fault::RejectBeforeCompute),
        fault(5, 2, Fault::DieAfterStart),
        fault(0, 9, Fault::Missing),
    ];
    assert_eq!(dense(admitted.clone()).code, "WORKER_IO");
    assert_eq!(compressed(admitted).code, "ARTIFACT_IO");
}

#[test]
fn a_fault_on_an_absent_rank_or_out_of_bounds_is_refused_before_any_worker() {
    // A53-R12: those faults were refused only at the faulted batch, after every earlier
    // batch ran. The routing, hence the participating ranks, is the same for every batch.
    let flood = 16 * 1024 * 1024 + 1;
    for (name, faults, message) in [
        (
            "rank 7 takes no part",
            vec![fault(5, 7, Fault::Missing)],
            "fault refers to a nonparticipating rank",
        ),
        (
            "delay over 60 s",
            vec![fault(
                5,
                9,
                Fault::Delay {
                    milliseconds: 60_001,
                },
            )],
            "fault injection outside bounded CPU test contract",
        ),
        (
            "stderr flood over one frame",
            vec![fault(5, 2, Fault::StderrFlood { bytes: flood })],
            "fault injection outside bounded CPU test contract",
        ),
    ] {
        let error = dense(faults);
        assert_eq!(
            (error.code.as_str(), error.message.as_str()),
            ("INVALID_INPUT", message),
            "dense: {name}"
        );
    }
    let bound = vec![fault(
        5,
        9,
        Fault::Delay {
            milliseconds: 60_000,
        },
    )];
    assert_eq!(dense(bound).code, "WORKER_IO");
    // A malformed problem keeps the refusal it had before R12 (moe::validate), not the
    // routing one the fault check would otherwise give first.
    let mut problem = common::problem();
    problem.stamp.run_id = "a53r12-malformed".into();
    let twin = problem.model.experts[0].clone();
    problem.model.experts.push(twin);
    let error = session::run_series(&SeriesInput {
        problem,
        workers: vec![ghost(2), ghost(9)],
        batches: 6,
        timeout_ms: 1000,
        epoch_every: 0,
        warmup: 0,
        tolerance_abs: 0.0,
        faults: vec![fault(5, 9, Fault::Missing)],
    })
    .unwrap_err();
    assert_eq!(
        (error.code.as_str(), error.message.as_str()),
        (
            "INVALID_INPUT",
            "duplicate global expert or invalid matrix shape"
        )
    );
}

#[test]
fn an_out_of_bounds_fault_is_refused_by_one_shot_and_per_batch_runs_before_any_exec() {
    // A53-R27: R12 refused these faults before a series starts. A one-shot `process::run`
    // and a per-batch `run_batch` still claimed the batch identity and sent every rank its
    // work: the other rank computed and only the faulted rank's worker refused. In the same
    // coordinator the batch could then not run again without recovery.
    use std::collections::BTreeMap;
    use std::time::Duration;
    use voltafabric::process::{self, ProcessProblem};
    let bounds = "fault injection outside bounded CPU test contract";
    let over = [
        (
            9,
            Fault::Delay {
                milliseconds: 60_001,
            },
        ),
        (
            2,
            Fault::StderrFlood {
                bytes: 16 * 1024 * 1024 + 1,
            },
        ),
    ];
    let worker = std::path::Path::new(env!("CARGO_BIN_EXE_vf-rank-worker"));
    let t = Duration::from_secs(20);
    for (i, (rank, fault)) in over.iter().enumerate() {
        let mut problem = common::problem();
        problem.stamp.run_id = format!("a53r27-one-shot-{i}");
        let run = |faults: BTreeMap<u32, Fault>| {
            process::run(
                &ProcessProblem {
                    problem: problem.clone(),
                    faults,
                    timeout_ms: 20_000,
                    recovery: false,
                },
                worker,
            )
        };
        let error = run(BTreeMap::from([(*rank, fault.clone())])).unwrap_err();
        assert_eq!(
            (error.code.as_str(), error.message.as_str()),
            ("INVALID_INPUT", bounds),
            "one-shot, rank {rank}"
        );
        // Nothing was attempted: the same identity still runs.
        let report = run(BTreeMap::new()).unwrap();
        assert!(report.ok, "one-shot retry, rank {rank}: {:?}", report.error);

        let mut p = common::problem();
        p.stamp.run_id = format!("a53r27-per-batch-{i}");
        let mut sessions = session::prepare(
            &[
                Endpoint {
                    rank: 2,
                    argv: vec![env!("CARGO_BIN_EXE_vf-rank-session").into()],
                },
                Endpoint {
                    rank: 9,
                    argv: vec![env!("CARGO_BIN_EXE_vf-rank-session").into()],
                },
            ],
            &p,
            t,
        )
        .unwrap();
        let faults = BTreeMap::from([(*rank, fault.clone())]);
        let error = session::run_batch(&mut sessions, &p, &faults, false, t).unwrap_err();
        assert_eq!(
            (error.code.as_str(), error.message.as_str()),
            ("INVALID_INPUT", bounds),
            "per-batch, rank {rank}"
        );
        assert!(sessions.iter().all(|s| s.traffic.execs == 0));
        let report = session::run_batch(&mut sessions, &p, &BTreeMap::new(), false, t).unwrap();
        assert!(
            report.ok,
            "per-batch retry, rank {rank}: {:?}",
            report.error
        );
        for s in sessions {
            assert_eq!(s.close(t).exit_code, Some(0));
        }
    }
}
