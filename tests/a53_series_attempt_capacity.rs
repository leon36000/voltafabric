//! A53-R4: a series claims one coordinator identity per batch, and the coordinator keeps
//! at most 65536 per process. A series longer than the identities left must be refused
//! before any file is read or any worker starts, not fail partway through. This file is
//! its own test process because the identity history is process-wide state.
#![cfg(target_os = "linux")]
mod common;

use std::collections::BTreeMap;
use std::path::Path;
use voltafabric::process::{self, ProcessProblem};
use voltafabric::session::{self, CompressedSeriesInput, Endpoint, SeriesInput};
use voltafabric::{routing, Error};

const LEFT: u64 = 3;

fn endpoint(rank: u32) -> Endpoint {
    Endpoint {
        rank,
        argv: vec![env!("CARGO_BIN_EXE_vf-rank-session").into()],
    }
}

fn series(batches: u64) -> SeriesInput {
    let mut problem = common::problem();
    problem.stamp.run_id = format!("a53-capacity-{batches}");
    SeriesInput {
        problem,
        workers: vec![endpoint(2), endpoint(9)],
        batches,
        timeout_ms: 10_000,
        epoch_every: 0,
        warmup: 0,
        tolerance_abs: 0.0,
        faults: vec![],
    }
}

fn assert_capacity_refusal(error: &Error) {
    assert_eq!(error.code, "INVALID_INPUT", "{error}");
    assert!(
        error
            .message
            .contains("more batches than identities this coordinator can still attempt"),
        "{error}"
    );
}

#[test]
fn a_series_longer_than_the_identities_left_is_refused_before_it_starts() {
    // Fill the history up to LEFT free identities; an empty batch spawns no worker.
    let worker = Path::new(env!("CARGO_BIN_EXE_vf-rank-worker"));
    let mut fill = ProcessProblem {
        problem: common::problem(),
        faults: BTreeMap::new(),
        timeout_ms: 5000,
        recovery: false,
    };
    fill.problem.batch.inputs.clear();
    fill.problem.batch.routes.clear();
    for i in 0..65536 - LEFT {
        fill.problem.stamp.run_id = format!("fill-{i}");
        assert!(process::run(&fill, worker).unwrap().ok, "identity {i}");
    }

    // One batch too many: refused whole, with real workers that would otherwise run.
    let error = session::run_series(&series(LEFT + 1))
        .expect_err("a series past the coordinator attempt limit must not start");
    assert_capacity_refusal(&error);

    // The compressed series shares the bound, and refuses before reading its artifacts.
    let compressed = CompressedSeriesInput {
        artifacts: vec!["/nonexistent/a53-capacity.safetensors".into()],
        routing: routing::from_dense(&common::problem()),
        inputs: vec![],
        inputs_fp16_bits: None,
        workers: vec![endpoint(2)],
        batches: LEFT + 1,
        timeout_ms: 10_000,
        epoch_every: 0,
        warmup: 0,
        tolerance_abs: 0.0,
        fp16_completions: false,
        faults: vec![],
        record: vec![],
    };
    let error = session::run_compressed_series(&compressed)
        .expect_err("a compressed series past the coordinator attempt limit must not start");
    assert_capacity_refusal(&error);

    // Exactly the identities left: the refusals above claimed none, and every batch runs.
    let report = session::run_series(&series(LEFT)).unwrap();
    assert!(report.ok, "{:?} {:?}", report.error, report.failure);
    assert_eq!((report.batches_ok, report.exact_batches), (LEFT, LEFT));

    // Nothing is left: even one batch is refused before it starts.
    let error = session::run_series(&series(1)).expect_err("no identity is left");
    assert_capacity_refusal(&error);
}
