//! A53-R9: batch b of a series is numbered base.batch + b and, with epoch_every > 0,
//! runs at epoch base.epoch + b / epoch_every. Before R9 both sums saturated: two batches
//! got one identity and the series failed mid-run as a replay, or epoch renewals silently
//! stopped at u64::MAX while the series reported ok. Both series kinds now refuse such
//! a numbering before any worker or artifact file.
mod common;

use voltafabric::session::{self, CompressedSeriesInput, Endpoint, SeriesInput};
use voltafabric::{routing, Error};

const NUMBERING: &str = "series batch or epoch numbers would overflow u64";

fn ghost(rank: u32) -> Endpoint {
    Endpoint {
        rank,
        argv: vec!["/nonexistent/vf-rank-session".into()],
    }
}

fn dense(batch: u64, epoch: u64, batches: u64, epoch_every: u64) -> Error {
    let mut problem = common::problem();
    problem.stamp.run_id = "a53r9-dense".into();
    problem.stamp.batch = batch;
    problem.stamp.epoch = epoch;
    problem.registry.epoch = epoch;
    session::run_series(&SeriesInput {
        problem,
        workers: vec![ghost(2), ghost(9)],
        batches,
        timeout_ms: 1000,
        epoch_every,
        warmup: 0,
        tolerance_abs: 0.0,
        faults: vec![],
    })
    .unwrap_err()
}

fn compressed(batch: u64, epoch: u64, batches: u64, epoch_every: u64) -> Error {
    let mut p = routing::from_dense(&common::problem());
    p.stamp.batch = batch;
    p.stamp.epoch = epoch;
    p.registry.epoch = epoch;
    session::run_compressed_series(&CompressedSeriesInput {
        artifacts: vec!["/nonexistent/a53r9.safetensors".into()],
        routing: p,
        inputs: vec![],
        inputs_fp16_bits: None,
        workers: vec![ghost(2)],
        batches,
        timeout_ms: 1000,
        epoch_every,
        warmup: 0,
        tolerance_abs: 0.0,
        fp16_completions: false,
        faults: vec![],
        record: vec![],
    })
    .unwrap_err()
}

#[test]
fn a_series_whose_batch_or_epoch_numbers_overflow_is_refused_before_any_worker() {
    const M: u64 = u64::MAX;
    // (name, base batch, base epoch, batches, epoch_every)
    let refused = [
        ("last batch past u64::MAX", M - 2, 1, 4, 0),
        ("batch u64::MAX, two batches", M, 1, 2, 0),
        ("last epoch past u64::MAX", 0, M - 1, 4, 1),
        ("epoch renewal every 2 batches past u64::MAX", 0, M, 3, 2),
    ];
    for (name, batch, epoch, batches, every) in refused {
        for (kind, error) in [
            ("dense", dense(batch, epoch, batches, every)),
            ("compressed", compressed(batch, epoch, batches, every)),
        ] {
            assert_eq!(
                (error.code.as_str(), error.message.as_str()),
                ("INVALID_INPUT", NUMBERING),
                "{kind}: {name}"
            );
        }
    }
    // Numberings that end exactly at u64::MAX, or never renew the epoch, are admitted;
    // the missing worker or artifact file then fails, as before.
    let admitted = [
        ("last batch is u64::MAX", M - 3, 1, 4, 0),
        ("one batch at u64::MAX", M, 1, 1, 0),
        ("last epoch is u64::MAX", 0, M - 3, 4, 1),
        ("epoch u64::MAX, no renewal", 0, M, 4, 0),
        ("epoch u64::MAX, renewal after the last batch", 0, M, 2, 2),
    ];
    for (name, batch, epoch, batches, every) in admitted {
        assert_eq!(
            dense(batch, epoch, batches, every).code,
            "WORKER_IO",
            "dense: {name}"
        );
        assert_eq!(
            compressed(batch, epoch, batches, every).code,
            "ARTIFACT_IO",
            "compressed: {name}"
        );
    }
}

#[test]
fn a_series_with_no_routed_token_is_refused_before_any_worker() {
    // A53-R19: a series whose routing has no token ran every batch with no EXEC and no
    // oracle expert, and reported each batch exact: a verdict on nothing.
    let empty = "a series needs at least one routed token";
    let mut problem = common::problem();
    problem.stamp.run_id = "a53r19-dense".into();
    problem.batch.inputs.clear();
    problem.batch.routes.clear();
    let error = session::run_series(&SeriesInput {
        problem,
        workers: vec![ghost(2), ghost(9)],
        batches: 3,
        timeout_ms: 1000,
        epoch_every: 0,
        warmup: 0,
        tolerance_abs: 0.0,
        faults: vec![],
    })
    .unwrap_err();
    assert_eq!(
        (error.code.as_str(), error.message.as_str()),
        ("INVALID_INPUT", empty),
        "dense"
    );
    let mut routing = routing::from_dense(&common::problem());
    routing.routes.clear();
    let error = session::run_compressed_series(&CompressedSeriesInput {
        artifacts: vec!["/nonexistent/a53r19.safetensors".into()],
        routing,
        inputs: vec![],
        inputs_fp16_bits: None,
        workers: vec![ghost(2)],
        batches: 3,
        timeout_ms: 1000,
        epoch_every: 0,
        warmup: 0,
        tolerance_abs: 0.0,
        fp16_completions: false,
        faults: vec![],
        record: vec![],
    })
    .unwrap_err();
    assert_eq!(
        (error.code.as_str(), error.message.as_str()),
        ("INVALID_INPUT", empty),
        "compressed"
    );
    // One routed token is enough: the missing worker or artifact file then fails.
    assert_eq!(dense(0, 1, 3, 0).code, "WORKER_IO");
    assert_eq!(compressed(0, 1, 3, 0).code, "ARTIFACT_IO");
}

#[test]
fn a_series_whose_derived_activations_overflow_is_refused_before_any_worker() {
    // A53-R24: batch b scales each base activation by up to 1 + 10/64. A finite base value
    // the scaling takes past f64::MAX passed every check, so the workers were started and
    // loaded and batch 0 ran before the series failed at the batch deriving it. Row 0,
    // column 0 is scaled by 1 at batch 0 and by 1 + 7/64 at batch 1.
    let derived = "series activations must stay finite in every derived batch";
    let dense = |batches: u64, x: f64| {
        let mut problem = common::problem();
        problem.stamp.run_id = "a53r24-dense".into();
        problem.batch.inputs[0][0] = x;
        session::run_series(&SeriesInput {
            problem,
            workers: vec![ghost(2), ghost(9)],
            batches,
            timeout_ms: 1000,
            epoch_every: 0,
            warmup: 0,
            tolerance_abs: 0.0,
            faults: vec![],
        })
        .unwrap_err()
    };
    let error = dense(2, 1.7e308);
    assert_eq!(
        (error.code.as_str(), error.message.as_str()),
        ("INVALID_INPUT", derived)
    );
    // A single batch never scales that value, and a value every factor keeps finite is
    // accepted: both reach the missing workers.
    assert_eq!(dense(1, 1.7e308).code, "WORKER_IO");
    assert_eq!(dense(2, f64::MAX / 1.25).code, "WORKER_IO");
}
