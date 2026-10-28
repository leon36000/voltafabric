//! A53-R15: the dense series' endpoints must cover every rank its routing sends work to,
//! and name only ranks that hold residents, as the compressed series already requires.
//! Before R15 a missing rank was found at batch 0, after every other rank was sent its
//! weights; this file is separate because it starts real session workers.
#![cfg(target_os = "linux")]
mod common;

use voltafabric::session::{self, Endpoint, SeriesInput};

fn endpoint(rank: u32) -> Endpoint {
    Endpoint {
        rank,
        argv: vec![env!("CARGO_BIN_EXE_vf-rank-session").into()],
    }
}

fn series(run_id: &str, workers: Vec<Endpoint>) -> SeriesInput {
    let mut problem = common::problem();
    problem.stamp.run_id = run_id.into();
    SeriesInput {
        problem,
        workers,
        batches: 2,
        timeout_ms: 10_000,
        epoch_every: 0,
        warmup: 0,
        tolerance_abs: 0.0,
        faults: vec![],
    }
}

#[test]
fn dense_series_endpoints_are_refused_before_any_weight_is_sent() {
    let ranks =
        "series endpoints must serve every rank the routing sends work to, and only resident ranks";
    // Rank 9 holds expert 25, which the routing uses: no endpoint serves it.
    let error = session::run_series(&series("a53r15-missing", vec![endpoint(2)]))
        .expect_err("a series missing a participating rank must not start");
    assert_eq!(
        (error.code.as_str(), error.message.as_str()),
        ("INVALID_INPUT", ranks)
    );
    // Rank 11 holds nothing in this problem.
    let error = session::run_series(&series(
        "a53r15-extra",
        vec![endpoint(2), endpoint(9), endpoint(11)],
    ))
    .expect_err("a series with an endpoint for a rank without residents must not start");
    assert_eq!(
        (error.code.as_str(), error.message.as_str()),
        ("INVALID_INPUT", ranks)
    );
    // A resident rank the routing never uses needs no endpoint: every token routed to
    // rank 2's experts 10 and 99, served by rank 2 alone, still runs.
    let mut only2 = series("a53r15-only2", vec![endpoint(2)]);
    for route in &mut only2.problem.batch.routes {
        route[0].expert = 10;
        route[1].expert = 99;
    }
    let report = session::run_series(&only2).unwrap();
    assert!(report.ok, "{:?} {:?}", report.error, report.failure);
    assert_eq!((report.batches_ok, report.exact_batches), (2, 2));
    // Exactly the resident ranks: the series runs, every batch exact.
    let report =
        session::run_series(&series("a53r15-exact", vec![endpoint(2), endpoint(9)])).unwrap();
    assert!(report.ok, "{:?} {:?}", report.error, report.failure);
    assert_eq!((report.batches_ok, report.exact_batches), (2, 2));
}

/// Hidden 512, intermediate 256: one expert is about 8 MB of LOAD JSON. Expert `id` (1..=4)
/// is held by rank `ranks[id - 1]`.
fn wide_dense(run_id: &str, ranks: [u32; 4]) -> voltafabric::moe::Problem {
    use voltafabric::moe::{Batch, Choice, Expert, Model, Problem, Registry, Replica};
    let (hidden, intermediate) = (512usize, 256usize);
    let weights = |seed: usize, n: usize| -> Vec<f64> {
        (0..n)
            .map(|i| (((i * 7919 + seed * 104_729) % 100_003) as f64) / 100_003.0 - 0.5)
            .collect()
    };
    let experts = (1..=4)
        .map(|id| Expert {
            id,
            gate: weights(3 * id as usize, hidden * intermediate),
            up: weights(3 * id as usize + 1, hidden * intermediate),
            down: weights(3 * id as usize + 2, hidden * intermediate),
        })
        .collect();
    let copies = (1..=4u32)
        .map(|id| Replica {
            expert: id,
            rank: ranks[id as usize - 1],
            local: id,
        })
        .collect();
    let mut problem = Problem {
        model: Model {
            hidden,
            intermediate,
            experts,
        },
        batch: Batch {
            inputs: vec![weights(90, hidden), weights(91, hidden)],
            routes: vec![
                vec![
                    Choice {
                        expert: 1,
                        weight: 0.5,
                    },
                    Choice {
                        expert: 2,
                        weight: 0.5,
                    },
                ],
                vec![
                    Choice {
                        expert: 3,
                        weight: 0.5,
                    },
                    Choice {
                        expert: 4,
                        weight: 0.5,
                    },
                ],
            ],
        },
        registry: Registry { epoch: 7, copies },
        stamp: common::problem().stamp,
        preferred_owners: std::collections::BTreeMap::new(),
    };
    problem.stamp.run_id = run_id.into();
    problem
}

#[test]
fn a_dense_rank_too_large_for_one_load_frame_is_refused_before_any_weight_is_sent() {
    // A53-R23: each rank's first LOAD carries its resident model as JSON in one 16 MiB
    // frame. A rank whose model is larger was refused only at its turn, after the ranks
    // before it had been sent their weights. Missing workers show any late refusal.
    let ghost = |rank| Endpoint {
        rank,
        argv: vec!["/nonexistent/vf-rank-session".into()],
    };
    let run = |problem, workers| {
        session::run_series(&SeriesInput {
            problem,
            workers,
            batches: 2,
            timeout_ms: 10_000,
            epoch_every: 0,
            warmup: 0,
            tolerance_abs: 0.0,
            faults: vec![],
        })
        .unwrap_err()
    };
    // Rank 9 holds three experts, about 24 MB of JSON.
    let error = run(
        wide_dense("a53r23-big", [2, 9, 9, 9]),
        vec![ghost(2), ghost(9)],
    );
    assert_eq!(
        (error.code.as_str(), error.message.as_str()),
        (
            "MESSAGE_TOO_LARGE",
            "private-pipe JSON message exceeds 16 MiB"
        )
    );
    // The same model, one expert per rank: every frame fits, and the missing worker fails.
    let error = run(
        wide_dense("a53r23-fits", [2, 9, 5, 12]),
        vec![ghost(2), ghost(5), ghost(9), ghost(12)],
    );
    assert_eq!(error.code, "WORKER_IO");
}

/// Rank 2's EXEC frame for batch `b` of `p` (every token is routed to it), as a session
/// sends it: 64-hex-digit digest, the batch's dispatch and activations, `attempt`, `fault`.
fn exec_frame_bytes(
    p: &voltafabric::moe::Problem,
    b: u64,
    attempt: u64,
    fault: voltafabric::process::Fault,
) -> usize {
    let batch = session::series_batch(p, b, 0);
    let dispatch = voltafabric::routing::dispatch(&voltafabric::routing::from_dense(&batch))
        .unwrap()
        .ranks
        .into_iter()
        .find(|r| r.rank == 2)
        .unwrap();
    let inputs = batch
        .batch
        .inputs
        .iter()
        .enumerate()
        .map(|(token, value)| voltafabric::process::TokenInput {
            token,
            value: value.clone(),
        })
        .collect();
    serde_json::to_vec(&session::Request::Exec {
        artifact_sha256: "0".repeat(64),
        dispatch,
        attempt,
        inputs,
        fault,
    })
    .unwrap()
    .len()
}

#[test]
fn a_dense_exec_frame_over_the_bound_is_refused_before_any_worker() {
    // A53-R26: each batch sends every routed rank one EXEC frame with its tokens'
    // activations, which the encoder refuses past 16 MiB. That refusal came only at the
    // batch whose frame was too large, after every rank had been loaded. Rank 2 gets all
    // 4096 tokens of hidden 990 (work 12 M of 100 M); activations are zero except some
    // 0.25s where batch 0 does not scale them, tuned so that the batch-0 frame, at the
    // widest attempt, is exactly 16 MiB. Missing workers make a late refusal WORKER_IO.
    use voltafabric::moe::{Batch, Choice, Expert, Model, Problem, Registry, Replica, Stamp};
    use voltafabric::process::{Fault, MAX_MESSAGE_BYTES};
    let (hidden, tokens) = (990, 4096);
    let mut p = Problem {
        model: Model {
            hidden,
            intermediate: 1,
            experts: vec![Expert {
                id: 1,
                gate: vec![0.5; hidden],
                up: vec![0.25; hidden],
                down: vec![0.125; hidden],
            }],
        },
        batch: Batch {
            inputs: vec![vec![0.0; hidden]; tokens],
            routes: vec![
                vec![Choice {
                    expert: 1,
                    weight: 1.0
                }];
                tokens
            ],
        },
        registry: Registry {
            epoch: 7,
            copies: vec![Replica {
                expert: 1,
                rank: 2,
                local: 0,
            }],
        },
        stamp: Stamp {
            run_id: "a53r26-exec".into(),
            model_revision: "fixture".into(),
            plan_id: "plan-1".into(),
            layer: 0,
            batch: 3,
            epoch: 7,
        },
        preferred_owners: Default::default(),
    };
    // Each 0.25 at a position batch 0 leaves unscaled ((3t + i) % 11 == 0) adds one byte.
    let short = MAX_MESSAGE_BYTES - exec_frame_bytes(&p, 0, u64::MAX, Fault::None);
    let unscaled = (0..tokens)
        .flat_map(|t| (0..hidden).map(move |i| (t, i)))
        .filter(|(t, i)| (3 * t + i) % 11 == 0);
    for (t, i) in unscaled.take(short) {
        p.batch.inputs[t][i] = 0.25;
    }
    let flood = Fault::StderrFlood { bytes: 1 << 24 };
    assert_eq!(
        exec_frame_bytes(&p, 0, u64::MAX, Fault::None),
        MAX_MESSAGE_BYTES
    );
    // Batch 1 scales those 0.25s into longer decimals, and the flood fault makes batch 0's
    // frame too large even with attempt 1: both would have failed after setup.
    assert!(exec_frame_bytes(&p, 1, 1, Fault::None) > MAX_MESSAGE_BYTES);
    assert!(exec_frame_bytes(&p, 0, 1, flood.clone()) > MAX_MESSAGE_BYTES);
    let run = |batches: u64, faults: Vec<session::SeriesFault>| {
        let error = session::run_series(&SeriesInput {
            problem: p.clone(),
            workers: vec![Endpoint {
                rank: 2,
                argv: vec!["/nonexistent/vf-rank-session".into()],
            }],
            batches,
            timeout_ms: 10_000,
            epoch_every: 0,
            warmup: 0,
            tolerance_abs: 0.0,
            faults,
        })
        .unwrap_err();
        (error.code, error.message)
    };
    let too_large = (
        "MESSAGE_TOO_LARGE".to_string(),
        "private-pipe JSON message exceeds 16 MiB".to_string(),
    );
    // Exactly 16 MiB fits: the missing worker is reached.
    assert_eq!(run(1, vec![]).0, "WORKER_IO");
    assert_eq!(run(2, vec![]), too_large, "batch 1");
    let fault = session::SeriesFault {
        batch: 0,
        rank: 2,
        fault: flood,
    };
    assert_eq!(run(1, vec![fault]), too_large, "faulted batch 0");
}

#[test]
fn a_dense_exec_frame_is_checked_at_its_widest_batch_number_and_attempt() {
    // A53-R26: the frames of one activation phase differ only in their numbers, which widen
    // with the batch: the last batch of each phase must be the one checked. Activations are
    // zero except some 64.0s, whose scaled values 64 + s keep 4 bytes in every phase, so all
    // phases have one frame size. Batches are numbered 89 + b: batch 11 (number 100) is one
    // byte wider than batch 0 of its phase. The attempt is checked at its widest.
    use voltafabric::moe::{Batch, Choice, Expert, Model, Problem, Registry, Replica, Stamp};
    use voltafabric::process::{Fault, MAX_MESSAGE_BYTES};
    let (hidden, tokens) = (990, 4096);
    let mut p = Problem {
        model: Model {
            hidden,
            intermediate: 1,
            experts: vec![Expert {
                id: 1,
                gate: vec![0.5; hidden],
                up: vec![0.25; hidden],
                down: vec![0.125; hidden],
            }],
        },
        batch: Batch {
            inputs: vec![vec![0.0; hidden]; tokens],
            routes: vec![
                vec![Choice {
                    expert: 1,
                    weight: 1.0
                }];
                tokens
            ],
        },
        registry: Registry {
            epoch: 7,
            copies: vec![Replica {
                expert: 1,
                rank: 2,
                local: 0,
            }],
        },
        stamp: Stamp {
            run_id: "a53r26-width".into(),
            model_revision: "fixture".into(),
            plan_id: "plan-1".into(),
            layer: 0,
            batch: 89,
            epoch: 7,
        },
        preferred_owners: Default::default(),
    };
    // Each 64.0 in place of a 0.0 adds one byte, in every phase.
    let short = MAX_MESSAGE_BYTES - exec_frame_bytes(&p, 0, u64::MAX, Fault::None);
    for n in 0..short {
        p.batch.inputs[n / hidden][n % hidden] = 64.0;
    }
    for b in 0..=10 {
        assert_eq!(
            exec_frame_bytes(&p, b, u64::MAX, Fault::None),
            MAX_MESSAGE_BYTES
        );
    }
    assert_eq!(
        exec_frame_bytes(&p, 11, u64::MAX, Fault::None),
        MAX_MESSAGE_BYTES + 1
    );
    let run = |p: &Problem, batches: u64| {
        let error = session::run_series(&SeriesInput {
            problem: p.clone(),
            workers: vec![Endpoint {
                rank: 2,
                argv: vec!["/nonexistent/vf-rank-session".into()],
            }],
            batches,
            timeout_ms: 10_000,
            epoch_every: 0,
            warmup: 0,
            tolerance_abs: 0.0,
            faults: vec![],
        })
        .unwrap_err();
        (error.code, error.message)
    };
    let too_large = (
        "MESSAGE_TOO_LARGE".to_string(),
        "private-pipe JSON message exceeds 16 MiB".to_string(),
    );
    assert_eq!(run(&p, 11).0, "WORKER_IO");
    assert_eq!(run(&p, 12), too_large, "batch 11, numbered 100");
    // One more byte: over the bound at the widest attempt, though attempt 1 would fit.
    p.batch.inputs[short / hidden][short % hidden] = 64.0;
    assert!(exec_frame_bytes(&p, 0, 1, Fault::None) < MAX_MESSAGE_BYTES);
    assert_eq!(run(&p, 1), too_large, "widest attempt");
}
