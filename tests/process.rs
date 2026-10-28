mod common;

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use voltafabric::{moe, process::*, routing};

fn worker() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_vf-rank-worker"))
}
fn fixture() -> ProcessProblem {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let mut p = common::problem();
    p.stamp.run_id = format!("ipc-test-{}", NEXT.fetch_add(1, Ordering::Relaxed));
    // Deliberately sparse rank-local logical IDs and reversed global stack order.
    p.registry.copies[0].local = 900;
    p.registry.copies[2].local = 3;
    p.model.experts.reverse();
    ProcessProblem {
        problem: p,
        faults: BTreeMap::new(),
        timeout_ms: 5000,
        recovery: false,
    }
}
fn failed(report: &ProcessReport) {
    assert!(!report.ok, "{report:?}");
    assert!(report.output.is_none());
    assert!(report.error.is_some());
    reaped(report);
}
fn reaped(report: &ProcessReport) {
    let spawned = report.receipts.iter().filter(|r| r.pid.is_some()).count();
    for receipt in &report.receipts {
        assert_eq!(receipt.children_spawned, spawned);
        assert_eq!(receipt.children_reaped, spawned);
        if let Some(pid) = receipt.pid {
            assert!(receipt.reaped);
            assert_eq!(receipt.reaped_pids, vec![pid]);
            #[cfg(target_os = "linux")]
            assert!(
                !Path::new(&format!("/proc/{pid}")).exists(),
                "child {pid} remains"
            );
        }
    }
}

#[test]
fn real_workers_combine_exactly_and_are_reaped() {
    let p = fixture();
    let report = run(&p, worker()).unwrap();
    assert!(report.ok, "{report:?}");
    assert_eq!(report.output.unwrap(), moe::reference(&p.problem).unwrap());
    let pids: std::collections::BTreeSet<_> =
        report.receipts.iter().filter_map(|r| r.pid).collect();
    assert_eq!(pids.len(), 2);
    assert!(!pids.contains(&std::process::id()));
    assert!(report
        .receipts
        .iter()
        .all(|r| r.state == ReceiptState::Completed && r.started));
    // Both participants receive subsets of the model and of the input batch.
    assert_eq!(
        report
            .receipts
            .iter()
            .find(|r| r.rank == 2)
            .unwrap()
            .resident_experts,
        2
    );
    assert_eq!(
        report
            .receipts
            .iter()
            .find(|r| r.rank == 9)
            .unwrap()
            .local_inputs,
        1
    );
    reaped(&ProcessReport {
        output: None,
        ..report
    });
}

#[test]
fn rank_request_is_compact_and_keeps_explicit_physical_order() {
    let p = fixture();
    let mut meta = routing::from_dense(&p.problem);
    meta.resident_order.get_mut(&2).unwrap().reverse();
    let dispatch = routing::dispatch(&meta).unwrap();
    let rank = dispatch.ranks.iter().find(|r| r.rank == 2).unwrap();
    let request = rank_request(&p.problem, rank, 17, Fault::None).unwrap();
    assert_eq!(request.dispatch, *rank);
    assert_eq!(
        request
            .model
            .experts
            .iter()
            .map(|e| e.id)
            .collect::<Vec<_>>(),
        vec![10, 99]
    );
    assert!(!request.model.experts.iter().any(|e| e.id == 25));
    for routed in &request.dispatch.slots {
        assert_eq!(
            request.model.experts[routed.physical_expert].id,
            routed.slot.expert
        );
    }
    let rank = dispatch.ranks.iter().find(|r| r.rank == 9).unwrap();
    let request = rank_request(&p.problem, rank, 17, Fault::None).unwrap();
    assert_eq!(request.inputs.len(), 1);
    assert_eq!(request.inputs[0].token, 1);
    assert_eq!(request.inputs[0].value, p.problem.batch.inputs[1]);
}

#[test]
fn delayed_and_reversed_responses_preserve_original_combine_order() {
    let mut p = fixture();
    p.faults.insert(2, Fault::Delay { milliseconds: 30 });
    p.faults.insert(9, Fault::Reverse);
    let report = run(&p, worker()).unwrap();
    assert!(report.ok, "{report:?}");
    assert_eq!(
        report.output.as_ref().unwrap(),
        &moe::reference(&p.problem).unwrap()
    );
    reaped(&report);
    let mut p = fixture();
    p.faults.insert(2, Fault::Reverse);
    let report = run(&p, worker()).unwrap();
    assert!(report.ok, "{report:?}");
    assert_eq!(
        report.output.as_ref().unwrap(),
        &moe::reference(&p.problem).unwrap()
    );
    reaped(&report);
}

#[test]
fn duplicate_missing_and_stale_epoch_abort_without_partial_output() {
    for fault in [Fault::Duplicate, Fault::Missing, Fault::StaleEpoch] {
        let mut p = fixture();
        p.faults.insert(2, fault);
        failed(&run(&p, worker()).unwrap());
    }
}

#[test]
fn explicit_rejection_proves_not_started() {
    let mut p = fixture();
    p.faults.insert(2, Fault::RejectBeforeCompute);
    let report = run(&p, worker()).unwrap();
    failed(&report);
    let receipt = report.receipts.iter().find(|r| r.rank == 2).unwrap();
    assert_eq!(receipt.state, ReceiptState::NotStarted);
    assert!(!receipt.started);
}

#[test]
fn death_before_started_is_unknown_without_a_rejection_proof() {
    let mut p = fixture();
    p.faults.insert(2, Fault::DieBeforeStart);
    let report = run(&p, worker()).unwrap();
    failed(&report);
    let receipt = report.receipts.iter().find(|r| r.rank == 2).unwrap();
    assert_eq!(receipt.state, ReceiptState::Unknown);
    assert!(!receipt.started);
}

#[test]
fn real_recovery_after_death_requires_new_attempt_and_rejects_old_completion() {
    for fault in [Fault::DieAfterStart, Fault::DieAfterCompute] {
        let mut p = fixture();
        p.faults.insert(2, fault);
        let first = run(&p, worker()).unwrap();
        failed(&first);
        let receipt = first.receipts.iter().find(|r| r.rank == 2).unwrap();
        assert_eq!(receipt.state, ReceiptState::Unknown);
        assert!(receipt.started);
        assert!(
            run(&p, worker()).is_err(),
            "implicit retry must be forbidden in this coordinator"
        );
        p.recovery = true;
        p.faults.clear();
        p.problem.stamp.epoch += 1;
        p.problem.registry.epoch += 1;
        let recovered = run(&p, worker()).unwrap();
        assert!(recovered.ok, "{recovered:?}");
        assert_ne!(first.attempt, recovered.attempt);
        assert_eq!(
            recovered.output.as_ref().unwrap(),
            &moe::reference(&p.problem).unwrap()
        );
        reaped(&recovered);
        p.faults.insert(
            2,
            Fault::OldAttempt {
                attempt: first.attempt,
            },
        );
        let stale = run(&p, worker()).unwrap();
        failed(&stale);
        assert_eq!(stale.error.unwrap().code, "STALE_ATTEMPT");
    }
}

#[test]
fn timeout_kills_waits_and_reaps_all_children() {
    let mut p = fixture();
    p.timeout_ms = 100;
    p.faults.insert(2, Fault::Delay { milliseconds: 5000 });
    p.faults.insert(9, Fault::Delay { milliseconds: 5000 });
    let start = Instant::now();
    let report = run(&p, worker()).unwrap();
    failed(&report);
    assert_eq!(report.error.as_ref().unwrap().code, "WORKER_TIMEOUT");
    assert!(start.elapsed() < Duration::from_secs(3));
    assert!(report
        .receipts
        .iter()
        .all(|r| r.state == ReceiptState::Unknown));
}

#[test]
fn stderr_is_drained_even_above_pipe_capacity() {
    let mut p = fixture();
    p.faults.insert(
        2,
        Fault::StderrFlood {
            bytes: 2 * 1024 * 1024,
        },
    );
    let report = run(&p, worker()).unwrap();
    assert!(report.ok, "{report:?}");
    assert!(
        report
            .receipts
            .iter()
            .find(|r| r.rank == 2)
            .unwrap()
            .stderr_bytes
            >= 2 * 1024 * 1024
    );
    reaped(&report);
}

#[test]
fn missing_executable_is_proven_not_started() {
    let p = fixture();
    let report = run(&p, Path::new("/definitely-missing-vf-rank-worker")).unwrap();
    failed(&report);
    assert!(report
        .receipts
        .iter()
        .all(|r| r.pid.is_none() && r.state == ReceiptState::NotStarted));
}

#[test]
fn malformed_or_oversized_problem_is_rejected_before_spawning() {
    let mut p = fixture();
    p.timeout_ms = 0;
    assert!(run(&p, worker()).is_err());
    let mut p = fixture();
    p.problem.registry.epoch += 1;
    assert!(run(&p, worker()).is_err());
    let mut p = fixture();
    p.faults.insert(400, Fault::None);
    assert!(run(&p, worker()).is_err());
    let mut p = fixture();
    p.problem.model.hidden = 1;
    p.problem.model.intermediate = 262144;
    p.problem.model.experts = vec![moe::Expert {
        id: 10,
        gate: vec![-1.2345678901234567e300; 262144],
        up: vec![-1.2345678901234567e300; 262144],
        down: vec![-1.2345678901234567e300; 262144],
    }];
    p.problem.registry.copies.retain(|r| r.expert == 10);
    p.problem.batch.inputs = vec![vec![1.0]];
    p.problem.batch.routes = vec![vec![moe::Choice {
        expert: 10,
        weight: 1.0,
    }]];
    assert_eq!(run(&p, worker()).unwrap_err().code, "MESSAGE_TOO_LARGE");
}

#[test]
fn empty_batch_spawns_no_worker() {
    let mut p = fixture();
    p.problem.batch.inputs.clear();
    p.problem.batch.routes.clear();
    let report = run(&p, worker()).unwrap();
    assert!(report.ok);
    assert_eq!(report.output, Some(vec![]));
    assert!(report.receipts.is_empty());
}

#[test]
fn worker_rejects_bad_physical_mapping_before_started() {
    let p = fixture();
    let dispatch = routing::dispatch(&routing::from_dense(&p.problem)).unwrap();
    let mut request = rank_request(&p.problem, &dispatch.ranks[0], 42, Fault::None).unwrap();
    request.dispatch.slots[0].physical_expert = 999;
    let mut child = Command::new(worker())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    serde_json::to_writer(&mut stdin, &request).unwrap();
    stdin.write_all(b"\n").unwrap();
    drop(stdin);
    let mut output = BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    output.read_line(&mut line).unwrap();
    let event: WorkerEvent = serde_json::from_str(&line).unwrap();
    assert_eq!(event.attempt, 42);
    assert!(matches!(event.event, EventKind::Rejected { .. }));
    assert!(child.wait().unwrap().success());
}

#[test]
fn worker_rejects_a_duplicate_compacted_input_before_started() {
    // Without the duplicate check the second entry would silently overwrite the
    // first and the token-coverage check would still pass, so the rank would
    // compute on a value the coordinator never dispatched.
    let p = fixture();
    let dispatch = routing::dispatch(&routing::from_dense(&p.problem)).unwrap();
    let mut request = rank_request(&p.problem, &dispatch.ranks[0], 7, Fault::None).unwrap();
    let mut duplicate = request.inputs[0].clone();
    duplicate.value = duplicate.value.iter().map(|v| v + 1.0).collect();
    assert_eq!(duplicate.token, request.inputs[0].token);
    assert_ne!(duplicate.value, request.inputs[0].value);
    request.inputs.push(duplicate);
    let mut child = Command::new(worker())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    serde_json::to_writer(&mut stdin, &request).unwrap();
    stdin.write_all(b"\n").unwrap();
    drop(stdin);
    let mut output = BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    output.read_line(&mut line).unwrap();
    let event: WorkerEvent = serde_json::from_str(&line).unwrap();
    assert_eq!(event.attempt, 7);
    assert!(matches!(event.event, EventKind::Rejected { .. }));
    assert!(child.wait().unwrap().success());
}

#[test]
fn worker_bounds_incoming_json_without_starting_calculation() {
    let mut child = Command::new(worker())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let writer = std::thread::spawn(move || {
        let buffer = [b' '; 8192];
        for _ in 0..=(MAX_MESSAGE_BYTES / buffer.len()) {
            if input.write_all(&buffer).is_err() {
                break;
            }
        }
    });
    let output = child.wait_with_output().unwrap();
    writer.join().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("MESSAGE_TOO_LARGE"));
}

// A40 guard sweep: one clause per case. With the clause alone disabled the
// coordinator built a request from a mismatched dispatch (one case panicked on a
// foreign local id), and the worker started computing on a malformed request (an
// empty one crashed it before any event). Each must be refused for its own reason.
fn mismatched_rank_request(edit: impl FnOnce(&mut routing::RankDispatch, &mut u64), reason: &str) {
    let p = fixture();
    let dispatch = routing::dispatch(&routing::from_dense(&p.problem)).unwrap();
    let mut rank = dispatch
        .ranks
        .into_iter()
        .find(|r| r.slots.len() >= 2)
        .unwrap();
    let mut attempt = 5;
    edit(&mut rank, &mut attempt);
    let error = rank_request(&p.problem, &rank, attempt, Fault::None).unwrap_err();
    assert!(
        error.message.contains(reason),
        "expected {reason:?}, got {error}"
    );
}
#[test]
fn coordinator_builds_requests_only_from_the_exact_dense_dispatch() {
    let oracle = "rank dispatch does not match the dense oracle";
    mismatched_rank_request(|_, attempt| *attempt = 0, oracle);
    mismatched_rank_request(|r, _| r.stamp.plan_id.push('x'), oracle);
    mismatched_rank_request(|r, _| r.hidden += 1, oracle);
    mismatched_rank_request(|r, _| r.tokens += 1, oracle);
    mismatched_rank_request(
        |r, _| {
            r.slots.pop();
        },
        oracle,
    );
    mismatched_rank_request(|r, _| r.slots[0].slot.weight += 0.5, oracle);
    mismatched_rank_request(
        |r, _| {
            let n = r.slots.len();
            r.slots[n - 1] = r.slots[0].clone();
        },
        oracle,
    );
    let order = "physical order does not cover this rank's residents";
    mismatched_rank_request(
        |r, _| {
            let x = r.resident_order[0];
            r.resident_order.push(x);
        },
        order,
    );
    mismatched_rank_request(|r, _| r.resident_order[0] = 4242, order);
    mismatched_rank_request(
        |r, _| {
            let n = r.resident_order.len();
            r.slots[0].physical_expert = (r.slots[0].physical_expert + 1) % n;
        },
        "rank slot does not match its physical resident",
    );
}
fn worker_rejection(edit: impl FnOnce(&mut RankRequest)) -> voltafabric::Error {
    let p = fixture();
    let dispatch = routing::dispatch(&routing::from_dense(&p.problem)).unwrap();
    let rank = dispatch.ranks.iter().find(|r| r.slots.len() >= 2).unwrap();
    let mut request = rank_request(&p.problem, rank, 9, Fault::None).unwrap();
    edit(&mut request);
    let mut bytes = serde_json::to_vec(&request).unwrap();
    bytes.push(b'\n');
    let mut child = Command::new(worker())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&bytes);
    });
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    let _ = child.kill();
    let _ = child.wait();
    writer.join().unwrap();
    let event: WorkerEvent = serde_json::from_str(&line).expect("worker sent no event");
    match event.event {
        EventKind::Rejected { error } => error,
        other => panic!("worker accepted a malformed request: {other:?}"),
    }
}
fn worker_refuses(edit: impl FnOnce(&mut RankRequest), reason: &str) {
    let error = worker_rejection(edit);
    assert!(
        error.message.contains(reason),
        "expected {reason:?}, got {error}"
    );
}
fn last_slot(r: &mut RankRequest) -> &mut moe::Slot {
    &mut r.dispatch.slots.last_mut().unwrap().slot
}
#[test]
fn worker_refuses_malformed_dimensions_and_order_itself() {
    let dims = "invalid rank-local dimensions or physical order";
    worker_refuses(|r| r.attempt = 0, dims);
    worker_refuses(
        |r| {
            r.dispatch.slots.clear();
            r.inputs.clear();
        },
        dims,
    );
    worker_refuses(|r| r.dispatch.tokens = 4097, dims);
    worker_refuses(|r| r.dispatch.hidden += 1, dims);
    worker_refuses(|r| r.dispatch.resident_order.push(4242), dims);
    worker_refuses(
        |r| {
            let x = r.dispatch.resident_order[0];
            r.dispatch.resident_order.push(x);
            let e = r.model.experts[0].clone();
            r.model.experts.push(e);
        },
        dims,
    );
}
#[test]
fn worker_refuses_work_and_fault_bounds_before_starting() {
    // 1024 tokens x 64 slots on one 16x32 expert: 512 * 65536 * 3 > 1e8 work units,
    // yet the request stays far below the 16 MiB pipe bound.
    worker_refuses(
        |r| {
            let local = r.dispatch.resident_order[0];
            let id = r.model.experts[0].id;
            r.model = moe::Model {
                hidden: 16,
                intermediate: 32,
                experts: vec![moe::Expert {
                    id,
                    gate: vec![0.0; 512],
                    up: vec![0.0; 512],
                    down: vec![0.0; 512],
                }],
            };
            r.dispatch.resident_order = vec![local];
            r.dispatch.hidden = 16;
            r.dispatch.tokens = 1024;
            let template = r.dispatch.slots[0].clone();
            r.dispatch.slots = (0..1024)
                .flat_map(|token| {
                    let template = template.clone();
                    (0..64).map(move |slot| {
                        let mut s = template.clone();
                        s.slot.token = token;
                        s.slot.slot = slot;
                        s.slot.expert = id;
                        s.slot.local = local;
                        s.physical_expert = 0;
                        s
                    })
                })
                .collect();
            r.inputs = (0..1024)
                .map(|token| TokenInput {
                    token,
                    value: vec![0.5; 16],
                })
                .collect();
        },
        "rank CPU work budget exceeded",
    );
    let fault = "fault injection outside bounded CPU test contract";
    worker_refuses(
        |r| {
            r.fault = Fault::Delay {
                milliseconds: 60_001,
            }
        },
        fault,
    );
    worker_refuses(
        |r| {
            r.fault = Fault::StderrFlood {
                bytes: MAX_MESSAGE_BYTES + 1,
            }
        },
        fault,
    );
}
#[test]
fn worker_refuses_malformed_inputs_and_slots_itself() {
    let input = "invalid or duplicate compacted input";
    worker_refuses(
        |r| {
            let token = r.dispatch.tokens;
            let value = r.inputs[0].value.clone();
            r.inputs.push(TokenInput { token, value });
        },
        input,
    );
    worker_refuses(|r| r.inputs.last_mut().unwrap().value.push(1.0), input);
    worker_refuses(
        |r| {
            r.inputs.pop();
        },
        "local inputs must cover exactly the dispatched tokens",
    );
    let slot = "rank slot has invalid identity or physical expert";
    worker_refuses(|r| last_slot(r).rank += 1, slot);
    worker_refuses(|r| last_slot(r).slot = 64, slot);
    worker_refuses(|r| last_slot(r).weight = -1.0, slot);
    worker_refuses(
        |r| {
            let first = r.dispatch.slots[0].clone();
            r.dispatch.slots.push(first);
        },
        slot,
    );
    worker_refuses(|r| last_slot(r).local = 4242, slot);
    worker_refuses(|r| last_slot(r).expert = 4242, slot);
}
#[test]
fn worker_timeout_above_one_minute_is_refused_before_spawning() {
    let mut p = fixture();
    p.timeout_ms = 60_001;
    assert_eq!(
        run(&p, worker()).unwrap_err().message,
        "worker timeout must be within 1..60000 ms"
    );
}

#[test]
fn worker_names_an_overflowing_work_size() {
    // A41 reason sweep: hidden * intermediate overflows before the work budget.
    worker_refuses(
        |r| {
            r.model.hidden = 1 << 40;
            r.model.intermediate = 1 << 40;
            r.dispatch.hidden = 1 << 40;
        },
        "rank CPU work overflow",
    );
}
#[test]
fn worker_without_a_request_says_so() {
    let output = Command::new(worker())
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert_eq!(
        String::from_utf8(output.stderr).unwrap().trim(),
        "INVALID_INPUT: missing rank request"
    );
}
#[test]
fn a_computation_failure_after_start_is_reported_with_the_workers_reason() {
    // Finite but huge weights and inputs: validation passes and the worker's
    // intermediate overflows after it has announced Started.
    let mut p = fixture();
    for e in &mut p.problem.model.experts {
        for w in e.gate.iter_mut().chain(&mut e.up).chain(&mut e.down) {
            *w = 1e200;
        }
    }
    for x in &mut p.problem.batch.inputs {
        x.fill(1e200);
    }
    let report = run(&p, worker()).unwrap();
    failed(&report);
    assert_eq!(
        report.error.as_ref().unwrap().message,
        "CPU intermediate is nonfinite"
    );
    assert!(report.receipts.iter().any(|r| r.started));
}
