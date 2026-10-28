//! A40 guard sweep: the coordinator keeps at most 65536 attempted identities. This
//! file is its own test process because the history is process-wide state.
mod common;
use std::collections::BTreeMap;
use std::path::Path;
use voltafabric::process::*;

#[test]
fn coordinator_refuses_a_new_identity_once_its_history_is_full() {
    let worker = Path::new(env!("CARGO_BIN_EXE_vf-rank-worker"));
    let mut p = ProcessProblem {
        problem: common::problem(),
        faults: BTreeMap::new(),
        timeout_ms: 5000,
        recovery: false,
    };
    // An empty batch spawns no worker but still records its identity.
    p.problem.batch.inputs.clear();
    p.problem.batch.routes.clear();
    for i in 0..65536 {
        p.problem.stamp.run_id = format!("limit-{i}");
        assert!(run(&p, worker).unwrap().ok, "identity {i}");
    }
    p.problem.stamp.run_id = "limit-65536".into();
    let error = run(&p, worker).unwrap_err();
    assert!(
        error.message.contains("coordinator attempt limit"),
        "{error}"
    );
    // A known identity is still governed by the explicit-recovery rule, not the limit.
    p.problem.stamp.run_id = "limit-0".into();
    p.recovery = true;
    assert!(run(&p, worker).unwrap().ok);
}
