//! A40 guard sweep: event identity checks. The real worker always echoes its own
//! identity, so only a foreign writer on the pipe can trip them; a sed wrapper rewrites
//! the event-level identity and leaves every completion untouched. Each generated worker
//! lives in a private per-invocation directory under CARGO_TARGET_TMPDIR, avoiding
//! collisions with concurrent test processes.
mod common;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use voltafabric::process::*;

// Keep script creation and execution in one fixture critical section. A fork in
// the other test can otherwise transiently inherit an open writable script fd,
// making exec fail with ETXTBSY even after fs::write closed in the parent.
// The coordinator's concurrent rank workers and all identity checks stay active.
static SCRIPT_FIXTURE: Mutex<()> = Mutex::new(());

fn worker() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_vf-rank-worker"))
}
#[cfg(unix)]
fn forging_worker(name: &str, sed: &str) -> (common::UniqueScriptDir, PathBuf) {
    use std::os::unix::fs::PermissionsExt;
    let directory = common::UniqueScriptDir::new().unwrap();
    let path = directory.path().join(name);
    let script = format!(
        "#!/bin/sh\n'{}' | sed -u '{}'\n",
        worker().display().to_string().replace('\'', "'\\''"),
        sed
    );
    std::fs::write(&path, script).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    (directory, path)
}
#[cfg(unix)]
#[test]
fn coordinator_refuses_events_with_a_forged_stamp_or_rank() {
    let _fixture = SCRIPT_FIXTURE.lock().unwrap();
    for (name, sed) in [
        (
            "forged-stamp-worker.sh",
            "s/\"plan_id\":\"plan-1\"/\"plan_id\":\"forged\"/",
        ),
        (
            "forged-rank-worker.sh",
            "s/},\"rank\":[0-9]*,\"attempt\"/},\"rank\":4242,\"attempt\"/",
        ),
    ] {
        let mut p = ProcessProblem {
            problem: common::problem(),
            faults: BTreeMap::new(),
            timeout_ms: 5000,
            recovery: false,
        };
        p.problem.stamp.run_id = format!("forged-{name}");
        let (_directory, worker) = forging_worker(name, sed);
        let report = run(&p, &worker).unwrap();
        assert!(!report.ok, "{report:?}");
        assert!(report.output.is_none());
        assert_eq!(
            report.error.as_ref().unwrap().message,
            "worker event has stale stamp or foreign rank",
            "{name}"
        );
    }
}

#[cfg(unix)]
#[test]
fn coordinator_refuses_a_completion_that_arrives_before_started() {
    let _fixture = SCRIPT_FIXTURE.lock().unwrap();
    // A41 reason sweep: only a foreign writer can break the event lifecycle.
    let mut p = ProcessProblem {
        problem: common::problem(),
        faults: BTreeMap::new(),
        timeout_ms: 5000,
        recovery: false,
    };
    p.problem.stamp.run_id = "forged-lifecycle".into();
    let (_directory, worker) = forging_worker("no-started-worker.sh", "/\"kind\":\"started\"/d");
    let report = run(&p, &worker).unwrap();
    assert!(!report.ok, "{report:?}");
    assert_eq!(
        report.error.as_ref().unwrap().message,
        "invalid worker event lifecycle"
    );
}
