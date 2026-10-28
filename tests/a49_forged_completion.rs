//! A49 return-Err sweep: the coordinator's own check that a completion names the
//! rank whose pipe carried it. The real worker always copies its rank, so a sed
//! wrapper rewrites the rank inside each completion and leaves the event identity
//! untouched. Its script is isolated in a unique per-invocation directory so parallel
//! test processes cannot overwrite an executable while another process launches it.
#![cfg(target_os = "linux")]
mod common;

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use voltafabric::process::*;

#[test]
fn completion_naming_a_foreign_rank_is_refused() {
    let worker = Path::new(env!("CARGO_BIN_EXE_vf-rank-worker"));
    let _directory = common::UniqueScriptDir::new().unwrap();
    let path = _directory.path().join("a49-foreign-completion-worker.sh");
    let script = format!(
        "#!/bin/sh\n'{}' | sed -u 's/\"expert\":\\([0-9]*\\),\"rank\":[0-9]*,\"local\"/\"expert\":\\1,\"rank\":4242,\"local\"/'\n",
        worker.display().to_string().replace('\'', "'\\''"),
    );
    std::fs::write(&path, script).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut problem = common::problem();
    problem.stamp.run_id = "a49-foreign-completion".into();
    let p = ProcessProblem {
        problem,
        faults: Default::default(),
        timeout_ms: 5000,
        recovery: false,
    };
    let report = run(&p, &path).unwrap();
    assert!(!report.ok, "{report:?}");
    assert!(report.output.is_none());
    let error = report.error.as_ref().unwrap();
    assert_eq!(error.code, "INVALID_COMPLETION", "{error}");
    assert_eq!(error.message, "completion has foreign rank", "{error}");
}
