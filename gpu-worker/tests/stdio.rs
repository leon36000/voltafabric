//! CPU-only subprocess regressions for real worker stderr boundaries.
//! No request/LOAD/EXEC is sent, and no CUDA context is created.
#![cfg(target_os = "linux")]
use std::io;
use std::os::fd::{FromRawFd, OwnedFd};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

fn closed_stderr() -> Stdio {
    let mut fds = [-1; 2];
    // SAFETY: fds is writable for two descriptors. CLOEXEC prevents unrelated
    // inheritance; Stdio duplicates only the owned writer into the child's fd2.
    assert_eq!(
        unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) },
        0,
        "{}",
        io::Error::last_os_error()
    );
    // SAFETY: successful pipe2 returned two new, uniquely owned descriptors.
    let reader = unsafe { OwnedFd::from_raw_fd(fds[0]) };
    let writer = unsafe { OwnedFd::from_raw_fd(fds[1]) };
    drop(reader);
    Stdio::from(writer)
}
fn invoke(args: &[&str], broken: bool) -> Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_vf-mxfp4-gpu-worker"));
    c.args(args).stdin(Stdio::null()).stdout(Stdio::piped());
    c.stderr(if broken {
        closed_stderr()
    } else {
        Stdio::piped()
    });
    let mut child = c.spawn().expect("worker subprocess");
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if child.try_wait().expect("child status").is_some() {
            break;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            child.wait().expect("reap timed-out owned child");
            panic!("CPU stderr contract exceeded five seconds");
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    child
        .wait_with_output()
        .expect("collect terminal subprocess")
}
#[test]
fn closed_usage_stderr_is_code2_without_panic() {
    let healthy = invoke(&["--invalid-r58-cpu-option"], false);
    assert_eq!(healthy.status.code(), Some(2));
    assert!(String::from_utf8(healthy.stderr)
        .unwrap()
        .starts_with("USAGE:"));
    let broken = invoke(&["--invalid-r58-cpu-option"], true);
    assert_eq!(broken.status.code(), Some(2));
    assert!(broken.stdout.is_empty());
}
#[test]
fn startup_audit_is_checked_before_session_or_owners() {
    let healthy = invoke(&["--audit-counters"], false);
    assert_eq!(healthy.status.code(), Some(2));
    assert!(healthy.stdout.is_empty());
    let text = String::from_utf8(healthy.stderr).unwrap();
    let startup: serde_json::Value =
        serde_json::from_str(text.lines().next().expect("startup audit")).unwrap();
    assert_eq!(startup["gpu_audit"], "startup");
    assert_eq!(startup["library_sha256"], env!("VF_MXFP4_LIBRARY_SHA256"));
    assert_eq!(startup["cuda_api_calls"], 0);
    assert_eq!(startup["lifecycle"], serde_json::json!([0, 0, 0]));
    assert!(!text.contains("panicked"));
    let broken = invoke(&["--audit-counters"], true);
    assert_eq!(broken.status.code(), Some(2));
    assert!(broken.stdout.is_empty());
}
#[test]
fn final_protocol_error_survives_closed_stderr() {
    let healthy = invoke(&[], false);
    assert_eq!(healthy.status.code(), Some(2));
    let text = String::from_utf8(healthy.stderr).unwrap();
    assert!(text.contains("SESSION_"), "{text}");
    assert!(!text.contains("panicked"));
    let broken = invoke(&[], true);
    assert_eq!(broken.status.code(), Some(2));
    assert!(broken.stdout.is_empty());
}
