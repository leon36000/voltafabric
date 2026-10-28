//! Backend selection must fail closed before a CPU worker publishes HELLO.
use std::ffi::OsString;
use std::io::Write;
use std::process::{Command, Output, Stdio};

fn invoke(arguments: &[OsString]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_vf-rank-session"))
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // The worker may reject arguments before this write; BrokenPipe is expected.
    let _ = child
        .stdin
        .take()
        .unwrap()
        .write_all(b"{\"op\":\"hello\",\"protocol\":1,\"rank\":7}\n{\"op\":\"shutdown\"}\n");
    child.wait_with_output().unwrap()
}

#[test]
fn cpu_worker_rejects_unknown_backend_arguments_before_any_output() {
    for argument in ["--mxfp4-gpu", "--gpu", "--library=unqualified.so"] {
        let output = invoke(&[argument.into()]);
        assert_eq!(output.status.code(), Some(2), "{argument}");
        assert!(output.stdout.is_empty(), "CPU HELLO escaped for {argument}");
        assert!(String::from_utf8(output.stderr).unwrap().contains("USAGE"));
    }
}

#[test]
fn default_cpu_worker_still_handshakes_and_closes() {
    let output = invoke(&[]);
    assert!(output.status.success());
    let rows: Vec<serde_json::Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["reply"], "hello");
    assert_eq!(rows[1]["reply"], "bye");
}

#[cfg(unix)]
#[test]
fn non_utf8_arguments_are_a_usage_refusal_not_a_panic() {
    use std::os::unix::ffi::OsStringExt;
    let output = invoke(&[OsString::from_vec(vec![0x80])]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8(output.stderr).unwrap().contains("USAGE"));
}
