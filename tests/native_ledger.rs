#![cfg(feature = "native-ggml-cpu")]
use std::{
    path::Path,
    process::{Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};
use voltafabric::{moe::Stamp, native_ledger::NativeBatchLedger};
fn stamp() -> Stamp {
    Stamp {
        run_id: "r384-restart-test".into(),
        model_revision: "model".into(),
        plan_id: "plan".into(),
        layer: 4,
        batch: 9,
        epoch: 3,
    }
}
#[test]
fn native_ledger_child() {
    let Some(path) = std::env::var_os("VF_R384_TEST_LEDGER") else {
        return;
    };
    NativeBatchLedger::open(Path::new(&path))
        .unwrap()
        .reserve(&stamp(), 7)
        .unwrap();
}
#[test]
fn reservation_survives_restart_and_is_exclusive_between_processes() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".runs")
        .join(format!(
            "r384-restart-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    std::fs::create_dir(&path).unwrap();
    let executable = std::env::current_exe().unwrap();
    let spawn = || {
        Command::new(&executable)
            .args(["--exact", "native_ledger_child"])
            .env("VF_R384_TEST_LEDGER", &path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap()
    };
    let mut first = spawn();
    let mut second = spawn();
    let statuses = [
        first.wait().unwrap().success(),
        second.wait().unwrap().success(),
    ];
    assert_ne!(statuses[0], statuses[1]);
    let third = spawn().wait().unwrap();
    assert!(!third.success(), "a fresh process reused a consumed stamp");
    let files = std::fs::read_dir(&path)
        .unwrap()
        .collect::<std::io::Result<Vec<_>>>()
        .unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(
        files[0].file_name(),
        "native-e91d43322e6d3739362f1c16e3b0310fca7c02dcf197a7538fab04eae0fe649b.claim.json"
    );
    let record: serde_json::Value =
        serde_json::from_slice(&std::fs::read(files[0].path()).unwrap()).unwrap();
    assert_eq!(record["status"], "RESERVED_UNKNOWN_NO_REPLAY");
    assert_eq!(record["attempt"], 7);
    assert_eq!(record["stamp"]["batch"], 9);
    assert!(NativeBatchLedger::open(Path::new("relative")).is_err());
}

#[test]
fn incomplete_claim_is_consumed_and_replaced_directory_is_refused() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".runs")
        .join(format!(
            "r384-incomplete-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    std::fs::create_dir(&path).unwrap();
    let ledger = NativeBatchLedger::open(&path).unwrap();
    std::fs::write(
        path.join(
            "native-e91d43322e6d3739362f1c16e3b0310fca7c02dcf197a7538fab04eae0fe649b.claim.json",
        ),
        [],
    )
    .unwrap();
    assert_eq!(
        ledger.reserve(&stamp(), 8).unwrap_err().code,
        "LEDGER_RESERVATION"
    );
    let moved = path.with_extension("original");
    std::fs::rename(&path, moved).unwrap();
    std::fs::create_dir(&path).unwrap();
    assert!(ledger.reserve(&stamp(), 9).is_err());
}
