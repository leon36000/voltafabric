//! Single-rank CPU research check of Session multipart load; not production.
use serde_json::{json, Value};
use std::{
    fs,
    io::{self, Read},
    time::Duration,
};
use voltafabric::{
    moe::Stamp,
    session::{CompressedArtifact, Endpoint, Session},
};
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut s = String::new();
    io::stdin().read_to_string(&mut s)?;
    let c: Value = serde_json::from_str(&s)?;
    let artifact =
        CompressedArtifact::from_bytes(fs::read(c["artifact"].as_str().ok_or("artifact")?)?)
            .map_err(|e| format!("{e:?}"))?;
    let stamp: Stamp = serde_json::from_value(c["stamp"].clone())?;
    let endpoint = Endpoint {
        rank: 2,
        argv: vec![
            c["python"].as_str().ok_or("python")?.into(),
            "-B".into(),
            c["worker"].as_str().ok_or("worker")?.into(),
            c["root"].as_str().ok_or("root")?.into(),
        ],
    };
    let timeout = Duration::from_secs(30);
    let mut session = Session::connect(&endpoint, timeout).map_err(|e| format!("{e:?}"))?;
    let result = session.load_compressed_multipart(&artifact, &stamp, 1, timeout);
    if let Some(expected) = c["expect_failure"].as_str() {
        let error = result.expect_err("expected load refusal");
        let usable = session.is_usable();
        if (expected == "missing_capability" && (error.code != "CAPABILITY_MISSING" || !usable))
            || (expected == "bad_ack" && usable)
        {
            return Err(format!("incorrect refusal state: {error:?}, usable={usable}").into());
        }
        let closed = session.close(timeout);
        println!(
            "{}",
            json!({"status":"EXPECTED_REFUSAL","code":error.code,"usable_before_close":usable,"closed":closed})
        );
        return Ok(());
    }
    let receipt = result.map_err(|e| format!("{e:?}"))?;
    let closed = session.close(timeout);
    if closed.exit_code != Some(0) || closed.bye_execs != Some(0) || closed.error.is_some() {
        return Err(format!("close failed: {closed:?}").into());
    }
    println!(
        "{}",
        json!({"status":"PASS","receipt":receipt,"closed":closed,"gpu":false})
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
