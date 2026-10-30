//! Fresh multipart fleet CPU load test; no expert EXEC.
use serde_json::{json, Value};
use std::{
    fs,
    io::{self, Read},
    time::Duration,
};
use voltafabric::{
    routing::RoutingProblem,
    session::{prepare_compressed_multipart, CompressedArtifact, Endpoint},
};
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::new();
    io::stdin().read_to_string(&mut text)?;
    let c: Value = serde_json::from_str(&text)?;
    let p: RoutingProblem = serde_json::from_value(c["routing"].clone())?;
    let workers: Vec<Endpoint> = serde_json::from_value(c["workers"].clone())?;
    let artifacts: Vec<CompressedArtifact> = c["artifacts"]
        .as_array()
        .ok_or("artifacts")?
        .iter()
        .map(|v| {
            let path = v.as_str().ok_or("path")?;
            let bytes = fs::read(path).map_err(|_| "read artifact")?;
            CompressedArtifact::from_bytes(bytes).map_err(|_| "invalid artifact")
        })
        .collect::<Result<_, _>>()?;
    let timeout = Duration::from_secs(30);
    let result = prepare_compressed_multipart(&workers, &artifacts, &p, 1, timeout);
    if c["expect_missing_capability"] == true {
        match result {
            Err(e) if e.code == "CAPABILITY_MISSING" => {
                println!("{}", json!({"status":"EXPECTED_REFUSAL","code":e.code}));
                return Ok(());
            }
            _ => return Err("fleet preflight did not reject missing capability".into()),
        }
    }
    if c["expect_bad_ack"] == true {
        match result {
            Err(e) if e.code == "INVALID_INPUT" => {
                println!(
                    "{}",
                    json!({"status":"EXPECTED_FLEET_FAILURE","code":e.code})
                );
                return Ok(());
            }
            _ => return Err("fleet did not reject bad acknowledgement".into()),
        }
    }
    let sessions = result.map_err(|e| format!("{e:?}"))?;
    let mut closes = Vec::new();
    for s in sessions {
        let c = s.close(timeout);
        if c.exit_code != Some(0) || c.bye_execs != Some(0) || c.error.is_some() {
            return Err(format!("bad close {c:?}").into());
        }
        closes.push(c);
    }
    println!("{}", json!({"status":"PASS","closed":closes,"gpu":false}));
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
