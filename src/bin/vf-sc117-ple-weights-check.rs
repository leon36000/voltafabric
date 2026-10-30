//! Bounded CPU diagnostic; synthetic residual and history, not generation.
use std::{fs::File, io::BufReader};
use voltafabric::{
    sc117_ple_post::{CHANNELS, HISTORY},
    sc117_ple_weights::{Manifest, Weights},
};
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: check MANIFEST SHARD GATHER_JSON".into());
    }
    let manifest: Manifest = serde_json::from_reader(BufReader::new(File::open(&args[1])?))?;
    let weights = Weights::load(
        &mut File::open(&args[2])?,
        &manifest,
        "sc117-local-observation",
    )?;
    let gather: serde_json::Value = serde_json::from_reader(BufReader::new(File::open(&args[3])?))?;
    let emb: Vec<f32> = serde_json::from_value(gather["values"].clone())?;
    let hidden: Vec<f32> = (0..CHANNELS)
        .map(|i| (i as i32 % 101 - 50) as f32 / 100.0)
        .collect();
    let history: Vec<f32> = (0..CHANNELS * HISTORY)
        .map(|i| (i as i32 % 37 - 18) as f32 / 50.0)
        .collect();
    let out = weights.evaluate(&emb, &hidden, &history)?;
    serde_json::to_writer(
        std::io::stdout(),
        &serde_json::json!({"residual":out.residual,"normalized":out.normalized,"next_history":out.next_history}),
    )?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
