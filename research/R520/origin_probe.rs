//! Read-only check of one locally observed source expert; no worker or ledger.
use std::{fs::File, path::Path};
use voltafabric::{sc117_expert_origin::Origin, strata_native_layout::NativeLayout};
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 || !args[1..].iter().all(|p| Path::new(p).is_absolute()) {
        return Err("absolute source and index required".into());
    }
    let index = std::fs::read_to_string(&args[2])?;
    let layout = NativeLayout::parse_sc117_v3(&index, 50_292_326_400)?;
    let mut origin = Origin::open(File::open(&args[1])?, &layout)?;
    let expected = "8a17e1a2a5f9a21e11d2bae64fc99a656c810e8ed16d15bed9a09800fccd9cb5";
    let first = origin.expert_digest(&layout, 2, 255)?;
    let second = origin.expert_digest(&layout, 2, 255)?;
    if first != expected || second != expected || origin.digest_cache_stats() != (1, 1, 1) {
        return Err("digest or cache counters mismatch".into());
    }
    println!(
        "{}",
        serde_json::json!({"layer":2,"expert":255,"sha256":first,"hits":1,"misses":1,"entries":1,"scope":"local source consistency, not publisher authentication"})
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("origin probe refused: {e}");
        std::process::exit(1);
    }
}
