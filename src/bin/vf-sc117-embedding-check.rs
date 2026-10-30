//! Bounded diagnostic; source and metadata must be authenticated by the caller.
use voltafabric::{sc117_embedding::Table, sha256};
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 3 {
        return Err("usage: check SHARD TOKEN".into());
    }
    let token: u32 = a[2].parse()?;
    let table = Table::open(std::fs::File::open(&a[1])?, 55_063_448_064, 545_644_544)?;
    let values = table.gather(token)?;
    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    serde_json::to_writer(
        std::io::stdout(),
        &serde_json::json!({"token":token,"values_sha256":sha256::hex(&bytes),"values":values}),
    )?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
