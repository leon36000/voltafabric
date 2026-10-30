//! Explicit bounded local CPU PLE gather diagnostic; no model execution.
use std::io::Write;
use voltafabric::{sc117_ple::Table, sha256};
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 {
        return Err("usage: ple-gather SHARD2 TOKEN OLDER NEWER".into());
    }
    let token: i32 = a[2].parse()?;
    let previous = [a[3].parse()?, a[4].parse()?];
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let table = Table::open(options.open(&a[1])?, 28_800_138_432, 192)?;
    let (rows, values) = table.gather(token, previous)?;
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    writeln!(
        std::io::stdout().lock(),
        "{}",
        serde_json::json!({"scope":"local CPU PLE gather only", "token":token,
        "previous":previous, "rows":rows, "values":values, "values_sha256":sha256::hex(&bytes)})
    )?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("PLE gather refused: {e}");
        std::process::exit(1);
    }
}
