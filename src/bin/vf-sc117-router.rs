//! Explicit CPU reference for a locally pinned SC117 router tensor.
use std::io::{Read, Seek, SeekFrom};
use voltafabric::{sc117_router_cpu, sha256};
fn open_regular(path: &str) -> Result<std::fs::File, Box<dyn std::error::Error>> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err("expected a regular file".into());
    }
    Ok(file)
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: vf-sc117-router DENSE INPUT_F32_FILE ROUTER_SHA256".into());
    }
    let mut dense = open_regular(&args[1])?;
    let before = dense.metadata()?;
    if !before.is_file() || before.len() != 1_538_035_200 {
        return Err("unexpected dense pack".into());
    }
    dense.seek(SeekFrom::Start(13_148_160))?;
    let mut bytes = vec![0; sc117_router_cpu::WEIGHT_BYTES];
    dense.read_exact(&mut bytes)?;
    if !sha256::is_hex_digest(&args[3]) || sha256::hex(&bytes) != args[3] {
        return Err("router digest mismatch".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let after = dense.metadata()?;
        if (
            before.dev(),
            before.ino(),
            before.len(),
            before.mtime(),
            before.mtime_nsec(),
            before.ctime(),
            before.ctime_nsec(),
        ) != (
            after.dev(),
            after.ino(),
            after.len(),
            after.mtime(),
            after.mtime_nsec(),
            after.ctime(),
            after.ctime_nsec(),
        ) {
            return Err("dense pack changed during read".into());
        }
    }
    let mut input = open_regular(&args[2])?;
    let mut raw = vec![0_u8; sc117_router_cpu::HIDDEN * 4];
    input.read_exact(&mut raw)?;
    let x: Vec<f32> = raw
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let result = sc117_router_cpu::route(&bytes, &x, 10)?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("router refused: {e}");
        std::process::exit(1);
    }
}
