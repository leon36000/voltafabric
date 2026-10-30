//! Explicit CPU diagnostic for the first attention-side hyperconnection read.
use std::io::{Read, Seek, SeekFrom};
use voltafabric::{hyperconnection_cpu, sha256};
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
        return Err("expected regular file".into());
    }
    Ok(file)
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 7 {
        return Err(
            "usage: vf-sc117-first-hc DENSE EMBEDDING NORM_SHA DOWN_SHA UP_SHA INJECT_SHA".into(),
        );
    }
    let mut f = open_regular(&args[1])?;
    let before = f.metadata()?;
    if !before.is_file() || before.len() != 1_538_035_200 {
        return Err("unexpected dense pack".into());
    }
    let spans = [
        (22_410_240_u64, 40_960_usize),
        (15_774_720, 6_553_600),
        (22_451_200, 6_553_600),
        (22_328_320, 81_920),
    ];
    let mut parts = Vec::new();
    for ((offset, len), expected) in spans.into_iter().zip(&args[3..]) {
        f.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0; len];
        f.read_exact(&mut bytes)?;
        if !sha256::is_hex_digest(expected) || sha256::hex(&bytes) != *expected {
            return Err("hyperconnection digest mismatch".into());
        }
        parts.push(bytes);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let after = f.metadata()?;
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
    let mut embedding = open_regular(&args[2])?;
    if embedding.metadata()?.len() != 2560 * 4 {
        return Err("embedding must contain exactly 2560 FP32 values".into());
    }
    let mut embed = vec![0; 2560 * 4];
    embedding.read_exact(&mut embed)?;
    let x: Vec<f32> = embed
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let mut residual = Vec::with_capacity(10240);
    for _ in 0..4 {
        residual.extend_from_slice(&x);
    }
    let gamma: Vec<f32> = parts[0]
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let result = hyperconnection_cpu::read(
        &residual, &gamma, &parts[1], &parts[2], &parts[3], 2560, 4, 320, 1e-6,
    )?;
    println!(
        "{}",
        serde_json::json!({"mixed":result.mixed,"inject":result.inject,"scope":"CPU first attention hyperconnection; no attention executed"})
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("hyperconnection refused: {e}");
        std::process::exit(1);
    }
}
