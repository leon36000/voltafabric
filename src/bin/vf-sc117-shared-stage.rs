//! Repeated resident shared-expert check on a pinned activation.
use std::io::{Read, Write};
use voltafabric::{
    gdn_cpu::Identity,
    sc117_shared_stage::{Manifest, Stage},
    sha256,
};
fn open(p: &str) -> Result<std::fs::File, Box<dyn std::error::Error>> {
    let mut o = std::fs::OpenOptions::new();
    o.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let f = o.open(p)?;
    if !f.metadata()?.is_file() {
        return Err("not regular".into());
    }
    Ok(f)
}
fn exact(p: &str, n: usize, h: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut f = open(p)?;
    if f.metadata()?.len() != n as u64 || !sha256::is_hex_digest(h) {
        return Err("input size or digest syntax".into());
    }
    let mut b = vec![0; n];
    f.read_exact(&mut b)?;
    if sha256::hex(&b) != h {
        return Err("input digest mismatch".into());
    }
    Ok(b)
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("usage: shared MANIFEST MANIFEST_SHA SHARD INPUT INPUT_SHA".into());
    }
    let n = open(&a[1])?.metadata()?.len();
    if n == 0 || n > 16384 {
        return Err("manifest length".into());
    }
    let m: Manifest = serde_json::from_slice(&exact(&a[1], n as usize, &a[2])?)?;
    let input: Vec<f32> = exact(&a[4], 10240, &a[5])?
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let id = Identity {
        run_id: "diagnostic-only".into(),
        model_revision: m.model_revision.clone(),
        layer: m.layer,
        epoch: 0,
    };
    let mut shard = open(&a[3])?;
    let mut stage = Stage::load(m, id.clone(), &mut shard)?;
    let mut wrong = id.clone();
    wrong.epoch = 1;
    if stage.compute(&wrong, &input).is_ok() {
        return Err("wrong epoch accepted".into());
    }
    if stage.compute(&id, &input[..2559]).is_ok() {
        return Err("wrong width accepted".into());
    }
    let mut bad = input.clone();
    bad[0] = f32::NAN;
    if stage.compute(&id, &bad).is_ok() {
        return Err("nonfinite accepted".into());
    }
    let out = stage.compute(&id, &input)?;
    if stage.compute(&id, &input)? != out {
        return Err("non-deterministic repeated compute".into());
    }
    let mut w = std::io::BufWriter::new(std::io::stdout().lock());
    for v in out {
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("resident shared refused: {e}");
        std::process::exit(1);
    }
}
