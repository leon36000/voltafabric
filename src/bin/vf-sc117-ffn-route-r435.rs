//! Layer-3 post-attention route diagnostic, without expert execution.
use std::io::{Read, Write};
use voltafabric::{
    gdn_cpu::Identity,
    sc117_ffn_stage::{Manifest, Stage},
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
    if a.len() != 5 {
        return Err("usage: ffn MANIFEST DENSE RESIDUAL QSA".into());
    }
    let n = open(&a[1])?.metadata()?.len();
    if n == 0 || n > 16384 {
        return Err("manifest size".into());
    }
    let m: Manifest = serde_json::from_slice(&exact(
        &a[1],
        n as usize,
        "79d44d7663231fea2a6c04c309d776cd9552af7d0ecb859c6428b8124351afd1",
    )?)?;
    let f = |b: Vec<u8>| -> Vec<f32> {
        b.chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect()
    };
    let residual = f(exact(
        &a[3],
        40960,
        "a4c6c803a64f367f0b5f4426321c2c7a2bc3f439ac42a58b5e8c696dc05cb38f",
    )?);
    let q = f(exact(
        &a[4],
        149520,
        "3025b1164864efa99f44a5b2d752c0e4a717d45afc77abd50a5aa5ef5eaa4e6e",
    )?);
    let id = Identity {
        run_id: "r435-route-only".into(),
        model_revision: m.model_revision.clone(),
        layer: 3,
        epoch: 1,
    };
    let mut d = open(&a[2])?;
    let stage = Stage::load(m, id.clone(), &mut d)?;
    let route = stage.compute(&id, &residual, &q[34820..37380], &q[2560..2564])?;
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for x in route
        .residual
        .iter()
        .chain(&route.mixed)
        .chain(&route.inject)
    {
        out.write_all(&x.to_le_bytes())?;
    }
    for c in route.choices {
        out.write_all(&c.expert.to_le_bytes())?;
        out.write_all(&c.weight.to_le_bytes())?;
        out.write_all(&c.logit.to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("FFN route refused: {e}");
        std::process::exit(1);
    }
}
