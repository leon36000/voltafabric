//! Local CPU head diagnostic; input is not claimed to be the final model layer.
use std::io::{Read, Write};
use voltafabric::{
    gdn_cpu::Identity,
    sc117_head_cpu::{Head, Manifest},
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
    if a.len() != 7 {
        return Err("usage: head MANIFEST SHA DENSE SHARD RESIDUAL RESIDUAL_SHA".into());
    }
    let n = open(&a[1])?.metadata()?.len();
    if n == 0 || n > 16384 {
        return Err("manifest size".into());
    }
    let m: Manifest = serde_json::from_slice(&exact(&a[1], n as usize, &a[2])?)?;
    let input = exact(&a[5], 40960, &a[6])?;
    let residual: Vec<f32> = input
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let id = Identity {
        run_id: "r429-head-only".into(),
        model_revision: m.model_revision.clone(),
        layer: m.layer_count,
        epoch: 0,
    };
    let mut d = open(&a[3])?;
    let mut s = open(&a[4])?;
    let mut head = Head::load(m, id.clone(), &mut d, &mut s)?;
    let mut wrong = id.clone();
    wrong.epoch = 1;
    if head.evaluate(&wrong, &residual).is_ok() {
        return Err("wrong epoch accepted".into());
    }
    let o = head.evaluate(&id, &residual)?;
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for x in o.mixed.iter().chain(&o.logits) {
        out.write_all(&x.to_le_bytes())?;
    }
    out.write_all(&o.greedy.to_le_bytes())?;
    out.flush()?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("head refused: {e}");
        std::process::exit(1);
    }
}
