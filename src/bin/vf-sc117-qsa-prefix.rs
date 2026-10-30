//! Observed first-token layer-3 QSA prefix only, not a complete layer.
use std::io::{Read, Write};
use voltafabric::{
    gdn_cpu::Identity,
    sc117_qsa_stage::{Manifest, Stage},
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
        return Err("usage: qsa MANIFEST DENSE SHARD RESIDUAL RESIDUAL_SHA".into());
    }
    let n = open(&a[1])?.metadata()?.len();
    if n == 0 || n > 16384 {
        return Err("manifest size".into());
    }
    let m: Manifest = serde_json::from_slice(&exact(
        &a[1],
        n as usize,
        "f0c4bdadf42bb502195a7314d3d73661d63687ca161e420e28381a45c4ddbd21",
    )?)?;
    let bytes = exact(&a[4], 40960, &a[5])?;
    let residual: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let id = Identity {
        run_id: "r430-cpu-prefix".into(),
        model_revision: m.model_revision.clone(),
        layer: 3,
        epoch: 1,
    };
    let mut d = open(&a[2])?;
    let mut s = open(&a[3])?;
    let mut stage = Stage::load(m, id.clone(), &mut d, &mut s, 2)?;
    let candidate = stage.prepare(&id, 0, &residual)?;
    let stale = stage.prepare(&id, 0, &residual)?;
    if !stage.is_empty() {
        return Err("candidate changed KV".into());
    }
    let out = stage.commit(candidate)?;
    if stage.commit(stale).is_ok() || stage.len() != 1 {
        return Err("stale cache committed".into());
    }
    let mut w = std::io::BufWriter::new(std::io::stdout().lock());
    for x in out
        .mixed
        .iter()
        .chain(&out.inject)
        .chain(&out.raw_query_gate)
        .chain(&out.query)
        .chain(&out.gate)
        .chain(&out.raw_key)
        .chain(&out.key)
        .chain(&out.value)
        .chain(&out.gated_attention)
        .chain(&out.attention)
    {
        w.write_all(&x.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("qsa prefix refused: {e}");
        std::process::exit(1);
    }
}
