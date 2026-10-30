//! One-shot validation of the reusable resident post-attention stage.
use std::io::{Read, Write};
use voltafabric::{
    gdn_cpu::Identity,
    sc117_gdn_stage::Output,
    sc117_post_attention::{Manifest, Stage},
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
    if a.len() != 9 {
        return Err("usage: post MANIFEST SHA DENSE SHARD RESIDUAL SHA PREFIX SHA".into());
    }
    let n = open(&a[1])?.metadata()?.len();
    if n == 0 || n > 16384 {
        return Err("manifest size".into());
    }
    let m: Manifest = serde_json::from_slice(&exact(&a[1], n as usize, &a[2])?)?;
    let floats = |b: Vec<u8>| -> Vec<f32> {
        b.chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect()
    };
    let residual = floats(exact(&a[5], 40960, &a[6])?);
    let g = floats(exact(&a[7], 76176, &a[8])?);
    let prefix = Output {
        mixed: g[..2560].to_vec(),
        inject: g[2560..2564].to_vec(),
        qkv: g[2564..12804].to_vec(),
        recurrent: g[12804..18948].to_vec(),
        gate: g[18948..18996].to_vec(),
        beta: g[18996..19044].to_vec(),
    };
    let id = Identity {
        run_id: "diagnostic-only".into(),
        model_revision: m.model_revision.clone(),
        layer: m.layer,
        epoch: 0,
    };
    let mut d = open(&a[3])?;
    let mut s = open(&a[4])?;
    let mut stage = Stage::load(m, id.clone(), &mut d, &mut s)?;
    let mut wrong = id.clone();
    wrong.epoch = 1;
    if stage.compute(&wrong, &prefix, &residual).is_ok() {
        return Err("wrong identity accepted".into());
    }
    let out = stage.compute(&id, &prefix, &residual)?;
    let mut w = std::io::BufWriter::new(std::io::stdout().lock());
    for x in out
        .z
        .iter()
        .chain(&out.normalized)
        .chain(&out.attention)
        .chain(&out.residual)
        .chain(&out.mixed)
        .chain(&out.inject)
    {
        w.write_all(&x.to_le_bytes())?;
    }
    for c in out.choices {
        w.write_all(&c.expert.to_le_bytes())?;
        w.write_all(&c.weight.to_le_bytes())?;
        w.write_all(&c.logit.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("post stage refused: {e}");
        std::process::exit(1);
    }
}
