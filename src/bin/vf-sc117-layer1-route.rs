//! Layer-1 CPU attention output and FFN route over fixed SC117 observations.
use std::io::{Read, Seek, SeekFrom, Write};
use voltafabric::{
    gdn_cpu::output_norm,
    hyperconnection_cpu,
    native_ggml_cpu::{MatrixBinding, QuantizedMatrix},
    sc117_router_cpu, sha256,
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
fn read(
    f: &mut std::fs::File,
    o: u64,
    n: usize,
    h: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    f.seek(SeekFrom::Start(o))?;
    let mut b = vec![0; n];
    f.read_exact(&mut b)?;
    if sha256::hex(&b) != h {
        return Err("digest mismatch".into());
    }
    Ok(b)
}
fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}
#[allow(clippy::too_many_arguments)]
fn matrix(
    s: &mut std::fs::File,
    o: u64,
    n: usize,
    h: &str,
    name: &str,
    kind: i32,
    rows: usize,
    cols: usize,
    x: &[f32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let b = read(s, o, n, h)?;
    let id = MatrixBinding {
        model_revision: "sc117-local-observation".into(),
        layer: 1,
        tensor: name.into(),
        ggml_type: kind,
        rows,
        columns: cols,
        sha256: h.into(),
    };
    Ok(QuantizedMatrix::load(id, &b)?.forward(x)?)
}
#[cfg(unix)]
fn same(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    (
        a.dev(),
        a.ino(),
        a.len(),
        a.mtime(),
        a.mtime_nsec(),
        a.ctime(),
        a.ctime_nsec(),
    ) == (
        b.dev(),
        b.ino(),
        b.len(),
        b.mtime(),
        b.mtime_nsec(),
        b.ctime(),
        b.ctime_nsec(),
    )
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 {
        return Err("usage: vf-sc117-layer1-route DENSE SHARD RESIDUAL GDN".into());
    }
    let mut d = open(&a[1])?;
    let db = d.metadata()?;
    if db.len() != 1_538_035_200 {
        return Err("dense size".into());
    }
    let mut s = open(&a[2])?;
    let sb = s.metadata()?;
    if sb.len() != 55_063_448_064 {
        return Err("shard size".into());
    }
    let mut r = open(&a[3])?;
    if r.metadata()?.len() != 40960 {
        return Err("residual size".into());
    }
    let residual = floats(&read(
        &mut r,
        0,
        40960,
        "f929a224afaf2832954242730520d71845cd16be3184b7ec0e7c192bed4bb0e8",
    )?);
    let mut g = open(&a[4])?;
    if g.metadata()?.len() != 76176 {
        return Err("GDN size".into());
    }
    let gdn = floats(&read(
        &mut g,
        0,
        76176,
        "f94d83875a943a0c2ffc3f196568ea96c4f26d699117fd6bec0baec277f7e86b",
    )?);
    let gamma = floats(&read(
        &mut d,
        138374400,
        512,
        "9f23158f98b697f85734f0c28a16ced888c5b71719b382636c1944eab4727dbe",
    )?);
    let z = matrix(
        &mut s,
        2077810560,
        12902400,
        "b37eed095b749a8847acd1fe220f7861e3e6d7893ad3bb43f0ebdf9c2c03aed8",
        "blk.1.attn_gate.weight",
        14,
        6144,
        2560,
        &gdn[..2560],
    )?;
    let y = output_norm(&gdn[12804..18948], &z, &gamma, 1e-6)?;
    let attn = matrix(
        &mut s,
        2981076736,
        16711680,
        "328fc3db6f58aa925912e00c5ab4d0f908f05b5373e76ca63f23358df86b8933",
        "blk.1.ssm_out.weight",
        8,
        2560,
        6144,
        &y,
    )?;
    let post = hyperconnection_cpu::write(&residual, &attn, &gdn[2560..2564])?;
    let norm = floats(&read(
        &mut d,
        65383296,
        40960,
        "7bdbcfdca1c995fbadd41c1001adeafe812a9f33cbacecee210b49951965a78c",
    )?);
    let down = read(
        &mut d,
        58747776,
        6553600,
        "2e8db98f0ed1142e4a8eb6c378b6b5d5d9c345ac677f4644aff5b2c8ec0b3e5e",
    )?;
    let up = read(
        &mut d,
        65424256,
        6553600,
        "6ce3cd925265b0ef3f852b1f4d0dc524a44dcdf7544863807fb0679e145909ff",
    )?;
    let inj = read(
        &mut d,
        65301376,
        81920,
        "7b6a31277d07b99a56612c7440b8ed8480f323ba2dbb7954925c9866f1da012a",
    )?;
    let hc = hyperconnection_cpu::read(&post, &norm, &down, &up, &inj, 2560, 4, 320, 1e-6)?;
    let rw = read(
        &mut d,
        42891136,
        2621440,
        "0721144c6eed2441e5cee395d2aa2700d8f9365f0b05d7dbd553003ffc4caab6",
    )?;
    let route = sc117_router_cpu::route(&rw, &hc.mixed, 10)?;
    #[cfg(unix)]
    if !same(&db, &d.metadata()?) || !same(&sb, &s.metadata()?) {
        return Err("checkpoint changed".into());
    }
    // z 6144, y 6144, attention 2560, post 10240, mixed 2560, injection 4; 10*(u32,f32,f32) route.
    let mut w = std::io::BufWriter::new(std::io::stdout().lock());
    for x in z
        .iter()
        .chain(&y)
        .chain(&attn)
        .chain(&post)
        .chain(&hc.mixed)
        .chain(&hc.inject)
    {
        w.write_all(&x.to_le_bytes())?;
    }
    for c in route {
        w.write_all(&c.expert.to_le_bytes())?;
        w.write_all(&c.weight.to_le_bytes())?;
        w.write_all(&c.logit.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("layer1 route refused: {e}");
        std::process::exit(1);
    }
}
