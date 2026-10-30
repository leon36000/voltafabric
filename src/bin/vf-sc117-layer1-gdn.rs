//! Bound SC117 layer-1 CPU path: residual -> hyperconnection -> QKV -> first GDN step.
use std::io::{Read, Seek, SeekFrom, Write};
use voltafabric::{
    gdn_cpu::{project_gates, GdnState, Identity},
    hyperconnection_cpu,
    native_ggml_cpu::{MatrixBinding, QuantizedMatrix},
    sha256,
};
fn open(path: &str) -> Result<std::fs::File, Box<dyn std::error::Error>> {
    let mut o = std::fs::OpenOptions::new();
    o.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let f = o.open(path)?;
    if !f.metadata()?.is_file() {
        return Err("not a regular file".into());
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
    if a.len() != 4 {
        return Err("usage: vf-sc117-layer1-gdn DENSE SHARD RESIDUAL".into());
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
    let norm = floats(&read(
        &mut d,
        52153216,
        40960,
        "5fe3379c9d308946d00a9b9d36de15838c7da651c3617e7735c236fb81c65e90",
    )?);
    let down = read(
        &mut d,
        45517696,
        6553600,
        "d0f5da6f9ad1d40205a25285ba73168ae9f99558f9852ea58b7c847eae926dec",
    )?;
    let up = read(
        &mut d,
        52194176,
        6553600,
        "61bc67d456c89c310ace1ab98b81723c1afd268923e658a430b5317683ee0438",
    )?;
    let inject = read(
        &mut d,
        52071296,
        81920,
        "0b62d311d080a8a9e65345446d2f0a1d2ae907347eabf581e72fe44605933e3e",
    )?;
    let hc = hyperconnection_cpu::read(&residual, &norm, &down, &up, &inject, 2560, 4, 320, 1e-6)?;
    let weights = read(
        &mut s,
        2090712960,
        18022400,
        "45014df1968838a460302a49ac12dd1ffae22f47a9fb92452783fa7bc4d295c3",
    )?;
    let binding = MatrixBinding {
        model_revision: "sc117-local-observation".into(),
        layer: 1,
        tensor: "blk.1.attn_qkv.weight".into(),
        ggml_type: 13,
        rows: 10240,
        columns: 2560,
        sha256: "45014df1968838a460302a49ac12dd1ffae22f47a9fb92452783fa7bc4d295c3".into(),
    };
    let qkv = QuantizedMatrix::load(binding, &weights)?.forward(&hc.mixed)?;
    let av = floats(&read(
        &mut d,
        137718656,
        192,
        "fccef1b89e2b638a58c7d8d1adee33693c39472d7378fff2f140935b3a8aae90",
    )?);
    let alpha = read(
        &mut d,
        137718848,
        245760,
        "eda02611ab474cd4828e02d4ce583e1e472a0667424630c5eab01efa0d7b7638",
    )?;
    let beta = read(
        &mut d,
        137964608,
        245760,
        "dc357360a7ffb23fdf30ce758f70a7d93d8e3240d7c2c60f9c8f591f6f69dece",
    )?;
    let conv = floats(&read(
        &mut d,
        138210368,
        163840,
        "d477ef4db5392ffd2aba61032fa3dbd71c9ad59e27c8edff31c876f81a0c57d5",
    )?);
    let dt = floats(&read(
        &mut d,
        138374208,
        192,
        "9d08f90b9e49085df92c3e216805f5f743e95a341896dc761fab36df355e2b5c",
    )?);
    let (g, b) = project_gates(&hc.mixed, &alpha, &beta, &dt, &av)?;
    let id = Identity {
        run_id: "r414-token42".into(),
        model_revision: "sc117-local-observation".into(),
        layer: 1,
        epoch: 0,
    };
    let mut st = GdnState::new(id.clone(), 16, 48, 128)?;
    let output = st.step(&id, &qkv, &conv, 1e-6, &g, &b)?;
    #[cfg(unix)]
    if !same(&db, &d.metadata()?) || !same(&sb, &s.metadata()?) {
        return Err("checkpoint changed".into());
    }
    // mixed 2560, inject 4, QKV 10240, recurrent output 6144, gate 48, beta 48.
    let mut w = std::io::BufWriter::new(std::io::stdout().lock());
    for x in hc
        .mixed
        .iter()
        .chain(&hc.inject)
        .chain(&qkv)
        .chain(&output)
        .chain(&g)
        .chain(&b)
    {
        w.write_all(&x.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("layer1 GDN refused: {e}");
        std::process::exit(1);
    }
}
