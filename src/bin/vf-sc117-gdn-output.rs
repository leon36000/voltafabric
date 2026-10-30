//! Local SC117 layer-0 GDN output diagnostic, using pinned observed bytes.
use std::io::{Read, Seek, SeekFrom, Write};
use voltafabric::{
    gdn_cpu::output_norm,
    native_ggml_cpu::{MatrixBinding, QuantizedMatrix},
    sha256,
};
fn open(path: &str) -> Result<std::fs::File, Box<dyn std::error::Error>> {
    let mut o = std::fs::OpenOptions::new();
    o.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let f = o.open(path)?;
    if !f.metadata()?.is_file() {
        return Err("not a regular file".into());
    }
    Ok(f)
}
fn read(
    f: &mut std::fs::File,
    offset: u64,
    len: usize,
    hash: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    f.seek(SeekFrom::Start(offset))?;
    let mut b = vec![0; len];
    f.read_exact(&mut b)?;
    if sha256::hex(&b) != hash {
        return Err("digest mismatch".into());
    }
    Ok(b)
}
fn floats(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
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
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: vf-sc117-gdn-output SHARD DENSE MIXED GDN_RESULT".into());
    }
    let mut shard = open(&args[1])?;
    let before = shard.metadata()?;
    if before.len() != 55_063_448_064 {
        return Err("shard size".into());
    }
    let mut dense = open(&args[2])?;
    let dense_before = dense.metadata()?;
    if dense_before.len() != 1_538_035_200 {
        return Err("dense size".into());
    }
    let mut mixed = open(&args[3])?;
    if mixed.metadata()?.len() != 10240 {
        return Err("mixed size".into());
    }
    let mut recurrence = open(&args[4])?;
    if recurrence.metadata()?.len() != 24960 {
        return Err("recurrence size".into());
    }
    let x = floats(&read(
        &mut mixed,
        0,
        10240,
        "9df9ccdc3759a810606a4bc9594177cb4c9eff411c123cd5b196200128f5ce4f",
    )?);
    let o = floats(&read(
        &mut recurrence,
        0,
        24960,
        "d8aa66b1a54d1bd08f9a5969df5fdeffb86330bcebee641d280d55a85fb7c2ba",
    )?);
    let gamma = floats(&read(
        &mut dense,
        42_890_624,
        512,
        "d27dd7b8d5dae1f3cc9880dd0367825008d8d7873ea0030aa92175634ae08e17",
    )?);
    let gate_hash = "2d591eddd7a04973f8306d8eb937f2c9593b02d558e093a7d3afdd2f85d9ada8";
    let gate = read(&mut shard, 883_359_744, 8_847_360, gate_hash)?;
    let binding = MatrixBinding {
        model_revision: "sc117-local-observation".into(),
        layer: 0,
        tensor: "blk.0.attn_gate.weight".into(),
        ggml_type: 12,
        rows: 6144,
        columns: 2560,
        sha256: gate_hash.into(),
    };
    let z = QuantizedMatrix::load(binding, &gate)?.forward(&x)?;
    let y = output_norm(&o[..6144], &z, &gamma, 1e-6)?;
    let out_hash = "d6255a7089c51a87f187c809833ba14619c96654f0c47f0b0a2b658096e8638b";
    let weights = read(&mut shard, 2_061_098_880, 16_711_680, out_hash)?;
    let binding = MatrixBinding {
        model_revision: "sc117-local-observation".into(),
        layer: 0,
        tensor: "blk.0.ssm_out.weight".into(),
        ggml_type: 8,
        rows: 2560,
        columns: 6144,
        sha256: out_hash.into(),
    };
    let output = QuantizedMatrix::load(binding, &weights)?.forward(&y)?;
    #[cfg(unix)]
    if !same(&before, &shard.metadata()?) || !same(&dense_before, &dense.metadata()?) {
        return Err("checkpoint changed during observation".into());
    }
    let mut result = std::io::BufWriter::new(std::io::stdout().lock());
    for v in z.iter().chain(&y).chain(&output) {
        result.write_all(&v.to_le_bytes())?;
    }
    result.flush()?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("SC117 GDN output refused: {e}");
        std::process::exit(1);
    }
}
