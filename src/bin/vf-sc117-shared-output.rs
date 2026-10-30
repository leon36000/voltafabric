//! Explicit single observed SC117 token shared-expert CPU diagnostic.
use std::io::{Read, Seek, SeekFrom, Write};
use voltafabric::{
    hyperconnection_cpu,
    native_ggml_cpu::{MatrixBinding, QuantizedMatrix},
    sha256, shared_expert_cpu,
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
    off: u64,
    len: usize,
    hash: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    f.seek(SeekFrom::Start(off))?;
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
#[allow(clippy::too_many_arguments)]
fn matrix(
    shard: &mut std::fs::File,
    off: u64,
    len: usize,
    hash: &str,
    tensor: &str,
    kind: i32,
    rows: usize,
    cols: usize,
    x: &[f32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = read(shard, off, len, hash)?;
    let b = MatrixBinding {
        model_revision: "sc117-local-observation".into(),
        layer: 0,
        tensor: tensor.into(),
        ggml_type: kind,
        rows,
        columns: cols,
        sha256: hash.into(),
    };
    Ok(QuantizedMatrix::load(b, &bytes)?.forward(x)?)
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 4 {
        return Err("usage: vf-sc117-shared-output SHARD ROUTER_INPUT COLLECTOR_JSON".into());
    }
    let mut shard = open(&a[1])?;
    let before = shard.metadata()?;
    if before.len() != 55_063_448_064 {
        return Err("shard size".into());
    }
    let mut input = open(&a[2])?;
    if input.metadata()?.len() != 51216 {
        return Err("input size".into());
    }
    let input = floats(&read(
        &mut input,
        0,
        51216,
        "2a49c7676a10972d3d3b887314968191cdf9d42aca352befd044fb8538fac7c4",
    )?);
    let mut cf = open(&a[3])?;
    if cf.metadata()?.len() != 57891 {
        return Err("collector size".into());
    }
    let raw = read(
        &mut cf,
        0,
        57891,
        "b5e1c63918b06800147224f112becd58e1d2ccc6dab3b190659df6f830b46e52",
    )?;
    let v: serde_json::Value = serde_json::from_slice(&raw)?;
    if v["ok"].as_bool() != Some(true) || v["result"]["accepted_rows"].as_u64() != Some(10) {
        return Err("collector evidence".into());
    }
    let row = v["result"]["output"][0]
        .as_array()
        .ok_or("missing collector output")?;
    if row.len() != 2560 {
        return Err("collector width".into());
    }
    let routed = row
        .iter()
        .map(|x| x.as_f64().ok_or("collector value"))
        .collect::<Result<Vec<_>, _>>()?;
    let ginp = read(
        &mut shard,
        1_711_058_944,
        5120,
        "a43a13457ded04d02a6ff9904c177b383e2d52ea34e3637c84e63dbb59d7dd45",
    )?;
    let scalar = shared_expert_cpu::scalar_gate(&input[..2560], &ginp)?;
    let gate = matrix(
        &mut shard,
        1_711_064_064,
        921600,
        "49827a29188f36738cfe7382f21d2c7d3ec68e4c0a1a91210134adc95eb0ddfa",
        "blk.0.ffn_gate_shexp.weight",
        12,
        640,
        2560,
        &input[..2560],
    )?;
    let up = matrix(
        &mut shard,
        2_033_112_064,
        870400,
        "b9133ea1a06cbd944bdb592f080812363dd64c2488c2e4e54cdf7fa40ed1dfab",
        "blk.0.ffn_up_shexp.weight",
        23,
        640,
        2560,
        &input[..2560],
    )?;
    let mid = shared_expert_cpu::activation(&gate, &up)?;
    let down = matrix(
        &mut shard,
        1_385_570_304,
        1740800,
        "24b490ee53349b9fff27e1f966663a335dfd59ca9f3979563196e124d185ab5e",
        "blk.0.ffn_down_shexp.weight",
        8,
        2560,
        640,
        &mid,
    )?;
    let shared: Vec<f32> = down.iter().map(|&x| x * scalar).collect();
    let combined = shared_expert_cpu::combine(&routed, &shared)?;
    let residual = hyperconnection_cpu::write(&input[2564..], &combined, &input[2560..2564])?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let after = shard.metadata()?;
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
            return Err("shard changed".into());
        }
    }
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for x in gate
        .iter()
        .chain(&up)
        .chain(&mid)
        .chain(&down)
        .chain(&shared)
        .chain(&combined)
        .chain(&residual)
        .chain(std::iter::once(&scalar))
    {
        out.write_all(&x.to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("shared output refused: {e}");
        std::process::exit(1);
    }
}
